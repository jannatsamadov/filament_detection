"""Turn spine / radius / semantic maps into filament instances.

    spine map -> threshold -> connected components  (one component, one filament)
             -> order the pixels into a path
             -> smoothing spline
             -> sweep a disk of the predicted radius along it
             -> intersect with the semantic mask

Each instance is a swept curve, so it is a single connected region by
construction: the fragmentation that Panoptic Quality punishes hardest cannot
occur. There are no anchors and no NMS -- separation comes from the spines
themselves, which are further apart than the masks (0.9% of spines come within
10 px of another filament's spine, against 2.8% of masks).

`assemble` is deliberately independent of any model, so it can be validated by
feeding it the ground-truth maps; see check_assembly().
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, binary_dilation, distance_transform_edt
from scipy.interpolate import splprep, splev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FULL = 2048


def order_path(points, max_points=400):
    """Order scattered centreline pixels into a path.

    Points are first sorted along the principal axis, which already orders a
    gently curved ribbon correctly, then thinned to at most `max_points` and
    refined by greedy nearest-neighbour, which handles the curvature the
    projection gets wrong.

    The thinning is not an optimisation detail. Greedy nearest-neighbour is
    O(n^2), and an untrained model produces spine blobs of tens of thousands of
    pixels rather than the few hundred a real spine has -- that combination hung
    the first validation pass with the GPU idle and no output. A ground-truth
    spine never triggers it, so it only appears once a model is in the loop.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 2:
        return pts

    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    order = np.argsort(c @ vt[0])
    pts = pts[order]

    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    remaining = list(range(1, len(pts)))
    path, cur = [0], 0
    while remaining:
        d = np.linalg.norm(pts[remaining] - pts[cur], axis=1)
        k = int(np.argmin(d))
        cur = remaining.pop(k)
        path.append(cur)
    return pts[path]


def fit_spline(path, smooth_per_point=2.0, n_out=None):
    """Smoothing spline through an ordered path; returns resampled points."""
    if len(path) < 4:
        return path
    x, y = path[:, 0], path[:, 1]
    keep = np.r_[True, (np.diff(x) ** 2 + np.diff(y) ** 2) > 1e-6]
    x, y = x[keep], y[keep]
    if len(x) < 4:
        return np.c_[x, y]
    k = min(3, len(x) - 1)
    try:
        tck, _ = splprep([x, y], s=smooth_per_point * len(x), k=k)
    except Exception:
        return np.c_[x, y]
    length = np.hypot(np.diff(x), np.diff(y)).sum()
    n = n_out or max(int(length), 8)
    u = np.linspace(0, 1, n)
    xs, ys = splev(u, tck)
    return np.c_[xs, ys]


def sweep(curve, radii, shape=(FULL, FULL)):
    """Draw a disk of the given radius at each curve point."""
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    for (x, y), r in zip(curve, radii):
        if r < 0.5:
            r = 0.5
        dr.ellipse([x - r, y - r, x + r, y + r], fill=1)
    return np.asarray(im, bool)


def assemble(spine_prob, radius_map, sem_prob, spine_thresh=0.4, sem_thresh=0.5,
             min_spine_px=12, min_area=60, smooth=2.0, dilate_sem=2,
             radius_scale=1.0):
    """Return a list of boolean instance masks.

    sem_prob gates the result: a swept disk is trimmed to where the semantic head
    also believes there is filament, which keeps the width honest where the
    radius head is unsure. The semantic mask is dilated slightly first so that a
    correct spine is not eroded by a marginally tight semantic boundary.
    """
    spine_bin = spine_prob >= spine_thresh
    if not spine_bin.any():
        return []
    lab, n = label(spine_bin, structure=np.ones((3, 3)))
    sem_bin = sem_prob >= sem_thresh
    if dilate_sem:
        sem_bin = binary_dilation(sem_bin, iterations=dilate_sem)

    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_spine_px:
            continue
        path = order_path(np.c_[xs, ys])
        curve = fit_spline(path, smooth_per_point=smooth)
        cx = np.clip(curve[:, 0].astype(int), 0, radius_map.shape[1] - 1)
        cy = np.clip(curve[:, 1].astype(int), 0, radius_map.shape[0] - 1)
        radii = radius_map[cy, cx] * radius_scale
        m = sweep(curve, radii, shape=spine_prob.shape)
        m &= sem_bin
        if m.sum() >= min_area:
            out.append(m)
    return out


# ----------------------------------------------------------------------------
def check_assembly(n_images=25, **kw):
    """Feed the assembly its own ground truth and score it.

    This bounds what the pipeline can achieve even with a perfect model, and
    catches bugs in ordering, spline fitting or sweeping before any training.
    """
    import collections, json, random
    from spine_data import rasterize, spine_heatmap, load_annotations, BASE  # noqa
    from metrics import pq_from_masks, aggregate
    from evaluate import gt_masks_full

    by_id, fname = load_annotations()
    keys = sorted(by_id)
    random.seed(0); random.shuffle(keys)

    results = []
    for k in keys[:n_images]:
        anns = by_id[k]
        mask, spine, radius = rasterize(anns)
        heat = spine_heatmap(spine, sigma=3.0)
        preds = assemble(heat, radius, mask.astype(np.float32), **kw)
        gts = gt_masks_full(anns)
        results.append(pq_from_masks(gts, preds))
    ds = aggregate(results)
    im = aggregate(results, mode="image")
    print(f"assembly on ground-truth maps ({n_images} images): "
          f"PQ={ds['PQ']:.4f} SQ={ds['SQ']:.4f} RQ={ds['RQ']:.4f} "
          f"TP={ds['TP']} FP={ds['FP']} FN={ds['FN']} | per-image PQ={im['PQ']:.4f}")
    return ds


if __name__ == "__main__":
    check_assembly()
