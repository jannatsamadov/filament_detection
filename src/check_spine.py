"""Is a spine-first representation viable for these filaments?

Mask R-CNN failed here because anchors match ground truth by box IoU, and a long
curved filament fills only 24% of its own box. A spine-first model sidesteps
anchors entirely: predict the centreline, predict a width at each point, then
sweep a disk along the curve to rebuild the mask.

That only works if two things hold, and both are measured here:
  1. Spines of different filaments are further apart than their masks, so
     connected components on a spine map separates instances cleanly.
  2. Sweeping a disk of the annotated width along the annotated spine
     reconstructs the annotated mask well enough to be worth scoring.
The second number is the ceiling of the whole approach: no model can beat the
reconstruction quality of its own representation.
"""
from __future__ import annotations

import collections, json, random, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
FULL = 2048


def poly_mask(poly, size=FULL):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.asarray(m, bool)


def spine_points(spine, step=2.0):
    """Dense points along the annotated polyline."""
    xs, ys = np.asarray(spine[0::2], float), np.asarray(spine[1::2], float)
    if len(xs) < 2:
        return xs, ys
    out_x, out_y = [], []
    for i in range(len(xs) - 1):
        d = np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        n = max(int(d / step), 1)
        t = np.linspace(0, 1, n, endpoint=False)
        out_x += list(xs[i] + t * (xs[i + 1] - xs[i]))
        out_y += list(ys[i] + t * (ys[i + 1] - ys[i]))
    out_x.append(xs[-1]); out_y.append(ys[-1])
    return np.array(out_x), np.array(out_y)


def sweep_disks(px, py, radii, size=FULL):
    """Rebuild a mask by drawing a disk of the given radius at each spine point."""
    im = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(im)
    for x, y, r in zip(px, py, radii):
        if r < 0.5:
            continue
        dr.ellipse([x - r, y - r, x + r, y + r], fill=1)
    return np.asarray(im, bool)


def main(n_images=60):
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    keys = sorted(by_img)
    random.seed(0); random.shuffle(keys)

    recon_iou, recon_dice = [], []
    widths, lengths = [], []
    spine_gaps, mask_gaps = [], []

    for k in keys[:n_images]:
        anns = [a for a in by_img[k] if len(a["spine"]) >= 4]
        if len(anns) < 2:
            continue
        masks, spines = [], []
        for a in anns:
            m = poly_mask(a["segmentation"][0])
            if m.sum() < 20:
                continue
            px, py = spine_points(a["spine"])
            masks.append(m); spines.append((px, py))

        # ---- 1. reconstruction ceiling ----
        for m, (px, py) in zip(masks, spines):
            # distance to the mask edge at each spine point = local half-width
            dt = distance_transform_edt(m)
            xi = np.clip(px.astype(int), 0, FULL - 1)
            yi = np.clip(py.astype(int), 0, FULL - 1)
            r = dt[yi, xi]
            r = np.maximum(r, 0.0)
            rec = sweep_disks(px, py, r)
            inter = (rec & m).sum(); union = (rec | m).sum()
            if union:
                recon_iou.append(inter / union)
                recon_dice.append(2 * inter / (rec.sum() + m.sum()))
            widths.append(2 * np.median(r[r > 0]) if (r > 0).any() else 0.0)
            lengths.append(np.hypot(np.diff(px), np.diff(py)).sum())

        # ---- 2. separation: spine-to-spine vs mask-to-mask ----
        for i in range(len(masks)):
            others_pts = []
            for j in range(len(masks)):
                if j != i:
                    others_pts.append(np.c_[spines[j][0], spines[j][1]])
            if not others_pts:
                continue
            others_pts = np.vstack(others_pts)
            mine = np.c_[spines[i][0], spines[i][1]]
            # nearest spine point of any other filament
            step = max(len(mine) // 60, 1)
            dmin = min(np.min(np.linalg.norm(others_pts - p, axis=1))
                       for p in mine[::step])
            spine_gaps.append(dmin)

            other_mask = np.zeros_like(masks[i])
            for j, mm in enumerate(masks):
                if j != i:
                    other_mask |= mm
            if other_mask.any():
                dt2 = distance_transform_edt(~other_mask)
                mask_gaps.append(dt2[masks[i]].min())

    ri = np.array(recon_iou); rd = np.array(recon_dice)
    w = np.array(widths); L = np.array(lengths)
    sg = np.array(spine_gaps); mg = np.array(mask_gaps)

    print(f"{len(ri)} filaments measured\n")
    print("1. CEILING -- rebuild the annotated mask by sweeping a disk along the")
    print("   annotated spine, using the true local half-width:")
    print(f"     IoU  : median {np.median(ri):.3f}   mean {ri.mean():.3f}   "
          f"p10 {np.percentile(ri,10):.3f}")
    print(f"     Dice : median {np.median(rd):.3f}   mean {rd.mean():.3f}")
    print(f"     share reconstructed at IoU > 0.5 : {100*np.mean(ri>0.5):.1f}%")
    print(f"     share reconstructed at IoU > 0.7 : {100*np.mean(ri>0.7):.1f}%")
    print()
    print("2. SHAPE")
    print(f"     spine length : median {np.median(L):.0f} px   p90 {np.percentile(L,90):.0f}")
    print(f"     full width   : median {np.median(w):.1f} px   p10 {np.percentile(w,10):.1f}"
          f"   p90 {np.percentile(w,90):.1f}")
    print(f"     length / width ratio: median {np.median(L/np.maximum(w,1e-6)):.1f}")
    print()
    print("3. SEPARATION -- can connected components on a spine map split instances?")
    print(f"     nearest OTHER spine : median {np.median(sg):.0f} px   "
          f"p10 {np.percentile(sg,10):.0f}   p1 {np.percentile(sg,1):.0f}")
    print(f"     nearest OTHER mask  : median {np.median(mg):.0f} px   "
          f"p10 {np.percentile(mg,10):.0f}   p1 {np.percentile(mg,1):.0f}")
    for t in (10, 20, 40):
        print(f"     within {t:>2} px: spines {100*np.mean(sg<t):5.1f}%   "
              f"masks {100*np.mean(mg<t):5.1f}%")


if __name__ == "__main__":
    main()
