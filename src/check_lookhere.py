"""Does yesterday's filament tell us where to look in today's image?

The first attempt at using time (v3) pasted the projected shape in as a new
label, and half of what it pasted landed on material that is not dark. The
proposal here is weaker and should be more robust: let the projection say only
*where* to look, and take the label from the day's own image.

Whether that works rests on three numbers, and none of them needs the GPU:

  1. how far the projection actually misses, measured against annotations that
     exist in both photographs -- not against a cross-correlation, which is what
     produced the unverified 3.5 degrees;
  2. whether there is dark material at the projected place at all;
  3. whether a local search finds the filament when the projection is off, which
     is the whole premise of "look here".

Darkness is read from the flattened images, so a number is comparable across
frames and stations. The reference is what real annotations score in the same
frame: if a projection is as dark as the filaments an expert drew, there is
something there.
"""
from __future__ import annotations

import collections, json, sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
PREP = ROOT / "preprocessed" / "train"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles                       # noqa: E402
from propagate_labels import (obs_time, summarise, project,     # noqa: E402
                              spine_arrays, MAX_GAP_DAYS)
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL, OFFSET, SCALE  # noqa: E402

import os
SEARCH_PX = int(os.environ.get("SEARCH_PX", 72))
SEARCH_STEP = 2      # local search half-window and grid
DEG_PER_PX = 1.0 / 15.8             # measured: 1 degree of longitude = 15.8 px


def load_flat(stem):
    p = PREP / f"{stem}.png"
    if not p.exists():
        return None
    return np.asarray(Image.open(p), dtype=np.float32) / SCALE + OFFSET


def darkness(img, px, py):
    """Median flattened brightness along a path, in units of the frame's own sigma.

    Negative means darker than the quiet Sun. Real filaments sit far below zero;
    empty disk sits at zero by construction.
    """
    xi = np.clip(np.round(px).astype(int), 0, FULL - 1)
    yi = np.clip(np.round(py).astype(int), 0, FULL - 1)
    return float(np.median(img[yi, xi]))


def ann_mask(anns):
    im = Image.new("L", (FULL, FULL), 0)
    dr = ImageDraw.Draw(im)
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.asarray(im, bool)


def best_offset(img, px, py, half=SEARCH_PX, step=SEARCH_STEP):
    """Shift the projected path around until it is darkest. Returns (dx, dy, value)."""
    offs = np.arange(-half, half + 1, step)
    best = (0, 0, darkness(img, px, py))
    for dy in offs:
        for dx in offs:
            v = darkness(img, px + dx, py + dy)
            if v < best[2]:
                best = (int(dx), int(dy), v)
    return best


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}

    # one entry per photograph; the same photograph appears once per annotator,
    # and for this question the annotators can be pooled
    per_photo = collections.defaultdict(list)
    for iid, anns in by_id.items():
        per_photo[stem[iid]].extend(anns)
    photos = sorted(per_photo)
    times = {s: obs_time(s) for s in photos}
    print(f"{len(photos)} photographs, {sum(len(v) for v in per_photo.values())} annotations")

    # the quiet-Sun spread of one frame, for scaling
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.95) ** 2

    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400.0 <= MAX_GAP_DAYS]
    print(f"{len(pairs)} photograph pairs within {MAX_GAP_DAYS} days\n")

    ref_dark, proj_dark, snap_dark = [], [], []
    offsets, gaps, on_ann, snapped_on_ann = [], [], [], []
    matched_shift = []
    ctrl_dark, ctrl_asis = [], []
    rng = np.random.default_rng(0)
    n_proj = 0

    cache = {}
    for k, (sa, sb) in enumerate(pairs, 1):
        for src_s, dst_s in ((sa, sb), (sb, sa)):
            img = cache.get(dst_s)
            if img is None:
                img = load_flat(dst_s)
                if img is None:
                    continue
                if len(cache) > 24:
                    cache.pop(next(iter(cache)))
                cache[dst_s] = img
            t_src, t_dst = times[src_s], times[dst_s]
            gap = abs((t_dst - t_src).total_seconds()) / 86400.0
            ang_src = solar_angles(t_src)
            dst_mask = ann_mask(per_photo[dst_s])

            # what a real filament looks like in this frame, for reference
            if dst_s not in getattr(main, "_seen", set()):
                main._seen = getattr(main, "_seen", set()) | {dst_s}
                sigma = float(img[ondisk].std())
                for a in per_photo[dst_s]:
                    sx, sy = spine_arrays(a)
                    if len(sx) >= 2:
                        ref_dark.append((darkness(img, sx, sy) - 1.0) / sigma)

            sigma = float(img[ondisk].std())
            for a in per_photo[src_s]:
                s = summarise(a, ang_src)
                if s is None:
                    continue
                p = project(s, t_src, t_dst)
                if p is None:
                    continue
                n_proj += 1
                z0 = (darkness(img, p["px"], p["py"]) - 1.0) / sigma
                dx, dy, v = best_offset(img, p["px"], p["py"])
                z1 = (v - 1.0) / sigma
                xi = np.clip(np.round(p["px"]).astype(int), 0, FULL - 1)
                yi = np.clip(np.round(p["py"]).astype(int), 0, FULL - 1)
                hit = float(dst_mask[yi, xi].mean())
                xj = np.clip(np.round(p["px"] + dx).astype(int), 0, FULL - 1)
                yj = np.clip(np.round(p["py"] + dy).astype(int), 0, FULL - 1)
                hit2 = float(dst_mask[yj, xj].mean())

                # Control. The local search tries 1681 positions and keeps the
                # darkest, and a solar disk is full of dark detail -- so it may
                # find something for any curve at all. The same curve is
                # therefore also placed somewhere it certainly does not belong,
                # far enough that it cannot be the same filament but still on
                # the disk, and searched the same way. If the control passes as
                # often as the real projection, the search is the whole signal.
                ctl = rng.uniform(0, 2 * np.pi)
                cd = rng.uniform(220, 420)
                cx, cy = p["px"] + cd * np.cos(ctl), p["py"] + cd * np.sin(ctl)
                if (np.hypot(cx - DISK_CX, cy - DISK_CY) / DISK_R).max() < 0.90:
                    _, _, cv = best_offset(img, cx, cy)
                    ctrl_dark.append((cv - 1.0) / sigma)
                    ctrl_asis.append((darkness(img, cx, cy) - 1.0) / sigma)

                proj_dark.append(z0); snap_dark.append(z1)
                offsets.append(np.hypot(dx, dy)); gaps.append(gap)
                on_ann.append(hit); snapped_on_ann.append(hit2)
                # for filaments annotated in both frames, the shift needed to
                # line them up IS the projection error -- measured, not inferred
                if hit > 0.5:
                    matched_shift.append((np.hypot(dx, dy), gap))
        if k % 200 == 0:
            print(f"  {k}/{len(pairs)} pairs, {n_proj} projections", flush=True)

    z_ref = np.array(ref_dark); z0 = np.array(proj_dark); z1 = np.array(snap_dark)
    off = np.array(offsets); g = np.array(gaps)
    ha = np.array(on_ann); hb = np.array(snapped_on_ann)

    print(f"\n{n_proj} projections from {len(pairs)} pairs\n")
    print("1. HOW DARK  (flattened brightness, in frame sigma; 0 = quiet Sun)")
    print(f"   real annotations       median {np.median(z_ref):+.2f}  "
          f"(10th {np.percentile(z_ref,10):+.2f}, 90th {np.percentile(z_ref,90):+.2f})")
    print(f"   projected, as-is       median {np.median(z0):+.2f}")
    print(f"   projected, after snap  median {np.median(z1):+.2f}")
    gate = float(np.percentile(z_ref, 90))       # as dark as the faintest real ones
    print(f"   gate (90th pct of real annotations): {gate:+.2f}")
    print(f"   pass as-is  {100*(z0 <= gate).mean():5.1f}%"
          f"   pass after snap {100*(z1 <= gate).mean():5.1f}%")

    print("\n2. HOW FAR THE PROJECTION MISSES")
    if len(matched_shift):
        ms = np.array([m[0] for m in matched_shift])
        mg = np.array([m[1] for m in matched_shift])
        print(f"   {len(ms)} projections landing on an existing annotation "
              f"(the ones where truth is known)")
        print(f"   shift to darkest: median {np.median(ms):5.1f} px "
              f"= {np.median(ms)*DEG_PER_PX:.2f} deg,  90th {np.percentile(ms,90):5.1f} px "
              f"= {np.percentile(ms,90)*DEG_PER_PX:.2f} deg")
        for lo, hi in ((0, 0.3), (0.3, 0.8), (0.8, 1.5)):
            m = (mg >= lo) & (mg < hi)
            if m.sum() > 5:
                print(f"     gap {lo}-{hi} d: n={m.sum():4d}  median "
                      f"{np.median(ms[m]):5.1f} px = {np.median(ms[m])*DEG_PER_PX:.2f} deg")
    else:
        print("   none landed on an annotation")

    print("\n3. WHAT 'LOOK HERE' WOULD RECOVER")
    already = ha > 0.5
    dark_now = z1 <= gate
    print(f"   projected onto an existing annotation      {100*already.mean():5.1f}%"
          f"   <- nothing new, but proves the geometry")
    print(f"   unannotated AND dark after local search    "
          f"{100*((~already) & dark_now).mean():5.1f}%   <- what this idea gains")
    print(f"   unannotated AND not dark (must be dropped) "
          f"{100*((~already) & ~dark_now).mean():5.1f}%   <- v3 kept these")
    print(f"   of the snapped-and-dark ones, {100*(hb[(~already) & dark_now] > 0.5).mean():.1f}%"
          f" land on an annotation after snapping")

    zc = np.array(ctrl_dark); zca = np.array(ctrl_asis)
    print(f"\n   CONTROL -- the same curve dropped 220-420 px away, searched the same way")
    print(f"     n={len(zc)}   as-is pass {100*(zca <= gate).mean():5.1f}%"
          f"   after snap pass {100*(zc <= gate).mean():5.1f}%")
    print(f"     real projection  as-is {100*(z0 <= gate).mean():5.1f}%"
          f"   after snap {100*(z1 <= gate).mean():5.1f}%")
    print(f"     -> the search finds dark material for a curve that does not belong "
          f"{100*(zc <= gate).mean():.1f}% of the time.")
    print(f"        Anything the real projection gains beyond that is the actual signal.")

    print("\n   for comparison, v3 pasted the projection as-is:")
    print(f"     on non-dark material: {100*(z0 > gate).mean():.1f}%   "
          f"(measured before as 54.3%)")

    json.dump({"n_proj": int(n_proj), "n_pairs": len(pairs), "gate": gate,
               "z_ref_median": float(np.median(z_ref)),
               "z_proj_median": float(np.median(z0)),
               "z_snap_median": float(np.median(z1)),
               "pass_asis": float((z0 <= gate).mean()),
               "pass_snap": float((z1 <= gate).mean()),
               "shift_median_px": float(np.median(off)),
               "matched_shift_median_px": float(np.median([m[0] for m in matched_shift]))
               if matched_shift else None,
               "recoverable_frac": float(((~already) & dark_now).mean())},
              open(ROOT / "lookhere_check.json", "w"), indent=1)
    print("\nwrote lookhere_check.json")


if __name__ == "__main__":
    main()
