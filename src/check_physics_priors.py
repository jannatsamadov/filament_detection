"""Do the physical regularities of filaments actually show up in this dataset?

Three claims from solar physics, each tested against the annotations rather than
assumed:

  1. Orientation depends on latitude. Polar crown filaments lie along parallels
     at high latitude; active-region filaments are shorter and sit at low
     latitude. If true, a detection running the wrong way for its latitude is
     physically suspect.

  2. Filaments sit in filament channels, where the chromospheric fibrils align
     with the filament axis. That alignment is visible in H-alpha even where the
     cool material is absent, so a local orientation field carries information a
     brightness-only model never sees.

  3. Filaments live for days and rotate with the Sun. If the dataset contains
     observations a few days apart, a detection should recur, displaced by solar
     rotation -- a consistency check no single-frame model can use.

Latitude here is the projected sin(lat) = y / R, ignoring the B0 angle (at most
7.25 degrees). That is coarse but enough to see whether a trend exists at all.
"""
from __future__ import annotations

import collections, json, math, sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
PRE = ROOT / "preprocessed" / "train"
sys.path.insert(0, str(HERE))
from spine_data import load_annotations, DISK_CX, DISK_CY, DISK_R, OFFSET, SCALE  # noqa


def spine_axis(spine):
    """Principal direction of a spine, as an angle from the +x axis in degrees."""
    x = np.asarray(spine[0::2], float)
    y = np.asarray(spine[1::2], float)
    if len(x) < 2:
        return None, None, None
    c = np.c_[x - x.mean(), y - y.mean()]
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    vx, vy = vt[0]
    ang = math.degrees(math.atan2(vy, vx)) % 180.0
    length = np.hypot(np.diff(x), np.diff(y)).sum()
    return ang, length, (x.mean(), y.mean())


def orientation_field(img, sigma=3.0, smooth=9.0):
    """Local ridge orientation and coherence from the structure tensor."""
    gy, gx = np.gradient(gaussian_filter(img, sigma))
    jxx = gaussian_filter(gx * gx, smooth)
    jyy = gaussian_filter(gy * gy, smooth)
    jxy = gaussian_filter(gx * gy, smooth)
    diff = jxx - jyy
    denom = np.sqrt(diff ** 2 + 4 * jxy ** 2) + 1e-12
    coherence = denom / (jxx + jyy + 1e-12)
    # dominant gradient direction; ridge runs perpendicular to it
    theta = 0.5 * np.arctan2(2 * jxy, diff)
    ridge = (np.degrees(theta) + 90.0) % 180.0
    return ridge, coherence


def circ_diff(a, b):
    """Smallest angle between two undirected orientations, in degrees."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def main(n_channel_images=25):
    by_id, fname = load_annotations()

    # ---------- 1. orientation vs latitude ----------
    rows = []
    for iid, anns in by_id.items():
        for a in anns:
            if len(a["spine"]) < 4:
                continue
            ang, length, (cx, cy) = spine_axis(a["spine"])
            if ang is None:
                continue
            sin_lat = (DISK_CY - cy) / DISK_R          # image y grows downward
            r = math.hypot(cx - DISK_CX, cy - DISK_CY) / DISK_R
            if r > 0.95:
                continue
            rows.append((abs(sin_lat), ang, length, a["area"]))
    rows = np.array(rows)
    lat, ang, length, area = rows.T
    # 0 deg == east-west in image coordinates (along a parallel, roughly)
    tilt = np.minimum(ang, 180 - ang)                  # 0 = horizontal, 90 = vertical

    print("1. ORIENTATION vs LATITUDE")
    print(f"   {len(rows)} filaments (r < 0.95)")
    print(f"   {'|sin(lat)|':>12} {'n':>6} {'median tilt':>12} {'frac within':>12} "
          f"{'median len':>11} {'median area':>12}")
    print(f"   {'':>12} {'':>6} {'from E-W':>12} {'30 deg of E-W':>12} {'px':>11} {'px':>12}")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (lat >= lo) & (lat < hi)
        if s.sum() < 10:
            continue
        print(f"   {lo:.1f}-{hi:.1f}      {s.sum():6d} {np.median(tilt[s]):12.1f} "
              f"{100*np.mean(tilt[s] < 30):11.1f}% {np.median(length[s]):11.0f} "
              f"{np.median(area[s]):12.0f}")
    from scipy.stats import spearmanr
    rho, p = spearmanr(lat, tilt)
    print(f"   Spearman(|sin lat|, tilt from E-W): rho={rho:+.3f}  p={p:.2e}")
    rho2, p2 = spearmanr(lat, length)
    print(f"   Spearman(|sin lat|, spine length) : rho={rho2:+.3f}  p={p2:.2e}")

    # ---------- 2. fibril alignment (filament channel) ----------
    print("\n2. FIBRIL ALIGNMENT AROUND FILAMENTS (filament channel)")
    from PIL import ImageDraw
    from scipy.ndimage import binary_dilation
    keys = sorted(by_id)[:n_channel_images]
    inside_align, ring_align, far_align = [], [], []
    inside_coh, ring_coh, far_coh = [], [], []
    for iid in keys:
        p = PRE / (Path(fname[iid]).stem + ".png")
        if not p.exists():
            continue
        img = np.asarray(Image.open(p), dtype=np.float32) / SCALE + OFFSET
        ridge, coh = orientation_field(img)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (0.9 * DISK_R) ** 2
        for a in by_id[iid]:
            if len(a["spine"]) < 4:
                continue
            axis, length, _ = spine_axis(a["spine"])
            if axis is None or length < 40:
                continue
            m = Image.new("L", img.shape[::-1], 0)
            pp = a["segmentation"][0]
            ImageDraw.Draw(m).polygon(list(zip(pp[0::2], pp[1::2])), fill=1)
            m = np.asarray(m, bool)
            if m.sum() < 50:
                continue
            near = binary_dilation(m, iterations=25) & ~m & ondisk
            if near.sum() < 200:
                continue
            inside_align.append(np.median(circ_diff(ridge[m], axis)))
            ring_align.append(np.median(circ_diff(ridge[near], axis)))
            inside_coh.append(np.median(coh[m]))
            ring_coh.append(np.median(coh[near]))
        # background reference: random on-disk patches away from any filament
        allm = np.zeros(img.shape, bool)
        for a in by_id[iid]:
            pp = a["segmentation"][0]
            m = Image.new("L", img.shape[::-1], 0)
            ImageDraw.Draw(m).polygon(list(zip(pp[0::2], pp[1::2])), fill=1)
            allm |= np.asarray(m, bool)
        far = ondisk & ~binary_dilation(allm, iterations=60)
        idx = np.nonzero(far.ravel())[0]
        if len(idx) > 5000:
            sel = np.random.default_rng(0).choice(idx, 5000, replace=False)
            far_coh.append(np.median(coh.ravel()[sel]))

    print(f"   {len(inside_align)} filaments over {n_channel_images} frames")
    print(f"   median |fibril orientation - filament axis|, inside the mask : "
          f"{np.median(inside_align):.1f} deg")
    print(f"   same, in a 25 px ring around it                             : "
          f"{np.median(ring_align):.1f} deg")
    print(f"   (45 deg would mean no alignment at all)")
    print(f"   orientation coherence  inside {np.median(inside_coh):.3f} | "
          f"ring {np.median(ring_coh):.3f} | far from any filament "
          f"{np.median(far_coh):.3f}")

    # ---------- 3. temporal recurrence ----------
    print("\n3. TEMPORAL RECURRENCE (solar rotation)")
    dates = sorted({datetime.strptime(f[:14], "%Y%m%d%H%M%S") for f in set(fname.values())})
    gaps = np.array([(dates[i + 1] - dates[i]).total_seconds() / 86400.0
                     for i in range(len(dates) - 1)])
    print(f"   {len(dates)} distinct observation times")
    for t in (1, 2, 3, 5, 10):
        print(f"   consecutive observations within {t:2d} day(s): "
              f"{int((gaps <= t).sum()):4d} pairs")
    print(f"   median gap between consecutive observations: {np.median(gaps):.1f} days")


if __name__ == "__main__":
    main()
