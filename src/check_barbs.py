"""Do all filaments have barbs, and how much of the mask do they hold?

A barb is material that leaves the spine sideways, so it is exactly what a disk
swept along the spine cannot reproduce. Sweeping the annotated spine with the
true local half-width rebuilds the annotated mask at median IoU 0.758; this asks
what the missing quarter is made of, and whether every filament has some.

barb_fraction = (mask area outside the swept ribbon) / (mask area)

Reported against size, latitude and spine length, since active-region filaments
are short and smooth while quiescent and polar-crown ones are long and barbed.
"""
from __future__ import annotations

import collections, json, math, random, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
sys.path.insert(0, str(HERE))
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL   # noqa: E402


def poly_mask_local(poly, pad=30):
    xs, ys = np.asarray(poly[0::2], float), np.asarray(poly[1::2], float)
    x0, y0 = xs.min() - pad, ys.min() - pad
    w = int(xs.max() - x0 + pad) + 1
    h = int(ys.max() - y0 + pad) + 1
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(list(zip(xs - x0, ys - y0)), fill=1)
    return np.asarray(im, bool), x0, y0


def densify(spine, step=1.5):
    xs, ys = np.asarray(spine[0::2], float), np.asarray(spine[1::2], float)
    ox, oy = [], []
    for i in range(len(xs) - 1):
        d = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        n = max(int(d / step), 1)
        t = np.linspace(0, 1, n, endpoint=False)
        ox += list(xs[i] + t * (xs[i + 1] - xs[i]))
        oy += list(ys[i] + t * (ys[i + 1] - ys[i]))
    ox.append(xs[-1]); oy.append(ys[-1])
    return np.array(ox), np.array(oy)


def main(n_images=120):
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    keys = sorted(by_img)
    random.seed(0); random.shuffle(keys)

    rows = []
    for k in keys[:n_images]:
        for a in by_img[k]:
            if len(a["spine"]) < 4 or a["area"] < 100:
                continue
            m, x0, y0 = poly_mask_local(a["segmentation"][0])
            if m.sum() < 60:
                continue
            px, py = densify(a["spine"])
            px, py = px - x0, py - y0
            dt = distance_transform_edt(m)
            xi = np.clip(px.astype(int), 0, m.shape[1] - 1)
            yi = np.clip(py.astype(int), 0, m.shape[0] - 1)
            r = dt[yi, xi]

            im = Image.new("L", (m.shape[1], m.shape[0]), 0)
            dr = ImageDraw.Draw(im)
            for x, y, rr in zip(px, py, r):
                if rr >= 0.5:
                    dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=1)
            ribbon = np.asarray(im, bool) & m
            outside = m & ~ribbon
            frac = outside.sum() / m.sum()

            # how many distinct protrusions, and how far they reach
            lab, n = label(outside, structure=np.ones((3, 3)))
            sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
            big = int((sizes >= 30).sum())
            reach = float(dt[m].max())

            cy = np.asarray(a["spine"][1::2], float).mean()
            cx = np.asarray(a["spine"][0::2], float).mean()
            sin_lat = (DISK_CY - cy) / DISK_R
            length = np.hypot(np.diff(px), np.diff(py)).sum()
            rows.append((frac, big, a["area"], length, abs(sin_lat),
                         2 * float(np.median(r[r > 0])) if (r > 0).any() else 0.0))

    rows = np.array(rows)
    frac, nbarb, area, length, lat, width = rows.T
    print(f"{len(rows)} filaments over {n_images} annotation sets\n")

    print("Share of mask area that a swept spine cannot reproduce (the barbs):")
    for q in (10, 25, 50, 75, 90):
        print(f"   p{q:<2} {np.percentile(frac, q):.3f}")
    print(f"   mean {frac.mean():.3f}")
    print()
    print("Do all filaments have them? (a protrusion of at least 30 px counts)")
    for t in (0, 1, 2, 3, 5):
        print(f"   with more than {t} distinct protrusions: {100*np.mean(nbarb > t):5.1f}%")
    print(f"   none at all (barb area < 5% of the mask): {100*np.mean(frac < 0.05):.1f}%")
    print()
    print(f"{'group':<22} {'n':>5} {'barb frac':>10} {'protrusions':>12} {'width px':>9}")
    for lo, hi, lab_ in ((0, 700, "area < 700 px"), (700, 1500, "700-1500"),
                         (1500, 4000, "1500-4000"), (4000, 1e9, "> 4000 px")):
        s = (area >= lo) & (area < hi)
        if s.sum() > 5:
            print(f"{lab_:<22} {s.sum():5d} {np.median(frac[s]):10.3f} "
                  f"{np.median(nbarb[s]):12.1f} {np.median(width[s]):9.1f}")
    print()
    for lo, hi in ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0)):
        s = (lat >= lo) & (lat < hi)
        if s.sum() > 5:
            print(f"|sin lat| {lo:.1f}-{hi:.1f}        {s.sum():5d} "
                  f"{np.median(frac[s]):10.3f} {np.median(nbarb[s]):12.1f} "
                  f"{np.median(width[s]):9.1f}")

    from scipy.stats import spearmanr
    print()
    for name, v in (("area", area), ("spine length", length), ("|sin lat|", lat)):
        rho, p = spearmanr(v, frac)
        print(f"   Spearman(barb fraction, {name:<13}) rho={rho:+.3f}  p={p:.2e}")


if __name__ == "__main__":
    main()
