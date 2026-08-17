"""Validate the disk fit across stations and produce a figure."""
from __future__ import annotations

import sys, time, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMGS = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from geometry import fit_disk, radial_coordinates  # noqa: E402


def load(p):
    return np.array(Image.open(p).convert("L")).astype(np.float32)


def main(n_per_station=10):
    by_station = collections.defaultdict(list)
    for f in sorted(IMGS.iterdir()):
        by_station[f.name[14:16]].append(f)

    rows, t0 = [], time.time()
    for st, files in sorted(by_station.items()):
        for f in files[:n_per_station]:
            img = load(f)
            fit = fit_disk(img)
            rr, _ = radial_coordinates(img.shape, fit)
            # brightness just inside vs just outside the fitted limb
            inside = np.median(img[(rr > 0.93) & (rr < 0.98)])
            outside = np.median(img[(rr > 1.02) & (rr < 1.07)])
            halo = np.median(img[(rr > 1.10) & (rr < 1.25)]) if (rr > 1.25).any() else np.nan
            rows.append((st, f.name, fit.cx, fit.cy, fit.r, fit.residual,
                         fit.n_points, inside, outside, halo))
    dt = time.time() - t0

    print(f"{len(rows)} images, {dt:.1f}s total, {dt/len(rows):.2f}s per image\n")
    print(f"{'st':>3} {'N':>3} {'R median':>9} {'R spread':>9} {'resid med':>10} "
          f"{'resid max':>10} {'in/out':>8} {'halo':>6}")
    for st in sorted({r[0] for r in rows}):
        g = [r for r in rows if r[0] == st]
        R = np.array([x[4] for x in g]); res = np.array([x[5] for x in g])
        io = np.array([x[7] / max(x[8], 1e-6) for x in g])
        hl = np.array([x[9] for x in g])
        print(f"{st:>3} {len(g):>3} {np.median(R):9.1f} {R.max()-R.min():9.1f} "
              f"{np.median(res):10.2f} {res.max():10.2f} {np.median(io):8.1f} "
              f"{np.nanmedian(hl):6.1f}")

    allR = np.array([r[4] for r in rows]); allres = np.array([r[5] for r in rows])
    print(f"\nALL: R {allR.min():.0f}-{allR.max():.0f} px (median {np.median(allR):.0f}, "
          f"spread {100*(allR.max()-allR.min())/np.median(allR):.1f}% of median)")
    print(f"     fit residual: median {np.median(allres):.2f} px, "
          f"90th pct {np.percentile(allres,90):.2f} px, max {allres.max():.2f} px")
    worst = rows[int(np.argmax(allres))]
    print(f"     worst fit: {worst[1]}  residual {worst[5]:.2f} px")

    # ---- figure: one image per station, fitted limb + radial profile
    fig, axes = plt.subplots(2, 6, figsize=(24, 8.5))
    for k, st in enumerate(sorted(by_station)):
        f = by_station[st][0]
        img = load(f)
        fit = fit_disk(img)
        rr, _ = radial_coordinates(img.shape, fit)
        ax = axes[0, k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.add_patch(plt.Circle((fit.cx, fit.cy), fit.r, fill=False, ec="lime", lw=1.1))
        ax.set_title(f"{st}   R={fit.r:.0f} px\nresidual {fit.residual:.2f} px", fontsize=10)
        ax.axis("off")

        ax = axes[1, k]
        bins = np.linspace(0.5, 1.3, 161)
        idx = np.digitize(rr.ravel(), bins); v = img.ravel()
        prof = np.array([np.median(v[idx == i]) if (idx == i).sum() > 40 else np.nan
                         for i in range(1, len(bins))])
        ax.plot(0.5 * (bins[1:] + bins[:-1]), prof, lw=1.8)
        ax.axvline(1.0, color="lime", ls="--", lw=1.2)
        ax.set_xlabel("r / R_sun"); ax.set_ylim(0, 255); ax.grid(alpha=.3)
        if k == 0:
            ax.set_ylabel("median brightness")
    fig.suptitle("Disk fit: the green circle must sit on the bright limb rim, "
                 "not on the outer edge of the halo", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / "03_disk_fit.png", dpi=85)
    print(f"\nwrote {FIGS / '03_disk_fit.png'}")


if __name__ == "__main__":
    main()
