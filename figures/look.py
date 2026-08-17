"""Look at raw GONG H-alpha images: disk geometry, radial profile, the halo band."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

D = Path(r"d:/Projects/Competitions/filament_detection/MAGFiLO_1.0_Kaggle_2026/train/train_images")
OUT = Path(r"C:/Users/Semed/AppData/Local/Temp/claude/d--Projects-Competitions/c0e77c17-49fc-42c3-b848-304022467901/scratchpad")

# one image per station
picks = {}
for f in sorted(D.iterdir()):
    st = f.name[14:16]
    if st not in picks:
        picks[st] = f
    if len(picks) == 6:
        break

def disk(img, thr=None):
    """Find disk centre and radius from a brightness threshold."""
    t = thr if thr else max(img.mean() * 0.5, 20)
    m = img > t
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    r = np.sqrt(m.sum() / np.pi)
    return cx, cy, r

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
info = []
for k, (st, f) in enumerate(sorted(picks.items())):
    img = np.array(Image.open(f).convert("L")).astype(np.float32)
    cx, cy, r = disk(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r

    axes[0, k].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, k].set_title(f"{st}  {f.name[:8]}\nR={r:.0f}px  c=({cx:.0f},{cy:.0f})", fontsize=9)
    axes[0, k].axis("off")
    for rad, c in ((1.0, "r"), (1.05, "y"), (1.15, "c")):
        axes[0, k].add_patch(plt.Circle((cx, cy), r * rad, fill=False, ec=c, lw=0.7))

    # radial profile: median brightness in each radial bin
    bins = np.linspace(0, 1.4, 141)
    idx = np.digitize(rr.ravel(), bins)
    v = img.ravel()
    prof = np.array([np.median(v[idx == i]) if (idx == i).sum() > 50 else np.nan
                     for i in range(1, len(bins))])
    ctr = 0.5 * (bins[1:] + bins[:-1])
    axes[1, k].plot(ctr, prof, lw=1.5)
    axes[1, k].axvline(1.0, color="r", ls="--", lw=0.8)
    axes[1, k].axvline(1.05, color="y", ls="--", lw=0.8)
    axes[1, k].axvline(1.15, color="c", ls="--", lw=0.8)
    axes[1, k].set_xlabel("r / R_sun"); axes[1, k].set_ylim(0, 255)
    if k == 0: axes[1, k].set_ylabel("median brightness")
    axes[1, k].grid(alpha=.3)

    # how bright is the ring just outside the limb?
    ring = np.median(img[(rr > 1.02) & (rr < 1.12)])
    far = np.median(img[rr > 1.25])
    onlimb = np.median(img[(rr > 0.95) & (rr < 1.0)])
    ctrb = np.median(img[rr < 0.2])
    info.append((st, f.name, r, ctrb, onlimb, ring, far))

plt.tight_layout(); plt.savefig(OUT / "01_stations.png", dpi=85); plt.close()

print(f"{'st':>3} {'file':>20} {'R_px':>6} {'centre':>7} {'r~0.97':>7} {'ring 1.02-1.12':>15} {'r>1.25':>7}")
for st, fn, r, c_, o, ring, far in info:
    print(f"{st:>3} {fn[:18]:>20} {r:6.0f} {c_:7.0f} {o:7.0f} {ring:15.0f} {far:7.0f}")
