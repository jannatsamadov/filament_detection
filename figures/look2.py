"""Zoom in: the limb ring, what a filament actually looks like, and a sunspot."""
import numpy as np, json, collections, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pathlib import Path

BASE = Path(r"d:/Projects/Competitions/filament_detection/MAGFiLO_1.0_Kaggle_2026")
D = BASE / "train/train_images"
OUT = Path(r"C:/Users/Semed/AppData/Local/Temp/claude/d--Projects-Competitions/c0e77c17-49fc-42c3-b848-304022467901/scratchpad")

d = json.load(open(BASE / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
by_img = collections.defaultdict(list)
for a in d["annotations"]:
    by_img[a["image_id"]].append(a)
fname = {i["id"]: i["file_name"] for i in d["images"]}

def fit_disk(img):
    """Proper disk fit: threshold at half of the disk's own median, then
    find centre from column/row extents (robust to the bright limb ring)."""
    t = np.percentile(img, 60) * 0.35
    m = img > t
    ys, xs = np.nonzero(m)
    cx = 0.5 * (xs.min() + xs.max()); cy = 0.5 * (ys.min() + ys.max())
    r = 0.25 * ((xs.max() - xs.min()) + (ys.max() - ys.min()))
    return cx, cy, r

# ---- pick an image with many filaments
iid = max(by_img, key=lambda k: len(by_img[k]))
f = D / fname[iid]
img = np.array(Image.open(f).convert("L")).astype(np.float32)
cx, cy, r = fit_disk(img)
anns = by_img[iid]
print(f"image {f.name}  filaments={len(anns)}  disk c=({cx:.0f},{cy:.0f}) R={r:.0f}")

fig = plt.figure(figsize=(20, 12))

# (1) full disk with annotations
ax = fig.add_subplot(2, 3, 1)
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
for a in anns:
    p = a["segmentation"][0]
    ax.plot(p[0::2] + p[0:1], p[1::2] + p[1:2], "r-", lw=1.0)
    s = a["spine"]
    ax.plot(s[0::2], s[1::2], "c-", lw=1.0)
ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec="lime", lw=1.0))
ax.set_title(f"{f.name}\n{len(anns)} filaments (red=mask, cyan=spine, green=fitted limb)", fontsize=10)
ax.axis("off")

# (2) limb ring zoom - a wedge at the right edge
ax = fig.add_subplot(2, 3, 2)
x0 = int(cx + r - 260); y0 = int(cy - 200)
ax.imshow(img[y0:y0 + 400, x0:x0 + 400], cmap="gray", vmin=0, vmax=255)
ax.set_title("LIMB ZOOM — the bright ring just inside the edge", fontsize=10)
ax.axis("off")

# (3) radial profile, fine sampling near the limb
ax = fig.add_subplot(2, 3, 3)
yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
bins = np.linspace(0.75, 1.10, 176)
idx = np.digitize(rr.ravel(), bins); v = img.ravel()
prof = np.array([np.median(v[idx == i]) if (idx == i).sum() > 30 else np.nan for i in range(1, len(bins))])
ax.plot(0.5 * (bins[1:] + bins[:-1]), prof, lw=2)
ax.axvline(1.0, color="lime", ls="--"); ax.grid(alpha=.3)
ax.set_xlabel("r / R_sun"); ax.set_ylabel("median brightness")
ax.set_title("radial profile near the limb", fontsize=10)

# (4,5) two filament close-ups
big = sorted(anns, key=lambda a: -a["area"])[:2]
for k, a in enumerate(big):
    ax = fig.add_subplot(2, 3, 4 + k)
    x, y, w, h = a["bbox"]; pad = 60
    X0, Y0 = int(max(x - pad, 0)), int(max(y - pad, 0))
    X1, Y1 = int(min(x + w + pad, 2048)), int(min(y + h + pad, 2048))
    sub = img[Y0:Y1, X0:X1]
    ax.imshow(sub, cmap="gray", vmin=sub.min(), vmax=sub.max())
    p = a["segmentation"][0]
    ax.plot([q - X0 for q in p[0::2]] + [p[0] - X0], [q - Y0 for q in p[1::2]] + [p[1] - Y0], "r-", lw=1.6)
    s = a["spine"]
    ax.plot([q - X0 for q in s[0::2]], [q - Y0 for q in s[1::2]], "c-", lw=1.6)
    ax.set_title(f"filament close-up  area={a['area']:.0f}px  ({X1-X0}x{Y1-Y0} crop)", fontsize=10)
    ax.axis("off")

# (6) brightness histogram: inside filaments vs quiet disk
ax = fig.add_subplot(2, 3, 6)
mask = Image.new("L", (2048, 2048), 0); dr = ImageDraw.Draw(mask)
for a in anns:
    p = a["segmentation"][0]
    dr.polygon(list(zip(p[0::2], p[1::2])), fill=1)
M = np.array(mask, bool)
ondisk = rr < 0.95
ax.hist(img[M & ondisk], bins=60, range=(0, 200), density=True, alpha=.6, label="inside filaments")
ax.hist(img[~M & ondisk], bins=60, range=(0, 200), density=True, alpha=.6, label="quiet disk (r<0.95)")
ax.legend(); ax.set_xlabel("pixel brightness"); ax.set_title("are filaments simply 'dark'?", fontsize=10)
q = np.percentile(img[~M & ondisk], [1, 5, 50])
print("quiet disk percentiles 1/5/50:", q)
print("filament pixels median:", np.median(img[M & ondisk]))
print("frac of quiet-disk px darker than filament median: %.3f" %
      np.mean(img[~M & ondisk] < np.median(img[M & ondisk])))

plt.tight_layout(); plt.savefig(OUT / "02_zoom.png", dpi=80); plt.close()
print("saved 02_zoom.png")
