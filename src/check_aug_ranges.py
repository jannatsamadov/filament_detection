"""How much do sharpness, brightness and contrast vary from frame to frame?

The spread is what a photometric / blur augmentation has to cover. Measured on
train and test separately, so we can see whether the test frames fall inside
the training range or beyond it (beyond is where augmentation earns its keep).

  brightness  median of the central disk, grey levels
  contrast    5-95% spread of the central disk, grey levels (large-scale)
  sharpness   robust sigma of the 9 px high-pass, divided by the disk median
              -- fine structure relative to brightness, which is what seeing blurs
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

root = Path(__file__).resolve().parents[1] / "MAGFiLO_1.0_Kaggle_2026"
yy, xx = np.mgrid[:2048, :2048]
inner = (xx - 1024) ** 2 + (yy - 1024) ** 2 < 700 ** 2


def stats(f):
    a = np.asarray(Image.open(f).convert("L")).astype(np.float32)
    d = a[inner]
    med = float(np.median(d))
    p5, p95 = np.percentile(d, [5, 95])
    c = a[724:1324, 724:1324]
    hp = c - uniform_filter(c, 9)
    sharp = float(np.median(np.abs(hp)) * 1.4826) / med
    return med, p95 - p5, sharp, f.stem[-2:]


rows = {}
for split, step in (("train", 2), ("test", 1)):
    files = sorted((root / split / f"{split}_images").glob("*.jpeg"))[::step]
    rows[split] = [stats(f) for f in files]

names = ("brightness", "contrast", "sharpness")
print(f"{'':12s} {'split':6s} {'n':>4s} {'p5':>8s} {'p50':>8s} {'p95':>8s}  p95/p5")
for i, nm in enumerate(names):
    for split in ("train", "test"):
        v = np.array([r[i] for r in rows[split]])
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        print(f"{nm:12s} {split:6s} {len(v):4d} {p5:8.4f} {p50:8.4f} {p95:8.4f}  {p95 / p5:5.2f}")

print("\nsharpness by station (train+test), median  -- seeing differs by site")
allr = rows["train"] + rows["test"]
for st in sorted({r[3] for r in allr}):
    v = [r[2] for r in allr if r[3] == st]
    print(f"  {st}: {np.median(v):.4f}  (n={len(v)})")
