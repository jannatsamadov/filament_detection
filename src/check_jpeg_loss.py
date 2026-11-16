"""What did the FITS -> 8-bit JPEG conversion throw away?

Two things could make FITS worth having for training: clipping (filament cores
or the bright disk pinned at 0/255) and heavy JPEG compression (8x8 blocks that
smear thin barbs). Bit depth itself was already ruled out (0.29 vs 4.4 levels).
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1] / "MAGFiLO_1.0_Kaggle_2026"
files = sorted((root / "train" / "train_images").glob("*.jpeg"))[::7] + \
        sorted((root / "test" / "test_images").glob("*.jpeg"))[::3]

low, high, qual = [], [], []
for f in files:
    im = Image.open(f)
    a = np.asarray(im.convert("L"))
    yy, xx = np.mgrid[:2048, :2048]
    disk = (xx - 1024) ** 2 + (yy - 1024) ** 2 < 800 ** 2      # well inside the limb
    d = a[disk]
    low.append(np.mean(d <= 1)); high.append(np.mean(d >= 254))
    # luminance quantisation table -> the encoder's quality setting (IJG scaling)
    q = getattr(im, "quantization", None)
    if q:
        t = np.array(q[0], float)
        std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                        14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                        18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                        49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
        s = np.median(t / std) * 100
        qual.append(100 - s / 2 if s <= 100 else 5000 / s)
print(f"frames checked: {len(files)}")
print(f"disk pixels clipped to black (<=1) : max {100*max(low):.3f}%  median {100*np.median(low):.4f}%")
print(f"disk pixels clipped to white (>=254): max {100*max(high):.3f}%  median {100*np.median(high):.4f}%")
if qual:
    print(f"JPEG quality estimate: min {min(qual):.0f}  median {np.median(qual):.0f}  max {max(qual):.0f}")
