"""How many distinct 8-bit grey levels does a filament actually get in the JPEGs?"""
import json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mu

root = Path(__file__).resolve().parents[1] / "MAGFiLO_1.0_Kaggle_2026" / "train"
coco = json.load(open(root / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
imgs = {i["id"]: i for i in coco["images"]}
by = {}
for a in coco["annotations"]:
    by.setdefault(a["image_id"], []).append(a)
random.seed(0)
ids = random.sample([k for k in by if len(by[k]) >= 3], 40)
fil_lv, disk_lv, fil_rng, gap = [], [], [], []
for iid in ids:
    im = np.asarray(Image.open(root / "train_images" / imgs[iid]["file_name"]).convert("L"))
    disk = im > 20
    disk_lv.append(len(np.unique(im[disk])))
    for a in by[iid]:
        rle = mu.frPyObjects(a["segmentation"], 2048, 2048)
        m = mu.decode(mu.merge(rle)).astype(bool)
        v = im[m]
        if v.size < 50:
            continue
        fil_lv.append(len(np.unique(v)))
        p5, p95 = np.percentile(v, [5, 95])
        fil_rng.append(p95 - p5)
        # darkness of the filament against its own bounding-box surroundings
        x, y, w, h = (int(round(t)) for t in a["bbox"])
        box = im[max(y - 20, 0):y + h + 20, max(x - 20, 0):x + w + 20]
        gap.append(np.median(box) - np.median(v))
print(f"images {len(ids)}, filaments {len(fil_lv)}")
print(f"grey levels on the whole disk        : median {np.median(disk_lv):.0f} of 256")
print(f"grey levels inside one filament      : median {np.median(fil_lv):.0f}, 10th pct {np.percentile(fil_lv, 10):.0f}")
print(f"5-95% spread inside one filament     : median {np.median(fil_rng):.0f} levels")
print(f"filament darker than its surroundings: median {np.median(gap):.0f} levels, 10th pct {np.percentile(gap, 10):.0f}")

# Is 8-bit quantisation the limit at all? Compare its step (1 level, rms error
# 1/sqrt(12) = 0.29) with the pixel-to-pixel scatter of the quiet Sun itself.
from scipy.ndimage import uniform_filter
sig = []
for iid in ids[:20]:
    im = np.asarray(Image.open(root / "train_images" / imgs[iid]["file_name"]).convert("L")).astype(np.float64)
    c = im[724:1324, 724:1324]                       # central 600x600, away from the limb
    hp = c - uniform_filter(c, 9)                    # remove large-scale shading
    sig.append(np.median(np.abs(hp)) * 1.4826)       # robust sigma, filaments barely move it
print(f"quiet-Sun small-scale scatter       : median {np.median(sig):.1f} levels "
      f"(8-bit rounding adds 0.29)")
