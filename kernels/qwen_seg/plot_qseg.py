"""figures/qwen_seg_v3.png: the Qwen-ViT segmenter's learning curve, and one fold-0 photograph
as the annotators drew it, a single U-Net R50 saw it, and Qwen without / with pretraining.

Reads kernels/qwen_seg/out_kaggle (logs) and the maps unpacked into kernels/aug_post/out/dump."""
import json, re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DUMP = HERE.parent / "aug_post" / "out" / "dump"
INK, MUTED, GRID, BLUE, ORANGE, GRAY = "#1f2328", "#6e7781", "#d0d7de", "#2f6fdb", "#d9822b", "#8c959f"

curve = {}
for arm in ("ft", "pre"):
    txt = (HERE / "out_kaggle" / f"log_qseg_{arm}.txt").read_text()
    curve[arm] = [(int(e), float(v)) for e, v in re.findall(r"finetune epoch (\d+): val PQ ([\d.]+)", txt)]
    curve[arm + "_final"] = json.load(open(HERE / "out_kaggle" / f"qwenseg_{arm}.json"))["best"]["PQ"]

ann = next((ROOT / "MAGFiLO_1.0_Kaggle_2026").rglob("*_train.json"))
tr_dir = next(p for p in (ROOT / "MAGFiLO_1.0_Kaggle_2026").rglob("train_images") if p.is_dir())
coco = json.load(open(ann))
val = json.load(open(HERE / "out_kaggle" / "dump_qwenseg_pre.json"))["val_files"]
by_file = {}
for im in coco["images"]:
    by_file.setdefault(im["file_name"], []).append(im["id"])
anns = {}
for a in coco["annotations"]:
    anns.setdefault(a["image_id"], []).append(a)
# a photograph with a typical number of filaments for its first annotation set
pick = next(f for f in val if 7 <= len(anns.get(by_file[f][0], [])) <= 9)
stem = Path(pick).stem
img = cv2.imread(str(tr_dir / pick), cv2.IMREAD_GRAYSCALE)
S = 512
small = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
rgb = np.dstack([small] * 3).copy()
for a in anns[by_file[pick][0]]:
    for poly in a["segmentation"]:
        pts = (np.array(poly).reshape(-1, 2) * S / img.shape[0]).round().astype(np.int32)
        cv2.polylines(rgb, [pts], True, (40, 220, 255), 1)


def load(model):
    p = cv2.imread(str(DUMP / model / "val_plain" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(p, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


fig = plt.figure(figsize=(14, 7.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.28, wspace=0.04)
panels = [(rgb, "foto + annotator 1-in konturları", None), (load("r50"), "U-Net R50 (tək model)", "magma"),
          (load("qwenseg_ft"), "Qwen, pretrain-siz", "magma"), (load("qwenseg_pre"), "Qwen, pretrain + fine-tune", "magma")]
for k, (im, title, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(im, cmap=cmap, vmin=0, vmax=1) if cmap else ax.imshow(im)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.axis("off")

ax = fig.add_subplot(gs[1, 1:3])
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6)
for arm, color, name in (("ft", GRAY, "Qwen, pretrain-siz"), ("pre", ORANGE, "Qwen, pretrain + fine-tune")):
    e, v = zip(*curve[arm])
    ax.plot(e, v, color=color, lw=2, marker="o", ms=8, mec="white", mew=2)
    ax.text(12.25, curve[arm + "_final"], f"{name}: {curve[arm + '_final']:.4f}", va="center", color=INK, fontsize=10)
ax.axhline(0.4248, color=BLUE, lw=2, ls="--")
ax.text(6, 0.4248 + 0.002, "tək U-Net R50: 0.4248", color=INK, fontsize=10)
ax.set(xlabel="fine-tune epoxası", ylabel="fold 0 val PQ", xticks=[6, 9, 12], xlim=(5.5, 12.3), ylim=(0.37, 0.432))
ax.set_title("Öyrənmə əyrisi (fold 0, 141 foto)", loc="left", color=INK, fontsize=12)
out = ROOT / "figures" / "qwen_seg_v3.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(out, stem)
