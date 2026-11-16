"""Report figure for the augmentation run: does heavy augmentation beat flips alone?

Reads what the Kaggle kernel wrote (fetch it with
`kaggle kernels output jannatsamadov/filament-aug-unet -p kernels/aug_unet/out_kaggle`)
and draws, for the two trainings side by side:

  top     validation PQ per epoch, and TP / FP / FN at each run's best epoch
  bottom  one validation frame, zoomed on its largest filament: what the
          annotators drew, what each model found (with its own tuned
          post-processing)

Every number here is our scorer on our held-out photographs -- validation,
not the leaderboard. It has run 0.05-0.09 above the official score before.
"""
from __future__ import annotations

import argparse, collections, json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mu
from scipy.ndimage import binary_dilation, binary_erosion

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN = ROOT / "MAGFiLO_1.0_Kaggle_2026/train/train_images"

# categorical slots 1-3 of the reference palette, validated together
COLOUR = {"flip": "#2a78d6", "full": "#eb6834", "gt": "#1baf7a"}
LABEL = {"flip": "flips only (control)", "full": "heavy augmentation"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FULL = 2048


def components(prob_u8, t, k, min_area):
    b = (prob_u8 >= int(round(255 * t))).astype(np.uint8)
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        lab = cv2.connectedComponents(cv2.dilate(b, ker), connectivity=8)[1]
        lab[b == 0] = 0
    else:
        lab = cv2.connectedComponents(b, connectivity=8)[1]
    area = np.bincount(lab.ravel())
    keep = area >= min_area
    keep[0] = False
    return keep[lab]


def rim(m, width=2):
    r = m ^ binary_erosion(m, iterations=1)
    return binary_dilation(r, iterations=width - 1) if width > 1 else r


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "out_kaggle"))
    ap.add_argument("--out", default=str(ROOT / "figures" / "aug_unet_report.png"))
    args = ap.parse_args()
    d = Path(args.dir)
    res = {a: json.load(open(d / f"result_{a}.json")) for a in ("flip", "full")
           if (d / f"result_{a}.json").exists()}

    fig = plt.figure(figsize=(16, 10.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1.25), hspace=0.28, wspace=0.18)

    # -- validation PQ per epoch
    ax = fig.add_subplot(gs[0, :2]); style(ax)
    for a, r in res.items():
        e = [v["epoch"] for v in r["val"]]; pq = [v["PQ"] for v in r["val"]]
        b = r["best"]
        ax.plot(e, pq, color=COLOUR[a], linewidth=2, marker="o", markersize=6,
                label=f"{LABEL[a]} -- best {b['PQ']:.3f} (epoch {b['epoch']})")
    ax.set_xlabel("epoch", color=INK2); ax.set_ylabel("validation PQ", color=INK2)
    ax.set_title("Validation PQ per epoch -- our scorer, held-out photographs, every annotation set",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    # -- TP / FP / FN at each run's best epoch
    ax = fig.add_subplot(gs[0, 2]); style(ax)
    keys, w = ("TP", "FP", "FN"), 0.38
    for i, (a, r) in enumerate(res.items()):
        vals = [r["best"][k] for k in keys]
        xs = np.arange(3) + (i - 0.5) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=COLOUR[a], label=LABEL[a])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{int(v)}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["found (TP)", "false (FP)", "missed (FN)"])
    ax.set_title("At each run's best epoch", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9)

    # -- one validation frame
    stems = sorted({p.stem.split("_", 2)[2] for p in d.glob("valprob_*_*.png")})
    if stems:
        stem = stems[0]
        coco = json.load(open(ANN))
        sets = [i["id"] for i in coco["images"] if Path(i["file_name"]).stem == stem]
        gt = np.zeros((FULL, FULL), bool)
        big = None
        for a in coco["annotations"]:
            if a["image_id"] in sets:
                m = mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], FULL, FULL))).astype(bool)
                gt |= m
                if big is None or m.sum() > big[0]:
                    big = (m.sum(), m)
        ys, xs = np.nonzero(big[1])
        cy, cx = int(ys.mean()), int(xs.mean())
        h = 300
        y0, x0 = np.clip(cy - h, 0, FULL - 2 * h), np.clip(cx - h, 0, FULL - 2 * h)
        win = (slice(y0, y0 + 2 * h), slice(x0, x0 + 2 * h))
        img = cv2.imread(str(TRAIN / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)[win]
        g = np.clip((img.astype(float) - np.percentile(img, 1)) / (np.ptp(np.percentile(img, [1, 99.5])) + 1e-6), 0, 1)
        base = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)

        panels = [("annotators (union)", gt[win], "gt")]
        for a, r in res.items():
            p = cv2.imread(str(d / f"valprob_{a}_{stem}.png"), cv2.IMREAD_GRAYSCALE)
            b = r["best"]
            panels.append((f"{LABEL[a]}\nthresh {b['thresh']} merge {b['merge']} min area {b['min_area']}",
                           components(p, b["thresh"], b["merge"], b["min_area"])[win], a))
        for j, (title, m, c) in enumerate(panels):
            ax = fig.add_subplot(gs[1, j])
            im = base.copy()
            im[rim(m)] = np.round(np.array(matplotlib.colors.to_rgb(COLOUR[c])) * 255)
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=11, color=INK, loc="left")
        fig.text(0.125, 0.02, f"{stem}, {2 * h} px around its largest filament; outlines only", fontsize=10, color=INK2)

    fig.suptitle("Heavy augmentation vs flips only -- full-frame U-Net (ResNet-34), Kaggle T4",
                 x=0.125, ha="left", fontsize=15, color=INK)
    fig.savefig(args.out, dpi=90, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    for a, r in res.items():
        b = r["best"]
        print(f"{a:5s} best epoch {b['epoch']:3d}  PQ {b['PQ']:.4f}  SQ {b['SQ']:.3f}  RQ {b['RQ']:.3f}  "
              f"TP {int(b['TP'])} FP {int(b['FP'])} FN {int(b['FN'])}  PQ(60 sets) {b['PQ60']:.4f}  "
              f"epochs run {len(r['history'])}")


if __name__ == "__main__":
    main()
