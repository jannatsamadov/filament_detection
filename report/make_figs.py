"""Figures of the technical report: report/fig_pipeline.pdf (drawn here) and report/fig_ceiling.png
(figures/ideal_model.png, made by kernels/aug_unet/ideal_model_fig.py).

    python report/make_figs.py
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
UNET, QWEN, YOLO, POST = "#DCEAF6", "#F6E3EF", "#FCEBD2", "#EEEEEE"


def box(ax, x, y, w, h, text, color, size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=color, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


def pipeline():
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.00, 0.40, 0.13, 0.20, "GONG\nH$\\alpha$\n2048$^2$", POST, 5.8)
    box(ax, 0.19, 0.70, 0.31, 0.26, "10 U-Nets\nR34 & R50 $\\times$ 5 folds\nsoft target, D4", UNET, 5.8)
    box(ax, 0.19, 0.37, 0.31, 0.26, "Qwen3-VL ViT\nsegmenter, pre-\ntrained on GONG", QWEN, 5.8)
    box(ax, 0.19, 0.04, 0.31, 0.26, "YOLO11l-seg\n$\\times$ 5 folds", YOLO, 5.8)
    box(ax, 0.57, 0.64, 0.24, 0.32, "mean map\n(Qwen weight 2/12)\n$\\downarrow$ candidates:\nt 0.6, join 3 px,\narea $\\geq$ 100 px", POST, 5.0)
    box(ax, 0.57, 0.04, 0.24, 0.50, "selection rule\nGBM, 25 features:\nappearance,\nheliographic\nlat. & long.,\naxis orientation,\nYOLO agreement;\ncut 0.18", POST, 5.0)
    box(ax, 0.87, 0.40, 0.13, 0.20, "masks\n(RLE)", POST, 5.8)
    for y in (0.83, 0.50, 0.17):
        arrow(ax, 0.13, 0.50, 0.19, y)
    arrow(ax, 0.50, 0.83, 0.57, 0.83)
    arrow(ax, 0.50, 0.50, 0.57, 0.72)
    arrow(ax, 0.69, 0.64, 0.69, 0.54)
    arrow(ax, 0.50, 0.17, 0.57, 0.20)
    arrow(ax, 0.81, 0.29, 0.87, 0.47)
    fig.savefig(HERE / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "fig_pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0.02)


def test_morphology():
    """fig_test.jpg: the submitted file's own masks on a test photograph (full disk and two close-ups)."""
    import csv
    import collections
    import cv2
    import numpy as np
    from pycocotools import mask as mu
    root = HERE.parent
    sub = root / "kernels" / "kfold_yolo" / "out_kaggle" / "submission_kf10q_rule_geo_yolo.csv"
    test_dir = next(p for p in (root / "MAGFiLO_1.0_Kaggle_2026").rglob("test_images") if p.is_dir())
    by = collections.defaultdict(list)
    with open(sub) as f:
        for row in csv.DictReader(f):
            by[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    stem = max(by, key=lambda s: len(by[s]))                     # the test photograph with most filaments
    img = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    masks = [mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(np.uint8) for r in by[stem]]
    for m in masks:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, (255, 140, 0), 2)
    areas = [int(m.sum()) for m in masks]
    big = [masks[i] for i in np.argsort(areas)[::-1][:2]]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.04))
    axs[0].imshow(rgb[::2, ::2])
    axs[0].set_title(f"{stem}: {len(masks)} filaments", fontsize=7)
    for ax, m in zip(axs[1:], big):
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = max(150, int(max(np.ptp(ys), np.ptp(xs)) * 0.65))
        y0, x0 = max(cy - h, 0), max(cx - h, 0)
        ax.imshow(rgb[y0:cy + h, x0:cx + h])
        axs[0].add_patch(plt.Rectangle((x0 / 2, y0 / 2), h, h, fill=False, ec="#56B4E9", lw=0.8))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(HERE / "fig_test.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02, pil_kwargs=dict(quality=88))
    print("test photograph", stem, len(masks), "filaments")


if __name__ == "__main__":
    test_morphology()
    pipeline()
    shutil.copy(HERE.parent / "figures" / "ideal_model.png", HERE / "fig_ceiling.png")
    print("wrote", HERE / "fig_pipeline.pdf", HERE / "fig_ceiling.png")
