"""Figure for the pixel classification (figures/pixel_cls.png), from kernels/aug_unet/out/pixel_cls.json.

A: precision, recall, F1 (= Dice) and IoU of every model at threshold 0.5, against the union of the
experts, with AUC, AP and accuracy beside; the human row is one expert against the other experts.
B: precision-recall curves. C: which pixels each model calls filament on two fold-0 photographs
(green = filament both by the model and by an expert, vermillion = only the model, blue = only an expert).

    python kernels/aug_unet/pixel_cls_fig.py
"""
import json, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel_cls as PC  # noqa: E402
import run  # noqa: E402

TP_C, FP_C, FN_C = (0x00, 0x9E, 0x73), (0xD5, 0x5E, 0x00), (0x56, 0xB4, 0xE9)      # Okabe-Ito
LINE = {"U-Net R50 (one model)": "#0072B2", "5 U-Nets + Qwen (submitted mix)": "#000000",
        "Qwen-seg (ViT)": "#CC79A7", "PixNet, 30 epochs": "#E69F00",
        "RF, pixel table only": "#56B4E9", "RF, table + U-Net pair": "#009E73"}
SHOW = list(LINE)
CROP = 800


def crop_of(gt):
    s = cv2.boxFilter(gt.astype(np.float32), -1, (CROP, CROP), normalize=False)
    cy, cx = np.unravel_index(np.argmax(s), s.shape)
    y0 = int(np.clip(cy - CROP // 2, 0, run.FULL - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, run.FULL - CROP))
    return slice(y0, y0 + CROP), slice(x0, x0 + CROP)


def error_rgb(img, pred, gt):
    base = (0.45 * img.astype(np.float32))[..., None].repeat(3, 2)
    for m, c in ((pred & gt, TP_C), (pred & ~gt, FP_C), (~pred & gt, FN_C)):
        base[m] = c
    return base.astype(np.uint8)


def main():
    res = json.loads(PC.OUT_JSON.read_text())
    models = list(res["models"])
    shown = [m for m in SHOW if m in models]
    pw = 3.0                                                     # width of one photograph panel, inches
    W = max(pw * (len(shown) + 1), 16)
    hc = 2 * (0.74 * W / (len(shown) + 1) + 0.75)                  # height of the two photograph rows
    fig = plt.figure(figsize=(W, 6 + hc + 1.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, hc], hspace=0.3)
    top = gs[0].subgridspec(1, 2, width_ratios=[2.0, 1], wspace=0.22)

    # A. the metrics, one row per model, the human reference at the bottom
    ax = fig.add_subplot(top[0])
    h = res["human_one_expert_vs_union_of_others"]
    rows = [(m, res["models"][m]) for m in models] + [("human: one expert vs the other experts", h)]
    marks = [("precision", "o", "white", 7), ("recall", "s", "#555555", 6), ("F1", "D", "black", 7),
             ("IoU", "^", "#999999", 7)]
    for i, (name, r) in enumerate(rows):
        ax.plot([min(r[k] for k, *_ in marks), max(r[k] for k, *_ in marks)], [i, i], color="#cccccc", lw=1, zorder=1)
        for k, mk, fc, ms in marks:
            ax.plot(r[k], i, mk, mfc=fc, mec="black", ms=ms, zorder=2, label=k if i == 0 else None)
        extra = (f"F1 {r['F1']:.3f}  AUC {r['AUC']:.3f}  AP {r['AP']:.3f}  acc {r['accuracy']:.4f}"
                 if "AUC" in r else f"F1 {r['F1']:.3f}  acc {r['accuracy']:.4f}")
        ax.text(0.86, i, extra, va="center", fontsize=8.5, family="monospace")
    ax.axvline(h["F1"], color="black", lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)), [n for n, _ in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0.35, 1.27)
    ax.set_xticks(np.arange(0.4, 0.81, 0.1))
    ax.grid(axis="x", color="#eeeeee")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, ncol=4, frameon=False, bbox_to_anchor=(0.0, -0.05))
    bg = res["models"][models[0]]["all_background_accuracy"]
    ax.set_title(f"A. Pixel classification, fold-0 validation ({res['photographs']} photographs, 2048 px), "
                 f"threshold {res['threshold']}, truth = any expert\n"
                 f"accuracy is {bg:.4f} if every pixel is called background, so it says almost nothing; "
                 "dotted line = human F1", fontsize=10.5, loc="left")

    # B. precision-recall curves
    ax = fig.add_subplot(top[1])
    for m in shown:
        c = res["curves"][m]
        ax.plot(c["recall"], c["precision"], color=LINE[m], lw=2, label=m)
    ax.plot(h["recall"], h["precision"], "X", color="black", ms=10, label="human (one expert)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall (share of expert pixels found)")
    ax.set_ylabel("precision (share of claimed pixels that are expert pixels)")
    ax.grid(color="#eeeeee")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("B. Precision-recall over every threshold", fontsize=10.5, loc="left")

    # C. which pixels each model calls filament, two photographs
    per = res["per_photo"]
    sub = "5 U-Nets + Qwen (submitted mix)"
    f1 = {s: 2 * tp / max(2 * tp + fp + fn, 1) for s, (tp, fp, fn) in per[sub].items()}
    nfil = res["filaments_per_photo"]
    nexp = res["experts_per_photo"]
    busy = max((s for s in nfil if nexp[s] >= 2), key=nfil.get)
    hard = min((s for s in f1 if nfil[s] >= 4 and s != busy), key=f1.get)
    D = PC.Dumps()
    by = PC.experts({busy, hard})
    _, tr_dir, _ = run.find_data()
    gs2 = gs[1].subgridspec(2, len(shown) + 1, wspace=0.04, hspace=0.32)
    c = int(round(255 * res["threshold"]))
    for row, s in enumerate((busy, hard)):
        gt = np.logical_or.reduce(PC.expert_masks(by[s]))
        win = crop_of(gt)
        img = cv2.imread(str(tr_dir / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        ax = fig.add_subplot(gs2[row, 0])
        ax.imshow(img[win], cmap="gray", vmin=0, vmax=255)
        for cnt in cv2.findContours(gt[win].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            q = cnt[:, 0]
            ax.plot(np.r_[q[:, 0], q[0, 0]], np.r_[q[:, 1], q[0, 1]], color="#56B4E9", lw=0.8)
        ax.set_title(f"{s}\nHα, {len(by[s])} experts (outline = any)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        leg_ax = ax
        if row == 0:
            ax.text(0, 1.3, f"C. Which pixels each model calls filament (threshold {res['threshold']}, {CROP} px crop). "
                    "Top: the busiest photograph with 2 or more experts; bottom: the submitted mix's worst "
                    "photograph with 4 or more filaments", transform=ax.transAxes, fontsize=10.5)
        cache = {}
        for k, m in enumerate(shown):
            pred = D.model(m, s, cache) >= c
            tp, fp, fn = per[m][s]
            ax = fig.add_subplot(gs2[row, k + 1])
            ax.imshow(error_rgb(img[win], pred[win], gt[win]))
            ax.set_title(f"{m}\nF1 {2 * tp / max(2 * tp + fp + fn, 1):.3f} on this photograph", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    leg_ax.legend(handles=[Patch(color=np.array(TP_C) / 255, label="filament: model and expert"),
                        Patch(color=np.array(FP_C) / 255, label="only the model (false positive)"),
                        Patch(color=np.array(FN_C) / 255, label="only an expert (missed)")],
               loc="upper left", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.0, -0.04))
    PC.OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(PC.OUT_FIG, dpi=95, bbox_inches="tight")
    print("wrote", PC.OUT_FIG)


if __name__ == "__main__":
    main()
