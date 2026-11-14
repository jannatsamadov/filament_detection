"""The user's pixel table, gradient boosting against random forest (figures/pixel_table.png).

Top: pixel-level AUC, object-level PQ out-of-fold on all 707 photographs, and what each adds to the
ensemble (difference with 95% bootstrap interval). Bottom: one fold-0 validation photograph, the
U-Net pair's map and the random forest's two maps, with one expert's outlines.

    python kernels/aug_unet/pixel_fig.py
"""
import collections, json, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

KER = HERE.parent
OUT = HERE.parents[1] / "figures" / "pixel_table.png"
GREY, GBM, RF = "#8a8a8a", "#0072B2", "#E69F00"      # Okabe-Ito blue and orange: apart under colour blindness


def main():
    g = json.load(open(KER / "pixel" / "out_kaggle" / "pixel.json"))
    r = json.load(open(KER / "pixel_rf" / "out_kaggle" / "pixel.json"))
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.22, wspace=0.45)

    # A. pixel AUC, mean over the five folds with their range
    ax = fig.add_subplot(gs[0, 0])
    rows = [("U-Net pair\n(its probability)", g["pixel_auc_oof"]["unet_p_alone"], GREY),
            ("table only\nGBM", g["pixel_auc_oof"]["pixA"], GBM), ("table only\nRF", r["pixel_auc_oof"]["pixA"], RF),
            ("table + U-Net\nGBM", g["pixel_auc_oof"]["pixB"], GBM), ("table + U-Net\nRF", r["pixel_auc_oof"]["pixB"], RF)]
    for i, (name, v, c) in enumerate(rows):
        m = np.mean(v)
        ax.barh(i, m, color=c, height=0.6)
        ax.plot([min(v), max(v)], [i, i], color="black", lw=1)
        ax.text(m + 0.003, i, f"{m:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [n for n, _, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.88, 1.01)
    ax.set_title("A. pixel level: AUC (filament or not)\nmean of 5 folds, line = range", fontsize=10)

    # B. object level PQ out-of-fold
    ax = fig.add_subplot(gs[0, 1])
    rows = [("U-Net pair", g["oof_grid"]["pair"], GREY), ("table only GBM", g["oof_grid"]["pixA"], GBM),
            ("table only RF", r["oof_grid"]["pixA"], RF), ("table + U-Net GBM", g["oof_grid"]["pixB"], GBM),
            ("table + U-Net RF", r["oof_grid"]["pixB"], RF)]
    for i, (name, v, c) in enumerate(rows):
        ax.barh(i, v["PQ"], color=c, height=0.6)
        ax.text(v["PQ"] + 0.008, i - 0.12, f"{v['PQ']:.3f}", va="center", fontsize=9)
        ax.text(v["PQ"] + 0.008, i + 0.2, f"SQ {v['SQ']:.3f}, RQ {v['RQ']:.3f}", va="center", fontsize=7, color="#444")
    ax.set_yticks(range(len(rows)), [n.replace(" GBM", "\nGBM").replace(" RF", "\nRF") for n, _, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.62)
    ax.set_title("B. object level: PQ, 707 photographs out-of-fold\n(each map at its own best threshold, no keep-rule)",
                 fontsize=10)

    # C. what the table adds to the ensemble: difference with its 95% interval
    ax = fig.add_subplot(gs[0, 2:])
    tests = [("pair + table map (25% weight), keep-rule, minus the pair alone: 707 photographs", "rule_oof_diff",
              "pmix25_25_minus_pair_25"),
             ("fold-0 ensemble + Qwen + table map, minus the same without it (0.4578): 141 photographs", "transfer_diff",
              "ens0qp_25_minus_ens0q_25")]
    for i, (name, sec, key) in enumerate(tests):
        ax.text(-0.0115, i - 0.42, name, fontsize=9, va="center")
        for dy, (src, c, lab) in zip((-0.15, 0.15), ((g, GBM, "GBM"), (r, RF, "RF"))):
            d = src[sec][key]
            ax.plot([d["lo"], d["hi"]], [i + dy, i + dy], color=c, lw=2)
            ax.plot(d["mean"], i + dy, "o", color=c, ms=8, label=lab if i == 0 else None)
            ax.text(d["hi"] + 0.0006, i + dy, f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]", va="center",
                    fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks([])
    ax.set_ylim(-0.65, len(tests) - 0.6)
    ax.invert_yaxis()
    ax.set_xlim(-0.012, 0.02)
    ax.set_xlabel("PQ difference (95% bootstrap interval)")
    ax.legend(loc="center right", fontsize=9)
    ax.set_title("C. what the pixel table adds: no measurable gain with either learner", fontsize=10)

    # D. one photograph
    ann, tr_dir, _ = run.find_data()
    coco = json.load(open(ann))
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    maps = KER / "pixel_rf" / "out_kaggle" / "maps"
    by_stem = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in coco["annotations"]:
        by_stem[Path(fname[a["image_id"]]).stem][a["image_id"]].append(a)
    shown = sorted({p.name.split("_")[0] for p in maps.glob("*_pair.png")})
    stem = max(shown, key=lambda s: max(len(v) for v in by_stem[s].values()))   # the busiest of the three
    expert = max(by_stem[stem].values(), key=len)
    gt = np.zeros((run.FULL, run.FULL), np.uint8)
    for a in expert:
        gt |= run.ann_mask(a).astype(np.uint8)
    s = cv2.boxFilter(gt.astype(np.float32), -1, (900, 900), normalize=False)
    cy, cx = np.unravel_index(np.argmax(s), s.shape)
    y0, x0 = int(np.clip(cy - 450, 0, run.FULL - 900)), int(np.clip(cx - 450, 0, run.FULL - 900))
    win = (slice(y0, y0 + 900), slice(x0, x0 + 900))
    cs = cv2.findContours(gt[win].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    img = cv2.imread(str(tr_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    panels = [("Hα image, one expert's outlines", img[win], "gray"),
              ("U-Net pair: probability", cv2.imread(str(maps / f"{stem}_pair.png"), 0)[win], "magma"),
              ("RF, table only", cv2.imread(str(maps / f"{stem}_pixA.png"), 0)[win], "magma"),
              ("RF, table + U-Net", cv2.imread(str(maps / f"{stem}_pixB.png"), 0)[win], "magma")]
    for k, (title, a, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(a, cmap=cmap, vmin=0, vmax=255)
        for c in cs:
            c = c[:, 0]
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="cyan", lw=0.8)
        ax.set_title(f"D{k + 1}. {title}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The pixel table (35 columns per pixel, 12 M rows): gradient boosting vs random forest; "
                 f"photograph {stem} (fold-0 validation, 900 px crop)", fontsize=12, y=0.995)
    fig.savefig(OUT, dpi=105, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
