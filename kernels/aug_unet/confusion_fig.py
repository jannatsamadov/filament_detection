"""Confusion matrices of the best RF, U-Net and PixNet models, per pixel and per filament (user, 11-15).

Fold 0: 141 validation photographs no model trained on.
  pixel level     truth = pixels drawn by at least one expert; from out/pixel_cls.npz (the 256-level histograms),
                  at threshold 0.5 and at each model's own best-F1 threshold
  filament level  a predicted filament is TP when IoU > 0.5 with one expert's filament (each expert set scored
                  on its own, as the official metric does); FP = predicted, no match; FN = expert's, no match.
                  There is no TN at this level: "not a filament" is not an object.
                  Numbers from stage2_baselines.py / rf_grouping.py / kfold_q2 (selector on held-out halves)
    python kernels/aug_unet/confusion_fig.py      -> figures/confusion_matrices.png, out/confusion.json
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H = np.load(HERE / "out" / "pixel_cls.npz")

# shown name -> pixel_cls model name, filament rows (setting, TP, FP, FN)
MODELS = {
    "RF: table + U-Net\n(best RF)": ("RF, table + U-Net pair", [
        ("fixed rules (t 0.5, join 24 px)", 1235, 1134, 614),
        ("+ learned selector", 1147, 526, 702)]),
    "RF: table only\n(no U-Net input)": ("RF, pixel table only", [
        ("fixed rules (t 0.4, join 24 px)", 789, 3222, 1060),
        ("+ learned selector (pool)", 647, 673, 1202)]),
    "U-Net R50\n(best single U-Net)": ("U-Net R50 (one model)", [
        ("fixed rules (t 0.6, join 6 px)", 1248, 1025, 601),
        ("+ learned selector", 1150, 536, 699)]),
    "5 U-Nets + Qwen\n(submitted)": ("5 U-Nets + Qwen (submitted mix)", [
        ("fixed rules (t 0.6, join 6 px)", 1261, 1008, 588),
        ("submitted rule (707 photos)", 1172, 456, 677)]),
    "PixNet\n(best PixNet)": ("PixNet, 30 epochs", [
        ("fixed rules (t 0.6, join 6 px)", 1178, 1420, 671),
        ("+ learned selector", 1001, 595, 848)]),
}


def at(pos, neg, t):
    c = int(round(255 * t))
    return int(pos[c:].sum()), int(neg[c:].sum()), int(pos[:c].sum()), int(neg[:c].sum())


def best_t(pos, neg):
    best = (0.0, 0.5)
    for c in range(1, 255):
        tp, fp = pos[c:].sum(), neg[c:].sum()
        fn = pos[:c].sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, c / 255)
    return round(best[1], 3)


def metrics(tp, fp, fn, tn=None):
    m = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), F1=2 * tp / max(2 * tp + fp + fn, 1),
             IoU=tp / max(tp + fp + fn, 1))
    if tn is not None:
        m["accuracy"] = (tp + tn) / (tp + fp + fn + tn)
    return {k: round(v, 4) for k, v in m.items()}


out = {}
fig, axes = plt.subplots(3, len(MODELS), figsize=(4.0 * len(MODELS), 11.5))
for j, (shown, (name, rows)) in enumerate(MODELS.items()):
    pos, neg = H[f"{name}|pos"], H[f"{name}|neg"]
    tb = best_t(pos, neg)
    rec = dict(pixel={}, filament={})
    for i, t in enumerate((0.5, tb)):
        tp, fp, fn, tn = at(pos, neg, t)
        rec["pixel"][f"t={t}"] = dict(TP=tp, FP=fp, FN=fn, TN=tn, **metrics(tp, fp, fn, tn))
        ax = axes[i, j]
        mat = np.array([[tp, fn], [fp, tn]], float)
        norm = mat / mat.sum(1, keepdims=True)                 # share of each true class
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(mat):
            ax.text(c, r, f"{v / 1e6:.2f} M\n{norm[r, c]:.1%}", ha="center", va="center",
                    color="white" if norm[r, c] > 0.6 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["predicted\nfilament", "predicted\nbackground"])
        ax.set_yticks([0, 1], ["expert:\nfilament", "expert:\nbackground"])
        mm = rec["pixel"][f"t={t}"]
        ax.set_title(f"{shown}\npixels, threshold {t}\nP {mm['precision']:.3f}  R {mm['recall']:.3f}  "
                     f"F1 {mm['F1']:.3f}  IoU {mm['IoU']:.3f}", fontsize=9)
    ax = axes[2, j]
    labels, y = [], np.arange(len(rows))
    for i, (setting, tp, fp, fn) in enumerate(rows):
        m = metrics(tp, fp, fn)
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        rec["filament"][setting] = dict(TP=tp, FP=fp, FN=fn, RQ=round(rq, 4), **m)
        left = 0
        for v, col, lab in ((tp, "#2a6fbb", "TP"), (fp, "#e3a33b", "FP"), (fn, "#b9b9b9", "FN")):
            ax.barh(i, v, left=left, color=col, edgecolor="white", linewidth=2, label=lab if i == 0 else None)
            ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=9)
            left += v
        labels.append(f"{setting}\nP {m['precision']:.2f} R {m['recall']:.2f} F1 {m['F1']:.2f}")
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("filaments (1849 expert filaments)")
    ax.set_title(f"{shown.splitlines()[0]}: filaments", fontsize=9)
    if j == 0:
        ax.legend(loc="lower right", fontsize=8)
    out[shown.replace("\n", " ")] = rec
fig.suptitle("Fold 0, 141 validation photographs. Rows 1-2: per pixel (truth = drawn by any expert). "
             "Row 3: per filament (match = IoU > 0.5; no TN at this level)", fontsize=11)
fig.tight_layout()
dst = ROOT / "figures" / "confusion_matrices.png"
fig.savefig(dst, dpi=110)
json.dump(out, open(HERE / "out" / "confusion.json", "w"), indent=1)
for k, v in out.items():
    print(k)
    for lvl in ("pixel", "filament"):
        for s, m in v[lvl].items():
            print(f"   {lvl:8s} {s:34s} " + " ".join(f"{a} {b}" for a, b in m.items()))
print(dst)
