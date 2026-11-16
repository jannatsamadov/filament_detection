"""figures/yolo_v2.png from kernels/yolo/out_kaggle/yolo/yolo.json: YOLO alone, and as a vote in the keep-rule."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "out_kaggle" / "yolo" / "yolo.json"))
INK, MUTED, GRID, BLUE, ORANGE = "#1f2328", "#6e7781", "#d0d7de", "#2f6fdb", "#d9822b"

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

cuts = sorted(r["yolo_alone_fold0_pq"], key=float)
pq = [r["yolo_alone_fold0_pq"][c] for c in cuts]
a.plot([float(c) for c in cuts], pq, color=ORANGE, lw=2, marker="o", ms=8, mec="white", mew=2)
a.axhline(0.4248, color=BLUE, lw=2, ls="--")
a.text(0.1, 0.4248 + 0.006, "tək U-Net R50: 0.4248", color=INK, fontsize=10)
k = max(range(len(pq)), key=pq.__getitem__)
a.annotate(f"YOLO tək başına: {pq[k]:.3f}", (float(cuts[k]), pq[k]), (0.42, 0.33), color=INK, fontsize=10,
           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set(xlabel="YOLO confidence həddi", ylabel="fold 0 val PQ", ylim=(0.18, 0.46))
a.set_title("YOLO11l-seg tək başına (fold 0)", loc="left", color=INK, fontsize=12)

n = r["ens0_nested_pq"]
names = ["mövcud seçici\n(GBM 14)", "seçici + YOLO səsi\n(+λ·y_near)"]
vals = [n["gbm14_all"], n["gbm14+vote"]]
# dots, not bars: the axis does not start at zero
b.scatter([0, 1], vals, s=110, color=[BLUE, ORANGE], edgecolor="white", lw=2, zorder=3)
for x, v in enumerate(vals):
    b.text(x + 0.08, v, f"{v:.4f}", va="center", color=INK, fontsize=11)
d = r["vote_minus_gbm14_all"]
b.text(0.5, 0.4562, f"fərq {d['mean']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]", ha="center", color=INK, fontsize=10.5)
b.axhline(0.4517, color=MUTED, lw=1, ls=":")
b.text(1.45, 0.4517 - 0.0009, "kf10_rule_geo-nun seçicisi: 0.4517", ha="right", color=MUTED, fontsize=9)
b.set_xticks([0, 1], names)
b.set(xlim=(-0.5, 1.5), ylabel="fold 0 ansambl, nested PQ", ylim=(0.440, 0.458))
b.set_title("YOLO seçiciyə əlavə edildikdə", loc="left", color=INK, fontsize=12)
fig.tight_layout()
out = HERE.parents[1] / "figures" / "yolo_v2.png"
fig.savefig(out, dpi=130)
print(out)
