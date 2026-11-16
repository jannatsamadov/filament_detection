"""Figure of a filament-kfold run: per-fold OOF PQ of each family, and the paired
bootstrap gaps.  Usage:  python plot_kfold.py out_kaggle_v3/kfold.json figures/kfold_v3.png"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1]); dst = Path(sys.argv[2])
k = json.loads(src.read_text())

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d8dee4"
SERIES = [("r50", "R50 (5 fold)", "#0072B2"),
          ("r34", "R34 (5 fold)", "#E69F00"),
          ("pair", "R50 + R34", "#009E73")]

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
for ax in (a, b):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)

# left: per fold + pooled OOF
rows = [f"fold {i}" for i in range(5)] + ["all 707"]
for j, (key, name, col) in enumerate(SERIES):
    vals = k[key]["by_fold"] + [k[key]["PQ"]]
    y = [i + (j - 1) * 0.22 for i in range(len(rows))]
    a.scatter(vals, y, s=46, color=col, edgecolor="white", linewidth=1.5, zorder=3, label=name)
    a.annotate(f"{vals[-1]:.4f}", (vals[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
               va="center", fontsize=9, color=INK)
a.set_yticks(range(len(rows))); a.set_yticklabels(rows, color=INK)
a.invert_yaxis(); a.axhline(4.5, color=GRID, lw=1)
a.grid(axis="x", color=GRID, lw=0.8); a.set_axisbelow(True)
a.set_xlabel("out-of-fold PQ (higher is better)", color=MUTED)
a.set_title("Each photograph scored by the models that never saw it", loc="left", color=INK,
            fontsize=11, pad=24)
a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9,
         borderaxespad=0.2, handletextpad=0.3)

# right: paired bootstrap gaps
r = k["rule"]
gaps = [("R50+R34 minus R50 alone", k["pair_minus_r50"]),
        ("keep-rule minus no rule", r["all17_minus_no_rule"]),
        ("17 features minus 10", r["all17_minus_old10"]),
        ("all data minus a quarter", r["all17_minus_quarter"])]
for i, (name, g) in enumerate(gaps):
    sig = g["lo"] > 0
    col = "#009E73" if sig else MUTED
    b.plot([g["lo"], g["hi"]], [i, i], color=col, lw=2, solid_capstyle="round")
    b.scatter([g["mean"]], [i], s=46, color=col, edgecolor="white", linewidth=1.5, zorder=3)
    b.annotate(f"{g['mean']:+.4f}  [{g['lo']:+.4f}, {g['hi']:+.4f}]", (g["hi"], i),
               xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
b.axvline(0, color=INK, lw=1)
b.set_yticks(range(len(gaps))); b.set_yticklabels([g[0] for g in gaps], color=INK)
b.invert_yaxis(); b.grid(axis="x", color=GRID, lw=0.8); b.set_axisbelow(True)
b.set_xlim(-0.006, 0.03)
b.set_xlabel("PQ gain, paired bootstrap 95% interval (grey = not distinguishable from 0)", color=MUTED)
b.set_title("What each step adds", loc="left", color=INK, fontsize=11, pad=24)

t = k["transfer"]
fig.suptitle(f"filament-kfold v3 — 707 photographs, {k['candidates']} rule candidates.  "
             f"Rule on the 5-model val ensemble: nested cut {t['on_ens0_nested_cut']['PQ']:.4f}, "
             f"best cut {t['on_ens0_best_cut']['cut']} → {t['on_ens0_best_cut']['PQ']:.4f} (optimistic)",
             x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95))
dst.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dst, dpi=130, facecolor="white")
print(dst)
