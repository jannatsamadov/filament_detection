"""figures/rf_importance.png: which columns the pixel forests use.

  A  the earlier forest (filament-pixel-rf0, fold 0, the 35-column table): permutation importance, i.e.
     how much the weighted log-loss on fold-0 pixels grows when one column is shuffled
  B  the user's 1358-column table (filament-pixel-nbc / -nb), when its output is in: the forest's own
     importance of every neighbour's contrast drawn back at its offset, and the shuffled-group importance

    python kernels/aug_unet/rf_importance_fig.py
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KER = HERE.parent
OUT = HERE.parents[1] / "figures" / "rf_importance.png"
GROUP_COLOR = {"where/when/station": "#0072B2", "darkness": "#E69F00", "shape (ridge)": "#009E73", "photograph": "#CC79A7"}
sys.path.insert(0, str(HERE))


def groups_of():
    import pixel_gbm as G
    return {c: g for g, cols in G.PIX_GROUPS.items() for c in cols}


def main():
    rf0 = json.loads((KER / "pixel_rf0/out_kaggle/pixel.json").read_text())["importance_pixA_fold0"]
    inc = {k: v for k, v in rf0["increase"].items() if not k.startswith("[")}
    grp = {k[1:-1]: v for k, v in rf0["increase"].items() if k.startswith("[")}
    g_of = groups_of()
    nb = None
    for p in (KER / "pixel_nbc/out_kaggle/pixel_nbc.json", KER / "pixel_nb/out_kaggle/pixel_nb.json"):
        if p.exists():
            nb = (p, json.loads(p.read_text()))
            break
    fig = plt.figure(figsize=(15, 9.5 if nb else 8.5))
    gs = fig.add_gridspec(1, 3 if nb else 2, width_ratios=[2.2, 1.1, 1.6] if nb else [2.2, 1.1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    names = sorted(inc, key=lambda k: inc[k])
    ax.barh(range(len(names)), [inc[n] for n in names], color=[GROUP_COLOR[g_of[n]] for n in names], height=0.7)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.set_xlabel("growth of the log-loss when the column is shuffled")
    ax.set_title(f"A. Earlier RF (35 columns, fold 0): one column at a time\nlog-loss {rf0['base_logloss']:.4f}",
                 loc="left", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COLOR.items()], loc="lower right", fontsize=8.5,
              frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[1])
    gn = sorted(grp, key=lambda k: grp[k])
    ax.barh(range(len(gn)), [grp[n] for n in gn], color=[GROUP_COLOR[n] for n in gn], height=0.6)
    ax.set_yticks(range(len(gn)), gn, fontsize=9)
    for i, n in enumerate(gn):
        ax.text(grp[n], i, f" {grp[n]:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("growth of the log-loss")
    ax.set_title("A'. The same, a whole group\nshuffled at once", loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    if nb:
        import pixel_nb as NB
        p, r = nb
        arm = "nbc" if "nbc" in r.get("mdi", {}) else "nb"
        sub = gs[2].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.45)
        ax = fig.add_subplot(sub[0])
        if arm in r.get("mdi", {}):
            pc = r["mdi"][arm]["per_column"]
            g = np.full((2 * NB.RADIUS + 1,) * 2, np.nan)
            for dy, dx in NB.OFFS:
                g[dy + NB.RADIUS, dx + NB.RADIUS] = pc[f"c_dx{dx}_dy{dy}"]
            im = ax.imshow(g, cmap="viridis", extent=(-12.5, 12.5, 12.5, -12.5))
            fig.colorbar(im, ax=ax, shrink=0.8, label="forest's own importance")
            ax.set_title(f"B. User's table ({arm}): importance of each\nneighbour's contrast, by its offset", loc="left",
                         fontsize=10.5)
            ax.set_xlabel("dx, px at 2048"); ax.set_ylabel("dy")
        ax = fig.add_subplot(sub[1])
        gi = r.get(f"importance_{arm}", {}).get("increase", {})
        k = sorted(gi, key=lambda n: gi[n])
        ax.barh(range(len(k)), [gi[n] for n in k], color="#0072B2", height=0.6)
        ax.set_yticks(range(len(k)), [n.strip("[]") for n in k], fontsize=8)
        ax.set_xlabel("growth of the squared error")
        ax.set_title("B'. Shuffled groups", loc="left", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Which columns the pixel random forests use", x=0.01, ha="left", fontsize=12)
    fig.savefig(OUT, dpi=100, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
