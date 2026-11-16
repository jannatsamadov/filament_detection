"""figures/ideal_model.png: PQ on fold 0 from the submitted mix up to perfect pixels and a perfect filter.

    python kernels/aug_unet/ideal_model_fig.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures" / "ideal_model.png"
OURS, FILTER, PIXELS, HUMAN = "#0072B2", "#56B4E9", "#E69F00", "#555555"


def main():
    r = json.loads((HERE / "out" / "ideal_model.json").read_text())
    mix = json.loads((HERE.parent / "pixnet_mix" / "out_kaggle" / "pixnet_mix.json").read_text())
    rule = mix["transfer"]["ens0q"]["on_ens0_nested_cut"]
    R, F = r["results"], r["perfect_filter"]
    rows = [("human: one expert scored against the other experts", R["human_one_vs_others"], HUMAN),
            ("submitted mix: threshold, grouping, minimum area", R["mix"], OURS),
            ("  + our 25-feature keep-rule (nested)", rule, OURS),
            ("  + a perfect filter on the same candidates", F["mix"], FILTER),
            ("perfect pixels = the experts' union, same grouping", R["union_k3"], PIXELS),
            ("perfect pixels = the experts' majority", R["major_k3"], PIXELS),
            ("  majority + a perfect filter", F["major_k3"], PIXELS)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (name, v, c) in enumerate(rows):
        ax.barh(i, v["PQ"], color=c, height=0.62)
        ax.text(v["PQ"] + 0.008, i, f"{v['PQ']:.3f}   (TP {int(v['TP'])}, FP {int(v['FP'])}, FN {int(v['FN'])})", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [n for n, _, _ in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("PQ, fold-0 validation (141 photographs, every expert's set scored)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Where PQ is lost: the filter can add up to +0.04, the pixels up to +0.33\n"
                 "(perfect pixels are built from the same experts who score them, so they are a ceiling, not a target)",
                 fontsize=10.5, loc="left")
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
