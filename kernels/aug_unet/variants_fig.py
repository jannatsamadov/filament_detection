"""One bar per post-processed variant: which model / TTA / ensemble scores best.

Reads post_local.json written by the post kernel and draws validation PQ (our
scorer, 141 held-out photographs, every annotation set) for every variant at its
own best grid point, sorted. TP / FP / FN sit beside each bar.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INK, INK2, GRID, BAR, KEY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
NAMES = {"flip": "R34, flips only", "full": "R34, heavy aug", "union": "R34, union target",
         "r50": "R50, heavy aug"}


def label(v):
    model, view = v.split("/")
    tta = "D4 TTA" if view == "d4" else "no TTA"
    if model == "ens":
        return f"ensemble of all, {tta}"
    if model.startswith("ens:"):
        return f"ensemble {' + '.join(model[4:].split('+'))}, {tta}"
    return f"{NAMES.get(model, model)}, {tta}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE.parent / "aug_post" / "out_kaggle_r2" / "post_local.json"))
    ap.add_argument("--out", default=str(ROOT / "figures" / "aug_unet_variants.png"))
    args = ap.parse_args()
    r = json.load(open(args.json))
    rows = sorted(r["results"], key=lambda b: b["PQ"])
    if r.get("used_keep_rule"):
        k, b = r["keep_rule"], r["best"]
        rows.append(dict(variant=b["variant"] + "+rule", PQ=k["PQ"], TP=k["TP"], FP=k["FP"], FN=k["FN"]))

    fig, ax = plt.subplots(figsize=(13, 0.48 * len(rows) + 1.6), facecolor="#fcfcfb")
    ys = range(len(rows))
    lo = min(b["PQ"] for b in rows) - 0.01
    for y, b in zip(ys, rows):
        rule = b["variant"].endswith("+rule")
        ax.barh(y, b["PQ"] - lo, left=lo, height=0.62, color=KEY if rule else BAR)
        ax.text(b["PQ"] + 0.0006, y, f"{b['PQ']:.4f}   TP {int(b['TP'])}  FP {int(b['FP'])}  FN {int(b['FN'])}",
                va="center", fontsize=10, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([label(b["variant"][:-5]) + " + learned keep-rule" if b["variant"].endswith("+rule")
                        else label(b["variant"]) for b in rows], fontsize=10, color=INK)
    ax.set_xlim(lo, max(b["PQ"] for b in rows) + 0.012)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(INK2); ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="x", colors=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("validation PQ (axis starts at %.2f)" % lo, color=INK2)
    ax.set_title("Every variant at its own best threshold / merge / min area -- our scorer, "
                 "141 held-out photographs, 236 annotation sets", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=90, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
