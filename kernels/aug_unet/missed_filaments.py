"""Which expert filaments the submitted mix misses: by expert agreement, by darkness and by size.

Fold 0, photographs with two or more experts. For every filament one expert drew:
  coverage   share of its pixels the mix calls filament (probability >= 0.5)
  agreement  share of the other experts on that photograph whose own filaments cover half of it
  contrast   1 - (its mean brightness / the median brightness of a 15 px ring around it)
  area       pixels at 2048

    python kernels/aug_unet/missed_filaments.py  -> out/missed_filaments.json, figures/missed_filaments.png
"""
import json, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel_cls as PC  # noqa: E402
import run  # noqa: E402

MIX = "5 U-Nets + Qwen (submitted mix)"
OUT = HERE / "out" / "missed_filaments.json"
FIG = HERE.parents[1] / "figures" / "missed_filaments.png"
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def main():
    D = PC.Dumps()
    stems = D.stems()
    by = PC.experts(set(stems))
    stems = [s for s in stems if s in by and len(by[s]) >= 2]
    _, tr_dir, _ = run.find_data()
    rows = []
    for i, s in enumerate(stems):
        pred = D.model(MIX, s, {}) >= 128
        img = cv2.imread(str(tr_dir / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        sets = [[run.ann_mask(a) for a in anns] for anns in by[s].values()]
        unions = [np.logical_or.reduce(ms) for ms in sets]
        for e, ms in enumerate(sets):
            others = [u for k, u in enumerate(unions) if k != e]
            for m in ms:
                n = int(m.sum())
                if n == 0:
                    continue
                ys, xs = np.nonzero(m)
                y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 21, run.FULL)
                x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 21, run.FULL)
                mc = m[y0:y1, x0:x1]
                ring = cv2.dilate(mc.astype(np.uint8), RING).astype(bool) & ~mc
                ic = img[y0:y1, x0:x1]
                bg = float(np.median(ic[ring])) if ring.any() else float(np.median(ic))
                rows.append(dict(stem=s, area=n, coverage=float(pred[y0:y1, x0:x1][mc].mean()),
                                 agreement=float(np.mean([o[y0:y1, x0:x1][mc].mean() >= 0.5 for o in others])),
                                 contrast=float(1 - ic[mc].mean() / max(bg, 1.0)), n_experts=len(sets)))
        if i % 20 == 0:
            print(f"  {i + 1}/{len(stems)} {s}", flush=True)
    found = np.array([r["coverage"] >= 0.5 for r in rows])
    agr = np.array([r["agreement"] for r in rows])
    con = np.array([r["contrast"] for r in rows])
    area = np.array([r["area"] for r in rows])

    def table(x, edges, labels):
        out = []
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            sel = (x >= lo) & (x < hi) if hi != edges[-1] else (x >= lo) & (x <= hi)
            out.append(dict(bin=lab, n=int(sel.sum()), found=round(float(found[sel].mean()), 3) if sel.any() else None))
        return out

    qc = np.quantile(con, [0, 0.2, 0.4, 0.6, 0.8, 1])
    qa = np.quantile(area, [0, 0.2, 0.4, 0.6, 0.8, 1])
    res = dict(photographs=len(stems), filaments=len(rows), found_overall=round(float(found.mean()), 3),
               by_agreement=table(agr, [0, 1e-9, 0.5, 0.999, 1.0], ["no other expert", "under half", "half or more",
                                                                    "every other expert"]),
               by_contrast=table(con, qc, [f"{a:.3f}-{b:.3f}" for a, b in zip(qc[:-1], qc[1:])]),
               by_area=table(area, qa, [f"{int(a)}-{int(b)}" for a, b in zip(qa[:-1], qa[1:])]))
    OUT.write_text(json.dumps(dict(res, rows=rows), indent=1))
    print(json.dumps(res, indent=1))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, key, title, xl in ((axs[0], "by_agreement", "A. by how many other experts drew it", "other experts who drew it"),
                               (axs[1], "by_contrast", "B. by darkness against its surroundings", "contrast (fifths)"),
                               (axs[2], "by_area", "C. by size", "area, px at 2048 (fifths)")):
        b = [x for x in res[key] if x["n"]]          # with two or three experts some agreement bins are empty
        ax.bar(range(len(b)), [x["found"] for x in b], color="#0072B2", width=0.65)
        for j, x in enumerate(b):
            ax.text(j, x["found"] + 0.02, f"{x['found']:.2f}\n(n {x['n']})", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(b)), [x["bin"] for x in b], fontsize=8.5, rotation=15)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 1.1)
    axs[0].set_ylabel("share of expert filaments the mix covers\n(half of its pixels or more)")
    fig.suptitle(f"Which expert filaments the submitted mix misses (fold 0, {len(stems)} photographs with 2+ experts, "
                 f"{len(rows)} filaments; found overall {res['found_overall']:.2f})", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=105, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
