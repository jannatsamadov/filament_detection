"""The standard report figure. Regenerate it for every model so results compare
at a glance: predictions against truth, where the score is lost, and how the
detection threshold trades false positives against missed filaments.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from data import FilamentDataset, load_annotations, make_split   # noqa: E402
from model import build_model                                    # noqa: E402
from evaluate import predict_masks, gt_masks_full, evaluate      # noqa: E402
from metrics import pq_from_masks, aggregate                     # noqa: E402

FULL = 2048
HUMAN_PQ, HUMAN_SQ, HUMAN_RQ = 0.348, 0.595, 0.542


def outline(ax, mask, color, lw=1.1):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="maskrcnn_r50_1024")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)
    ds = FilamentDataset(val_ids, by_id, fname, size=args.size, augment=False)

    model = build_model(size=args.size).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']})")

    # ---- threshold sweep
    sweep = []
    for t in args.thresholds:
        _, dsm, imm = evaluate(model, ds, by_id, device, score_thresh=t, limit=args.limit)
        sweep.append((t, dsm))
        print(f"  thr {t:.2f}: PQ={dsm['PQ']:.4f} SQ={dsm['SQ']:.4f} RQ={dsm['RQ']:.4f} "
              f"TP={dsm['TP']} FP={dsm['FP']} FN={dsm['FN']}", flush=True)
    best_t, best = max(sweep, key=lambda kv: kv[1]["PQ"])
    print(f"\nbest threshold {best_t:.2f}: PQ={best['PQ']:.4f}")

    # ---- per-image results at the best threshold, for the distributions
    results, dsm, imm = evaluate(model, ds, by_id, device,
                                 score_thresh=best_t, limit=args.limit)
    matched_iou = [v for r in results for v in r.ious]

    # sizes of matched vs missed ground-truth filaments
    tp_area, fn_area = [], []
    for i, r in enumerate(results):
        anns = by_id[ds.ids[i]]
        matched_gt = {m[0] for m in r.matches}
        for j, a in enumerate(anns, start=1):
            (tp_area if j in matched_gt else fn_area).append(a["area"])

    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.35, 1, 1])

    # ---- row 1: two full-disk comparisons
    show = sorted(range(len(results)), key=lambda i: -results[i].pq)[:1] + \
           sorted(range(len(results)), key=lambda i: results[i].pq)[:1]
    for col, idx in enumerate(show):
        image, _ = ds[idx]
        preds, _ = predict_masks(model, image, device, best_t)
        gts = gt_masks_full(by_id[ds.ids[idx]])
        arr = np.asarray(Image.open(ds.dir / (Path(ds.fname[ds.ids[idx]]).stem + ".png")))
        ax = fig.add_subplot(gs[0, col * 2:col * 2 + 2])
        ax.imshow(arr, cmap="gray", vmin=60, vmax=190)
        for m in gts:
            outline(ax, m, "#e8442f")
        for m in preds:
            outline(ax, m, "#2fd4e8")
        r = results[idx]
        ax.set_title(f"{ds.fname[ds.ids[idx]]}   PQ={r.pq:.3f}  "
                     f"TP={r.tp} FP={r.fp} FN={r.fn}\n"
                     f"red = ground truth, cyan = prediction", fontsize=11)
        ax.axis("off")

    # ---- row 2
    ax = fig.add_subplot(gs[1, 0])
    ts = [s[0] for s in sweep]
    ax.plot(ts, [s[1]["PQ"] for s in sweep], "o-", label="PQ", lw=2)
    ax.plot(ts, [s[1]["SQ"] for s in sweep], "s--", label="SQ", alpha=.7)
    ax.plot(ts, [s[1]["RQ"] for s in sweep], "^--", label="RQ", alpha=.7)
    ax.axhline(HUMAN_PQ, color="k", ls=":", lw=1.5)
    ax.text(ts[0], HUMAN_PQ + .01, "human-vs-human PQ", fontsize=8)
    ax.axvline(best_t, color="grey", lw=1, alpha=.6)
    ax.set_xlabel("detection score threshold"); ax.set_ylabel("score")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("calibration: the main knob", fontsize=11)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts, [s[1]["FP"] for s in sweep], "o-", label="false positives")
    ax.plot(ts, [s[1]["FN"] for s in sweep], "s-", label="missed")
    ax.plot(ts, [s[1]["TP"] for s in sweep], "^-", label="matched")
    ax.axvline(best_t, color="grey", lw=1, alpha=.6)
    ax.set_xlabel("detection score threshold"); ax.set_ylabel("count")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("what the threshold trades", fontsize=11)

    ax = fig.add_subplot(gs[1, 2])
    if matched_iou:
        ax.hist(matched_iou, bins=24, range=(0.5, 1.0), color="#2fd4e8", edgecolor="k", lw=.4)
        ax.axvline(np.mean(matched_iou), color="r", lw=1.5,
                   label=f"mean {np.mean(matched_iou):.3f}")
        ax.axvline(HUMAN_SQ, color="k", ls=":", lw=1.5, label=f"human {HUMAN_SQ}")
        ax.legend(fontsize=8)
    ax.set_xlabel("IoU of matched pairs"); ax.set_title("boundary quality (SQ)", fontsize=11)

    ax = fig.add_subplot(gs[1, 3])
    bins = np.logspace(np.log10(50), np.log10(20000), 26)
    if tp_area:
        ax.hist(tp_area, bins=bins, alpha=.65, label=f"found ({len(tp_area)})")
    if fn_area:
        ax.hist(fn_area, bins=bins, alpha=.65, label=f"missed ({len(fn_area)})")
    ax.set_xscale("log"); ax.set_xlabel("ground-truth filament area, px")
    ax.legend(fontsize=8); ax.set_title("which filaments are missed", fontsize=11)

    # ---- row 3: training curves and the headline table
    ax = fig.add_subplot(gs[2, 0:2])
    log = ROOT / f"trainlog_{args.tag}.csv"
    if log.exists():
        import csv as _csv
        rows = list(_csv.DictReader(open(log)))
        if rows:
            x = np.arange(len(rows))
            ax.plot(x, [float(r["loss"]) for r in rows], lw=1.4, label="total loss")
            for key, lab in (("loss_mask", "mask"), ("loss_classifier", "cls"),
                             ("loss_objectness", "obj")):
                ax.plot(x, [float(r[key]) for r in rows], lw=.9, alpha=.65, label=lab)
            ax.set_xlabel("logged step (every 50)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("training loss", fontsize=11)

    ax = fig.add_subplot(gs[2, 2])
    vlog = ROOT / f"valog_{args.tag}.json"
    if vlog.exists():
        recs = [json.loads(l) for l in open(vlog) if l.strip()]
        if recs:
            ep = [r["epoch"] for r in recs]
            ax.plot(ep, [r["dataset"]["PQ"] for r in recs], "o-", label="PQ")
            ax.plot(ep, [r["dataset"]["SQ"] for r in recs], "s--", label="SQ", alpha=.7)
            ax.plot(ep, [r["dataset"]["RQ"] for r in recs], "^--", label="RQ", alpha=.7)
            ax.axhline(HUMAN_PQ, color="k", ls=":", lw=1.5)
            ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_xlabel("epoch")
    ax.set_title("validation by epoch (thr 0.50)", fontsize=11)

    ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
    txt = (f"MODEL   Mask R-CNN R50-FPN v2, {args.size}px\n"
           f"        flattened input, small anchors\n"
           f"CKPT    epoch {ck['epoch']}\n"
           f"VAL     {args.limit} annotation sets, grouped split\n\n"
           f"best threshold        {best_t:.2f}\n"
           f"  PQ                  {best['PQ']:.4f}\n"
           f"  SQ                  {best['SQ']:.4f}\n"
           f"  RQ                  {best['RQ']:.4f}\n"
           f"  TP / FP / FN        {best['TP']} / {best['FP']} / {best['FN']}\n"
           f"  per-image PQ        {imm['PQ']:.4f}\n\n"
           f"human vs human        {HUMAN_PQ:.3f} (SQ {HUMAN_SQ}, RQ {HUMAN_RQ})\n"
           f"  our scorer, not the organisers'\n")
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=10.5)

    fig.suptitle(f"Filament segmentation — {args.tag} @ epoch {ck['epoch']}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGS / f"report_{args.tag}_e{ck['epoch']}.png"
    plt.savefig(out, dpi=78); plt.close()
    print(f"wrote {out}")

    json.dump({"tag": args.tag, "epoch": ck["epoch"], "best_threshold": best_t,
               "best": best, "sweep": [(t, s) for t, s in sweep]},
              open(ROOT / f"result_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
