"""Standard report figure for the spine model.

Regenerated for every checkpoint so results compare at a glance: what the three
heads predict, how the assembled instances line up with truth, where the score is
lost, and how the two thresholds trade false positives against missed filaments.

Two reference lines appear on every panel that carries a score:
  0.348  two human annotators against each other
  0.734  this pipeline fed its own ground-truth maps (its ceiling)
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from spine_data import load_annotations, make_split, DISK_CX, DISK_CY, DISK_R, FULL  # noqa
from spine_model import build_spine_model                    # noqa: E402
from spine_infer import assemble                             # noqa: E402
from spine_train import predict_frame, load_frame, validate  # noqa: E402
from metrics import pq_from_masks, aggregate                 # noqa: E402
from evaluate import gt_masks_full                           # noqa: E402

HUMAN_PQ, CEILING_PQ = 0.348, 0.734


def outline(ax, mask, color, lw=1.0):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_unet_r34_best.pt"))
    ap.add_argument("--tag", default="spine_unet_r34")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--spine-thresholds", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_spine_model(ck["args"].get("encoder", "resnet34"), pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']})")

    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    # ---- threshold sweep on the spine map
    sweep = []
    for t in args.spine_thresholds:
        dsm, _ = validate(model, val_ids, by_id, fname, device,
                          limit=args.limit, spine_thresh=t)
        sweep.append((t, dsm))
        print(f"  spine_thresh {t:.2f}: PQ={dsm['PQ']:.4f} SQ={dsm['SQ']:.4f} "
              f"RQ={dsm['RQ']:.4f} TP={dsm['TP']} FP={dsm['FP']} FN={dsm['FN']}", flush=True)
    best_t, best = max(sweep, key=lambda kv: kv[1]["PQ"])
    print(f"best spine threshold {best_t:.2f}: PQ={best['PQ']:.4f}")

    # ---- per-image results at the best threshold
    results, preds_cache = [], {}
    for iid in val_ids[:args.limit]:
        img = load_frame(fname[iid])
        sem, spn, rad = predict_frame(model, img, device)
        sem *= ondisk; spn *= ondisk
        preds = assemble(spn, rad, sem, spine_thresh=best_t)
        gts = gt_masks_full(by_id[iid])
        r = pq_from_masks(gts, preds)
        results.append(r)
        preds_cache[iid] = (img, sem, spn, rad, preds, gts, r)
    matched_iou = [v for r in results for v in r.ious]

    tp_area, fn_area = [], []
    for r, iid in zip(results, val_ids[:args.limit]):
        matched_gt = {m[0] for m in r.matches}
        for j, a in enumerate(by_id[iid], start=1):
            (tp_area if j in matched_gt else fn_area).append(a["area"])

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(21, 15))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 1])

    ranked = sorted(preds_cache, key=lambda k: -preds_cache[k][6].pq)
    show = [ranked[0], ranked[-1]]

    for col, iid in enumerate(show):
        img, sem, spn, rad, preds, gts, r = preds_cache[iid]
        ax = fig.add_subplot(gs[0, col * 2])
        ax.imshow(img, cmap="gray", vmin=0.75, vmax=1.15)
        for m in gts:
            outline(ax, m, "#e8442f", 1.2)
        for m in preds:
            outline(ax, m, "#2fd4e8", 0.9)
        ax.set_title(f"{fname[iid]}\nPQ={r.pq:.3f}  TP={r.tp} FP={r.fp} FN={r.fn}\n"
                     f"red = truth, cyan = predicted", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(gs[0, col * 2 + 1])
        rgb = np.zeros((*spn.shape, 3), np.float32)
        rgb[..., 0] = np.clip(spn, 0, 1)
        rgb[..., 1] = np.clip(sem, 0, 1) * 0.7
        ax.imshow(rgb)
        ax.set_title("heads: red = spine, green = semantic", fontsize=10)
        ax.axis("off")

    # zoom on the best image
    iid = show[0]
    img, sem, spn, rad, preds, gts, r = preds_cache[iid]
    if gts:
        big = max(gts, key=lambda m: m.sum())
        ys, xs = np.nonzero(big)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = 220
        sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
        for j, (arr, name, kw) in enumerate((
                (img[sl], "image", dict(cmap="gray", vmin=0.75, vmax=1.15)),
                (spn[sl], "predicted spine", dict(cmap="magma", vmin=0, vmax=1)),
                (rad[sl], "predicted radius", dict(cmap="viridis")))):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(arr, **kw)
            ax.set_title(name, fontsize=10); ax.axis("off")
        ax = fig.add_subplot(gs[1, 3])
        ax.imshow(img[sl], cmap="gray", vmin=0.75, vmax=1.15)
        for m in gts:
            outline(ax, m[sl], "#e8442f", 1.4)
        for m in preds:
            outline(ax, m[sl], "#2fd4e8", 1.1)
        ax.set_title("close-up: truth vs assembled", fontsize=10); ax.axis("off")

    ts = [s[0] for s in sweep]
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(ts, [s[1]["PQ"] for s in sweep], "o-", lw=2, label="PQ")
    ax.plot(ts, [s[1]["SQ"] for s in sweep], "s--", alpha=.7, label="SQ")
    ax.plot(ts, [s[1]["RQ"] for s in sweep], "^--", alpha=.7, label="RQ")
    ax.axhline(HUMAN_PQ, color="k", ls=":", lw=1.4)
    ax.axhline(CEILING_PQ, color="g", ls=":", lw=1.4)
    ax.text(ts[0], HUMAN_PQ + .012, "human 0.348", fontsize=7)
    ax.text(ts[0], CEILING_PQ + .012, "ceiling 0.734", fontsize=7, color="g")
    ax.axvline(best_t, color="grey", lw=1, alpha=.6)
    ax.set_xlabel("spine threshold"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("calibration", fontsize=10)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(ts, [s[1]["FP"] for s in sweep], "o-", label="false positives")
    ax.plot(ts, [s[1]["FN"] for s in sweep], "s-", label="missed")
    ax.plot(ts, [s[1]["TP"] for s in sweep], "^-", label="matched")
    ax.axvline(best_t, color="grey", lw=1, alpha=.6)
    ax.set_xlabel("spine threshold"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax.set_title("what the threshold trades", fontsize=10)

    ax = fig.add_subplot(gs[2, 2])
    if matched_iou:
        ax.hist(matched_iou, bins=22, range=(0.5, 1.0), color="#2fd4e8",
                edgecolor="k", lw=.4)
        ax.axvline(np.mean(matched_iou), color="r", lw=1.5,
                   label=f"mean {np.mean(matched_iou):.3f}")
        ax.axvline(0.595, color="k", ls=":", lw=1.4, label="human 0.595")
        ax.legend(fontsize=8)
    ax.set_xlabel("IoU of matched pairs"); ax.set_title("boundary quality", fontsize=10)

    ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
    txt = (f"spine U-Net ({ck['args'].get('encoder','resnet34')}), 2048 px native\n"
           f"epoch {ck['epoch']}   val {args.limit} frames\n\n"
           f"best spine threshold  {best_t:.2f}\n"
           f"  PQ                  {best['PQ']:.4f}\n"
           f"  SQ                  {best['SQ']:.4f}\n"
           f"  RQ                  {best['RQ']:.4f}\n"
           f"  TP / FP / FN        {best['TP']} / {best['FP']} / {best['FN']}\n\n"
           f"pipeline ceiling      {CEILING_PQ:.3f}\n"
           f"human vs human        {HUMAN_PQ:.3f}\n\n"
           f"our scorer, not the organisers'\n")
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=10.5)

    fig.suptitle(f"Filament segmentation - {args.tag} @ epoch {ck['epoch']}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGS / f"report_{args.tag}_e{ck['epoch']}.png"
    plt.savefig(out, dpi=74); plt.close()
    print(f"wrote {out}")

    json.dump({"tag": args.tag, "epoch": ck["epoch"], "best_spine_thresh": best_t,
               "best": best, "sweep": [(t, s) for t, s in sweep],
               "ceiling": CEILING_PQ, "human": HUMAN_PQ},
              open(ROOT / f"result_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
