"""Sweep the assembly thresholds on cached predictions.

Validation plateaued at PQ 0.197 while the training loss kept falling, and the
reason was visible in the counts: true positives tripled and misses halved
between epoch 1 and 5, but false positives barely moved (367 -> 331 against 196
real filaments). More training does not fix that. Deciding *which* spine
component counts as a filament does, and that is pure post-processing.

Two knobs matter:
  spine_thresh   how confident the spine head must be
  min_spine_px   how long a component must be to be a filament at all
                 (real spines are a median 125 px long; the default 12 px let
                 specks through)

The network runs once per frame and every threshold combination is scored on the
cached maps, so the whole sweep costs one pass.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
from spine_data import load_annotations, make_split          # noqa: E402
from spine_model import build_spine_model                    # noqa: E402
from spine_infer import assemble                             # noqa: E402
from spine_train2 import load_stack, predict_frame as predict_frame_v2  # noqa: E402
from metrics import pq_from_masks, aggregate                 # noqa: E402
from evaluate import gt_masks_full                           # noqa: E402

HUMAN_PQ, CEILING_PQ = 0.348, 0.734


def sweep(maps, by_id, spine_threshs, min_spine_pxs, sem_thresh=0.5, min_area=60):
    gts = {iid: gt_masks_full(by_id[iid]) for iid in maps}
    rows = []
    for st in spine_threshs:
        for mp in min_spine_pxs:
            res = []
            for iid, (sem, spn, rad) in maps.items():
                preds = assemble(spn, rad, sem, spine_thresh=st, sem_thresh=sem_thresh,
                                 min_spine_px=mp, min_area=min_area)
                res.append(pq_from_masks(gts[iid], preds))
            a = aggregate(res)
            rows.append((st, mp, a))
            print(f"  spine_thresh {st:.2f}  min_spine_px {mp:3d} -> "
                  f"PQ {a['PQ']:.4f}  SQ {a['SQ']:.4f}  RQ {a['RQ']:.4f}  "
                  f"TP {a['TP']:4d} FP {a['FP']:4d} FN {a['FN']:4d}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_unet_r34_best.pt"))
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--tag", default="spine_unet_r34")
    ap.add_argument("--spine-threshs", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--min-spine-px", type=int, nargs="+", default=[12, 30, 60, 100])
    ap.add_argument("--sem-thresh", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = int(ck["model"]["stem.0.weight"].shape[1])
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"{args.ckpt} (epoch {ck['epoch']}, val PQ {ck.get('PQ', float('nan')):.4f})")

    print(f"predicting {args.limit} frames once...", flush=True)
    from spine_data import DISK_CX, DISK_CY, DISK_R, FULL
    from pathlib import Path as _P
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    maps = {}
    for iid in val_ids[:args.limit]:
        stem = _P(fname[iid]).stem
        stack = load_stack(stem, "train", in_ch > 1)
        sem, spn, rad = predict_frame_v2(model, stack, device, tta=args.tta)
        maps[iid] = (sem * ondisk, spn * ondisk, rad)

    rows = sweep(maps, by_id, args.spine_threshs, args.min_spine_px,
                 sem_thresh=args.sem_thresh)
    st, mp, best = max(rows, key=lambda r: r[2]["PQ"])
    print(f"\nbest: spine_thresh {st:.2f}, min_spine_px {mp} -> PQ {best['PQ']:.4f} "
          f"(human {HUMAN_PQ}, ceiling {CEILING_PQ})")

    json.dump({"tag": args.tag, "epoch": ck["epoch"],
               "best_spine_thresh": st, "best_min_spine_px": mp,
               "best": best, "human": HUMAN_PQ, "ceiling": CEILING_PQ,
               "sweep": [{"spine_thresh": a, "min_spine_px": b, **c} for a, b, c in rows]},
              open(ROOT / f"calibration_{args.tag}.json", "w"), indent=1)
    print(f"wrote calibration_{args.tag}.json")


if __name__ == "__main__":
    main()
