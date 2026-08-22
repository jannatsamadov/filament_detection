"""Is it worth predicting a filament only one annotator drew?

The scorer runs against each annotator's outlines separately and adds the totals
up. So on a photograph three people annotated, predicting something only one of
them drew scores one true positive and two false positives.

Arithmetic says that still pays. The prediction adds about 0.8 to the numerator
and 1.5 to the denominator -- one true positive, two false positives at half
weight each, less the missed detection it replaces -- for a marginal ratio near
0.53, well above the 0.29 the submissions score. On that reasoning the model
should chase every filament any annotator drew.

Threshold calibration says the opposite. It keeps choosing the strictest setting
on offer, throwing away detections to cut false positives, and the score goes up
when it does.

Both cannot be right. This settles it by measurement: score the predictions as
they are, then score them again with every prediction that matches a
singly-drawn filament removed, and see which way the number moves.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))
from spine_data import load_annotations, make_split, DISK_CX, DISK_CY, DISK_R, FULL  # noqa: E402
from spine_model import build_spine_model                        # noqa: E402
from spine_infer import assemble                                 # noqa: E402
from spine_train2 import load_stack, predict_frame               # noqa: E402
from metrics import pq_from_masks, aggregate                     # noqa: E402
from evaluate import gt_masks_full                               # noqa: E402
from gpu_guard import Governor                                   # noqa: E402
from gpu_lock import GpuLock                                     # noqa: E402


def iou(a, b):
    i = np.logical_and(a, b).sum()
    if i == 0:
        return 0.0
    return float(i) / float(np.logical_or(a, b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_v4_best.pt"))
    ap.add_argument("--spine-thresh", type=float, default=0.90)
    ap.add_argument("--min-spine-px", type=int, default=700)
    ap.add_argument("--loose-spine-thresh", type=float, default=0.85)
    ap.add_argument("--loose-min-spine-px", type=int, default=300)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--tta", action="store_true", default=True)
    args = ap.parse_args()

    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    # which of a photograph's annotators drew each of its image_ids
    photo_of = {}
    for im in d["images"]:
        pref, st = im["id"].split("-")
        photo_of[im["id"]] = st
    n_annotators = collections.Counter(photo_of.values())

    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)
    val_ids = val_ids[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = int(ck["model"]["stem.0.weight"].shape[1])
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"{Path(args.ckpt).name}, epoch {ck['epoch']}, {len(val_ids)} validation sets")

    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    gov = Governor(target_util=55, check_every=2)
    lock = GpuLock("filament_detection", "single-vote check", wait=True).acquire()

    settings = {"strict": (args.spine_thresh, args.min_spine_px),
                "loose": (args.loose_spine_thresh, args.loose_min_spine_px)}
    res = {k: {"all": [], "no_single": []} for k in settings}
    counts = collections.Counter()

    try:
        for k, iid in enumerate(val_ids, 1):
            stem = Path(fname[iid]).stem
            stack = load_stack(stem, "train", in_ch > 1)
            with gov.step():
                sem, spn, rad = predict_frame(model, stack, device, tta=args.tta)
            sem *= ondisk; spn *= ondisk

            gts = gt_masks_full(by_id[iid])
            n_ann = n_annotators[stem]
            # a GT filament is "single" when this photograph had several
            # annotators and the others drew nothing on top of it
            others = [gt_masks_full(by_id[j]) for j in by_id
                      if j != iid and photo_of.get(j) == stem]
            single = []
            for g in gts:
                support = sum(1 for om in others if any(iou(g, o) > 0.3 for o in om))
                single.append(support == 0 and n_ann >= 2)

            for name, (st, mp) in settings.items():
                preds = assemble(spn, rad, sem, spine_thresh=st, sem_thresh=0.5,
                                 min_spine_px=mp, min_area=60)
                keep = []
                for p in preds:
                    best, bi = 0.0, -1
                    for j, g in enumerate(gts):
                        v = iou(p, g)
                        if v > best:
                            best, bi = v, j
                    matches_single = best > 0.5 and bi >= 0 and single[bi]
                    if matches_single:
                        counts[name] += 1
                    else:
                        keep.append(p)
                res[name]["all"].append(pq_from_masks(gts, preds))
                res[name]["no_single"].append(pq_from_masks(gts, keep))
            if k % 10 == 0:
                lock.progress(f"{k}/{len(val_ids)}")
                print(f"  {k}/{len(val_ids)}", flush=True)
    finally:
        lock.release()

    print(f"\nvalidation sets: {len(val_ids)}   "
          f"(a photograph counts once per annotator)")
    for name, (st, mp) in settings.items():
        a = aggregate(res[name]["all"])
        b = aggregate(res[name]["no_single"])
        print(f"\n{name}  (spine {st}, min_spine_px {mp})")
        print(f"  as predicted             PQ {a['PQ']:.4f}  "
              f"TP {a['TP']:4d} FP {a['FP']:4d} FN {a['FN']:4d}")
        print(f"  minus singly-drawn hits  PQ {b['PQ']:.4f}  "
              f"TP {b['TP']:4d} FP {b['FP']:4d} FN {b['FN']:4d}"
              f"   ({counts[name]} predictions removed)")
        d_ = b["PQ"] - a["PQ"]
        verdict = ("removing them HELPS -- chasing singly-drawn filaments costs score"
                   if d_ > 0 else
                   "removing them HURTS -- singly-drawn filaments are worth predicting")
        print(f"  difference {d_:+.4f}   -> {verdict}")

    json.dump({name: {"all": aggregate(res[name]["all"]),
                      "no_single": aggregate(res[name]["no_single"]),
                      "removed": counts[name]} for name in settings},
              open(ROOT / "single_vote_value.json", "w"), indent=1)
    print(f"\ngovernor: {gov.summary()}")
    print("wrote single_vote_value.json")


if __name__ == "__main__":
    main()
