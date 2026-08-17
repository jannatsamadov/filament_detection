"""Are our false positives really false, or filaments a different annotator marked?

Every annotator's version of a photograph is a separate training sample, so a
filament that only one of three experts marked is shown to the model once as
foreground and twice as background. Worse, at validation the model sees the
photograph once and produces one prediction, which is then scored separately
against each annotator's own labels -- so anything the model detects that this
particular annotator skipped counts as a false positive, even when a colleague
marked it.

This measures how much of the false-positive count that explains, by scoring the
same prediction three ways:

  own      against the annotator whose set is being scored (what we report now)
  union    against every filament any annotator of that photograph marked
  consensus against filaments at least two annotators marked

If FP collapses under `union`, the model is finding real filaments and the metric
is punishing annotator disagreement rather than model error.
"""
from __future__ import annotations

import collections, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
from spine_data import load_annotations, make_split, DISK_CX, DISK_CY, DISK_R, FULL  # noqa
from spine_model import build_spine_model            # noqa: E402
from spine_infer import assemble                     # noqa: E402
from spine_train import predict_frame, load_frame    # noqa: E402
from metrics import pq_from_masks, aggregate         # noqa: E402
from evaluate import gt_masks_full                   # noqa: E402


def merge_sets(mask_lists, min_votes=1, iou_link=0.3):
    """Merge several annotators' masks into one list.

    Masks from different annotators are linked when they overlap at IoU above
    iou_link; a group kept if it has at least min_votes distinct annotators, and
    represented by the mask from the annotator who drew it largest.
    """
    flat = [(ai, m) for ai, ms in enumerate(mask_lists) for m in ms]
    n = len(flat)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        ai, mi = flat[i]
        idx = np.nonzero(mi.ravel())[0]
        for j in range(i + 1, n):
            aj, mj = flat[j]
            if ai == aj:
                continue
            inter = int(mj.ravel()[idx].sum())
            if inter == 0:
                continue
            u = int(mi.sum()) + int(mj.sum()) - inter
            if u and inter / u >= iou_link:
                union(i, j)

    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out = []
    for members in groups.values():
        voters = {flat[i][0] for i in members}
        if len(voters) >= min_votes:
            out.append(max((flat[i][1] for i in members), key=lambda m: m.sum()))
    return out


def main(limit=18, spine_thresh=0.8, min_spine_px=100, sem_thresh=0.5):
    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)

    by_file = collections.defaultdict(list)
    for iid in val_ids:
        by_file[fname[iid]].append(iid)
    multi = {f: ids for f, ids in by_file.items() if len(ids) >= 2}
    files = sorted(multi)[:limit]
    print(f"{len(files)} validation photographs with 2+ annotators "
          f"(of {len(by_file)} total)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "checkpoints" / "spine_unet_r34_best.pt",
                    map_location=device, weights_only=False)
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    res_own, res_union, res_cons = [], [], []
    n_ann = []
    for f in files:
        ids = multi[f]
        sem, spn, rad = predict_frame(model, load_frame(f), device)
        sem *= ondisk; spn *= ondisk
        preds = assemble(spn, rad, sem, spine_thresh=spine_thresh,
                         sem_thresh=sem_thresh, min_spine_px=min_spine_px)

        per_annotator = [gt_masks_full(by_id[i]) for i in ids]
        n_ann.append(len(ids))
        for gts in per_annotator:
            res_own.append(pq_from_masks(gts, preds))
        res_union.append(pq_from_masks(merge_sets(per_annotator, 1), preds))
        res_cons.append(pq_from_masks(merge_sets(per_annotator, 2), preds))

    print(f"{'scored against':<28} {'PQ':>7} {'SQ':>7} {'RQ':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    for name, res in (("own annotator (reported)", res_own),
                      ("union of all annotators", res_union),
                      ("consensus (2+ annotators)", res_cons)):
        a = aggregate(res)
        print(f"{name:<28} {a['PQ']:7.4f} {a['SQ']:7.4f} {a['RQ']:7.4f} "
              f"{a['TP']:5d} {a['FP']:5d} {a['FN']:5d}")

    ao, au = aggregate(res_own), aggregate(res_union)
    # own is scored once per annotator, union once per photograph: normalise
    per_photo_own_fp = ao["FP"] / sum(n_ann) * len(files)
    print(f"\nfalse positives per photograph: {per_photo_own_fp / len(files):.1f} "
          f"against the annotator's own labels, "
          f"{au['FP'] / len(files):.1f} against the union")
    drop = 1 - (au["FP"] / len(files)) / max(per_photo_own_fp / len(files), 1e-9)
    print(f"{100*drop:.0f}% of the false positives are filaments that some other "
          f"annotator of the same photograph did mark")


if __name__ == "__main__":
    main()
