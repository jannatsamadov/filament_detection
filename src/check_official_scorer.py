"""Run the organisers' own PQ code on our validation data and compare.

Every score reported so far came from src/metrics.py, written from the formula on
the competition page. The organisers' Self Evaluation notebook holds what the
host called "the exact implementation of PQ", so the two can now be compared
directly rather than assumed equal.

Reading the notebook first shows the scoring is per *annotator-image*, not per
image: ground truth ids are "<annotator>-<image>", predictions are matched on the
image part alone, and TP/FP/FN are pooled over every annotator-image. One
prediction for a three-annotator photograph is therefore scored three times, so a
filament only one expert marked scores one TP and two FP. That is what our
validation already did, but it is worth having confirmed.

The two implementations should also agree on matching. The official code takes
every pair above IoU 0.5 without resolving a matching; ours matches greedily.
At that threshold the two coincide -- a prediction cannot exceed 0.5 IoU with two
disjoint ground-truth masks -- but this checks it on real data instead.
"""
from __future__ import annotations

import json, sys, types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = ROOT / "reference" / "self-evaluation-notebook.ipynb"
sys.path.insert(0, str(HERE))
from spine_data import load_annotations, make_split                # noqa: E402
from evaluate import gt_masks_full                                 # noqa: E402
from metrics import pq_from_masks, aggregate                       # noqa: E402
from pycocotools import mask as maskutil                           # noqa: E402


def load_official():
    """Execute the notebook's helper and scoring cells in a fresh module."""
    nb = json.load(open(NB, encoding="utf-8"))
    mod = types.ModuleType("official")
    mod.__dict__.update({"np": np, "pd": pd, "torch": torch, "maskutil": maskutil})
    src = ["import numpy as np", "import pandas as pd", "import torch",
           "from pycocotools import mask as mask_util"]
    for i in (1, 3, 4):
        src.append("".join(nb["cells"][i]["source"]))
    exec(compile("\n".join(src), "<official>", "exec"), mod.__dict__)
    return mod


def rle_of(mask):
    r = maskutil.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("ascii")


def main(limit=25):
    official = load_official()
    print("loaded the organisers' functions:",
          ", ".join(sorted(k for k in ("get_pq_score", "get_overlap_df",
                                       "get_overlap_matrices", "fp_count_hit",
                                       "fn_count_hit", "rles_to_layers")
                           if hasattr(official, k))))

    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)
    val_ids = val_ids[:limit]

    # Predictions come from the saved v2 maps if present; otherwise reuse the
    # ground truth of a *different* annotator as a stand-in, which still
    # exercises both scorers on identical inputs.
    gt_rows, pred_rows = [], []
    ours = []
    for iid in val_ids:
        stem = Path(fname[iid]).stem
        annot = iid.split("-", 1)[0]
        gts = gt_masks_full(by_id[iid])
        for j, m in enumerate(gts, 1):
            gt_rows.append({"filament_id": f"{annot}-{stem}_{j}",
                            "segmentation_rle": rle_of(m)})

    # one prediction set per photograph: take the first annotator's masks
    seen = set()
    preds_by_stem = {}
    for iid in val_ids:
        stem = Path(fname[iid]).stem
        if stem in seen:
            continue
        seen.add(stem)
        preds_by_stem[stem] = gt_masks_full(by_id[iid])
        for j, m in enumerate(preds_by_stem[stem], 1):
            pred_rows.append({"filament_id": f"{stem}_{j}",
                              "segmentation_rle": rle_of(m)})

    gt_df = pd.DataFrame(gt_rows)
    pred_df = pd.DataFrame(pred_rows)
    print(f"{len(val_ids)} annotator-images, {len(gt_df)} GT filaments, "
          f"{len(pred_df)} predictions over {len(preds_by_stem)} photographs\n")

    overlap = official.get_overlap_df(gt_df, pred_df)
    pq_official = official.get_pq_score(overlap)

    for iid in val_ids:
        stem = Path(fname[iid]).stem
        ours.append(pq_from_masks(gt_masks_full(by_id[iid]), preds_by_stem[stem]))
    a = aggregate(ours)

    print(f"{'':<22} {'PQ':>9} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'organisers code':<22} {pq_official:9.6f}")
    print(f"{'src/metrics.py':<22} {a['PQ']:9.6f} {a['TP']:6d} {a['FP']:6d} {a['FN']:6d}")
    diff = abs(pq_official - a["PQ"])
    print(f"\ndifference: {diff:.2e}")
    if diff < 1e-6:
        print("The two implementations agree. Our numbers were never the problem.")
    else:
        print("They disagree -- every score reported so far needs revisiting.")


if __name__ == "__main__":
    main()
