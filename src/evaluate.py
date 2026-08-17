"""Run a trained model over a set of images and score it with Panoptic Quality.

Scoring happens at the full 2048x2048 resolution the competition uses, not at the
training resolution, so validation numbers mean the same thing as leaderboard
numbers. Note the scorer here is ours, not the organisers'; see src/metrics.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import pq_from_masks, aggregate      # noqa: E402
from model import forward_images                  # noqa: E402

FULL = 2048


def gt_masks_full(anns, full=FULL):
    out = []
    for a in anns:
        p = a["segmentation"][0]
        m = Image.new("L", (full, full), 0)
        ImageDraw.Draw(m).polygon(list(zip(p[0::2], p[1::2])), fill=1)
        arr = np.asarray(m, bool)
        if arr.any():
            out.append(arr)
    return out


def predict_masks(model, image, device, score_thresh, mask_thresh=0.5,
                  min_area=1, full=FULL, governor=None):
    """image: [1,S,S] float tensor. Returns (masks at full res, scores).

    min_area is at least 1 on purpose. A confident box whose soft mask never
    crosses `mask_thresh` decodes to an empty mask; keeping it would add a false
    positive that can never match anything. Early in training most masks are
    like that, which is how a first evaluation reported 782 false positives.
    """
    min_area = max(int(min_area), 1)
    model.eval()
    with torch.inference_mode():
        imgs = forward_images(model, [image.to(device)])
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs)[0]
    scores = out["scores"].float().cpu().numpy()
    keep = scores >= score_thresh
    if keep.sum() == 0:
        return [], np.array([])
    soft = out["masks"][keep].float()                # [N,1,S,S] soft
    scores = scores[keep]

    # Upsample a few at a time. All at once is what pushed peak memory to
    # 5751 MiB of 6144 during the first validation: N masks at 2048x2048 in
    # float32 is 16 MB each, and N can be in the hundreds early in training.
    out_masks, out_scores = [], []
    chunk = 8
    for a in range(0, soft.shape[0], chunk):
        big = torch.nn.functional.interpolate(
            soft[a:a + chunk], size=(full, full), mode="bilinear", align_corners=False)
        big = (big[:, 0] > mask_thresh).cpu().numpy()
        for j, m in enumerate(big):
            if m.sum() >= min_area:
                out_masks.append(m); out_scores.append(scores[a + j])
        del big
    return out_masks, np.array(out_scores)


def evaluate(model, dataset, by_id, device, score_thresh=0.5, mask_thresh=0.5,
             min_area=0, limit=None, governor=None, verbose=False):
    results = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        image, _ = dataset[i]
        iid = dataset.ids[i]
        preds, _ = predict_masks(model, image, device, score_thresh,
                                 mask_thresh, min_area)
        gts = gt_masks_full(by_id[iid])
        results.append(pq_from_masks(gts, preds))
        if verbose and (i + 1) % 25 == 0:
            a = aggregate(results)
            print(f"    [{i+1}/{n}] running PQ={a['PQ']:.4f}", flush=True)
        if governor is not None:
            governor._regulate()
    return results, aggregate(results), aggregate(results, mode="image")


def sweep_threshold(model, dataset, by_id, device, thresholds, mask_thresh=0.5,
                    min_area=0, limit=None):
    """Detection threshold is the main calibration knob for PQ: too low floods
    false positives, too high loses real filaments. Both halve the score."""
    rows = []
    for t in thresholds:
        _, ds, im = evaluate(model, dataset, by_id, device, score_thresh=t,
                             mask_thresh=mask_thresh, min_area=min_area, limit=limit)
        rows.append((t, ds, im))
        print(f"  score_thresh={t:.2f}  PQ={ds['PQ']:.4f}  SQ={ds['SQ']:.4f} "
              f"RQ={ds['RQ']:.4f}  TP={ds['TP']} FP={ds['FP']} FN={ds['FN']}", flush=True)
    return rows
