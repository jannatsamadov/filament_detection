"""Panoptic Quality for filament instance masks.

PQ = sum(IoU over matched pairs) / (|TP| + |FP|/2 + |FN|/2), matching at IoU > 0.5.
At that threshold a match is unique, so greedy matching by descending IoU is exact.

This is our own implementation. The organisers' Self Evaluation notebook holds the
official one; every number produced here must be cross-checked against it before
it is reported as a competition score.
"""
from __future__ import annotations

import collections
import numpy as np

__all__ = ["pq_from_labels", "pq_from_masks", "PQResult", "aggregate"]


class PQResult:
    __slots__ = ("pq", "sq", "rq", "tp", "fp", "fn", "ious", "matches")

    def __init__(self, pq, sq, rq, tp, fp, fn, ious, matches):
        self.pq, self.sq, self.rq = pq, sq, rq
        self.tp, self.fp, self.fn = tp, fp, fn
        self.ious = ious          # IoU of each matched pair
        self.matches = matches    # list of (gt_index, pred_index, iou), 1-based labels

    def __repr__(self):
        return (f"PQ={self.pq:.4f} SQ={self.sq:.4f} RQ={self.rq:.4f} "
                f"TP={self.tp} FP={self.fp} FN={self.fn}")


def pq_from_labels(gt_lab: np.ndarray, pred_lab: np.ndarray, n_gt: int, n_pred: int,
                   iou_thr: float = 0.5) -> PQResult:
    """gt_lab / pred_lab: int label images, 0 = background, 1..n = instances.

    Label images cannot represent overlapping instances. Use pq_from_masks when
    predictions may overlap.
    """
    if n_gt == 0 and n_pred == 0:
        return PQResult(1.0, 1.0, 1.0, 0, 0, 0, [], [])

    area_g = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    area_p = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

    both = (gt_lab > 0) & (pred_lab > 0)
    if both.any():
        key = gt_lab[both].astype(np.int64) * (n_pred + 1) + pred_lab[both].astype(np.int64)
        counts = collections.Counter(key.tolist())
    else:
        counts = {}

    ious = {}
    for k, inter in counts.items():
        g, p = divmod(k, n_pred + 1)
        union = area_g[g] + area_p[p] - inter
        if union > 0:
            ious[(g, p)] = inter / union

    matches, used_g, used_p = [], set(), set()
    for (g, p), v in sorted(ious.items(), key=lambda kv: -kv[1]):
        if v > iou_thr and g not in used_g and p not in used_p:
            matches.append((g, p, v)); used_g.add(g); used_p.add(p)

    tp = len(matches)
    fp = n_pred - len(used_p)
    fn = n_gt - len(used_g)
    denom = tp + 0.5 * fp + 0.5 * fn
    vals = [m[2] for m in matches]
    pq = (sum(vals) / denom) if denom else 0.0
    sq = float(np.mean(vals)) if vals else 0.0
    rq = (tp / denom) if denom else 0.0
    return PQResult(pq, sq, rq, tp, fp, fn, vals, matches)


def pq_from_masks(gt_masks, pred_masks, iou_thr: float = 0.5) -> PQResult:
    """Lists/arrays of boolean masks of identical shape. Handles overlap."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return PQResult(1.0, 1.0, 1.0, 0, 0, 0, [], [])

    g_flat = [np.asarray(m, bool).ravel() for m in gt_masks]
    p_flat = [np.asarray(m, bool).ravel() for m in pred_masks]
    g_area = np.array([m.sum() for m in g_flat], dtype=np.int64)
    p_area = np.array([m.sum() for m in p_flat], dtype=np.int64)

    ious = {}
    for i, gm in enumerate(g_flat, start=1):
        gi = np.nonzero(gm)[0]
        if gi.size == 0:
            continue
        for j, pm in enumerate(p_flat, start=1):
            inter = int(pm[gi].sum())
            if inter == 0:
                continue
            union = g_area[i - 1] + p_area[j - 1] - inter
            if union > 0:
                ious[(i, j)] = inter / union

    matches, used_g, used_p = [], set(), set()
    for (g, p), v in sorted(ious.items(), key=lambda kv: -kv[1]):
        if v > iou_thr and g not in used_g and p not in used_p:
            matches.append((g, p, v)); used_g.add(g); used_p.add(p)

    tp = len(matches)
    fp = n_pred - len(used_p)
    fn = n_gt - len(used_g)
    denom = tp + 0.5 * fp + 0.5 * fn
    vals = [m[2] for m in matches]
    pq = (sum(vals) / denom) if denom else 0.0
    sq = float(np.mean(vals)) if vals else 0.0
    rq = (tp / denom) if denom else 0.0
    return PQResult(pq, sq, rq, tp, fp, fn, vals, matches)


def aggregate(results, mode: str = "dataset"):
    """Combine per-image results.

    mode='dataset' : pool TP/FP/FN over all images, then form PQ once. This is
                     what the panoptic-segmentation paper specifies.
    mode='image'   : mean of per-image PQ. Reported alongside because it is what
                     a per-image leaderboard would show.
    """
    if mode == "image":
        return {
            "PQ": float(np.mean([r.pq for r in results])),
            "SQ": float(np.mean([r.sq for r in results])),
            "RQ": float(np.mean([r.rq for r in results])),
            "TP": int(sum(r.tp for r in results)),
            "FP": int(sum(r.fp for r in results)),
            "FN": int(sum(r.fn for r in results)),
        }
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    s = sum(sum(r.ious) for r in results)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "PQ": (s / denom) if denom else 0.0,
        "SQ": (s / tp) if tp else 0.0,
        "RQ": (tp / denom) if denom else 0.0,
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
    }
