"""Where PQ is lost: pixels, grouping, or the filter (the user's "ideal model", 11-14).

Fold-0 validation, 141 photographs, scored the way the competition scores (every expert's set of
filaments on a photograph is one ground truth; TP, FP, FN and IoU pooled over all sets).

  1. the submitted mix's map (5 U-Nets + Qwen) through threshold, grouping and minimum area;
  2. the same candidates with a perfect filter: a candidate is kept exactly when its true matched
     IoU, averaged over the experts, pays (the ceiling for any keep-rule on these candidates);
  3. perfect pixels: the experts' own union (or majority) as the map, then the same grouping;
  4. perfect pixels and a perfect filter;
  5. one expert's own filaments scored against the other experts (the human level).

    python kernels/aug_unet/ideal_model.py   -> kernels/aug_unet/out/ideal_model.json
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel_cls as PC  # noqa: E402
import run  # noqa: E402

OUT = HERE / "out" / "ideal_model.json"
MIX = "5 U-Nets + Qwen (submitted mix)"
POINT = (0.65, 7, 350)                  # the mix's best grid point on fold 0 (pixnet_mix.json, ens0q)
CUTS = np.round(np.arange(0.0, 0.61, 0.01), 3)


def crops_of(mask_list):
    out = []
    for m in mask_list:
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out.append((int(y0), int(x0), m[y0:y1, x0:x1].copy()))
    return out


def summary(c):
    tp, fp, fn, s = c
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=round(s / den, 4) if den else 0.0, SQ=round(s / tp, 4) if tp else 0.0,
                RQ=round(tp / den, 4) if den else 0.0, TP=int(tp), FP=int(fp), FN=int(fn))


def counts(lab, area, keep, sets):
    c = np.zeros(4)
    for crops in sets:
        c += run.pq_counts(crops, lab, area, keep)
    return c


def matched(lab, area, ids, sets):
    """Per candidate, its matched IoU in each set (0 where it matches nothing)."""
    Y = np.zeros((len(ids), len(sets)))
    pos = {j: n for n, j in enumerate(ids)}
    for s_i, crops in enumerate(sets):
        for y0, x0, c in crops:
            lv = lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c]
            u, n = np.unique(lv[lv > 0], return_counts=True)
            for j, it in zip(u.tolist(), n.tolist()):
                if j in pos:
                    iou = it / (int(c.sum()) + int(area[j]) - it)
                    if iou > 0.5:
                        Y[pos[j], s_i] = max(Y[pos[j], s_i], iou)
    return Y


class Oracle:
    """Collects candidates' matched IoUs; the best single cut on their mean is the perfect filter."""
    def __init__(self):
        self.frames = []

    def add(self, lab, area, keep, sets):
        ids = np.flatnonzero(keep)
        self.frames.append((matched(lab, area, ids, sets), [len(c) for c in sets]))

    def best(self):
        res = None
        for cut in CUTS:
            tp = fp = fn = s = 0.0
            for Y, n_gt in self.frames:
                k = Y.mean(axis=1) > cut if len(Y) else np.zeros(0, bool)
                for s_i, ng in enumerate(n_gt):
                    iou = Y[k, s_i]
                    t_ = int((iou > 0).sum())
                    tp += t_; s += float(iou.sum()); fp += int(k.sum()) - t_; fn += ng - t_
            r = summary((tp, fp, fn, s))
            if res is None or r["PQ"] > res["PQ"]:
                res = dict(r, cut=float(cut))
        return res


def main():
    D = PC.Dumps()
    stems = D.stems()
    by = PC.experts(set(stems))
    stems = [s for s in stems if s in by]
    t, k, a = POINT
    acc = {n: np.zeros(4) for n in ("mix", "union_k0", "union_k3", "union_k7", "major_k0", "major_k3", "major_k7",
                                    "human_one_vs_others")}
    orc = {n: Oracle() for n in ("mix", "union_k3", "major_k3")}
    for i, stem in enumerate(stems):
        masks_by_set = [[run.ann_mask(x) for x in anns] for anns in by[stem].values()]
        sets = [crops_of(ms) for ms in masks_by_set]
        unions = [np.logical_or.reduce(ms) if ms else np.zeros((run.FULL, run.FULL), bool) for ms in masks_by_set]
        # 1-2. the submitted mix
        p = D.model(MIX, stem, {})
        lab, area = run.components(p, t, k)
        keep = area >= a
        keep[0] = False
        acc["mix"] += counts(lab, area, keep, sets)
        orc["mix"].add(lab, area, keep, sets)
        # 3-4. perfect pixels: the experts' union, and their majority
        votes = np.sum(unions, axis=0)
        for name, m in (("union", votes >= 1), ("major", votes * 2 >= len(unions))):
            pm = m.astype(np.uint8) * 255
            for kk in (0, 3, 7):
                lab, area = run.components(pm, 0.5, kk)
                keep = area > 0
                keep[0] = False
                acc[f"{name}_k{kk}"] += counts(lab, area, keep, sets)
                if kk == 3:
                    orc[f"{name}_k3"].add(lab, area, keep, sets)
        # 5. each expert's own filaments against every other expert's set on the photograph
        if len(sets) > 1:
            for j, ms in enumerate(masks_by_set):
                lab = np.zeros((run.FULL, run.FULL), np.int32)
                for n, m in enumerate(ms, 1):
                    lab[m & (lab == 0)] = n
                area = np.bincount(lab.ravel(), minlength=len(ms) + 1)
                keep = area > 0
                keep[0] = False
                acc["human_one_vs_others"] += counts(lab, area, keep, [c for q, c in enumerate(sets) if q != j])
        if i % 20 == 0:
            print(f"  {i + 1}/{len(stems)} {stem}", flush=True)
    res = dict(photographs=len(stems), point=POINT, results={n: summary(c) for n, c in acc.items()},
               perfect_filter={n: o.best() for n, o in orc.items()})
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
