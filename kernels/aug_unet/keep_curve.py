"""Would more candidates help the keep-rule, and do the grid features really add anything?

Reads the frames cached by candidate_pool.py, at the one setting t 0.6, merge 3.
The cut is nested: chosen on the other folds, applied to the held-out one, so these
numbers carry no selection bias, unlike the in-sample cuts in candidate_pool.json.

  learning curve: train on 25 / 50 / 100% of the other folds' photographs
  features: the keep-rule's ten against all seventeen, paired bootstrap over photographs
"""
from __future__ import annotations

import json, pickle, sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/aug_post/out")
CUTS = np.round(np.arange(0.02, 0.6, 0.01), 2)
OLD, ALL = list(range(10)), list(range(17))
SEEDS = 3


def load():
    frames = []
    for p in sorted((OUT / "pool").glob("*.pkl")):
        f = pickle.load(open(p, "rb"))
        k = np.flatnonzero(f["one"])
        frames.append(dict(X=f["X"][k], Y=f["Y"][k], n_gt=f["n_gt"]))
    return frames


def counts(frames, scores, cuts):
    """Per photograph and cut: TP, FP, FN, summed IoU. Components at one setting are disjoint."""
    c = np.zeros((len(frames), len(cuts), 4))
    for n, (f, s) in enumerate(zip(frames, scores)):
        for i, cut in enumerate(cuts):
            keep = s >= cut
            y = f["Y"][keep]
            for q, g in enumerate(f["n_gt"]):
                tp = (y[:, q] > 0).sum()
                c[n, i] += (tp, keep.sum() - tp, g - tp, y[:, q].sum())
    return c


def pq(c):
    den = c[..., 0] + 0.5 * c[..., 1] + 0.5 * c[..., 2]
    return np.where(den > 0, c[..., 3] / np.maximum(den, 1e-9), 0.0)


def oof(frames, cols, frac, seed):
    """Out-of-fold scores, training on a random fraction of the other folds' photographs."""
    rng = np.random.default_rng(seed)
    scores = [None] * len(frames)
    for fold in range(5):
        others = [n for n in range(len(frames)) if n % 5 != fold]
        take = rng.choice(others, int(round(len(others) * frac)), replace=False)
        X = np.concatenate([frames[n]["X"][:, cols] for n in take])
        y = np.concatenate([frames[n]["Y"].mean(axis=1) for n in take])
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=fold).fit(X, y)
        for n in range(len(frames)):
            if n % 5 == fold:
                scores[n] = m.predict(frames[n]["X"][:, cols]) if len(frames[n]["X"]) else np.zeros(0)
    return scores


def nested(c):
    """Per photograph counts with each fold's cut chosen on the other four folds."""
    idx = np.arange(len(c))
    out = np.zeros((len(c), 4))
    for fold in range(5):
        te = idx % 5 == fold
        out[te] = c[te, int(np.argmax(pq(c[~te].sum(axis=0))))]
    return out


def main():
    frames = load()
    res = dict(photographs=len(frames), candidates=int(sum(len(f["X"]) for f in frames)), curve={})
    print(res, flush=True)
    per = {}
    for name, cols in (("old10", OLD), ("all17", ALL)):
        for frac in (0.25, 0.5, 1.0):
            vals, last = [], None
            for seed in range(SEEDS if frac < 1 else 1):
                last = nested(counts(frames, oof(frames, cols, frac, seed), CUTS))
                vals.append(float(pq(last.sum(axis=0))))
            res["curve"][f"{name}_{int(frac * 100)}"] = dict(PQ=round(float(np.mean(vals)), 4), runs=[round(v, 4) for v in vals])
            print(name, frac, vals, flush=True)
            if frac == 1.0:
                per[name] = last
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        b = rng.integers(0, len(frames), len(frames))
        d.append(float(pq(per["all17"][b].sum(axis=0)) - pq(per["old10"][b].sum(axis=0))))
    res["all17_minus_old10"] = dict(mean=round(float(np.mean(d)), 4), lo=round(float(np.percentile(d, 2.5)), 4),
                                    hi=round(float(np.percentile(d, 97.5)), 4))
    json.dump(res, open(OUT / "keep_curve.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
