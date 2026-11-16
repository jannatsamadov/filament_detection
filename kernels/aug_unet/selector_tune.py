"""The keep-rule, other ways (11-16): on the single R50's 707 out-of-fold frames (single_r50_707.py's rows).

Five-fold out-of-fold by the U-Nets' own folds; every variant's cut chosen the same way (on its own out-of-fold
scores), so they compare fairly with each other.
    python kernels/aug_unet/selector_tune.py   -> out/selector_tune.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import json, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
from run import np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesRegressor  # noqa: E402

FR = pickle.load(open(HERE / "out" / "r50_707_frames.pkl", "rb"))
CUTS = np.round(np.arange(0.02, 0.8, 0.01), 2)


def rows(frames, target, pool=False):
    X, y = [], []
    for f in frames:
        r = np.ones(len(f["one"]), bool) if pool else f["one"]
        if r.any():
            X.append(f["X"][r])
            Y = f["Y"][r]
            y.append({"mean_iou": Y.mean(1), "hit_rate": (Y > 0).mean(1), "max_iou": Y.max(1),
                      "any_hit": (Y > 0).any(1).astype(float)}[target] if Y.shape[1] else np.zeros(len(Y)))
    return np.vstack(X), np.concatenate(y)


def make(kind):
    if kind == "base":
        return HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=0)
    if kind == "bigger":
        return HistGradientBoostingRegressor(max_iter=800, max_leaf_nodes=31, learning_rate=0.03,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    if kind == "smaller":
        return HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=7, learning_rate=0.05,
                                             min_samples_leaf=60, l2_regularization=3.0, random_state=0)
    if kind == "classifier":
        return HistGradientBoostingClassifier(max_iter=300, max_leaf_nodes=15, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=0)
    if kind == "extratrees":
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=10, max_features=0.5, n_jobs=3, random_state=0)


def oof(kind, target, pool=False):
    scores = [None] * len(FR)
    for k in range(5):
        fit = [f for f in FR if f["fold"] != k]
        X, y = rows(fit, target, pool)
        m = make(kind)
        if kind == "classifier":
            m.fit(X, (y > 0.5).astype(int))
        else:
            m.fit(X, y)
        for n, f in enumerate(FR):
            if f["fold"] == k:
                if not len(f["one"]):
                    scores[n] = np.zeros(0)
                elif kind == "classifier":
                    scores[n] = m.predict_proba(f["X"])[:, 1]
                else:
                    scores[n] = m.predict(f["X"])
    pq = [S.summary(S.tally(FR, scores, c, not pool)) for c in CUTS]
    b = int(np.argmax([r["PQ"] for r in pq]))
    return dict(pq[b], cut=float(CUTS[b]))


def main():
    t0 = time.time()
    arms = [("base", "mean_iou", False), ("base", "hit_rate", False), ("base", "max_iou", False),
            ("base", "any_hit", False), ("bigger", "mean_iou", False), ("smaller", "mean_iou", False),
            ("classifier", "hit_rate", False), ("extratrees", "mean_iou", False), ("base", "mean_iou", True),
            ("bigger", "mean_iou", True)]
    res = {}
    for kind, target, pool in arms:
        name = f"{kind}, target {target}, {'whole pool' if pool else 'one outline'}"
        res[name] = oof(kind, target, pool)
        r = res[name]
        print(f"{name:48s} PQ {r['PQ']:.4f} SQ {r['SQ']:.3f} RQ {r['RQ']:.3f} TP {r['TP']} FP {r['FP']} FN {r['FN']} "
              f"cut {r['cut']}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(HERE / "out" / "selector_tune.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
