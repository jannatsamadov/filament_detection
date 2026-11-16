"""How much could a perfect "panel of other annotators" add to the keep-rule? (11-16)

On the single R50's 707 out-of-fold frames (single_r50_707.py). For photographs with 2-3 annotation sets, each
set is scored in turn; the keep score of an outline is what the OTHER experts of the same photograph did with it
(their mean IoU-if-matched), i.e. a perfect simulator of other annotators, but blind to the one being scored.
Compared on the same (photograph, set) pairs with the 21-number selector (5-fold out of fold) and the true oracle.
    python kernels/aug_unet/panel_oracle.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import pickle, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_pool as C  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

FR = pickle.load(open(HERE / "out" / "r50_707_frames.pkl", "rb"))
CUTS = np.round(np.arange(0.01, 0.8, 0.01), 2)


def selector_oof(pool):
    sc = [None] * len(FR)
    for k in range(5):
        X, y = [], []
        for f in FR:
            if f["fold"] != k:
                r = np.ones(len(f["one"]), bool) if pool else f["one"]
                if r.any():
                    X.append(f["X"][r]); y.append(f["Y"][r].mean(1))
        m = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
                                          random_state=0).fit(np.vstack(X), np.concatenate(y))
        for n, f in enumerate(FR):
            if f["fold"] == k:
                sc[n] = m.predict(f["X"]) if len(f["one"]) else np.zeros(0)
    return sc


def tally(pairs, cut, pool):
    """pairs: (frame, set index q, scores). Each scored against set q only."""
    tot = np.zeros(4)
    for f, q, s in pairs:
        allowed = np.ones(len(s), bool) if pool else f["one"]
        chosen = [n for n in C.select(s, f["conflict"], allowed) if s[n] >= cut]
        y = f["Y"][chosen, q] if chosen else np.zeros(0)
        tp = int((y > 0).sum())
        tot += (tp, len(chosen) - tp, f["n_gt"][q] - tp, float(y.sum()))
    return tot


def best(pairs, pool):
    r = []
    for c in CUTS:
        tp, fp, fn, iou = tally(pairs, c, pool)
        r.append((iou / max(tp + 0.5 * fp + 0.5 * fn, 1e-9), c, tp, fp, fn, iou / max(tp, 1)))
    return max(r)


def main():
    rng = np.random.default_rng(0)
    for pool in (False, True):
        sel = selector_oof(pool)
        multi = [(n, f) for n, f in enumerate(FR) if len(f["n_gt"]) >= 2 and len(f["one"])]
        arms = {"21-number selector (out of fold)": [], "panel: the other experts' mean (perfect simulator)": [],
                "panel: one other expert": [], "selector + panel mean, averaged": [],
                "selector, fixed-rule photographs' cut check": None, "true oracle (the scored expert's own answer)": []}
        for n, f in multi:
            Y = f["Y"]
            for q in range(Y.shape[1]):
                others = [j for j in range(Y.shape[1]) if j != q]
                pm = Y[:, others].mean(1)
                p1 = Y[:, rng.choice(others)]
                arms["21-number selector (out of fold)"].append((f, q, sel[n]))
                arms["panel: the other experts' mean (perfect simulator)"].append((f, q, pm + 1e-6 * sel[n]))
                arms["panel: one other expert"].append((f, q, p1 + 1e-6 * sel[n]))
                arms["selector + panel mean, averaged"].append((f, q, 0.5 * sel[n] + 0.5 * pm))
                arms["true oracle (the scored expert's own answer)"].append((f, q, Y[:, q] + 1e-6 * sel[n]))
        print(f"\n{'whole pool' if pool else 'one outline (t 0.6, join 3)'}: {len(multi)} photographs with 2-3 experts, "
              f"{len(arms['panel: one other expert'])} (photograph, expert) pairs")
        for name, pairs in arms.items():
            if pairs is None:
                continue
            pq, c, tp, fp, fn, sq = best(pairs, pool)
            print(f"  {name:52s} PQ {pq:.4f}  SQ {sq:.3f}  TP {tp:.0f} FP {fp:.0f} FN {fn:.0f}  cut {c}")
        # by the number of other experts
        for k in (2, 3):
            sub = {a: [p for p in pairs if len(p[0]["n_gt"]) == k] for a, pairs in arms.items() if pairs}
            print(f"   photographs with {k} experts: " + "; ".join(
                f"{a.split(':')[0].split('(')[0].strip()[:22]} {best(p, pool)[0]:.4f}" for a, p in sub.items()))


if __name__ == "__main__":
    main()
