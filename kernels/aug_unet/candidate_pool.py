"""Can a different threshold or merge radius rescue the filaments we find but mis-shape?

Half of the misses and 43% of the false positives are one object: found, but with
IoU 0.2-0.5 (error_analysis.py). So every (threshold, merge) pair proposes its own
outline, and each filament region picks one. Masks in a frame stay disjoint.

  1. pool recall: how many ground-truth filaments have any candidate with IoU > 0.5
  2. oracle selection: the ceiling for any selector of this pool, against the same
     oracle on one setting (the keep-rule's ceiling, 0.5187 at t 0.6, merge 3)
  3. learned selection: gradient boosting, out-of-fold over photographs, on the same
     pool against the same model restricted to the one setting

One process; each frame is cached, so a local crash loses at most one frame.
    python kernels/aug_unet/candidate_pool.py --dir kernels/aug_post/out
"""
from __future__ import annotations

import argparse, collections, json, pickle, sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import post_local as P  # noqa: E402
import run  # noqa: E402

# ---------------------------------------------------------------- pool settings
VARIANT = "ens:flip+full+r50+solar+union/plain"
POOL_T = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
POOL_K = (0, 3, 7, 12)
ONE_SETTING = (0.6, 3)
POOL_MIN_AREA = 100
POOL_CUTS = np.round(np.arange(0.02, 0.6, 0.01), 2)
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
RIM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


N_FEAT = 21


def candidates(p, img, b0=0.0):
    """Distinct components over the whole grid, with features and where they came from.
    b0: the frame's heliographic latitude of the disk centre (run.b0_deg), for the geometry."""
    labs = {(t, k): run.components(p, t, k) for t in POOL_T for k in POOL_K}
    out, index = [], {}
    for (t, k), (lab, area) in labs.items():
        boxes = ndimage.find_objects(lab)
        for j, sl in enumerate(boxes, start=1):
            if sl is None or area[j] < POOL_MIN_AREA:
                continue
            y0, x0 = sl[0].start, sl[1].start
            m = lab[sl] == j
            key = (int(area[j]), y0, x0, m.shape, m.tobytes())
            if key in index:
                out[index[key]]["from"].append((t, k))
                continue
            index[key] = len(out)
            out.append(dict(y0=y0, x0=x0, m=m, area=int(area[j]), t=t, k=k, sl=sl, label=j, **{"from": [(t, k)]}))
    busy = float(sum(c["from"].count(ONE_SETTING) for c in out))
    X = np.array([features(c, p, img, labs, busy, b0) for c in out], np.float32) if out else np.zeros((0, N_FEAT), np.float32)
    return out, X


def features(c, p, img, labs, busy, b0=0.0):
    """The keep-rule's ten, plus how this outline sits among its neighbours in the grid
    (11-17), plus where it sits on the solar sphere and how it lies there (18-21)."""
    pad = 10
    sl = (slice(max(c["sl"][0].start - pad, 0), c["sl"][0].stop + pad),
          slice(max(c["sl"][1].start - pad, 0), c["sl"][1].stop + pad))
    t, k = c["t"], c["k"]
    lab = labs[(t, k)][0]
    m = lab[sl] == c["label"]
    u = m.astype(np.uint8)
    pv = p[sl][m].astype(np.float32) / 255.0
    ring = (cv2.dilate(u, RING) > 0) & ~m
    rim = (cv2.dilate(u, RIM) > 0) & ~m
    iv, rv = img[sl][m].astype(np.float32), img[sl][ring].astype(np.float32)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
    cov = np.cov(np.stack([ys, xs])) if ys.size > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    i = POOL_T.index(t)
    parent = 1.0
    if i > 0:
        pl, pa = labs[(POOL_T[i - 1], k)]
        q = pl[sl][m]
        q = q[q > 0]
        parent = float(pa[np.bincount(q).argmax()]) / c["area"] if q.size else 1.0
    n_child, big_child = 0, 0.0
    if i < len(POOL_T) - 1:
        cl, ca = labs[(POOL_T[i + 1], k)]
        ids = np.unique(cl[sl][m])
        ids = ids[ids > 0]
        ids = ids[ca[ids] >= POOL_MIN_AREA]
        n_child = int(ids.size)
        big_child = float(ca[ids].max()) / c["area"] if ids.size else 0.0
    # the frames are solar-north-up (run.b0_deg), so image x is east-west
    dx, dy = (cx - run.CX) / run.R_DISK, -(cy - run.CY) / run.R_DISK
    z = np.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))
    b = np.radians(b0)
    sinlat = float(np.clip(dy * np.cos(b) + z * np.sin(b), -1.0, 1.0))
    lon = float(np.arctan2(dx, z * np.cos(b) - dy * np.sin(b)) / (np.pi / 2))
    w, V = np.linalg.eigh(cov)
    vy, vx = V[:, int(np.argmax(w))]
    th = np.arctan2(-vy, vx)                  # long axis against east-west; undirected, so 2*theta
    return [np.log(c["area"]), pv.mean(), pv.max(), np.percentile(pv, 90), pv.std(), (pv > 0.8).mean(),
            np.sqrt(ev[0] / max(ev[1], 1e-6)), np.hypot(cx - run.CX, cy - run.CY) / run.R_DISK,
            (rv.mean() - iv.mean()) if rv.size else 0.0, busy,
            t, k, len(c["from"]), np.log(parent), n_child, big_child,
            p[sl][rim].mean() / 255.0 if rim.any() else 0.0,
            sinlat, lon, np.cos(2 * th), np.sin(2 * th)]


def targets(cands, sets):
    """Per annotation set, each candidate's IoU if above 0.5, else 0; and each GT's best IoU."""
    Y = np.zeros((len(cands), len(sets)))
    best = []
    for s, crops in enumerate(sets):
        for y0, x0, g in crops:
            ga, top = int(g.sum()), 0.0
            for n, c in enumerate(cands):
                ya, xa = max(y0, c["y0"]), max(x0, c["x0"])
                yb = min(y0 + g.shape[0], c["y0"] + c["m"].shape[0])
                xb = min(x0 + g.shape[1], c["x0"] + c["m"].shape[1])
                if ya >= yb or xa >= xb:
                    continue
                it = int((g[ya - y0:yb - y0, xa - x0:xb - x0] & c["m"][ya - c["y0"]:yb - c["y0"], xa - c["x0"]:xb - c["x0"]]).sum())
                if it:
                    iou = it / (ga + c["area"] - it)
                    top = max(top, iou)
                    if iou > 0.5:
                        Y[n, s] = max(Y[n, s], iou)
            best.append(top)
    return Y, best


def conflicts(cands):
    """Pairs of candidates that share a pixel, so that only one of them can be submitted."""
    out = [[] for _ in cands]
    for a in range(len(cands)):
        ca = cands[a]
        for b in range(a + 1, len(cands)):
            cb = cands[b]
            ya, xa = max(ca["y0"], cb["y0"]), max(ca["x0"], cb["x0"])
            yb = min(ca["y0"] + ca["m"].shape[0], cb["y0"] + cb["m"].shape[0])
            xb = min(ca["x0"] + ca["m"].shape[1], cb["x0"] + cb["m"].shape[1])
            if ya < yb and xa < xb and (ca["m"][ya - ca["y0"]:yb - ca["y0"], xa - ca["x0"]:xb - ca["x0"]]
                                        & cb["m"][ya - cb["y0"]:yb - cb["y0"], xa - cb["x0"]:xb - cb["x0"]]).any():
                out[a].append(b)
                out[b].append(a)
    return out


N_PTS = 24


def sample_points(c):
    """Up to N_PTS of a candidate's own pixels, (y, x) in the frame, spread along it."""
    ys, xs = np.nonzero(c["m"])
    k = np.linspace(0, ys.size - 1, min(N_PTS, ys.size)).round().astype(int)
    return np.stack([ys[k] + c["y0"], xs[k] + c["x0"]], 1).astype(np.int16)


def cand_rle(c):
    """A candidate as a full-frame COCO RLE, for overlaps with another model's instances."""
    m = np.zeros((run.FULL, run.FULL), np.uint8)
    m[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
    return run.mu.encode(np.asfortranarray(m))


def frame(d, stem, img, sets, cache, variant=VARIANT, with_conflicts=True, with_rle=False):
    path = cache / f"{stem}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    p = P.load_map(d, variant, "val", stem)
    cands, X = candidates(p, img, run.b0_deg(stem))
    Y, best = targets(cands, sets)
    one = np.array([ONE_SETTING in c["from"] for c in cands], bool)
    r = dict(stem=stem, X=X, Y=Y, best=best, n_gt=[len(s) for s in sets],
             conflict=conflicts(cands) if with_conflicts else None, one=one,
             pts=[sample_points(c) for c in cands],
             rle=[cand_rle(c) if o else None for c, o in zip(cands, one)] if with_rle else None)
    pickle.dump(r, open(path, "wb"))
    return r


def select(score, conflict, allowed):
    """Greedy by descending score, skipping anything touching a chosen outline."""
    taken, blocked = [], set()
    for n in np.argsort(-score):
        if allowed[n] and n not in blocked:
            taken.append(int(n))
            blocked.update(conflict[n])
    return taken


def pq_at_cuts(frames, scores, restrict=False):
    """Best PQ over POOL_CUTS. The chosen set at a cut is a prefix of the greedy order."""
    tot = np.zeros((len(POOL_CUTS), 4))
    for f, s in zip(frames, scores):
        allowed = f["one"] if restrict else np.ones(len(s), bool)
        order = select(s, f["conflict"], allowed)
        for i, cut in enumerate(POOL_CUTS):
            chosen = [n for n in order if s[n] >= cut]
            y = f["Y"][chosen] if chosen else np.zeros((0, len(f["n_gt"])))
            for q, n_gt in enumerate(f["n_gt"]):
                tp = int((y[:, q] > 0).sum())
                tot[i] += (tp, len(chosen) - tp, n_gt - tp, float(y[:, q].sum()))
    pq = P.pq_of(tot)
    b = int(np.argmax(pq))
    return dict(PQ=round(float(pq[b]), 4), cut=float(POOL_CUTS[b]), TP=int(tot[b, 0]), FP=int(tot[b, 1]), FN=int(tot[b, 2]),
                SQ=round(float(tot[b, 3] / max(tot[b, 0], 1)), 4))


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="kernels/aug_post/out")
    a = ap.parse_args()
    d = Path(a.dir)
    cache = d / "pool"
    cache.mkdir(parents=True, exist_ok=True)
    meta, gt = P.ground_truth()
    stems = [Path(f).stem for f in meta["val_files"]]
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    val_img = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    frames = []
    for n, (f, sets) in enumerate(sorted(by_frame.items())):
        frames.append(frame(d, stems[f], np.asarray(val_img[f]), sets, cache))
        if (n + 1) % 10 == 0:
            print(f"frame {n + 1}/{len(by_frame)}, candidates so far {sum(len(x['X']) for x in frames)}", flush=True)

    best = np.concatenate([f["best"] for f in frames])
    y_mean = [f["Y"].mean(axis=1) for f in frames]
    res = dict(n_candidates=int(sum(len(y) for y in y_mean)), n_gt=int(best.size),
               gt_with_iou_above_half_in_pool=round(float((best > 0.5).mean()), 4),
               gt_with_iou_above_half_at_one_setting=None,
               oracle_one_setting=pq_at_cuts(frames, y_mean, restrict=True),
               oracle_pool=pq_at_cuts(frames, y_mean))

    X = np.concatenate([f["X"] for f in frames])
    y = np.concatenate(y_mean)
    groups = np.concatenate([[n] * len(f["X"]) for n, f in enumerate(frames)])
    oof = np.zeros(len(y))
    for fold in range(5):
        te = groups % 5 == fold
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=fold)
        oof[te] = m.fit(X[~te], y[~te]).predict(X[te])
    scores = [oof[groups == n] for n in range(len(frames))]
    res["learned_pool"] = pq_at_cuts(frames, scores)
    res["learned_one_setting"] = pq_at_cuts(frames, scores, restrict=True)

    one = np.concatenate([f["one"] for f in frames])
    oof1 = np.zeros(len(y))
    for fold in range(5):
        te = groups % 5 == fold
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, l2_regularization=1.0, random_state=fold)
        oof1[te] = m.fit(X[~te & one], y[~te & one]).predict(X[te])
    res["learned_one_setting_own_model"] = pq_at_cuts(frames, [oof1[groups == n] for n in range(len(frames))], restrict=True)
    json.dump(res, open(d / "candidate_pool.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
