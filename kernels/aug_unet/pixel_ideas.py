"""The user's questions about the pixel forest (2026-11-14), each tested as its own arm on fold 0.

  base     the forest as filament-pixel-rf0 ran it: the 35-column pixel table, target = the share
           of the photograph's experts who drew the pixel
  onehot   the station as seven 0/1 columns instead of one number 0-6
  photo    the same number of training rows, a third from the photograph and a third from each of
           two copies with only brightness, contrast, gamma, blur or sharpening, noise and JPEG
           changed (run.photometric) -- no rotation or mirror, so the location columns stay true
  count    target by how many experts drew the pixel (user): 1 -> 0.45, 2 -> 0.65, 3 -> 0.95
  count1   the same, except that on a photograph with one expert his pixels are 0.95: 58% of the
           photographs have a single expert, and under `count` all their filaments are 0.45
  context  auto-context: a second forest that reads the table plus a first forest's map around the
           pixel -- how sure it is there and nearby, and whether its filament pixels join into an
           elongated piece (the user's "classify the pixels, then keep those that form a filament").
           A training photograph's first-forest map comes from the half of the training set that
           does not hold it; a validation photograph's is the mean of both halves' maps (`stage1`).

Trained on fold 0's 566 training photographs, scored on its 141 validation photographs: row AUC and
permutation importance here, the full pixel metrics locally by aug_unet/pixel_cls.py from the tarred
maps. CPU only.
    kernels/build.py builds kernels/pixel_ideas/run.py;  FIL_IDEAS_SMOKE=1 python pixel_ideas.py  locally
"""
from __future__ import annotations

import collections, gc, json, multiprocessing as mp, os, random, shutil, sys, tarfile, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402
import pixel_gbm as G  # noqa: E402

SMOKE = os.environ.get("FIL_IDEAS_SMOKE") == "1"
ARMS = ("base", "onehot", "photo", "count", "count1")        # one forest each, on the table alone
N_AUG = 2
COUNT_SCALE = np.array([0.0, 0.45, 0.65, 0.95], np.float32)   # experts who drew the pixel -> target (user, 11-14)
RF = dict(G.PIX_RF, n_estimators=5 if SMOKE else G.PIX_RF["n_estimators"])
N_ROWS = {k: v // 10 for k, v in G.PIX_N.items()} if SMOKE else dict(G.PIX_N)
NB_NAMES = ["p1", "p1_g2", "p1_g4", "p1_g8", "p1_max5", "p1_max11", "p1_max21", "p1_min5",
            "piece05_area", "piece05_length", "piece05_elong", "piece05_area_near", "piece05_length_near",
            "piece05_dist", "piece03_area", "piece03_length", "piece03_elong"]
FEAT_CTX = G.PIX_FEAT_A + NB_NAMES
WORK = run.WORK
MAPS = (Path("/tmp") if run.ON_KAGGLE else WORK) / ("ideas_maps_smoke" if SMOKE else "ideas_maps")
SETS = {}        # file name -> one list of annotations per expert; filled before the workers fork
MODELS = {}      # filled before the prediction passes fork
TR_DIR = None
_YX = None


# ---------------------------------------------------------------- truth and rows

def truth(f):
    """Per pixel at 2048, how many of the photograph's experts drew it; and how many experts there are."""
    cnt = np.zeros((run.FULL, run.FULL), np.uint8)
    for anns in SETS[f]:
        u = np.zeros((run.FULL, run.FULL), bool)
        for a in anns:
            u |= run.ann_mask(a)
        cnt += u
    return cnt, len(SETS[f])


def targets(cnt, n):
    """The three targets at the table's 1024 px, disk pixels only."""
    on = G.PIX_G[3]

    def half(a):
        return cv2.resize(a.astype(np.float32), (G.PIX_HALF, G.PIX_HALF), interpolation=cv2.INTER_AREA)[on]
    scale = COUNT_SCALE[np.minimum(cnt, 3)]
    single = np.where(cnt > 0, COUNT_SCALE[3], 0.0) if n == 1 else scale
    return dict(share=half(cnt.astype(np.float32) / n), count=half(scale), count1=half(single))


def sample(t, rng, frac=1.0):
    """Disk-pixel indices of one photograph's training rows: pixels an expert drew, a 4 px ring around
    them and the rest, sampled separately and weighted back to their true counts (as G.pix_rows)."""
    on = G.PIX_G[3]
    pos2d = np.zeros(on.shape, bool)
    pos2d[on] = t > 0
    ring = (cv2.dilate(pos2d.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~pos2d)[on]
    pos = t > 0
    idx, w = [], []
    for name, sel in (("pos", pos), ("ring", ring), ("far", ~pos & ~ring)):
        cand = np.flatnonzero(sel)
        if cand.size == 0:
            continue
        take = rng.choice(cand, min(max(1, int(N_ROWS[name] * frac)), cand.size), replace=False)
        idx.append(take)
        w.append(np.full(take.size, cand.size / take.size, np.float32))
    return np.concatenate(idx), np.concatenate(w)


def expand(X, t, w):
    """A soft target t as a positive copy of weight w t and a negative one of weight w (1 - t): cross-entropy on t."""
    soft = (t > 0) & (t < 1)
    Xr = np.concatenate([X, X[soft]])
    y = np.concatenate([(t > 0).astype(np.int8), np.zeros(int(soft.sum()), np.int8)])
    wr = np.concatenate([np.where(t > 0, w * t, w), w[soft] * (1.0 - t[soft])]).astype(np.float32)
    return Xr, y, wr


def onehot(X):
    """The station column (0-6) as seven 0/1 columns at the end."""
    j = G.PIX_STATION
    st = np.clip(X[:, j].astype(np.int64), 0, len(G.PIX_STATIONS))
    return np.concatenate([np.delete(X, j, axis=1), np.eye(len(G.PIX_STATIONS) + 1, dtype=np.float32)[st]], axis=1)


def seed_of(stem):
    return int(stem[:14]) % (2 ** 32)


def rows_job(job):
    """Pass 1, one training photograph: table rows with all three targets, and the `photo` arm's rows."""
    f, stem = job
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    T = targets(cnt, n)
    rng = np.random.default_rng(seed_of(stem))
    X = G.pix_features(img, stem)
    idx, w = sample(T["share"], rng)
    out = dict(X=X[idx], w=w, t={a: T[a][idx] for a in T})
    k = 1.0 / (1 + N_AUG)
    i3, w3 = sample(T["share"], rng, k)
    px, pt, pw = [X[i3]], [T["share"][i3]], [w3]
    random.seed(seed_of(stem))
    for _ in range(N_AUG):
        Xa = G.pix_features(run.photometric(img), stem)
        ia, wa = sample(T["share"], rng, k)
        px.append(Xa[ia]); pt.append(T["share"][ia]); pw.append(wa)
    out["photo"] = (np.concatenate(px), np.concatenate(pt), np.concatenate(pw) * k)
    return out


# ---------------------------------------------------------------- the first forest's map around a pixel

def pieces(p, thr):
    """Per pixel: area (px at 1024), length (px) and elongation of the 8-connected piece of p >= thr it is in."""
    global _YX
    n, lab = cv2.connectedComponents((p >= thr).astype(np.uint8), connectivity=8)
    if n <= 1:
        z = np.zeros_like(p)
        return z, z, z
    if _YX is None:
        yy, xx = np.indices(p.shape, dtype=np.float64)
        _YX = (yy.ravel(), xx.ravel())
    yy, xx = _YX
    lb = lab.ravel()
    a = np.bincount(lb, minlength=n).astype(np.float64)
    m = {}
    for key, v in (("x", xx), ("y", yy), ("xx", xx * xx), ("yy", yy * yy), ("xy", xx * yy)):
        m[key] = np.bincount(lb, v, n) / a
    cxx, cyy = m["xx"] - m["x"] ** 2, m["yy"] - m["y"] ** 2
    cxy = m["xy"] - m["x"] * m["y"]
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0.0))
    l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1.0 / 12)
    length, elong = np.sqrt(12.0 * np.maximum(l1, 1.0 / 12)), np.sqrt(np.maximum(l1, 1.0 / 12) / l2)
    a[0] = length[0] = elong[0] = 0.0
    return (a[lab].astype(np.float32), length[lab].astype(np.float32), elong[lab].astype(np.float32))


def full_map(prob_on):
    a = np.zeros((G.PIX_HALF, G.PIX_HALF), np.float32)
    a[G.PIX_G[3]] = prob_on
    return a


def neighbours(p):
    """(n_disk, 17): the first forest's map at and around each disk pixel, and the pieces it forms."""
    cols = [p] + [cv2.GaussianBlur(p, (0, 0), s) for s in (2.0, 4.0, 8.0)]
    cols += [cv2.dilate(p, np.ones((k, k), np.uint8)) for k in (5, 11, 21)] + [cv2.erode(p, np.ones((5, 5), np.uint8))]
    a5, l5, e5 = pieces(p, 0.5)
    k11 = np.ones((11, 11), np.uint8)
    dist = np.minimum(cv2.distanceTransform((p < 0.5).astype(np.uint8), cv2.DIST_L2, 5), 64.0)
    a3, l3, e3 = pieces(p, 0.3)
    cols += [a5, l5, e5, cv2.dilate(a5, k11), cv2.dilate(l5, k11), dist, a3, l3, e3]
    on = G.PIX_G[3]
    return np.stack([c[on] for c in cols], axis=1).astype(np.float32)


def ctx_rows_job(job):
    """Pass 2, one training photograph: the same rows as pass 1, with the other half's map around them."""
    f, stem, half = job
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    T = targets(cnt, n)
    rng = np.random.default_rng(seed_of(stem))
    X = G.pix_features(img, stem)
    idx, w = sample(T["share"], rng)
    nb = neighbours(full_map(MODELS[f"h{1 - half}"].predict_proba(X)[:, 1]))
    return np.concatenate([X[idx], nb[idx]], axis=1), T["share"][idx], w


def write_map(arm, stem, prob_on):
    full = cv2.resize(full_map(prob_on), (run.FULL, run.FULL), interpolation=cv2.INTER_LINEAR)
    (MAPS / arm).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MAPS / arm / f"{stem}.png"), np.round(np.clip(full, 0.0, 1.0) * 255.0).astype(np.uint8))


def val_job(job):
    """Pass 3, one validation photograph: every arm's map, and sampled rows for the AUCs and importance."""
    f, stem = job
    img = cv2.imread(str(TR_DIR / f), cv2.IMREAD_GRAYSCALE)
    cnt, n = truth(f)
    T = targets(cnt, n)
    X = G.pix_features(img, stem)
    pr = {a: MODELS[a].predict_proba(onehot(X) if a == "onehot" else X)[:, 1] for a in ARMS}
    pr["stage1"] = 0.5 * (MODELS["h0"].predict_proba(X)[:, 1] + MODELS["h1"].predict_proba(X)[:, 1])
    nb = neighbours(full_map(pr["stage1"]))
    XC = np.concatenate([X, nb], axis=1)
    pr["context"] = MODELS["context"].predict_proba(XC)[:, 1]
    for a, p in pr.items():
        write_map(a, stem, p)
    idx, w = sample(T["share"], np.random.default_rng(seed_of(stem) + 1))
    return dict(X=XC[idx], w=w, t={a: T[a][idx] for a in T}, p={a: p[idx] for a, p in pr.items()})


# ---------------------------------------------------------------- forests and scores

def fit(X, t, w, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    Xr, y, wr = expand(X, t, w)
    m = RandomForestClassifier(**RF, n_jobs=-1, random_state=seed).fit(Xr, y, sample_weight=wr)
    del Xr, y, wr
    gc.collect()
    return m


def auc(t, w, s):
    from sklearn.metrics import roc_auc_score
    soft = (t > 0) & (t < 1)
    y = np.concatenate([(t > 0).astype(np.int8), np.zeros(int(soft.sum()), np.int8)])
    ww = np.concatenate([np.where(t > 0, w * t, w), w[soft] * (1.0 - t[soft])])
    return round(float(roc_auc_score(y, np.concatenate([s, s[soft]]), sample_weight=ww)), 4)


def importance(model, X, t, w, rng):
    """Increase of the weighted log-loss (target: share of experts) when a column or group is shuffled."""
    def loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(w * (t * np.log(p) + (1 - t) * np.log(1 - p))).sum() / w.sum())
    base = loss(model.predict_proba(X)[:, 1])
    groups = [(n, [n]) for n in NB_NAMES] + [("[pixel table]", G.PIX_FEAT_A), ("[first forest's map]", NB_NAMES)]
    out = {}
    for name, members in groups:
        Xs = X.copy()
        perm = rng.permutation(len(Xs))
        for mname in members:
            j = FEAT_CTX.index(mname)
            Xs[:, j] = Xs[perm, j]
        out[name] = round(loss(model.predict_proba(Xs)[:, 1]) - base, 5)
    return dict(base_logloss=round(base, 5), increase=dict(sorted(out.items(), key=lambda kv: -kv[1])))


def main(procs):
    global TR_DIR
    t_start = time.time()

    def log(*a):
        print(f"[{(time.time() - t_start) / 60:6.1f} min]", *a, flush=True)

    ann, TR_DIR, _ = run.find_data()
    coco = json.load(open(ann))
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    for sid in sorted(fname):
        SETS.setdefault(fname[sid], []).append(by_id[sid])
    train_files, val_files = run.photograph_split(fname, 0)
    if SMOKE:
        train_files, val_files = train_files[:6], val_files[:3]
    half_of = {f: i % 2 for i, f in enumerate(train_files)}
    res = dict(arms=list(ARMS) + ["stage1", "context"], train=len(train_files), val=len(val_files), rf=RF,
               n_aug=N_AUG, count_scale=COUNT_SCALE.tolist(), neighbour_columns=NB_NAMES, smoke=SMOKE)
    if MAPS.exists():
        shutil.rmtree(MAPS)

    # 1. rows of the training photographs
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    parts = list(imap(rows_job, [(f, Path(f).stem) for f in train_files]))
    if pool:
        pool.close(); pool.join()
    row_half = np.concatenate([[half_of[f]] * len(p["w"]) for f, p in zip(train_files, parts)])
    X = np.concatenate([p["X"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    T = {a: np.concatenate([p["t"][a] for p in parts]) for a in ("share", "count", "count1")}
    PX = np.concatenate([p["photo"][0] for p in parts])
    PT = np.concatenate([p["photo"][1] for p in parts])
    PW = np.concatenate([p["photo"][2] for p in parts])
    del parts
    gc.collect()
    res["rows"] = dict(table=int(len(w)), photo=int(len(PW)), MB=round((X.nbytes + PX.nbytes) / 2 ** 20))
    log("rows", res["rows"])

    # 2. the forests
    res["fit_minutes"] = {}
    for name, args in (("base", (X, T["share"], w)), ("onehot", None), ("photo", (PX, PT, PW)),
                       ("count", (X, T["count"], w)), ("count1", (X, T["count1"], w)),
                       ("h0", None), ("h1", None)):
        t0 = time.time()
        if name == "onehot":
            MODELS[name] = fit(onehot(X), T["share"], w)
        elif name in ("h0", "h1"):
            sel = row_half == int(name[1])
            MODELS[name] = fit(X[sel], T["share"][sel], w[sel], seed=1 + int(name[1]))
        else:
            MODELS[name] = fit(*args)
        res["fit_minutes"][name] = round((time.time() - t0) / 60, 1)
        log("fitted", name, res["fit_minutes"][name], "min")
    del X, w, T, PX, PT, PW, row_half
    gc.collect()

    # 3. the second forest: each training photograph read through the half that never saw it
    for m in MODELS.values():
        m.n_jobs = 1
    pool = mp.Pool(procs, initializer=G.pix_one_thread) if procs > 1 else None
    imap = pool.imap if pool else map
    parts = list(imap(ctx_rows_job, [(f, Path(f).stem, half_of[f]) for f in train_files]))
    if pool:
        pool.close(); pool.join()
    XC = np.concatenate([p[0] for p in parts])
    tC = np.concatenate([p[1] for p in parts])
    wC = np.concatenate([p[2] for p in parts])
    del parts
    gc.collect()
    log("context rows", XC.shape)
    t0 = time.time()
    MODELS["context"] = fit(XC, tC, wC)
    res["fit_minutes"]["context"] = round((time.time() - t0) / 60, 1)
    log("fitted context", res["fit_minutes"]["context"], "min")
    del XC, tC, wC
    gc.collect()

    # 4. the validation photographs: maps of every arm, and sampled rows
    for m in MODELS.values():
        m.n_jobs = 1
    pool = mp.Pool(procs, initializer=G.pix_one_thread) if procs > 1 else None
    imap = pool.imap if pool else map
    vparts = []
    for i, p in enumerate(imap(val_job, [(f, Path(f).stem) for f in val_files])):
        vparts.append(p)
        if i % 20 == 0:
            log(f"validation {i + 1}/{len(val_files)}")
    if pool:
        pool.close(); pool.join()
    vt = {a: np.concatenate([p["t"][a] for p in vparts]) for a in ("share", "count", "count1")}
    vw = np.concatenate([p["w"] for p in vparts])
    vp = {a: np.concatenate([p["p"][a] for p in vparts]) for a in res["arms"]}
    res["row_auc_vs_share"] = {a: auc(vt["share"], vw, vp[a]) for a in res["arms"]}
    log("row AUC on fold 0", res["row_auc_vs_share"])
    VX = np.concatenate([p["X"] for p in vparts])
    del vparts
    gc.collect()
    rng = np.random.default_rng(0)
    sub = rng.choice(len(vw), min(300000, len(vw)), replace=False)
    MODELS["context"].n_jobs = -1
    res["importance_context"] = importance(MODELS["context"], VX[sub], vt["share"][sub], vw[sub], rng)
    log("importance", list(res["importance_context"]["increase"].items())[:10])

    tar = WORK / ("dump_ideas_smoke.tar" if SMOKE else "dump_ideas.tar")
    with tarfile.open(tar, "w") as tf:
        for arm in res["arms"]:
            tf.add(str(MAPS / arm), arcname=arm)
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(WORK / ("pixel_ideas_smoke.json" if SMOKE else "pixel_ideas.json"), "w"), indent=1)
    log("done", tar)
    return res


def kaggle_main():
    main(os.cpu_count() or 4)


# ---------------------------------------------------------------- CLI
# Everything below is local only.

if __name__ == "__main__":
    main(1)
