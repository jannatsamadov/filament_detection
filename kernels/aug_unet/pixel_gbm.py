"""The user's idea (2026-11-12): every pixel is a row of a table and a classifier decides
filament or not.

Columns: where the pixel sits (x, y, distance from the centre, mu, heliographic latitude and
longitude, the rotation rate at that latitude), when and from where the photograph was taken (B0,
date as solar-cycle phase, hour, station), and how dark the pixel is -- against the whole disk,
against the quiet Sun at the same mu (limb darkening removed), and against its neighbourhood at
several radii -- plus the shape of the darkness around it (Hessian ridge strength and direction).

  pixA  those columns only: a model family with no neural network in it
  pixB  the same plus the U-Net pair's out-of-fold probability and YOLO's masks: a per-pixel
        combiner that can learn where each family is right, instead of a fixed-weight average

Out-of-fold over all 707 photographs: the classifier of fold k never saw fold k, and pixB reads only
maps of models that never saw that photograph. Then (1) PQ of each map alone; (2) the pair's map with
pixA averaged in, at three weights; (3) the keep-rule (21 and 25 columns) on the best of them, against
the pair; (4) carried to fold 0's ensemble, with and without the Qwen map; (5) test CSVs. CPU only, so
on Kaggle it spends no GPU quota.
    kernels/build.py builds kernels/pixel/run.py (and pixel_smoke/) from this file
"""
from __future__ import annotations

import json, multiprocessing as mp, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_pool as C  # noqa: E402,F401
import kfold as K  # noqa: E402
import post_local as P  # noqa: E402
import run  # noqa: E402

# ---------------------------------------------------------------- pixel settings
import gc, time  # noqa: E402,E401
from datetime import datetime  # noqa: E402

PIX_HALF = 1024                               # features and classifier at half resolution; maps go back to 2048
PIX_STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
PIX_SIGMAS = (1.0, 2.0, 4.0, 8.0)             # px at 1024: smoothing of the contrast
PIX_BOXES = (8, 16, 32, 64)                   # px at 1024: radius of the neighbourhood the pixel is compared with
PIX_HESS = (1.5, 3.0, 6.0)                    # px at 1024: scale of the ridge (dark lane) detector
PIX_NBINS = 50                                # limb-darkening profile: radial bins
PIX_N = dict(pos=8000, ring=3000, far=8000)   # training pixels per photograph, weighted back to the true counts
PIX_ITERS = 200
PIX_MIX_W = (0.25, 0.5, 1.0)                  # pixA's weight next to the pair's two U-Nets
PIX_SHOW = 3                                  # fold-0 photographs whose maps are copied out for the figure
PIX_SMOKE = False                             # kernels/build.py sets this in pixel_smoke/run.py
PIX_LEARNER = "gbm"                           # "rf" in pixel_rf/run.py: a random forest on the same table (user, 11-12)
PIX_FOLD0 = False                             # pixel_rf0/run.py: fold 0's classifiers only, its 141 maps tarred out (11-14)
# pixel_rfb/run.py (user, 11-15): the forest with the U-Net pair as columns (pixB) on its own, as a submission of
# its own: no pixA forest and no mixes; its 25-column rule fitted on its 707 out-of-fold maps, then its test CSV
PIX_B_ONLY = False
# pixel_rf1/run.py (user, 11-15: "1 U-Net + 1 RF, why ten U-Nets?"): at test time ONE U-Net (r50all, trained on all
# 707 photographs) and ONE forest (trained on all 707), no YOLO; the 21-number rule. In training the forest's rows
# read each photograph's own fold R50 (the model that did not see it), so it learns from honest U-Net maps
PIX_ONE_UNET = False
PIX_ONE_SETTING = (0.5, 12)                   # the forest's maps want a wider join (fold 0: 0.398 -> 0.441 with it)
PIX_RF = dict(n_estimators=100, max_samples=0.1, min_samples_leaf=50, max_features="sqrt")
PIX_FEAT_A = ["x_west", "y_north", "rho", "mu", "sin_lat", "lon_cm", "omega_rot", "b0", "year", "hour",
              "station", "I", "I_over_disk_median", "contrast_mu"]
PIX_FEAT_A += [f"contrast_g{s:g}" for s in PIX_SIGMAS] + [f"vs_box{r}" for r in PIX_BOXES]
PIX_FEAT_A += ["local_min5", "local_std8"]
PIX_FEAT_A += [f"hess{s:g}_{e}" for s in PIX_HESS for e in ("max", "min")] + ["ridge_cos2", "ridge_sin2"]
PIX_FEAT_A += ["disk_median", "disk_std", "dark_fraction"]
PIX_FEAT_B = PIX_FEAT_A + ["unet_p", "unet_g4", "unet_max7", "yolo_conf_cover", "yolo_n"]
PIX_COLS_A = list(range(len(PIX_FEAT_A)))
PIX_COLS_B = list(range(len(PIX_FEAT_B)))
PIX_STATION = PIX_FEAT_A.index("station")
PIX_GROUPS = {"where/when/station": PIX_FEAT_A[:11], "darkness": PIX_FEAT_A[11:24],
              "shape (ridge)": PIX_FEAT_A[24:32], "photograph": PIX_FEAT_A[32:]}


def pix_grid():
    """Disk coordinates of the half-resolution pixels, in units of the radius, as run.geo_channels."""
    i = ((2.0 * np.arange(PIX_HALF) + 0.5 - run.CX) / run.R_DISK).astype(np.float32)
    u, v = np.meshgrid(i, i)                  # u to the right (west), v downward
    rho = np.hypot(u, v)
    return u, v, rho, rho < 1.0


PIX_G = pix_grid()
PIX_YOLO, PIX_MODELS = {}, {}                 # set in pixel_main before the workers fork


def pix_features(img_u8, stem, pmap=None, yinst=None):
    """(n_disk, F) float32: one row per disk pixel of one photograph, columns PIX_FEAT_A (+ B's five)."""
    u, v, rho, on = PIX_G
    im = cv2.resize(img_u8, (PIX_HALF, PIX_HALF), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vals = im[on]
    med, sd = float(np.median(vals)), float(vals.std())
    # limb darkening of this photograph: the median at each distance from the centre
    bins = np.minimum((rho * PIX_NBINS).astype(np.int32), PIX_NBINS - 1)
    b_on = bins[on]
    order = np.argsort(b_on, kind="stable")
    edges = np.searchsorted(b_on[order], np.arange(PIX_NBINS + 1))
    prof = np.full(PIX_NBINS, med, np.float32)
    for b in range(PIX_NBINS):
        s = vals[order[edges[b]:edges[b + 1]]]
        if s.size > 50:
            prof[b] = np.median(s)
    c = np.where(on, im / np.maximum(prof[bins], 1e-3), 1.0).astype(np.float32)   # 1 = quiet Sun; 1 off the disk

    b0 = np.deg2rad(run.b0_deg(stem))
    z = np.sqrt(np.clip(1.0 - rho * rho, 0.0, None))
    dy = -v
    sinlat = np.clip(dy * np.cos(b0) + z * np.sin(b0), -1.0, 1.0)
    lon = np.arctan2(u, z * np.cos(b0) - dy * np.sin(b0))
    s2 = sinlat * sinlat
    omega = 14.366 - 2.297 * s2 - 1.624 * s2 * s2          # deg/day, Snodgrass: magnetic features
    t = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    year = t.year + (t.timetuple().tm_yday - 1) / 365.25
    hour = t.hour + t.minute / 60.0
    st = PIX_STATIONS.index(stem[14:16]) if stem[14:16] in PIX_STATIONS else len(PIX_STATIONS)

    cols = [u, dy, rho, z, sinlat, lon, omega, float(np.rad2deg(b0)), year, hour, float(st), im, im / max(med, 1e-3), c]
    g1 = None
    for s in PIX_SIGMAS:
        g = cv2.GaussianBlur(c, (0, 0), s)
        g1 = g if g1 is None else g1
        cols.append(g)
    for r in PIX_BOXES:
        cols.append(g1 / np.maximum(cv2.blur(c, (2 * r + 1, 2 * r + 1)), 1e-3))
    m8 = cv2.blur(c, (17, 17))
    cols += [cv2.erode(g1, np.ones((5, 5), np.uint8)), np.sqrt(np.maximum(cv2.blur(c * c, (17, 17)) - m8 * m8, 0.0))]
    orient = None
    for s in PIX_HESS:
        q = cv2.GaussianBlur(c, (0, 0), s)
        dxx = cv2.Sobel(q, cv2.CV_32F, 2, 0, ksize=3) * (s * s)
        dyy = cv2.Sobel(q, cv2.CV_32F, 0, 2, ksize=3) * (s * s)
        dxy = cv2.Sobel(q, cv2.CV_32F, 1, 1, ksize=3) * (s * s)
        tr, dd = 0.5 * (dxx + dyy), np.sqrt(0.25 * (dxx - dyy) ** 2 + dxy * dxy)
        cols += [tr + dd, tr - dd]            # a dark lane: large positive curvature across it
        if s == 3.0:
            ang = np.arctan2(2.0 * dxy, dxx - dyy)            # twice the direction of strongest curvature
            orient = [np.cos(ang), np.sin(ang)]
    cols += orient + [med, sd, float((c[on] < 0.9).mean())]
    if pmap is not None:
        p = cv2.resize(pmap, (PIX_HALF, PIX_HALF), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        yc = np.zeros((PIX_HALF, PIX_HALF), np.float32)
        yn = np.zeros((PIX_HALF, PIX_HALF), np.float32)
        for conf, rle in yinst or []:
            m = cv2.resize(run.mu.decode(rle).astype(np.float32), (PIX_HALF, PIX_HALF), interpolation=cv2.INTER_AREA)
            yc = np.maximum(yc, float(conf) * m)
            if conf >= K.Y_STRONG:
                yn += (m > 0.5)
        cols += [p, cv2.GaussianBlur(p, (0, 0), 4.0), cv2.dilate(p, np.ones((7, 7), np.uint8)), yc, yn]
    n = int(on.sum())
    X = np.empty((n, len(cols)), np.float32)
    for j, a in enumerate(cols):
        X[:, j] = a[on] if isinstance(a, np.ndarray) else a
    return X


def pix_rows(X, t2d, rng):
    """Training rows of one photograph. Pixels a filament covers (t > 0), a ring around them and the
    rest are sampled separately and weighted back to their true counts; a soft target t becomes a
    positive copy of weight t and a negative copy of weight 1 - t, which is cross-entropy on t."""
    on = PIX_G[3]
    pos2d = t2d > 0
    ring2d = cv2.dilate(pos2d.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~pos2d
    t, pos, ring = t2d[on], pos2d[on], ring2d[on]
    idx, w = [], []
    for name, sel in (("pos", pos), ("ring", ring), ("far", ~pos & ~ring)):
        cand = np.flatnonzero(sel)
        if cand.size == 0:
            continue
        take = rng.choice(cand, min(PIX_N[name], cand.size), replace=False)
        idx.append(take)
        w.append(np.full(take.size, cand.size / take.size, np.float32))
    idx, w = np.concatenate(idx), np.concatenate(w)
    tt = t[idx]
    soft = (tt > 0) & (tt < 1)
    Xr = np.concatenate([X[idx], X[idx[soft]]])
    y = np.concatenate([(tt > 0).astype(np.int8), np.zeros(int(soft.sum()), np.int8)])
    wr = np.concatenate([np.where(tt > 0, w * tt, w), w[soft] * (1.0 - tt[soft])]).astype(np.float32)
    return Xr, y, wr


def unet_variant(fold):
    """The U-Net map the forest reads for a photograph of this fold: the pair, or (one-U-Net mode) the fold's R50."""
    return f"{K.FAMILIES['r50'][fold]}/{K.VIEW}" if PIX_ONE_UNET else f"{K.PAIR[fold]}/{K.VIEW}"


def pix_val_inputs(d, cache, f, stem, fold, split="val"):
    img = np.asarray(np.load(Path(cache) / f"{split}_img.npy", mmap_mode="r")[f])
    pmap = P.load_map(Path(d), unet_variant(fold), split, stem)
    return img, pmap, [] if PIX_ONE_UNET else PIX_YOLO.get(fold, {}).get((split, stem), [])


def pix_row_job(job):
    d, cache, f, stem, fold = job
    img, pmap, yinst = pix_val_inputs(d, cache, f, stem, fold)
    X = pix_features(img, stem, pmap, yinst)
    msk = np.asarray(np.load(Path(cache) / "val_msk.npy", mmap_mode="r")[f])
    t2d = cv2.resize(msk, (PIX_HALF, PIX_HALF), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return pix_rows(X, t2d, np.random.default_rng(int(stem[:14]) % (2 ** 32)))


def pix_fit(X, y, w, cols, seed=0):
    if PIX_LEARNER == "rf":
        from sklearn.ensemble import RandomForestClassifier
        n = max(2, PIX_ITERS // 20) if PIX_SMOKE else PIX_RF["n_estimators"]
        return RandomForestClassifier(**{**PIX_RF, "n_estimators": n}, n_jobs=-1, random_state=seed).fit(
            X[:, cols], y, sample_weight=w)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=PIX_ITERS, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=500,
                                          l2_regularization=1.0, categorical_features=[cols.index(PIX_STATION)],
                                          early_stopping=False, random_state=seed).fit(X[:, cols], y, sample_weight=w)


def pix_write(d, name, split, stem, prob_on):
    a = np.zeros((PIX_HALF, PIX_HALF), np.float32)
    a[PIX_G[3]] = prob_on
    full = cv2.resize(a, (run.FULL, run.FULL), interpolation=cv2.INTER_LINEAR)
    out = Path(d) / "dump" / name / f"{split}_{K.VIEW}"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{stem}.png"), np.round(np.clip(full, 0.0, 1.0) * 255.0).astype(np.uint8))


def pix_one_thread():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def pix_pred_job(job):
    """Writes pixA and pixB for one photograph: a validation one with its own fold's classifiers,
    a test one averaged over the five folds' (each pixB_k reads fold k's U-Net pair and YOLO)."""
    kind, d, stem, fold, cache, f, test_dir = job
    if kind == "val":
        img, pmap, yinst = pix_val_inputs(d, cache, f, stem, fold)
        X = pix_features(img, stem, pmap, yinst)
        pa = None if PIX_B_ONLY else PIX_MODELS["A"][fold].predict_proba(X[:, PIX_COLS_A])[:, 1]
        pb = PIX_MODELS["B"][fold].predict_proba(X)[:, 1]
    else:
        img = cv2.imread(str(Path(test_dir) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        pa, pb = 0.0, 0.0
        if PIX_ONE_UNET:                           # one U-Net, one forest
            t0 = time.time()
            X = pix_features(img, stem, P.load_map(Path(d), f"r50all/{K.VIEW}", "test", stem), [])
            pb = PIX_MODELS["B"]["all"].predict_proba(X)[:, 1]
            pix_write(d, "pixB", kind, stem, pb)
            return stem, time.time() - t0
        for k in range(5):
            X = pix_features(img, stem, P.load_map(Path(d), f"{K.PAIR[k]}/{K.VIEW}", "test", stem),
                             PIX_YOLO.get(k, {}).get(("test", stem), []))
            if not PIX_B_ONLY:
                pa = pa + PIX_MODELS["A"][k].predict_proba(X[:, PIX_COLS_A])[:, 1] / 5.0
            pb = pb + PIX_MODELS["B"][k].predict_proba(X)[:, 1] / 5.0
    if not PIX_B_ONLY:
        pix_write(d, "pixA", kind, stem, pa)
    pix_write(d, "pixB", kind, stem, pb)
    return stem


def pix_mix(d, name, members, extras, split, stems, view=None):
    """d/dump/<name>/<split>_<view>/: the members' mean with each (variant, weight) in extras added."""
    view = view or K.VIEW
    out = Path(d) / "dump" / name / f"{split}_{view}"
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        acc = sum(P.load_map(Path(d), f"{m}/{view}", split, s).astype(np.float32) for m in members)
        wsum = float(len(members))
        for v, w in extras:
            acc = acc + w * P.load_map(Path(d), v, split, s).astype(np.float32)
            wsum += w
        cv2.imwrite(str(out / f"{s}.png"), np.round(acc / wsum).astype(np.uint8))
    return f"{name}/{view}"


def pix_auc(y, w, s):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, s, sample_weight=w)), 4)


def pix_logloss(y, w, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def pix_importance(model, X, y, w, cols, names, rng):
    """Increase of the weighted log-loss when one column (or one group) is shuffled."""
    base = pix_logloss(y, w, model.predict_proba(X[:, cols])[:, 1])
    out = {}
    for name, members in [(n, [n]) for n in names] + [(f"[{g}]", m) for g, m in PIX_GROUPS.items()]:
        Xs = X[:, cols].copy()
        perm = rng.permutation(len(Xs))
        for m in members:
            j = names.index(m)
            Xs[:, j] = Xs[perm, j]
        out[name] = round(pix_logloss(y, w, model.predict_proba(Xs)[:, 1]) - base, 5)
    return dict(base_logloss=round(base, 5), increase=dict(sorted(out.items(), key=lambda kv: -kv[1])))


def pix_rule_frames(d, truth, var_of_fold, tag, imap, yolo, folds):
    cache = Path(d) / f"pix_pool_{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), var_of_fold(n), stem, t["cache"], f, sets, str(cache), True)
            for n, t in enumerate(truth) for f, stem, sets in t["frames"]]
    frames = K.one_setting(list(imap(K.pool_job, jobs)))
    frames = K.add_neighbours(frames, Path(d), [var_of_fold(g) for g in folds], "val", folds, imap)
    return K.add_yolo(frames, [yolo[g] for g in folds], "val")


def pix_ens0_frames(d, t0, variant, tag, imap, yolo):
    cache = Path(d) / f"pix_pool_ens0_{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    ens = K.one_setting(list(imap(K.pool_job, [(str(d), variant, stem, t0["cache"], f, sets, str(cache), True)
                                               for f, stem, sets in t0["frames"]])))
    ens = K.add_neighbours(ens, Path(d), [variant] * len(ens), "val", np.zeros(len(ens), int), imap)
    return K.add_yolo(ens, [yolo[0]] * len(ens), "val")


def pix_b_only(d, out, truth, folds, yolo, imap, test_files, res, log, t_start):
    """The forest with the U-Net pair as columns, alone: its fixed-chain grid and its 25-column rule, both out of
    fold on all 707 photographs, then the rule fitted on all of them and the test CSV (maps averaged over the five
    folds' forests, each reading its own fold's U-Net pair and YOLO)."""
    vf = lambda n: "pixB/d4"
    g = np.stack(list(imap(K.grid_job, [(str(d), vf(n), stem, sets) for n, t in enumerate(truth)
                                        for _, stem, sets in t["frames"]])))
    pq = P.pq_of(g.sum(axis=0))
    i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
    res["oof_grid_pixB"] = dict(point=[P.THRESH[i], P.MERGE[j], P.MIN_AREA[a]], **K.summary(g[:, i, j, a].sum(axis=0)))
    log("pixB fixed chain", res["oof_grid_pixB"])
    # one-U-Net mode has no YOLO at test time, so its rule reads the 21 numbers, not 25
    cols, ytest = (K.GEO_COLS, None) if PIX_ONE_UNET else (K.GEO_YOLO_COLS, yolo)
    fr = pix_rule_frames(d, truth, vf, "pixB", imap, yolo, folds)
    oof = K.rule_oof(fr, folds, cols)
    res["rule_oof_pixB"] = dict(columns=len(cols), one_setting=list(C.ONE_SETTING), all_707=K.summary(oof.sum(axis=0)),
                                fold0_141=K.summary(oof[folds == 0].sum(axis=0)))
    log("pixB rule, out of fold", res["rule_oof_pixB"])
    final = K.fit(fr, np.arange(len(fr)), cols)
    cut_i = int(np.argmax(K.pq_of(K.counts_at_cuts(fr, K.score(final, fr, cols)).sum(axis=0))))
    cut = float(C.POOL_CUTS[cut_i])
    name = "submission_rf1_unet1_rule21.csv" if PIX_ONE_UNET else "submission_rf_pixB_rule_geo_yolo.csv"
    res["file"] = K.write_csv(d, "pixB/d4", test_files, out / name, final, cut, cols, yolo=ytest)
    res["file"].update(cut=cut, verify=K.verify(out / name))
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(out / ("pixel_rf1.json" if PIX_ONE_UNET else "pixel_rfb.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def pixel_main(src, d, out, procs, smoke=False):
    global PIX_ITERS
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(*a):
        print(f"[{(time.time() - t_start) / 60:6.1f} min]", *a, flush=True)

    if smoke:
        PIX_ITERS = 10
        for k in PIX_N:
            PIX_N[k] //= 10
    # a classifier's probabilities may want a lower threshold than a U-Net's
    P.THRESH = (0.15, 0.2, 0.25) + tuple(t for t in P.THRESH if t > 0.25)
    if PIX_ONE_UNET:            # before any worker forks: the candidate pool marks its outlines with this setting
        C.ONE_SETTING = PIX_ONE_SETTING
    log("models:", P.extract(src, d))
    PIX_YOLO.update(K.load_yolo(src))
    if smoke:
        for f in range(5):
            PIX_YOLO.setdefault(f, {})
    assert sorted(PIX_YOLO) == [0, 1, 2, 3, 4], "every fold needs its own YOLO"
    yolo = PIX_YOLO
    truth = K.fold_truth(smoke)
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    stems_of = [[s for _, s, _ in t["frames"]] for t in truth]
    test_files = truth[0]["meta"]["test_files"]
    test_stems = [Path(n).stem for n in test_files]
    test_dir = str(run.find_data()[2])
    res = dict(learner=PIX_LEARNER, photographs=len(folds), features_A=PIX_FEAT_A, features_B_extra=PIX_FEAT_B[len(PIX_FEAT_A):])

    # 1. the table: sampled rows of every photograph, its own fold's U-Net pair and YOLO for pixB
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    jobs = [(str(d), t["cache"], f, stem, n) for n, t in enumerate(truth) for f, stem, _ in t["frames"]]
    parts = list(imap(pix_row_job, jobs))
    row_fold = np.concatenate([[folds[i]] * len(p[1]) for i, p in enumerate(parts)])
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([p[2] for p in parts])
    del parts
    gc.collect()
    res["rows"] = dict(n=int(len(y)), MB=round(X.nbytes / 2 ** 20), positive_weight_share=round(float(w[y == 1].sum() / w.sum()), 5))
    log("rows", res["rows"])

    # 2. five classifiers per arm, each without its fold
    PIX_MODELS["A"], PIX_MODELS["B"] = {}, {}
    auc = {"pixA": [], "pixB": [], "unet_p_alone": []}
    for k in ((0,) if PIX_FOLD0 else range(5)):
        tr, te = row_fold != k, row_fold == k
        if not PIX_B_ONLY:
            PIX_MODELS["A"][k] = pix_fit(X[tr], y[tr], w[tr], PIX_COLS_A, k)
            auc["pixA"].append(pix_auc(y[te], w[te], PIX_MODELS["A"][k].predict_proba(X[te][:, PIX_COLS_A])[:, 1]))
        PIX_MODELS["B"][k] = pix_fit(X[tr], y[tr], w[tr], PIX_COLS_B, k)
        auc["pixB"].append(pix_auc(y[te], w[te], PIX_MODELS["B"][k].predict_proba(X[te])[:, 1]))
        auc["unet_p_alone"].append(pix_auc(y[te], w[te], X[te][:, PIX_FEAT_B.index("unet_p")]))
        log(f"fold {k}: pixel AUC", {a: v[-1] for a, v in auc.items() if v})
    res["pixel_auc_oof"] = auc
    if PIX_ONE_UNET:                               # the one forest the test photographs go through
        t_fit = time.time()
        PIX_MODELS["B"]["all"] = pix_fit(X, y, w, PIX_COLS_B, 99)
        res["fit_minutes_all"] = round((time.time() - t_fit) / 60, 1)
        log("forest on all 707 photographs", res["fit_minutes_all"], "min")
    rng = np.random.default_rng(0)
    te = np.flatnonzero(row_fold == 0)
    sub = rng.choice(te, min(300000, te.size), replace=False)
    if not PIX_B_ONLY:
        res["importance_pixA_fold0"] = pix_importance(PIX_MODELS["A"][0], X[sub], y[sub], w[sub], PIX_COLS_A, PIX_FEAT_A, rng)
        log("importance", list(res["importance_pixA_fold0"]["increase"].items())[:12])
    del X, y, w
    gc.collect()

    # 3. out-of-fold maps of all 707 photographs and the test maps (workers fork with the classifiers)
    for arm in PIX_MODELS.values():                # a forest predicts with one thread in each worker
        for m in arm.values():
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1
    if pool:
        pool.close()
        pool.join()
    pool = mp.Pool(procs, initializer=pix_one_thread) if procs > 1 else None
    imap = pool.imap_unordered if pool else map
    jobs = [("val", str(d), stem, n, t["cache"], f, test_dir) for n, t in enumerate(truth) for f, stem, _ in t["frames"]
            if n == 0 or not PIX_FOLD0]
    jobs += [] if PIX_FOLD0 else [("test", str(d), s, -1, "", -1, test_dir) for s in test_stems]
    test_sec = []
    for i, r in enumerate(imap(pix_pred_job, jobs)):
        if isinstance(r, tuple):                    # one-U-Net mode: seconds of table + forest for a test photograph
            test_sec.append(r[1])
        if i % 100 == 0:
            log(f"maps {i}/{len(jobs)}")
    if test_sec:
        res["test_seconds_per_photo_table_and_forest_1_core"] = round(float(np.median(test_sec)), 2)
    pool.close() if pool else None
    PIX_MODELS.clear()
    gc.collect()
    if PIX_FOLD0:                                  # fold 0's maps and the pair's, for aug_unet/pixel_cls.py
        import tarfile
        with tarfile.open(out / "dump_rf0.tar", "w") as tf:
            for s in stems_of[0]:
                for name, v in (("pair", f"{K.PAIR[0]}/{K.VIEW}"), ("pixA", "pixA/d4"), ("pixB", "pixB/d4")):
                    p = d / f"show_{name}.png"
                    cv2.imwrite(str(p), P.load_map(d, v, "val", s))
                    tf.add(str(p), arcname=f"{name}/{s}.png")
        res["minutes"] = round((time.time() - t_start) / 60, 1)
        json.dump(res, open(out / "pixel.json", "w"), indent=1, default=float)
        log("fold-0 maps tarred", res)
        return res
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    log("maps written")
    if PIX_B_ONLY:
        return pix_b_only(d, out, truth, folds, yolo, imap, test_files, res, log, t_start)

    # 4. PQ of each map, and of the pair with pixA averaged in, all out-of-fold at one grid point
    for wt in PIX_MIX_W:
        for k in range(5):
            pix_mix(d, f"pmix{int(wt * 100)}", [K.FAMILIES["r50"][k], K.FAMILIES["r34"][k]], [("pixA/d4", wt)],
                    "val", stems_of[k])
    variants = {"pair": lambda n: f"{K.PAIR[n]}/{K.VIEW}", "pixA": lambda n: "pixA/d4", "pixB": lambda n: "pixB/d4"}
    variants.update({f"pmix{int(wt * 100)}": (lambda n, wt=wt: f"pmix{int(wt * 100)}/d4") for wt in PIX_MIX_W})
    at = {}
    res["oof_grid"] = {}
    for name, vf in variants.items():
        g = np.stack(list(imap(K.grid_job, [(str(d), vf(n), stem, sets) for n, t in enumerate(truth)
                                            for _, stem, sets in t["frames"]])))
        pq = P.pq_of(g.sum(axis=0))
        i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
        at[name] = g[:, i, j, a]
        res["oof_grid"][name] = dict(point=[P.THRESH[i], P.MERGE[j], P.MIN_AREA[a]], **K.summary(at[name].sum(axis=0)),
                                     minus_pair=K.boot(at[name], at["pair"]) if name != "pair" else None)
        log(name, res["oof_grid"][name])
    mixes = [f"pmix{int(wt * 100)}" for wt in PIX_MIX_W]
    best = max(mixes, key=lambda m: res["oof_grid"][m]["PQ"])      # chosen on the same 707: a mild optimism
    w_best = PIX_MIX_W[mixes.index(best)]
    res["mix_chosen"] = dict(name=best, weight=w_best, pixA_share=round(w_best / (2 + w_best), 4))

    # 5. the keep-rule on candidates of the pair, of the chosen mix and of pixB
    GY, G21 = K.GEO_YOLO_COLS, K.GEO_COLS
    fr = {"pair": pix_rule_frames(d, truth, variants["pair"], "pair", imap, yolo, folds),
          best: pix_rule_frames(d, truth, variants[best], best, imap, yolo, folds),
          "pixB": pix_rule_frames(d, truth, variants["pixB"], "pixB", imap, yolo, folds)}
    oof = {}
    for name, f in fr.items():
        oof[(name, 21)], oof[(name, 25)] = K.rule_oof(f, folds, G21), K.rule_oof(f, folds, GY)
    res["rule_oof"] = {f"{n}_{c}": K.summary(v.sum(axis=0)) for (n, c), v in oof.items()}
    res["rule_oof_diff"] = {f"{n}_{c}_minus_pair_{c}": K.boot(oof[(n, c)], oof[("pair", c)])
                            for (n, c) in oof if n != "pair"}
    log("rule", res["rule_oof"], res["rule_oof_diff"])

    # 6. carried to fold 0's ensemble (five U-Nets), with the Qwen map and with pixA at the same share
    t0 = truth[0]
    ens_members = K.ENS0.split("+")
    w5 = 2.5 * w_best                          # five members: the same share as w next to two
    s0 = stems_of[0]
    have_q = (d / "dump" / K.QSEG.split("/")[0]).exists()
    e = {"ens0": pix_ens0_frames(d, t0, f"ens:{K.ENS0}/{K.VIEW}", "ens0", imap, yolo),
         "ens0p": pix_ens0_frames(d, t0, pix_mix(d, "pens0", ens_members, [("pixA/d4", w5)], "val", s0), "ens0p", imap, yolo)}
    if have_q:
        e["ens0q"] = pix_ens0_frames(d, t0, pix_mix(d, "qens0", ens_members, [(K.QSEG, K.QSEG_W)], "val", s0),
                                     "ens0q", imap, yolo)
        e["ens0qp"] = pix_ens0_frames(d, t0, pix_mix(d, "qpens0", ens_members, [(K.QSEG, K.QSEG_W), ("pixA/d4", w5)],
                                                     "val", s0), "ens0qp", imap, yolo)
    tr_res, nested = {}, {}
    tr_res["ens0_21"], nested["ens0_21"] = K.transfer(fr["pair"], folds, e["ens0"], G21)
    for name in e:
        rule_frames = fr[best] if name.endswith("p") else fr["pair"]
        tr_res[f"{name}_25"], nested[f"{name}_25"] = K.transfer(rule_frames, folds, e[name], GY)
    res["transfer"] = tr_res
    res["transfer_diff"] = {f"{a}_minus_{b}": K.boot(nested[a], nested[b]) for a, b in
                            [("ens0p_25", "ens0_25"), ("ens0p_25", "ens0_21")]
                            + ([("ens0qp_25", "ens0q_25"), ("ens0qp_25", "ens0_21"), ("ens0q_25", "ens0_21")] if have_q else [])}
    log("transfer", {k: v["on_ens0_nested_cut"] for k, v in tr_res.items()}, res["transfer_diff"])

    # 7. test CSVs: the ten U-Nets with pixA at the same share, with and without the Qwen map
    ten = K.FAMILIES["r50"] + K.FAMILIES["r34"]
    w10 = 5.0 * w_best
    final = K.fit(fr[best], np.arange(len(fr[best])), GY)
    files = {"kf10p_rule_geo_yolo": (pix_mix(d, "pten", ten, [("pixA/d4", w10)], "test", test_stems), "ens0p_25")}
    if have_q:
        files["kf10qp_rule_geo_yolo"] = (pix_mix(d, "pqten", ten, [(K.QSEG, 2 * K.QSEG_W), ("pixA/d4", w10)], "test",
                                                 test_stems), "ens0qp_25")
    res["files"] = {}
    for fname, (variant, tr_key) in files.items():
        cut = tr_res[tr_key]["on_ens0_best_cut"]["cut"]
        res["files"][fname] = K.write_csv(d, variant, test_files, out / f"submission_{fname}.csv", final, cut, GY, yolo=yolo)
        res["files"][fname]["cut"] = cut
        res["files"][fname]["verify"] = K.verify(out / f"submission_{fname}.csv")

    # the maps of a few fold-0 photographs, for the figure
    show = out / "maps"
    show.mkdir(exist_ok=True)
    for s in s0[:PIX_SHOW]:
        for name, v in (("pair", f"{K.PAIR[0]}/{K.VIEW}"), ("pixA", "pixA/d4"), ("pixB", "pixB/d4"), (best, f"{best}/d4")):
            cv2.imwrite(str(show / f"{s}_{name}.png"), P.load_map(d, v, "val", s))
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(out / "pixel.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def kaggle_pixel():
    """filament-pixel: the dump kernels' tars, both YOLO kernels' and the Qwen segmenter's outputs under /kaggle/input."""
    pixel_main(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, run.os.cpu_count() or 4, smoke=PIX_SMOKE)


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    """Local check of the feature table on two photographs (no maps needed)."""
    ann, tr_dir, _ = run.find_data()
    files = sorted(p.name for p in tr_dir.glob("*.jpeg"))[:2]
    for name in files:
        img = cv2.imread(str(tr_dir / name), cv2.IMREAD_GRAYSCALE)
        t = time.time()
        X = pix_features(img, Path(name).stem)
        Xb = pix_features(img, Path(name).stem, np.zeros((run.FULL, run.FULL), np.uint8), [])
        assert X.shape[1] == len(PIX_FEAT_A) and Xb.shape[1] == len(PIX_FEAT_B), (X.shape, Xb.shape)
        assert np.isfinite(X).all()
        print(name, X.shape, f"{time.time() - t:.2f}s",
              {n: round(float(np.median(X[:, j])), 3) for j, n in enumerate(PIX_FEAT_A)})
        t2d = np.zeros((PIX_HALF, PIX_HALF), np.float32)
        t2d[500:510, 400:520] = 1.0
        t2d[505:515, 400:520] = np.maximum(t2d[505:515, 400:520], 0.5)
        Xr, y, w = pix_rows(X, t2d, np.random.default_rng(0))
        print("rows", Xr.shape, "pos weight", round(float(w[y == 1].sum()), 1), "total weight", round(float(w.sum())),
              "disk px", int(PIX_G[3].sum()))


if __name__ == "__main__":
    main()
