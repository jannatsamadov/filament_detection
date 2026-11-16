"""Every photograph out-of-fold: what five R50 folds can and cannot tell us.

  1. out-of-fold PQ over all 707 photographs, D4, one setting for all: R50 and R34
     alone, and each fold's pair averaged (version 2 also compared plain with D4)
  2. the keep-rule on the pair's candidates: no rule against ten and seventeen
     features, and training on a quarter of the data against all of it. The cut is
     nested (chosen on the training folds), so these carry no selection bias
  3. does a rule trained on single-model maps carry over to an ensemble's maps?
     Fold 0's five-model ensemble was trained without fold 0, so this is honest
  4. test CSVs, each put through src/verify_submission.py: all ten fold models
     averaged, with and without the rule

An averaged ten-model map cannot be validated, since each photograph has only one
model that never saw it: its settings come from step 3 and its score from the
leaderboard. CPU only; on Kaggle it spends no GPU quota.
    python kernels/aug_unet/kfold.py --dir kernels/aug_post/out --out kernels/kfold/out --smoke --procs 1
"""
from __future__ import annotations

import argparse, collections, json, multiprocessing as mp, pickle, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_pool as C  # noqa: E402
import post_local as P  # noqa: E402
import run  # noqa: E402
from post_local import extract, frame_job, pq_of  # noqa: E402

# ---------------------------------------------------------------- k-fold settings
FAMILIES = {"r50": ("r50", "r50f1", "r50f2", "r50f3", "r50f4"),
            "r34": ("full", "r34f1", "r34f2", "r34f3", "r34f4")}
# each fold's two models average to one out-of-fold map: the ensemble is tuned on all 707
PAIR = tuple(f"ens:{a}+{b}" for a, b in zip(FAMILIES["r50"], FAMILIES["r34"]))
VIEW = "d4"                              # +0.0054 [+0.002, +0.009] over plain, 707 photographs
ENS0 = "flip+full+r50+solar+union"       # all trained without fold 0
ENS_POINT = {"plain": (0.65, 3, 350), "d4": (0.65, 7, 350)}   # that ensemble's tuned points on fold 0
OLD_COLS, ALL_COLS = list(range(10)), list(range(17))
GEO_COLS = list(range(21))               # + latitude, longitude from the CM, long-axis angle (v4)
GEO_EXTRA_CUTS = (0.20, 0.24, 0.28, 0.32, 0.36)   # v6: stricter than the ens0 cut (0.16)


def fold_truth(smoke):
    """Each fold's validation photographs, with every annotation set."""
    truth = []
    for fold in range(5):
        run.set_fold(fold)
        if smoke or not (run.CACHE / "meta.json").exists():
            run.prepare(smoke, with_train=False)
        meta = json.load(open(run.CACHE / "meta.json"))
        gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
        by = collections.defaultdict(list)
        for i, g in enumerate(gt):
            by[g["frame"]].append((i, g["crops"]))
        stems = [Path(f).stem for f in meta["val_files"]]
        truth.append(dict(cache=str(run.CACHE), meta=meta,
                          frames=[(f, stems[f], by[f]) for f in sorted(by)]))
    return truth


def grid_job(job):
    return frame_job(job)[0]


def pool_job(job):
    d, variant, stem, cache_dir, f, sets, out, *rle = job
    img = np.asarray(np.load(Path(cache_dir) / "val_img.npy", mmap_mode="r")[f])
    return C.frame(Path(d), stem, img, [c for _, c in sets], Path(out), variant, with_conflicts=False,
                   with_rle=bool(rle and rle[0]))


def one_setting(frames):
    """The candidates of the one setting, which are disjoint within a photograph."""
    out = []
    for f in frames:
        k = np.flatnonzero(f["one"])
        out.append(dict(X=f["X"][k], Y=f["Y"][k], n_gt=f["n_gt"], stem=f["stem"], pts=[f["pts"][i] for i in k],
                        rle=[f["rle"][i] for i in k] if f.get("rle") is not None else None))
    return out


# ---- the same filament on another day: the user's rotation idea, within the rules.
# A candidate's points are carried along the sphere by the differential rotation to
# a neighbouring frame's time, and that frame's own map is read there. Neighbours
# are only ever frames the model saw no labels of: on validation the same fold
# (one model never trained on either, and ~141 frames, as dense in time as test's
# 180); on test, the other test frames -- inference may use only test/ images.
NB_DAYS = 3.5
NB_TOL = 15                                   # max-filter, px: what the filament may have moved
NB_COLS = GEO_COLS + [21, 22, 23]            # support, |gap| in days, neighbours seen


def stamp(stem):
    from datetime import datetime
    t = datetime.strptime(Path(stem).stem[:14], "%Y%m%d%H%M%S")
    return (t - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0


def carry(pts, b0a, b0b, dt):
    """(y, x) points of frame a to frame b, dt days later; and whether b sees them."""
    y, x = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    dx, dy = (x - run.CX) / run.R_DISK, -(y - run.CY) / run.R_DISK
    z = np.sqrt(np.clip(1.0 - dx * dx - dy * dy, 0.0, None))
    a, b = np.radians(b0a), np.radians(b0b)
    lat = np.arcsin(np.clip(dy * np.cos(a) + z * np.sin(a), -1.0, 1.0))
    lon = np.arctan2(dx, z * np.cos(a) - dy * np.sin(a))
    s2 = np.sin(lat) ** 2
    lon = lon + np.radians((14.713 - 2.396 * s2 - 1.787 * s2 * s2 - 0.9856) * dt)   # Snodgrass, synodic
    xh, yh, zh = np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)
    yr, zr = yh * np.cos(b) - zh * np.sin(b), yh * np.sin(b) + zh * np.cos(b)
    return run.CY - yr * run.R_DISK, run.CX + xh * run.R_DISK, zr > 0.15


def neighbours(stems):
    """For each frame, the others within NB_DAYS: (index, signed gap in days)."""
    t = np.array([stamp(s) for s in stems])
    return [[(j, float(t[j] - t[i])) for j in range(len(stems)) if j != i and abs(t[j] - t[i]) <= NB_DAYS]
            for i in range(len(stems))]


def nb_job(job):
    """One frame's candidates: the best support any neighbour gives, its gap, how many saw it."""
    d, variant, split, stem, pts, nbs = job
    out = np.full((len(pts), 3), np.nan, np.float32)
    if not pts or not nbs:
        return out
    b0a = run.b0_deg(stem)
    best, gap, seen = np.full(len(pts), -1.0), np.full(len(pts), np.nan), np.zeros(len(pts))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NB_TOL + 1, 2 * NB_TOL + 1))
    for nstem, dt in nbs:
        q = cv2.dilate(P.load_map(Path(d), variant, split, nstem), ker).astype(np.float32) / 255.0
        b0b = run.b0_deg(nstem)
        for n, pp in enumerate(pts):
            yy, xx, vis = carry(pp, b0a, b0b, dt)
            if vis.mean() < 0.5:
                continue
            yy = np.clip(np.round(yy[vis]), 0, run.FULL - 1).astype(int)
            xx = np.clip(np.round(xx[vis]), 0, run.FULL - 1).astype(int)
            s = float(q[yy, xx].mean())
            seen[n] += 1
            if s > best[n]:
                best[n], gap[n] = s, abs(dt)
    ok = seen > 0
    out[ok, 0], out[ok, 1] = best[ok], gap[ok]
    out[:, 2] = seen
    return out


def add_neighbours(frames, d, variants, split, groups, imap):
    """Append the three neighbour columns; variants[n] is the map frame n's neighbours are read from."""
    jobs = []
    for g in sorted(set(groups)):
        idx = [n for n in range(len(frames)) if groups[n] == g]
        nb = neighbours([frames[n]["stem"] for n in idx])
        for k, n in enumerate(idx):
            jobs.append((n, (str(d), variants[n], split, frames[n]["stem"], frames[n]["pts"],
                             [(frames[idx[j]]["stem"], dt) for j, dt in nb[k]])))
    feats = dict(zip([n for n, _ in jobs], imap(nb_job, [j for _, j in jobs])))
    for n, f in enumerate(frames):
        f["X"] = np.concatenate([f["X"], feats[n]], 1)
    return frames


# ---- a second model family (v8). Every model so far is a U-Net, so their mistakes are
# alike; YOLO11l-seg, trained per fold on the same splits (aug_unet/yolo_seg.py, kernels
# filament-yolo: folds 0-1, filament-yolo-b: 2-4), says whether it draws each candidate
# too. On fold 0 alone, as a vote on top of the rule, it gave +0.0059 [+0.0015, +0.0104].
Y_NEAR, Y_STRONG = 0.3, 0.25
YOLO_COLS = [24, 25, 26, 27]                  # best IoU, its confidence, best confidence at IoU >= 0.3, cover
GEO_YOLO_COLS = GEO_COLS + YOLO_COLS


def load_yolo(src):
    """{fold: {(split, stem): [(confidence, rle), ...]}} from every yolo_preds_f*.pkl under src."""
    out = {}
    for p in sorted(Path(src).rglob("yolo_preds_f*.pkl")):
        out[int(p.stem[-1])] = pickle.load(open(p, "rb"))
    return out


def yolo_cols(cr, inst):
    """(n, 4) for n candidate RLEs against one photograph's YOLO instances."""
    f = np.zeros((len(cr), 4), np.float32)
    if not cr or not inst:
        return f
    conf = np.array([c for c, _ in inst])
    yr = [e for _, e in inst]
    iou = np.asarray(run.mu.iou(cr, yr, [0] * len(yr))).reshape(len(cr), len(yr))
    f[:, 0] = iou.max(1)
    f[:, 1] = np.where(f[:, 0] > 0, conf[iou.argmax(1)], 0.0)
    f[:, 2] = np.where(iou >= Y_NEAR, conf[None, :], 0.0).max(1)
    strong = [e for c, e in inst if c >= Y_STRONG]
    if strong:
        u = run.mu.merge(strong)
        for i, e in enumerate(cr):
            f[i, 3] = run.mu.area(run.mu.merge([e, u], intersect=1)) / max(run.mu.area(e), 1)
    return f


def add_yolo(frames, preds, split):
    """Append the four YOLO columns; preds[n] is the YOLO model that never saw frame n."""
    for f, pr in zip(frames, preds):
        f["X"] = np.concatenate([f["X"], yolo_cols(f["rle"], pr.get((split, f["stem"]), []))], 1)
    return frames


# ---- a third family (v8): the Qwen3-VL-2B vision tower with a mask head, pretrained on
# 2000 unlabelled GONG frames (fold-0 U-Net maps as targets) and fine-tuned on fold 0's
# training photographs (aug_unet/qwen_seg.py, kernel filament-qwen-seg). Fold 0 only, so
# it enters as one more member of the averaged map, not as a keep-rule column.
QSEG, QSEG_FT = "qwenseg_pre/plain", "qwenseg_ft/plain"
QSEG_W = 1.0                                  # one member of six in ens0; its share is kept in the ten


def mix_dump(d, name, members, extra, w, split, stems, view=VIEW):
    """d/dump/<name>/<split>_<view>/: the members' mean with `extra` (a '<model>/<view>', or a
    tuple of them) at weight w each. One extra gives exactly the v8 maps (the 0.39 file's)."""
    extras = (extra,) if isinstance(extra, str) else tuple(extra)
    out = d / "dump" / name / f"{split}_{view}"
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        acc = sum(P.load_map(d, f"{m}/{view}", split, s).astype(np.float32) for m in members)
        for e in extras:
            acc = acc + w * P.load_map(d, e, split, s).astype(np.float32)
        cv2.imwrite(str(out / f"{s}.png"), np.round(acc / (len(members) + w * len(extras))).astype(np.uint8))
    return f"{name}/{view}"


# ---- 11-15: which Qwen members the map should carry. "q" is the submitted 0.39 file's map
# (and so the pipeline check: its CSV must come out byte for byte the same); the others add
# the no-pretraining arm, or swap in / add the arms continued for 9 more epochs
# (filament-qwen-seg2). Each Qwen map is one member next to fold 0's five U-Nets, two next
# to the ten. A variant whose dump is not attached is skipped.
QSEG2, QSEG2_FT = "qwenseg_pre2/plain", "qwenseg_ft2/plain"
Q2_MIXES = (("q", (QSEG,)), ("qf", (QSEG, QSEG_FT)), ("q2", (QSEG2,)), ("q2f", (QSEG2, QSEG2_FT)),
            ("qq2", (QSEG, QSEG2)))


def kfold_q2(src, d, out, procs, ysrc):
    """The 25-column rule (geo + YOLO) fitted on all 707 out-of-fold photographs, as in
    kfold_yolo; then for each Q2_MIXES map: fold 0's ensemble scored with the nested cut,
    against "q" by bootstrap, and its test CSV, verified."""
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("models:", extract(src, d), flush=True)
    yolo = load_yolo(ysrc)
    assert sorted(yolo) == [0, 1, 2, 3, 4], "every fold needs its own YOLO"
    truth = fold_truth(False)
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    view = VIEW
    cache = d / f"kfold_pool_rle_{view}"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), f"{PAIR[n]}/{view}", stem, t["cache"], f, sets, str(cache), True)
            for n, t in enumerate(truth) for f, stem, sets in t["frames"]]
    frames = one_setting(list(imap(pool_job, jobs)))
    frames = add_neighbours(frames, d, [f"{PAIR[g]}/{view}" for g in folds], "val", folds, imap)
    frames = add_yolo(frames, [yolo[g] for g in folds], "val")
    final_gy = fit(frames, np.arange(len(frames)), GEO_YOLO_COLS)
    res = dict(photographs=len(folds), candidates=int(sum(len(f["X"]) for f in frames)), files={})
    print(res, flush=True)
    t0 = truth[0]
    stems0 = [s for _, s, _ in t0["frames"]]
    test_files = t0["meta"]["test_files"]
    base = None
    for tag, extras in Q2_MIXES:
        if not all((d / "dump" / e.split("/")[0]).exists() for e in extras):
            res[tag] = "dump not attached"
            print(tag, res[tag], flush=True)
            continue
        mixed = mix_dump(d, f"{tag}mix0", ENS0.split("+"), extras, QSEG_W, "val", stems0)
        c0 = d / f"kfold_pool_rle_{tag}mix0_{view}"
        c0.mkdir(parents=True, exist_ok=True)
        e0 = one_setting(list(imap(pool_job, [(str(d), mixed, stem, t0["cache"], f, sets, str(c0), True)
                                              for f, stem, sets in t0["frames"]])))
        e0 = add_neighbours(e0, d, [mixed] * len(e0), "val", np.zeros(len(e0), int), imap)
        e0 = add_yolo(e0, [yolo[0]] * len(e0), "val")
        res[tag], n25 = transfer(frames, folds, e0, GEO_YOLO_COLS)
        if base is None:
            base = (tag, n25)
        else:
            res[f"{tag}_minus_{base[0]}"] = boot(n25, base[1])
        ten = mix_dump(d, f"{tag}mix10", FAMILIES["r50"] + FAMILIES["r34"], extras, 2 * QSEG_W,
                       "test", [Path(n).stem for n in test_files])
        res["files"][tag] = write_csv(d, ten, test_files, out / f"submission_kf10{tag}_rule_geo_yolo.csv", final_gy,
                                      res[tag]["on_ens0_best_cut"]["cut"], GEO_YOLO_COLS, yolo=yolo)
        res["files"][tag]["verify"] = verify(out / res["files"][tag]["file"])
        print(tag, res[tag], {k: v for k, v in res.items() if k.startswith(tag + "_minus")}, flush=True)
        json.dump(res, open(out / "kfold_q2.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def counts_at_cuts(frames, scores):
    c = np.zeros((len(frames), len(C.POOL_CUTS), 4))
    for n, (f, s) in enumerate(zip(frames, scores)):
        for i, cut in enumerate(C.POOL_CUTS):
            keep = s >= cut
            y = f["Y"][keep]
            for q, g in enumerate(f["n_gt"]):
                tp = (y[:, q] > 0).sum()
                c[n, i] += (tp, keep.sum() - tp, g - tp, y[:, q].sum())
    return c


def fit(frames, idx, cols, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([frames[n]["X"][:, cols] for n in idx])
    y = np.concatenate([frames[n]["Y"].mean(axis=1) for n in idx])
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=seed).fit(X, y)


def score(model, frames, cols):
    return [model.predict(f["X"][:, cols]) if len(f["X"]) else np.zeros(0) for f in frames]


def rule_oof(frames, folds, cols, frac=1.0, seed=0):
    """Per photograph counts: train on the other folds, take their best cut, apply it to this one."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(frames), 4))
    for fold in sorted(set(folds)):
        tr = np.flatnonzero(folds != fold)
        te = np.flatnonzero(folds == fold)
        take = rng.choice(tr, int(round(len(tr) * frac)), replace=False)
        m = fit(frames, take, cols, fold)
        cut = int(np.argmax(pq_of(counts_at_cuts([frames[n] for n in tr], score(m, [frames[n] for n in tr], cols)).sum(axis=0))))
        out[te] = counts_at_cuts([frames[n] for n in te], score(m, [frames[n] for n in te], cols))[:, cut]
    return out


def boot(a, b, n=2000):
    rng = np.random.default_rng(0)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a))
        d.append(float(pq_of(a[s].sum(axis=0)) - pq_of(b[s].sum(axis=0))))
    return dict(mean=round(float(np.mean(d)), 4), lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))


def summary(c):
    return {k: (round(v, 4) if isinstance(v, float) else int(v)) for k, v in run.summarise(list(c)).items()}


def transfer(frames, folds, ens0, cols):
    """Trained on folds 1-4, scored on fold 0's ensemble: the cut an averaged map needs."""
    tr = np.flatnonzero(folds != 0)
    model = fit(frames, tr, cols)
    trf = [frames[n] for n in tr]
    tr_cut = int(np.argmax(pq_of(counts_at_cuts(trf, score(model, trf, cols)).sum(axis=0))))
    s0 = score(model, ens0, cols)
    c0 = counts_at_cuts(ens0, s0)
    p0 = pq_of(c0.sum(axis=0))
    f0 = np.arange(len(ens0)) % 5
    # a map averaged over models is softer, so the rule's scores run lower on it and the
    # single-model cut is far too strict (0.409, against 0.448): the cut comes from this ensemble
    best_cut = int(np.argmax(p0))
    nested_cut = np.zeros((len(ens0), 4))
    for fold in range(5):
        te = f0 == fold
        nested_cut[te] = c0[te, int(np.argmax(pq_of(c0[~te].sum(axis=0))))]
    return dict(single_rule_cut=float(C.POOL_CUTS[tr_cut]), on_ens0_at_that_cut=summary(c0[:, tr_cut].sum(axis=0)),
                on_ens0_best_cut=dict(cut=float(C.POOL_CUTS[best_cut]), PQ=round(float(p0.max()), 4),
                                      kept_per_photo=round(float(np.mean([(s >= C.POOL_CUTS[best_cut]).sum() for s in s0])), 3)),
                on_ens0_nested_cut=summary(nested_cut.sum(axis=0)),
                rule_trained_on_ens0_itself=summary(rule_oof(ens0, f0, cols).sum(axis=0))), nested_cut


def write_csv(d, variant, test_files, out, model=None, cut=None, cols=ALL_COLS, yolo=None, filv=None):
    """Test CSV for a map; with a rule, its setting; without one, the ensemble's tuned point.

    With yolo ({fold: predictions}), the rule scores each candidate once per YOLO fold
    model and keeps the mean, as the ten U-Nets' maps are averaged. filv(stem) gives the
    FilNet fold models' votes on a test photograph; each YOLO fold is paired with one."""
    te_dir = run.find_data()[2]
    rows, per = [], []
    stems = [Path(n).stem for n in test_files]
    nbs = neighbours(stems)                   # test frames only: inference may use only test/ images
    for i, name in enumerate(test_files):
        stem = Path(name).stem
        p = P.load_map(d, variant, "test", stem)
        masks = []
        if model is None:
            t, k, a = ENS_POINT[variant.split("/")[1]]
            lab, area = run.components(p, t, k)
            masks = [lab == j for j in range(1, len(area)) if area[j] >= a]
        else:
            img = cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE)
            cands, X = C.candidates(p, img, run.b0_deg(name))
            k = [n for n, c in enumerate(cands) if C.ONE_SETTING in c["from"]]
            if k and max(cols) >= X.shape[1]:
                nb = nb_job((str(d), variant, "test", stem, [C.sample_points(cands[n]) for n in k],
                             [(stems[j], dt) for j, dt in nbs[i]]))
                X = np.concatenate([X[k], nb], 1)
                k_rows = np.arange(len(k))
            else:
                k_rows = k
            if k and yolo:
                rl = [C.cand_rle(cands[n]) for n in k]
                fv = [yolo_cols(rl, v) for v in filv(stem)] if filv else None      # one per FilNet fold model
                s = np.mean([model.predict(np.concatenate([X[k_rows], yolo_cols(rl, pr.get(("test", stem), []))]
                                                          + ([fv[i % len(fv)]] if fv else []), 1)[:, cols])
                             for i, pr in enumerate(yolo.values())], axis=0)
            elif k:
                s = model.predict(X[k_rows][:, cols])
            if k:
                for n, q in zip(k, s):
                    if q >= cut:
                        c = cands[n]
                        m = np.zeros((run.FULL, run.FULL), bool)
                        m[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
                        masks.append(m)
        for n, m in enumerate(masks, 1):
            r = run.mu.encode(np.asfortranarray(m.astype(np.uint8)))
            rows.append(f"{stem}_{n},{r['counts'].decode('ascii')}")
        per.append(len(masks))
    Path(out).write_text("filament_id,segmentation_rle\n" + "\n".join(rows) + "\n")
    print(f"wrote {Path(out).name}: {len(rows)} rows, {np.mean(per):.2f} per image, {sum(x == 0 for x in per)} empty", flush=True)
    return dict(file=Path(out).name, rows=len(rows), per_image=round(float(np.mean(per)), 3), empty=int(sum(x == 0 for x in per)))


def verify(path):
    """src/verify_submission.py on one CSV, so that the check needs no local CPU; embedded by kernels/build.py."""
    src = globals().get("VERIFY_SRC") or (HERE.parents[1] / "src" / "verify_submission.py").read_text(encoding="utf-8")
    script = Path(path).parent / "verify_submission.py"
    script.write_text(src, encoding="utf-8")
    r = run.subprocess.run([sys.executable, str(script), str(path), "--test-dir", str(run.find_data()[2])],
                           capture_output=True, text=True)
    print(f"verify {Path(path).name}: exit {r.returncode}\n{r.stdout[-600:]}", flush=True)
    return dict(exit=r.returncode, tail=r.stdout[-300:])


def kfold_main(src, d, out, procs, smoke=False):
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("models:", extract(src, d), flush=True)
    truth = fold_truth(smoke)
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    res = dict(photographs=len(folds), sets=int(sum(len(s) for t in truth for _, _, s in t["frames"])))

    # 1. out-of-fold PQ: each architecture alone, and the pair averaged
    view, at_best = VIEW, {}
    for name, variants in (("r50", FAMILIES["r50"]), ("r34", FAMILIES["r34"]), ("pair", PAIR)):
        jobs = [(str(d), f"{variants[n]}/{view}", stem, sets) for n, t in enumerate(truth) for _, stem, sets in t["frames"]]
        g = np.stack(list(imap(grid_job, jobs)))
        pq = pq_of(g.sum(axis=0))
        i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
        at_best[name] = g[:, i, j, a]
        res[name] = dict(point=[P.THRESH[i], P.MERGE[j], P.MIN_AREA[a]], **summary(at_best[name].sum(axis=0)),
                         by_fold=[round(float(pq_of(at_best[name][folds == n].sum(axis=0))), 4) for n in range(5)])
        print(name, res[name], flush=True)
    res["pair_minus_r50"] = boot(at_best["pair"], at_best["r50"])
    baseline = at_best["pair"]

    # 2. the keep-rule on all of it
    cache = d / f"kfold_pool_{view}"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), f"{PAIR[n]}/{view}", stem, t["cache"], f, sets, str(cache))
            for n, t in enumerate(truth) for f, stem, sets in t["frames"]]
    frames = one_setting(list(imap(pool_job, jobs)))
    res["candidates"] = int(sum(len(f["X"]) for f in frames))
    print("candidates", res["candidates"], flush=True)
    frames = add_neighbours(frames, d, [f"{PAIR[g]}/{view}" for g in folds], "val", folds, imap)
    nbX = np.concatenate([f["X"][:, 21:24] for f in frames])
    res["neighbours"] = dict(candidates_with_support=round(float(np.isfinite(nbX[:, 0]).mean()), 4),
                             frames_with_a_neighbour=int(sum(np.isfinite(f["X"][:, 21]).any() for f in frames if len(f["X"]))))
    print("neighbours", res["neighbours"], flush=True)
    old, new = rule_oof(frames, folds, OLD_COLS), rule_oof(frames, folds, ALL_COLS)
    geo, nb = rule_oof(frames, folds, GEO_COLS), rule_oof(frames, folds, NB_COLS)
    small = [rule_oof(frames, folds, ALL_COLS, 0.25, seed) for seed in range(3)]
    res["rule"] = dict(old10=summary(old.sum(axis=0)), all17=summary(new.sum(axis=0)), geo21=summary(geo.sum(axis=0)),
                       nb24=summary(nb.sum(axis=0)),
                       all17_quarter_data=[round(float(pq_of(s.sum(axis=0))), 4) for s in small],
                       all17_minus_no_rule=boot(new, baseline), all17_minus_old10=boot(new, old),
                       all17_minus_quarter=boot(new, small[0]), geo21_minus_all17=boot(geo, new),
                       nb24_minus_geo21=boot(nb, geo), nb24_minus_all17=boot(nb, new))
    print(res["rule"], flush=True)

    # 3. does the rule carry over to an ensemble? trained on folds 1-4, scored on fold 0's ensemble
    cache0 = d / f"kfold_pool_ens0_{view}"
    cache0.mkdir(parents=True, exist_ok=True)
    t0 = truth[0]
    ens0 = one_setting(list(imap(pool_job, [(str(d), f"ens:{ENS0}/{view}", stem, t0["cache"], f, sets, str(cache0))
                                            for f, stem, sets in t0["frames"]])))
    ens0 = add_neighbours(ens0, d, [f"ens:{ENS0}/{view}"] * len(ens0), "val", np.zeros(len(ens0), int), imap)
    res["transfer"], n17 = transfer(frames, folds, ens0, ALL_COLS)
    res["transfer_geo21"], n21 = transfer(frames, folds, ens0, GEO_COLS)
    res["transfer_nb24"], n24 = transfer(frames, folds, ens0, NB_COLS)
    res["transfer_geo21_minus_all17"] = boot(n21, n17)
    res["transfer_nb24_minus_geo21"] = boot(n24, n21)
    print(res["transfer"], res["transfer_geo21"], res["transfer_nb24"], flush=True)

    # 4. test CSVs, with the rule trained on every photograph
    final = fit(frames, np.arange(len(frames)), ALL_COLS)
    final_geo = fit(frames, np.arange(len(frames)), GEO_COLS)
    final_nb = fit(frames, np.arange(len(frames)), NB_COLS)
    cut = float(C.POOL_CUTS[int(np.argmax(pq_of(counts_at_cuts(frames, score(final, frames, ALL_COLS)).sum(axis=0))))])
    res["final_cut_in_sample"] = cut
    cut = res["transfer"]["on_ens0_best_cut"]["cut"]
    cut_geo = res["transfer_geo21"]["on_ens0_best_cut"]["cut"]
    cut_nb = res["transfer_nb24"]["on_ens0_best_cut"]["cut"]
    res["final_cut"], res["final_cut_geo21"], res["final_cut_nb24"] = cut, cut_geo, cut_nb
    test_files = truth[0]["meta"]["test_files"]
    ten = "ens:" + "+".join(FAMILIES["r50"] + FAMILIES["r34"])
    res["files"] = dict(
        folds10=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10.csv"),
        folds10_rule=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10_rule.csv", final, cut),
        folds10_rule_geo=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10_rule_geo.csv",
                                   final_geo, cut_geo, GEO_COLS),
        folds10_rule_nb=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10_rule_nb.csv",
                                  final_nb, cut_nb, NB_COLS),
    )
    # the public leaderboard did not move when the 230 weakest of 1206 filaments were dropped
    # (0.38 both), where validation says that costs 0.04: test seems to reward fewer. A
    # series of stricter cuts on the best rule, nested in one another, to find its peak there
    for c in GEO_EXTRA_CUTS:
        res["files"][f"folds10_rule_geo_c{c:.2f}"] = write_csv(
            d, f"{ten}/{view}", test_files, out / f"submission_kf10_rule_geo_c{c:.2f}.csv", final_geo, c, GEO_COLS)
    for v in res["files"].values():
        v["verify"] = verify(out / v["file"])
    json.dump(res, open(out / "kfold.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def kfold_yolo(src, d, out, procs, ysrc, smoke=False):
    """v8: the geo rule with and without YOLO's four columns -- out-of-fold on all
    photographs, and carried to fold 0's ensemble -- and its test CSV. When the Qwen
    segmenter's dump is there, also: it alone against its control and an R50, and fold 0's
    ensemble with its map averaged in, under both rules, with the two test CSVs of the ten
    plus it. Only what these comparisons need, so ~1.5 h of CPU instead of kfold_main's 2.5."""
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print("models:", extract(src, d), flush=True)
    yolo = load_yolo(ysrc)
    print("yolo folds:", sorted(yolo), {f: len(v) for f, v in yolo.items()}, flush=True)
    if smoke:                                 # the plumbing check may run before every fold is trained
        yolo = {f: yolo.get(f, {}) for f in range(5)}
    assert sorted(yolo) == [0, 1, 2, 3, 4], "every fold needs its own YOLO"
    truth = fold_truth(smoke)
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    view = VIEW
    cache = d / f"kfold_pool_rle_{view}"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(str(d), f"{PAIR[n]}/{view}", stem, t["cache"], f, sets, str(cache), True)
            for n, t in enumerate(truth) for f, stem, sets in t["frames"]]
    frames = one_setting(list(imap(pool_job, jobs)))
    frames = add_neighbours(frames, d, [f"{PAIR[g]}/{view}" for g in folds], "val", folds, imap)
    frames = add_yolo(frames, [yolo[g] for g in folds], "val")
    res = dict(photographs=len(folds), candidates=int(sum(len(f["X"]) for f in frames)))
    hit = np.concatenate([(f["Y"] > 0).any(axis=1) for f in frames]).astype(int)
    yX = np.concatenate([f["X"][:, YOLO_COLS] for f in frames])
    from sklearn.metrics import roc_auc_score
    res["yolo_auc"] = {c: round(float(roc_auc_score(hit, yX[:, i])), 4)
                       for i, c in enumerate(("y_iou", "y_conf", "y_near", "y_cover"))}
    print(res, flush=True)

    geo, gy = rule_oof(frames, folds, GEO_COLS), rule_oof(frames, folds, GEO_YOLO_COLS)
    res["rule"] = dict(geo21=summary(geo.sum(axis=0)), geo_yolo25=summary(gy.sum(axis=0)),
                       geo_yolo25_minus_geo21=boot(gy, geo),
                       by_fold={k: [round(float(pq_of(v[folds == n].sum(axis=0))), 4) for n in range(5)]
                                for k, v in (("geo21", geo), ("geo_yolo25", gy))})
    print(res["rule"], flush=True)

    cache0 = d / f"kfold_pool_rle_ens0_{view}"
    cache0.mkdir(parents=True, exist_ok=True)
    t0 = truth[0]
    ens0 = one_setting(list(imap(pool_job, [(str(d), f"ens:{ENS0}/{view}", stem, t0["cache"], f, sets, str(cache0), True)
                                            for f, stem, sets in t0["frames"]])))
    ens0 = add_neighbours(ens0, d, [f"ens:{ENS0}/{view}"] * len(ens0), "val", np.zeros(len(ens0), int), imap)
    ens0 = add_yolo(ens0, [yolo[0]] * len(ens0), "val")
    res["transfer_geo21"], n21 = transfer(frames, folds, ens0, GEO_COLS)
    res["transfer_geo_yolo25"], n25 = transfer(frames, folds, ens0, GEO_YOLO_COLS)
    res["transfer_geo_yolo25_minus_geo21"] = boot(n25, n21)
    print(res["transfer_geo21"], res["transfer_geo_yolo25"], res["transfer_geo_yolo25_minus_geo21"], flush=True)

    test_files = truth[0]["meta"]["test_files"]
    ten = "ens:" + "+".join(FAMILIES["r50"] + FAMILIES["r34"])
    qseg = (d / "dump" / QSEG.split("/")[0]).exists()
    if qseg:
        # the Qwen segmenter alone against its no-pretraining control and a single R50, all
        # plain on fold 0's photographs, each at its own best point of the grid
        jobs = [(str(d), v, stem, sets) for v in (QSEG, QSEG_FT, "r50/plain") for _, stem, sets in t0["frames"]]
        g = np.stack(list(imap(grid_job, jobs))).reshape(3, len(t0["frames"]), len(P.THRESH), len(P.MERGE),
                                                         len(P.MIN_AREA), 4)
        at = []
        for v, gv in zip(("qseg_pre", "qseg_ft", "r50_plain"), g):
            pq = pq_of(gv.sum(axis=0))
            i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
            at.append(gv[:, i, j, a])
            res[v] = dict(point=[P.THRESH[i], P.MERGE[j], P.MIN_AREA[a]], **summary(at[-1].sum(axis=0)))
        res["qseg_pre_minus_ft"], res["qseg_pre_minus_r50"] = boot(at[0], at[1]), boot(at[0], at[2])
        # one more member of the averaged map: fold 0's five U-Nets plus the Qwen map
        # (not "ens...": load_map reads every model name starting so as an ensemble)
        mixed = mix_dump(d, "qmix0", ENS0.split("+"), QSEG, QSEG_W, "val", [s for _, s, _ in t0["frames"]])
        cache0q = d / f"kfold_pool_rle_ens0q_{view}"
        cache0q.mkdir(parents=True, exist_ok=True)
        ens0q = one_setting(list(imap(pool_job, [(str(d), mixed, stem, t0["cache"], f, sets, str(cache0q), True)
                                                 for f, stem, sets in t0["frames"]])))
        ens0q = add_neighbours(ens0q, d, [mixed] * len(ens0q), "val", np.zeros(len(ens0q), int), imap)
        ens0q = add_yolo(ens0q, [yolo[0]] * len(ens0q), "val")
        res["qseg_transfer_geo21"], q21 = transfer(frames, folds, ens0q, GEO_COLS)
        res["qseg_transfer_geo_yolo25"], q25 = transfer(frames, folds, ens0q, GEO_YOLO_COLS)
        res["qseg_geo21_minus_geo21"], res["qseg_geo_yolo25_minus_geo21"] = boot(q21, n21), boot(q25, n21)
        res["qseg_geo_yolo25_minus_geo_yolo25"] = boot(q25, n25)
        print({k: res[k] for k in res if k.startswith("qseg")}, flush=True)
        # the ten keep the Qwen map's share: 2 of 12, as 1 of 6
        ten_q = mix_dump(d, "qmix10", FAMILIES["r50"] + FAMILIES["r34"], QSEG, 2 * QSEG_W,
                         "test", [Path(n).stem for n in test_files])
    all_ = np.arange(len(frames))
    final_geo, final_gy = fit(frames, all_, GEO_COLS), fit(frames, all_, GEO_YOLO_COLS)
    cut_geo = res["transfer_geo21"]["on_ens0_best_cut"]["cut"]
    cut_gy = res["transfer_geo_yolo25"]["on_ens0_best_cut"]["cut"]
    # the first file must reproduce submission_kf10_rule_geo.csv (1255 rows): the pipeline check
    res["files"] = dict(
        folds10_rule_geo=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10_rule_geo.csv",
                                   final_geo, cut_geo, GEO_COLS),
        folds10_rule_geo_yolo=write_csv(d, f"{ten}/{view}", test_files, out / "submission_kf10_rule_geo_yolo.csv",
                                        final_gy, cut_gy, GEO_YOLO_COLS, yolo=yolo))
    if qseg:
        res["files"].update(
            folds10q_rule_geo=write_csv(d, ten_q, test_files, out / "submission_kf10q_rule_geo.csv", final_geo,
                                        res["qseg_transfer_geo21"]["on_ens0_best_cut"]["cut"], GEO_COLS),
            folds10q_rule_geo_yolo=write_csv(d, ten_q, test_files, out / "submission_kf10q_rule_geo_yolo.csv", final_gy,
                                             res["qseg_transfer_geo_yolo25"]["on_ens0_best_cut"]["cut"], GEO_YOLO_COLS,
                                             yolo=yolo))
    for v in res["files"].values():
        v["verify"] = verify(out / v["file"])
    json.dump(res, open(out / "kfold_yolo.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


# ---- 11-16: FilNet (aug_unet/filnet.py: a U-Net trained from nothing, no ImageNet) in the 0.39 file's
# pipeline, two ways: as a third member of every fold's map, and as "votes" -- its own filaments beside each
# candidate, the way YOLO's are. Kernels filament-filnet (fold 0, and its `raw` arm), -b (folds 1-2), -c (3-4).
FIL = ("filnet_phys", "filnet_phys_f1", "filnet_phys_f2", "filnet_phys_f3", "filnet_phys_f4")
FIL_RAW = "filnet_raw"
FILV_COLS = [28, 29, 30, 31]                  # as YOLO_COLS, against FilNet's own filaments
KFF_SMOKE = False                             # kernels/build.py sets this in kfold_fil_smoke/run.py
KFF_CSV = ("qfil", "qv", "qfilv")             # the variants whose test CSV is written


def fil_votes(d, model, split, stem):
    """One FilNet's own filaments on a photograph (the one setting), as YOLO's are kept: [(mean probability, rle)]."""
    p = P.load_map(d, f"{model}/{VIEW}", split, stem)
    lab, area = run.components(p, *C.ONE_SETTING)
    out = []
    for j in range(1, len(area)):
        if area[j] >= C.POOL_MIN_AREA:
            m = lab == j
            out.append((float(p[m].mean()) / 255.0, run.mu.encode(np.asfortranarray(m.astype(np.uint8)))))
    return out


def add_votes(frames, votes):
    """Append the four FilNet columns; votes[n] is None where that fold has no FilNet (the rule reads NaN)."""
    for f, v in zip(frames, votes):
        cols = yolo_cols(f["rle"], v) if v is not None else np.full((len(f["X"]), 4), np.nan, np.float32)
        f["X"] = np.concatenate([f["X"], cols], 1)
    return frames


def kfold_fil(src, d, out, procs, ysrc, smoke=False):
    """The 0.39 file's pipeline with FilNet added, each way and both.
      rows   each fold's two U-Nets (pair), or the two plus that fold's FilNet (triple); 25 columns, or 29 with
             the FilNet votes
      check  out of fold on all photographs, the rule fitted and cut as in the 0.39 file, against pair/25 by
             bootstrap; and fold 0's ensemble (5 U-Nets + Qwen, with or without fold 0's FilNets), nested cut
      test   the ten U-Nets (+ every FilNet) + Qwen, the rule fitted on all rows; written and verified
    A fold whose FilNet is not attached keeps its pair and has no votes."""
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    models = extract(src, d)
    print("models:", models, flush=True)
    have = [m in models for m in FIL]
    print("FilNet folds attached:", [n for n in range(5) if have[n]], flush=True)
    assert have[0], "fold 0's FilNet is needed for the fold-0 check"
    yolo = load_yolo(ysrc)
    if smoke:
        yolo = {f: yolo.get(f, {}) for f in range(5)}
    assert sorted(yolo) == [0, 1, 2, 3, 4], "every fold needs its own YOLO"
    truth = fold_truth(smoke)
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    view = VIEW

    def frames_of(variants, tag):
        cache = d / f"kfold_pool_rle_{tag}_{view}"
        cache.mkdir(parents=True, exist_ok=True)
        fr = one_setting(list(imap(pool_job, [(str(d), variants[n], stem, t["cache"], f, sets, str(cache), True)
                                              for n, t in enumerate(truth) for f, stem, sets in t["frames"]])))
        fr = add_neighbours(fr, d, [variants[g] for g in folds], "val", folds, imap)
        return add_yolo(fr, [yolo[g] for g in folds], "val")

    pair = [f"{PAIR[n]}/{view}" for n in range(5)]
    triple = [f"{PAIR[n]}+{FIL[n]}/{view}" if have[n] else pair[n] for n in range(5)]
    fr = {"pair": frames_of(pair, "pair")}
    print("pair rows ready", flush=True)
    fr["triple"] = frames_of(triple, "triple")
    votes = [fil_votes(d, FIL[g], "val", f["stem"]) if have[g] else None for f, g in zip(fr["pair"], folds)]
    for k in fr:
        add_votes(fr[k], votes)
    print("triple rows and votes ready", flush=True)
    COLS = {"25": GEO_YOLO_COLS, "29": GEO_YOLO_COLS + FILV_COLS}
    oof = {f"{k}/{c}": rule_oof(fr[k], folds, COLS[c]) for k in fr for c in COLS}
    on = np.array([have[g] for g in folds])
    res = dict(photographs=len(folds), filnet_folds=[n for n in range(5) if have[n]],
               oof={k: summary(v.sum(axis=0)) for k, v in oof.items()},
               oof_minus_pair25={k: boot(v, oof["pair/25"]) for k, v in oof.items() if k != "pair/25"},
               oof_on_filnet_folds={k: summary(v[on].sum(axis=0)) for k, v in oof.items()},
               oof_on_filnet_folds_minus_pair25={k: boot(v[on], oof["pair/25"][on]) for k, v in oof.items() if k != "pair/25"},
               files={})
    print(json.dumps(res, indent=1, default=float), flush=True)
    json.dump(res, open(out / "kfold_fil.json", "w"), indent=1, default=float)
    t0 = truth[0]
    stems0 = [s for _, s, _ in t0["frames"]]
    test_files = t0["meta"]["test_files"]
    test_stems = [Path(n).stem for n in test_files]
    raw = [FIL_RAW] if FIL_RAW in models else []
    fil_all = [FIL[n] for n in range(5) if have[n]]
    extra0 = {"pair": [], "triple": [FIL[0]] + raw}
    extra_test = {"pair": [], "triple": fil_all + raw}
    e0, mix_test = {}, {}
    for key in ("pair", "triple"):
        m0 = ENS0.split("+") + extra0[key]
        mixed = mix_dump(d, f"{key}mix0", m0, (QSEG,), QSEG_W * len(m0) / 5.0, "val", stems0)
        c0 = d / f"kfold_pool_rle_{key}mix0_{view}"
        c0.mkdir(parents=True, exist_ok=True)
        e = one_setting(list(imap(pool_job, [(str(d), mixed, stem, t0["cache"], f, sets, str(c0), True)
                                             for f, stem, sets in t0["frames"]])))
        e = add_neighbours(e, d, [mixed] * len(e), "val", np.zeros(len(e), int), imap)
        e = add_yolo(e, [yolo[0]] * len(e), "val")
        e0[key] = add_votes(e, [fil_votes(d, FIL[0], "val", f["stem"]) for f in e])
    tv = {}

    def test_votes(stem):
        if stem not in tv:
            tv[stem] = [fil_votes(d, m, "test", stem) for m in fil_all]
        return tv[stem]

    base = None
    for tag, key, c in (("q", "pair", "25"), ("qfil", "triple", "25"), ("qv", "pair", "29"), ("qfilv", "triple", "29")):
        res[tag], n25 = transfer(fr[key], folds, e0[key], COLS[c])
        if base is None:
            base = n25
        else:
            res[f"{tag}_minus_q"] = boot(n25, base)
        print(tag, res[tag], {k: v for k, v in res.items() if k == f"{tag}_minus_q"}, flush=True)
        json.dump(res, open(out / "kfold_fil.json", "w"), indent=1, default=float)
    for tag, key, c in (("qfil", "triple", "25"), ("qv", "pair", "29"), ("qfilv", "triple", "29")):
        if tag not in KFF_CSV:
            continue
        if key not in mix_test:
            members = list(FAMILIES["r50"] + FAMILIES["r34"]) + extra_test[key]
            mix_test[key] = (mix_dump(d, f"{key}mixtest", members, (QSEG,), QSEG_W * len(members) / 5.0, "test",
                                      test_stems), len(members))
        mix, n_mem = mix_test[key]
        final = fit(fr[key], np.arange(len(folds)), COLS[c])
        name = f"submission_kf{n_mem}{tag}_rule_geo_yolo.csv"
        res["files"][tag] = write_csv(d, mix, test_files, out / name, final, res[tag]["on_ens0_best_cut"]["cut"],
                                      COLS[c], yolo=yolo, filv=test_votes if c == "29" else None)
        res["files"][tag]["verify"] = verify(out / name)
        json.dump(res, open(out / "kfold_fil.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def kaggle_kfold_fil():
    """filament-kfold-fil: filament-kfold-q2's inputs plus the three FilNet kernels' dumps."""
    kfold_fil(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, run.os.cpu_count() or 4, Path("/kaggle/input"),
              smoke=KFF_SMOKE)


def kaggle_kfold():
    """The kernel's entry point: the dump kernels' tars arrive under /kaggle/input."""
    kfold_main(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, procs=run.os.cpu_count() or 4)


KFY_SMOKE = False                             # kernels/build.py sets this in kfold_yolo_smoke/run.py


def kaggle_kfold_yolo():
    """filament-kfold-yolo: the dump kernels' tars and both YOLO kernels' predictions under /kaggle/input."""
    kfold_yolo(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, run.os.cpu_count() or 4, Path("/kaggle/input"),
               smoke=KFY_SMOKE)


def kaggle_kfold_q2():
    """filament-kfold-q2: as filament-kfold-yolo's inputs, plus filament-qwen-seg2 when it has run."""
    kfold_q2(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, run.os.cpu_count() or 4, Path("/kaggle/input"))


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="kernels/aug_post/out")
    ap.add_argument("--out", default="kernels/kfold/out")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="ten validation photographs per fold")
    ap.add_argument("--yolo", help="folder holding yolo_preds_f0..4.pkl: run the v8 YOLO comparison instead")
    a = ap.parse_args()
    if a.yolo:
        kfold_yolo(a.dir, a.dir, a.out, a.procs, a.yolo, a.smoke)
    else:
        kfold_main(a.dir, a.dir, a.out, a.procs, a.smoke)


if __name__ == "__main__":
    main()
