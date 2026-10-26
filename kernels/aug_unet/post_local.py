"""Post-processing, TTA and ensembling explored offline, on dumped probability maps.

The Kaggle dump (run.py with MODE = "dump") holds every validation and test
probability map of both trained models, plain and with D4 TTA. Everything here
is CPU work on those maps, so it spends no GPU quota:

  1. a wide grid over threshold x merge radius x minimum area, for each variant:
     flip / full / the two averaged, each plain and with D4 TTA
  2. a learned keep-rule: predict each component's expected matched IoU from its
     own features and keep it only if that exceeds PQ/2. Adding a prediction
     always adds 0.5 to PQ's denominator (TP+FN or FP, either way) and q*IoU to
     its numerator, so PQ/2 is exactly where a component starts to pay
  3. a submission CSV for the best variant, checked by src/verify_submission.py
  4. the outline on its own setting: detection threshold and mask extent apart
  5. oracles: PQ with a perfect keep-rule, at several detection thresholds
  6. models trained on every photograph (no validation maps) stand in on test
     for the validated model they extend, with that model's tuned settings

Scores are our scorer on the 141 held-out photographs, every annotation set --
validation, not the leaderboard. Run:
    python kernels/aug_unet/post_local.py --dir kernels/aug_unet/out_kaggle
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, pickle, subprocess, sys, tarfile, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- off Kaggle this hides the GPU and refuses if it is visible

THRESH = tuple(round(t, 2) for t in np.arange(0.30, 0.96, 0.05))
MERGE = (0, 3, 7, 12)
MIN_AREA = (100, 200, 350, 500, 750, 1000, 1500, 2500)

# SQ sat at 0.67-0.68 for every model and mix while the grid tied a filament's
# outline to the threshold that finds it. At 8 px wide, one pixel on each edge
# is about a quarter of the IoU, so the outline gets a setting of its own:
#   ("t", e)     e below the detection threshold: grow each kept component through
#                pixels >= e, at most GROW px out; e above it: keep its pixels >= e
#   ("morph", r) dilate (r > 0) or erode (r < 0) it by |r| px; 2 px or more cost
#                0.08-0.12 PQ on the 10-frame smoke test, so only 1 px is tried
NO_EXT = ("none", 0)
EXTENTS = ([NO_EXT] + [("t", e) for e in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
           + [("morph", r) for r in (-1, 1)])
GROW = 8
ORACLE_T = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------- inputs

def extract(src, d):
    """Unpack every dump_<model>.tar found under src into d/dump/<model>/."""
    for tar in sorted(Path(src).rglob("dump_*.tar")):
        out = d / "dump" / tar.stem[len("dump_"):]
        if not out.exists():
            out.mkdir(parents=True)
            with tarfile.open(tar) as t:
                t.extractall(out, filter="data")
    return sorted(p.name for p in (d / "dump").iterdir()) if (d / "dump").exists() else []


def ground_truth(smoke=False):
    """The same val split and GT crops the Kaggle run scored against."""
    if not run.ON_KAGGLE:
        run.CACHE = HERE / "out" / ("cache_smoke" if smoke else "cache_val")
    if not (run.CACHE / "val_gt.pkl").exists():
        run.prepare(smoke, with_train=False)
    meta = json.load(open(run.CACHE / "meta.json"))
    return meta, pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))


def load_map(d, variant, split, stem):
    """variant = '<model>/<view>'; model "ens" is the mean of every dumped model."""
    model, view = variant.split("/")
    if model.startswith("ens"):              # "ens" = every model, "ens:a+b" = those two
        ms = (model.split(":", 1)[1].split("+") if ":" in model
              else sorted(q.name for q in (d / "dump").iterdir()))
        acc = sum(load_map(d, f"{m}/{view}", split, stem).astype(np.float32) for m in ms)
        return np.round(acc / len(ms)).astype(np.uint8)
    path = d / "dump" / model / f"{split}_{view}" / f"{stem}.png"
    p = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if p is None:
        raise FileNotFoundError(path)
    return p


# ---------------------------------------------------------------- 1. grid

def frame_job(job):
    """All grid points for one frame: counts summed over its annotation sets."""
    d, variant, stem, sets = job
    p = load_map(Path(d), variant, "val", stem)
    tot = np.zeros((len(THRESH), len(MERGE), len(MIN_AREA), 4))
    first = np.zeros_like(tot)
    for i, t in enumerate(THRESH):
        for j, k in enumerate(MERGE):
            lab, area = run.components(p, t, k)
            for a_i, a in enumerate(MIN_AREA):
                keep = area >= a
                keep[0] = False
                for idx, crops in sets:
                    c = np.array(run.pq_counts(crops, lab, area, keep), float)
                    tot[i, j, a_i] += c
                    if idx < 60:
                        first[i, j, a_i] += c
    return tot, first


def pq_of(c):
    tp, fp, fn, s = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    den = tp + 0.5 * fp + 0.5 * fn
    return np.where(den > 0, s / np.maximum(den, 1e-9), 0.0)


def grid(d, variant, meta, gt, pool):
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    stems = [Path(f).stem for f in meta["val_files"]]
    jobs = [(str(d), variant, stems[f], sets) for f, sets in by_frame.items()]
    tot = first = 0
    for t, f in pool.imap_unordered(frame_job, jobs, chunksize=2):
        tot, first = tot + t, first + f
    pq = pq_of(tot)
    i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
    c = tot[i, j, a]
    best = dict(variant=variant, PQ=float(pq[i, j, a]), PQ60=float(pq_of(first)[i, j, a]),
                SQ=float(c[3] / max(c[0], 1)), RQ=float(c[0] / max(c[0] + 0.5 * c[1] + 0.5 * c[2], 1e-9)),
                TP=int(c[0]), FP=int(c[1]), FN=int(c[2]),
                thresh=THRESH[i], merge=MERGE[j], min_area=MIN_AREA[a])
    return best, pq


# ---------------------------------------------------------------- 2. learned keep-rule

def comp_features(p, img, lab, area, ids):
    """Per component: what the model and the image say about it."""
    feats = []
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for j in ids:
        ys, xs = np.nonzero(lab == j)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 10
        sl = (slice(max(y0 - pad, 0), y1 + pad), slice(max(x0 - pad, 0), x1 + pad))
        m = lab[sl] == j
        pv = p[sl][m].astype(np.float32) / 255.0
        ring = (cv2.dilate(m.astype(np.uint8), ker) > 0) & ~m
        iv, rv = img[sl][m].astype(np.float32), img[sl][ring].astype(np.float32)
        cy, cx = ys.mean(), xs.mean()
        cov = np.cov(np.stack([ys - cy, xs - cx])) if ys.size > 2 else np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        feats.append([
            np.log(area[j]), pv.mean(), pv.max(), np.percentile(pv, 90), pv.std(), (pv > 0.8).mean(),
            np.sqrt(ev[0] / max(ev[1], 1e-6)),                       # elongation
            np.hypot(cx - run.CX, cy - run.CY) / run.R_DISK,          # where on the disk
            (rv.mean() - iv.mean()) if rv.size else 0.0,              # darker than its surroundings
            float(len(ids)),                                          # how busy the frame is
        ])
    return np.array(feats, np.float32)


def keep_rule_job(job):
    """Components of one val frame at a fixed (thresh, merge), with features and targets.

    Features describe the detection outline; targets (matched IoU) are scored on
    the outline `ext` draws. feats=False skips the features, for the oracle.
    """
    d, variant, stem, sets, t, k, a_min, tr_dir, ext, feats = job
    p = load_map(Path(d), variant, "val", stem)
    lab0, area0 = run.components(p, t, k)
    cand = area0 >= a_min
    cand[0] = False
    lab, area = extend(p, lab0, cand, tuple(ext), t)
    ids = [j for j in range(1, len(area0)) if cand[j] and area[j] > 0]
    if not ids:
        return stem, np.zeros((0, 10), np.float32), np.zeros((0, len(sets))), [len(c) for _, c in sets]
    if feats:
        img = cv2.imread(str(Path(tr_dir) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        X = comp_features(p, img, lab0, area0, ids)
    else:
        X = np.zeros((len(ids), 10), np.float32)
    return stem, X, matched_iou(lab, area, ids, sets), [len(c) for _, c in sets]


def matched_iou(lab, area, ids, sets):
    """Per set, each component's matched IoU (0 if it matches nothing there)."""
    Y = np.zeros((len(ids), len(sets)))
    pos = {j: n for n, j in enumerate(ids)}
    for s_i, (_, crops) in enumerate(sets):
        for y0, x0, c in crops:
            lv = lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c]
            if lv.size == 0:
                continue
            u, n = np.unique(lv[lv > 0], return_counts=True)
            for j, it in zip(u.tolist(), n.tolist()):
                if j in pos:
                    iou = it / (int(c.sum()) + int(area[j]) - it)
                    if iou > 0.5:
                        Y[pos[j], s_i] = iou
    return Y


def pq_for(frames, score, cut):
    """Pooled PQ when a frame's components are kept where score > cut."""
    tp = fp = fn = s = 0.0
    start = 0
    for _, Xf, Yf, n_gt in frames:
        keep = score[start:start + len(Xf)] > cut
        start += len(Xf)
        for s_i, ng in enumerate(n_gt):
            iou = Yf[keep, s_i] if len(Xf) else np.zeros(0)
            t_ = int((iou > 0).sum())
            tp += t_; s += float(iou.sum()); fp += int(keep.sum()) - t_; fn += ng - t_
    den = tp + 0.5 * fp + 0.5 * fn
    return s / den if den else 0.0, dict(TP=int(tp), FP=int(fp), FN=int(fn))


def keep_rule(d, best, meta, gt, pool, tr_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    stems = [Path(f).stem for f in meta["val_files"]]
    jobs = [(str(d), best["variant"], stems[f], sets, best["thresh"], best["merge"], 100, str(tr_dir),
             tuple(best.get("ext", NO_EXT)), True)
            for f, sets in by_frame.items()]
    frames = list(pool.imap(keep_rule_job, jobs, chunksize=2))
    X = np.concatenate([f[1] for f in frames])
    y = np.concatenate([f[2].mean(axis=1) for f in frames])          # expected matched IoU
    if len(y) < 100:
        return dict(PQ=0.0, cut=None, n_components=int(len(y)), TP=0, FP=0, FN=0, curve=[]), None
    groups = np.concatenate([[n] * len(f[1]) for n, f in enumerate(frames)])
    oof = np.zeros(len(y))
    for fold in range(5):                                             # grouped by photograph
        te = groups % 5 == fold
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(X[~te], y[~te])
        oof[te] = m.predict(X[te])

    cuts = np.round(np.arange(0.05, 0.50, 0.025), 3)
    scores = [(c, *pq_for(frames, oof, c)) for c in cuts]
    c_best, pq_best, cnt = max(scores, key=lambda r: r[1])
    final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(X, y)
    return dict(PQ=float(pq_best), cut=float(c_best), n_components=int(len(y)), **cnt,
                curve=[(float(c), float(p)) for c, p, _ in scores]), final


# ---------------------------------------------------------------- 3. submission

def write_submission(d, best, rule, model, meta, out):
    rows, per = [], []
    te_dir = run.find_data()[2]
    ext = tuple(best.get("ext", NO_EXT))
    for f in meta["test_files"]:
        stem = Path(f).stem
        p = load_map(d, best["variant"], "test", stem)
        lab, area = run.components(p, best["thresh"], best["merge"])
        if rule is not None:
            ids = [j for j in range(1, len(area)) if area[j] >= 100]
            if ids:
                img = cv2.imread(str(te_dir / f), cv2.IMREAD_GRAYSCALE)
                pred = model.predict(comp_features(p, img, lab, area, ids))
                ids = [j for j, q in zip(ids, pred) if q > rule["cut"]]
        else:
            ids = [j for j in range(1, len(area)) if area[j] >= best["min_area"]]
        keep = np.zeros(len(area), bool)
        keep[ids] = True
        lab, area = extend(p, lab, keep, ext, best["thresh"])
        ids = [j for j in ids if area[j] > 0]
        for n, j in enumerate(ids, 1):
            r = run.mu.encode(np.asfortranarray((lab == j).astype(np.uint8)))
            rows.append(f"{stem}_{n},{r['counts'].decode('ascii')}")
        per.append(len(ids))
    out.write_text("filament_id,segmentation_rle\n" + "".join(r + "\n" for r in rows))
    return len(rows), float(np.mean(per))


# ---------------------------------------------------------------- 4. mask extent

_DISK = None


def on_disk():
    global _DISK
    if _DISK is None:
        yy, xx = np.mgrid[:run.FULL, :run.FULL]
        _DISK = ((xx - run.CX) ** 2 + (yy - run.CY) ** 2) <= run.R_DISK ** 2
    return _DISK


def extend(p, lab, keep, ext, t):
    """Redraw the kept components' outlines: a new label image and its areas.

    What is found and kept is decided on the detection outline, before this;
    a component the new outline empties is dropped (its area becomes 0).
    """
    kind, v = ext
    lab = np.where(keep[lab], lab, 0).astype(np.int32)
    if kind == "none":
        pass
    elif (kind == "t" and v < t) or (kind == "morph" and v > 0):
        allowed = on_disk() & (p >= int(round(255 * v))) if kind == "t" else on_disk()
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        f = lab.astype(np.float32)               # cv2 dilates float32; labels stay exact
        for _ in range(GROW if kind == "t" else v):
            g = cv2.dilate(f, cross)
            new = (f == 0) & allowed & (g > 0)
            if not new.any():
                break
            f[new] = g[new]                      # where two meet, the higher label wins
        lab = f.astype(np.int32)
    elif kind == "t":
        lab[p < int(round(255 * v))] = 0
    else:                                        # erode each component by |v| px
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * -v + 1, 2 * -v + 1))
        f = lab.astype(np.float32)
        lab[(cv2.erode(f, ker) != f) | (cv2.dilate(f, ker) != f)] = 0
    return lab, np.bincount(lab.ravel(), minlength=len(keep))


def ext_name(ext):
    return "" if ext[0] == "none" else f"_t{ext[1]}" if ext[0] == "t" else f"_m{ext[1]:+d}"


def extent_job(job):
    """Detection grid x outline settings for one frame, counts summed over its sets."""
    d, variant, stem, sets = job
    p = load_map(Path(d), variant, "val", stem)
    tot = np.zeros((len(THRESH), len(MERGE), len(EXTENTS), len(MIN_AREA), 4))
    first = np.zeros_like(tot)
    for i, t in enumerate(THRESH):
        for j, k in enumerate(MERGE):
            lab0, area0 = run.components(p, t, k)
            cand = area0 >= MIN_AREA[0]
            cand[0] = False
            for e_i, ext in enumerate(EXTENTS):
                lab, area = extend(p, lab0, cand, ext, t)
                for a_i, a in enumerate(MIN_AREA):
                    keep = (area0 >= a) & (area > 0)
                    keep[0] = False
                    for idx, crops in sets:
                        c = np.array(run.pq_counts(crops, lab, area, keep), float)
                        tot[i, j, e_i, a_i] += c
                        if idx < 60:
                            first[i, j, e_i, a_i] += c
    return tot, first


def extent_grid(d, variant, meta, gt, pool):
    """The best detection point with its own outline, and the best PQ per outline."""
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    stems = [Path(f).stem for f in meta["val_files"]]
    jobs = [(str(d), variant, stems[f], sets) for f, sets in by_frame.items()]
    tot = first = 0
    for t, f in pool.imap_unordered(extent_job, jobs, chunksize=1):
        tot, first = tot + t, first + f
    pq = pq_of(tot)
    i, j, e, a = np.unravel_index(np.argmax(pq), pq.shape)
    c = tot[i, j, e, a]
    best = dict(variant=variant, ext=EXTENTS[e], PQ=float(pq[i, j, e, a]),
                PQ60=float(pq_of(first)[i, j, e, a]),
                SQ=float(c[3] / max(c[0], 1)), RQ=float(c[0] / max(c[0] + 0.5 * c[1] + 0.5 * c[2], 1e-9)),
                TP=int(c[0]), FP=int(c[1]), FN=int(c[2]),
                thresh=THRESH[i], merge=MERGE[j], min_area=MIN_AREA[a])
    per_ext = []
    for e_i, ext in enumerate(EXTENTS):
        i, j, a = np.unravel_index(np.argmax(pq[:, :, e_i]), pq[:, :, e_i].shape)
        c = tot[i, j, e_i, a]
        per_ext.append(dict(ext=ext, PQ=float(pq[i, j, e_i, a]), SQ=float(c[3] / max(c[0], 1)),
                            RQ=float(c[0] / max(c[0] + 0.5 * c[1] + 0.5 * c[2], 1e-9)),
                            thresh=THRESH[i], merge=MERGE[j], min_area=MIN_AREA[a]))
    return best, per_ext


# ---------------------------------------------------------------- 5. oracles

def oracle(d, best, meta, gt, pool, tr_dir):
    """PQ if each component were kept exactly when its true mean matched IoU pays.

    The ceiling for any keep-rule, learned or image-based, on these candidates;
    lower detection thresholds give it more of them to choose from.
    """
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    stems = [Path(f).stem for f in meta["val_files"]]
    ext = tuple(best.get("ext", NO_EXT))
    out = []
    for t in sorted(set(ORACLE_T) | {best["thresh"]}):
        jobs = [(str(d), best["variant"], stems[f], sets, t, best["merge"], 100, str(tr_dir), ext, False)
                for f, sets in by_frame.items()]
        frames = list(pool.imap(keep_rule_job, jobs, chunksize=2))
        y = np.concatenate([f[2].mean(axis=1) if f[2].shape[1] else np.zeros(len(f[2])) for f in frames])
        cuts = np.round(np.arange(0.0, 0.60, 0.01), 3)
        pq, cnt, c_best = max(((*pq_for(frames, y, c), c) for c in cuts), key=lambda r: r[0])
        out.append(dict(thresh=t, merge=best["merge"], ext=ext, PQ=float(pq), cut=float(c_best),
                        n_components=int(len(y)), **cnt))
        print(f"  oracle keep-rule at t {t} merge {best['merge']}{ext_name(ext)}: PQ {pq:.4f}  "
              f"TP {cnt['TP']} FP {cnt['FP']} FN {cnt['FN']}  ({len(y)} candidates)", flush=True)
    return out


def swap(variant, h, m):
    """The variant with model h replaced by m; unchanged if h is not in it."""
    model, view = variant.split("/")
    if model.startswith("ens:"):
        model = "ens:" + "+".join(m if x == h else x for x in model[4:].split("+"))
    elif model == h:
        model = m
    return f"{model}/{view}"


def post_main(src, d, out_dir, procs, smoke=False):
    """src: where dump_*.tar live; d: where they are unpacked; out_dir: results."""
    t0 = time.time()
    dumped = extract(src, d)
    # a model trained on every photograph has no honest validation maps: it can be
    # ensembled on test, but nothing can be tuned or scored on it here
    have = [m for m in dumped if (d / "dump" / m / "val_plain").exists()]
    if set(dumped) - set(have):
        print(f"test-only models (no validation maps): {sorted(set(dumped) - set(have))}", flush=True)
    meta, gt = ground_truth(smoke)
    print(f"{len(meta['val_files'])} val photographs, {len(gt)} annotation sets; models {have}",
          flush=True)

    results, grids = [], {}

    def score(v, pool):
        b, pq = grid(d, v, meta, gt, pool)
        results.append(b); grids[v] = pq.tolist()
        print(f"  {v:18s} PQ {b['PQ']:.4f}  PQ60 {b['PQ60']:.4f}  SQ {b['SQ']:.3f} RQ {b['RQ']:.3f}  "
              f"TP {b['TP']} FP {b['FP']} FN {b['FN']}  at t {b['thresh']} merge {b['merge']} "
              f"area {b['min_area']}   ({time.time() - t0:.0f}s)", flush=True)
        return b

    with mp.Pool(procs) as pool:
        single = {m: {v: score(f"{m}/{v}", pool) for v in ("plain", "d4")} for m in have}
        if len(have) >= 2:
            everyone = "+".join(have)
            score(f"ens:{everyone}/plain", pool); score(f"ens:{everyone}/d4", pool)
        # with several models the mean of all of them need not be the best mix:
        # also try the strongest two and three, ranked by their own D4 score
        ranked = sorted(have, key=lambda m: -single[m]["d4"]["PQ"])
        for k in range(2, len(have)):
            score(f"ens:{'+'.join(ranked[:k])}/d4", pool)
        grid_best = max(results, key=lambda r: r["PQ"])

        # 4. the outline on its own setting, for the best single model and the best mix
        best_single = max((single[m]["d4"] for m in have), key=lambda r: r["PQ"])
        ext_results, per_ext = [], {}
        for v in dict.fromkeys([best_single["variant"], grid_best["variant"]]):
            b, rows = extent_grid(d, v, meta, gt, pool)
            ext_results.append(b); per_ext[v] = rows
            print(f"\noutline settings on {v} (best detection point for each; {time.time() - t0:.0f}s):",
                  flush=True)
            for r in rows:
                print(f"    {str(tuple(r['ext'])):16s} PQ {r['PQ']:.4f}  SQ {r['SQ']:.3f} RQ {r['RQ']:.3f}  "
                      f"at t {r['thresh']} merge {r['merge']} area {r['min_area']}", flush=True)
        ext_best = max(ext_results, key=lambda r: r["PQ"])
        use_ext = ext_best["PQ"] > grid_best["PQ"] + 0.005     # below that it is noise, not a gain
        best = ext_best if use_ext else grid_best
        print(f"\noutline: best {ext_best['variant']}{ext_name(ext_best['ext'])} PQ {ext_best['PQ']:.4f} "
              f"(SQ {ext_best['SQ']:.3f}) vs grid best {grid_best['PQ']:.4f} -> "
              f"{'used' if use_ext else 'not used'}", flush=True)

        _, tr_dir, _ = run.find_data()
        rule, model = keep_rule(d, best, meta, gt, pool, tr_dir)
        print(f"\nlearned keep-rule on {best['variant']}{ext_name(best.get('ext', NO_EXT))}: "
              f"PQ {rule['PQ']:.4f} at cut {rule['cut']} (without it {best['PQ']:.4f}; "
              f"{rule['n_components']} components; TP {rule['TP']} FP {rule['FP']} FN {rule['FN']})",
              flush=True)
        # 5. how much a perfect keep-rule would add, i.e. what an image-based one could reach
        print("\noracles:", flush=True)
        oracles = oracle(d, best, meta, gt, pool, tr_dir)

    def csv_name(variant, ext=NO_EXT, rule=False):
        return (f"submission_{variant.replace('/', '_').replace(':', '-').replace('+', '-')}"
                f"{ext_name(ext)}{'_rule' if rule else ''}.csv")

    # one CSV per variant at its own best grid point, so the choice stays open
    files = {}
    for b in results + [b for b in ext_results if tuple(b["ext"]) != NO_EXT]:
        ext = tuple(b.get("ext", NO_EXT))
        out = out_dir / csv_name(b["variant"], ext)
        n, mean = write_submission(d, b, None, None, meta, out)
        files[b["variant"] + ext_name(ext)] = dict(file=out.name, rows=n, per_image=mean)
    use_rule = rule["PQ"] > best["PQ"] + 0.005          # below that it is noise, not a gain
    if use_rule:
        out = out_dir / csv_name(best["variant"], tuple(best.get("ext", NO_EXT)), True)
        n, mean = write_submission(d, best, rule, model, meta, out)
        files[best["variant"] + ext_name(best.get("ext", NO_EXT)) + "+rule"] = dict(
            file=out.name, rows=n, per_image=mean)

    # 6. test-only models stand in for the validated model whose name they extend
    # (r50all for r50), with everything that model was tuned to
    stand_ins = {}
    for m in sorted(set(dumped) - set(have)):
        h = max((h for h in have if m.startswith(h)), key=len, default=None)
        if h is None:
            continue
        stand_ins[m] = h
        for b, with_rule in [(single[h]["d4"], False), (best, use_rule)]:
            if swap(b["variant"], h, m) == b["variant"]:
                continue                                  # h is not in this mix
            b2 = dict(b, variant=swap(b["variant"], h, m))
            ext = tuple(b2.get("ext", NO_EXT))
            if b2["variant"] + ext_name(ext) + ("+rule" if with_rule else "") in files:
                continue                                  # the best is that model's own D4
            out = out_dir / csv_name(b2["variant"], ext, with_rule)
            n, mean = write_submission(d, b2, rule if with_rule else None, model if with_rule else None,
                                       meta, out)
            files[b2["variant"] + ext_name(ext) + ("+rule" if with_rule else "")] = dict(
                file=out.name, rows=n, per_image=mean, stands_in_for=b["variant"])

    for k, v in files.items():
        print(f"  {k:16s} -> {v['file']}: {v['rows']} rows, {v['per_image']:.1f} per image", flush=True)
    json.dump(dict(results=results, keep_rule=rule, used_keep_rule=use_rule, best=best, files=files,
                   extent=dict(results=ext_results, per_ext=per_ext, used=use_ext, axes=EXTENTS, grow=GROW),
                   oracle=oracles, stand_ins=stand_ins,
                   grid_axes=dict(thresh=THRESH, merge=MERGE, min_area=MIN_AREA), grids=grids),
              open(out_dir / "post_local.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return files


def kaggle_post():
    """MODE = "post" on Kaggle: the dump kernel's tars arrive under /kaggle/input."""
    post_main(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, procs=run.os.cpu_count() or 4)


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "out_kaggle"))
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="the local 10-frame dump")
    args = ap.parse_args()
    d = Path(args.dir)
    files = post_main(d, d, d, args.procs, args.smoke)
    for v in files.values():
        subprocess.call([sys.executable, str(ROOT / "src" / "verify_submission.py"), str(d / v["file"]),
                         "--test-dir", str(run.find_data()[2])])


if __name__ == "__main__":
    main()
