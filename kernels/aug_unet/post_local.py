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
    """Components of one val frame at a fixed (thresh, merge), with features and targets."""
    d, variant, stem, sets, t, k, a_min, tr_dir = job
    p = load_map(Path(d), variant, "val", stem)
    img = cv2.imread(str(Path(tr_dir) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    lab, area = run.components(p, t, k)
    ids = [j for j in range(1, len(area)) if area[j] >= a_min]
    if not ids:
        return stem, np.zeros((0, 10), np.float32), np.zeros((0, len(sets))), [len(c) for _, c in sets]
    X = comp_features(p, img, lab, area, ids)
    # per set, each component's matched IoU (0 if it matches nothing there)
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
    return stem, X, Y, [len(c) for _, c in sets]


def keep_rule(d, best, meta, gt, pool, tr_dir):
    from sklearn.ensemble import HistGradientBoostingRegressor
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    stems = [Path(f).stem for f in meta["val_files"]]
    jobs = [(str(d), best["variant"], stems[f], sets, best["thresh"], best["merge"], 100, str(tr_dir))
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

    def pq_for(cut):
        tp = fp = fn = s = 0.0
        start = 0
        for _, Xf, Yf, n_gt in frames:
            keep = oof[start:start + len(Xf)] > cut
            start += len(Xf)
            for s_i, ng in enumerate(n_gt):
                iou = Yf[keep, s_i] if len(Xf) else np.zeros(0)
                t_ = int((iou > 0).sum())
                tp += t_; s += float(iou.sum()); fp += int(keep.sum()) - t_; fn += ng - t_
        den = tp + 0.5 * fp + 0.5 * fn
        return s / den if den else 0.0, dict(TP=int(tp), FP=int(fp), FN=int(fn))

    cuts = np.round(np.arange(0.05, 0.50, 0.025), 3)
    scores = [(c, *pq_for(c)) for c in cuts]
    c_best, pq_best, cnt = max(scores, key=lambda r: r[1])
    final = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=20, random_state=0).fit(X, y)
    return dict(PQ=float(pq_best), cut=float(c_best), n_components=int(len(y)), **cnt,
                curve=[(float(c), float(p)) for c, p, _ in scores]), final


# ---------------------------------------------------------------- 3. submission

def write_submission(d, best, rule, model, meta, out):
    rows, per = [], []
    te_dir = run.find_data()[2]
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
        for n, j in enumerate(ids, 1):
            r = run.mu.encode(np.asfortranarray((lab == j).astype(np.uint8)))
            rows.append(f"{stem}_{n},{r['counts'].decode('ascii')}")
        per.append(len(ids))
    out.write_text("filament_id,segmentation_rle\n" + "".join(r + "\n" for r in rows))
    return len(rows), float(np.mean(per))


def post_main(src, d, out_dir, procs, smoke=False):
    """src: where dump_*.tar live; d: where they are unpacked; out_dir: results."""
    t0 = time.time()
    have = extract(src, d)
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
            score("ens/plain", pool); score("ens/d4", pool)
        # with several models the mean of all of them need not be the best mix:
        # also try the strongest two and three, ranked by their own D4 score
        ranked = sorted(have, key=lambda m: -single[m]["d4"]["PQ"])
        for k in range(2, len(have)):
            score(f"ens:{'+'.join(ranked[:k])}/d4", pool)
        best = max(results, key=lambda r: r["PQ"])
        _, tr_dir, _ = run.find_data()
        rule, model = keep_rule(d, best, meta, gt, pool, tr_dir)
    print(f"\nlearned keep-rule on {best['variant']}: PQ {rule['PQ']:.4f} at cut {rule['cut']} "
          f"(grid best {best['PQ']:.4f}; {rule['n_components']} components; "
          f"TP {rule['TP']} FP {rule['FP']} FN {rule['FN']})", flush=True)

    # one CSV per variant at its own best grid point, so the choice stays open
    files = {}
    for b in results:
        out = out_dir / f"submission_{b['variant'].replace('/', '_').replace(':', '-').replace('+', '-')}.csv"
        n, mean = write_submission(d, b, None, None, meta, out)
        files[b["variant"]] = dict(file=out.name, rows=n, per_image=mean)
    use_rule = rule["PQ"] > best["PQ"] + 0.005          # below that it is noise, not a gain
    if use_rule:
        out = out_dir / f"submission_{best['variant'].replace('/', '_').replace(':', '-').replace('+', '-')}_rule.csv"
        n, mean = write_submission(d, best, rule, model, meta, out)
        files[best["variant"] + "+rule"] = dict(file=out.name, rows=n, per_image=mean)
    for k, v in files.items():
        print(f"  {k:16s} -> {v['file']}: {v['rows']} rows, {v['per_image']:.1f} per image", flush=True)
    json.dump(dict(results=results, keep_rule=rule, used_keep_rule=use_rule, best=best, files=files,
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
