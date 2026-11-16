"""Qwen3-VL as the decision model: each candidate's picture and its record, together.

qwen_finetune.py taught Qwen3-VL-2B to judge a candidate from its picture alone:
AUC 0.795 on fold 0 against the 10-feature GBM's 0.814, and a 50/50 blend
+0.0025 [-0.002, +0.007] over the GBM -- close, not proven. Here the model also
reads the candidate's record as text: date, heliographic latitude, distance
from the central meridian, long-axis angle to east-west, length, width, area,
contrast with its surroundings, the detector's confidence. Two arms, one per T4,
trained on the same 4911 candidates of folds 1-4 (each fold's own R50 maps):

  img   the picture only (as qwen_finetune.py)
  geo   the picture and the record

and both, unlike qwen_finetune.py, also score what a submission needs:

  fold 0, R50      the 1340 candidates of the earlier comparison (AUC, PQ)
  fold 0, ens0     the five-model ensemble trained without fold 0: the rule's
                   cut is set here, as in kfold.py (nested and best)
  test, ten maps   all ten fold models averaged, as submission_kf10_rule

A GBM on the same 14 numbers (10 hand + 4 geometry) is trained alongside, and
every scorer is put through the same cut and the same test CSV writer. CSVs are
checked with src/verify_submission.py after download, before any upload.
"""
from __future__ import annotations

import json, math, multiprocessing as mp, pickle, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen_finetune as QF  # noqa: E402  -- qwen_probe first: env, run (torch before cv2)
from qwen_finetune import QP, run, torch, cv2, np  # noqa: E402
import post_local as P  # noqa: E402

TEN = "ens:r50+r50f1+r50f2+r50f3+r50f4+full+r34f1+r34f2+r34f3+r34f4/d4"
ENS0 = "ens:flip+full+r50+solar+union/d4"
FOLD_MODEL = {0: "r50", 1: "r50f1", 2: "r50f2", 3: "r50f3", 4: "r50f4"}
MAPS = Path("/tmp/qwen_geo_maps") if run.ON_KAGGLE else HERE / "out" / "qwen_geo_maps"
OUT = QP.OUT / "geo"
RES = QP.RES
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
ARMS = ("img", "geo")
MODEL = "Qwen/Qwen3-VL-2B-Instruct"


# ---------------------------------------------------------------- candidates and records

def record(stem, img, p, lab, j, sl):
    """The candidate's numbers: 14 for the GBM, and the same as a sentence for Qwen."""
    pad = 10
    s = (slice(max(sl[0].start - pad, 0), sl[0].stop + pad), slice(max(sl[1].start - pad, 0), sl[1].stop + pad))
    m = lab[s] == j
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean() + s[0].start, xs.mean() + s[1].start
    cov = np.cov(np.stack([ys, xs])) if ys.size > 2 else np.eye(2)
    w, V = np.linalg.eigh(cov)
    vy, vx = V[:, int(np.argmax(w))]
    ang = math.degrees(math.atan2(-vy, vx)) % 180.0            # long axis against east-west
    length, width = math.sqrt(12 * max(w.max(), 0)), math.sqrt(12 * max(w.min(), 0))
    dx, dy = (cx - run.CX) / run.R_DISK, -(cy - run.CY) / run.R_DISK
    z = math.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))
    b = math.radians(run.b0_deg(stem))
    lat = math.degrees(math.asin(max(-1.0, min(1.0, dy * math.cos(b) + z * math.sin(b)))))
    lon = math.degrees(math.atan2(dx, z * math.cos(b) - dy * math.sin(b)))
    ring = (cv2.dilate(m.astype(np.uint8), RING) > 0) & ~m
    iv, rv = img[s][m].astype(np.float32), img[s][ring].astype(np.float32)
    contrast = float((rv.mean() - iv.mean()) / max(rv.mean(), 1.0)) if rv.size else 0.0
    pv = p[s][m].astype(np.float32) / 255.0
    date = f"{stem[0:4]}-{stem[4:6]}-{stem[6:8]}"
    text = (f"Detector record for this candidate: observed {date}; heliographic latitude {lat:+.0f} deg, "
            f"{abs(lon):.0f} deg {'west' if lon >= 0 else 'east'} of the central meridian "
            f"({math.hypot(dx, dy):.2f} of the way from disk centre to limb); long axis {ang:.0f} deg from east-west; "
            f"length {length:.0f} px, width {width:.0f} px, area {int(m.sum())} px (1 px is about 1.1 Mm at disk centre); "
            f"{100 * contrast:.0f}% darker than its surroundings; detector confidence mean {pv.mean():.2f}, max {pv.max():.2f}.")
    geo = [math.sin(math.radians(lat)), lon / 90.0, math.cos(math.radians(2 * ang)), math.sin(math.radians(2 * ang))]
    return text, geo


def frame_job(job):
    """One frame's candidates at the rule's setting: crops, records, features, matched IoU, masks."""
    variant, split, stem, path, sets = job
    from scipy import ndimage
    p = P.load_map(MAPS, variant, split, stem)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lab, area = run.components(p, QP.T, QP.K)
    ids = [j for j in range(1, len(area)) if area[j] >= QP.AMIN]
    out = dict(stem=stem, crops=[], text=[], masks=[], X=np.zeros((0, 14), np.float32),
               Y=np.zeros((0, len(sets or []))), n_gt=[len(c) for _, c in (sets or [])])
    if not ids:
        return out
    boxes = ndimage.find_objects(lab)
    geo = []
    for j in ids:
        sy, sx = boxes[j - 1]
        side = int(np.clip(max(sy.stop - sy.start, sx.stop - sx.start) + 2 * QP.PAD, *QP.WIN))
        y0 = int(np.clip((sy.start + sy.stop - side) // 2, 0, run.FULL - side))
        x0 = int(np.clip((sx.start + sx.stop - side) // 2, 0, run.FULL - side))
        out["crops"].append((img[y0:y0 + side, x0:x0 + side].copy(), lab[y0:y0 + side, x0:x0 + side] == j))
        out["masks"].append((sy.start, sx.start, lab[sy, sx] == j))
        t, g = record(stem, img, p, lab, j, (sy, sx))
        out["text"].append(t)
        geo.append(g)
    out["X"] = np.concatenate([P.comp_features(p, img, lab, area, ids), np.array(geo, np.float32)], 1)
    if sets is not None:
        out["Y"] = P.matched_iou(lab, area, ids, sets)
    return out


def fold_jobs(fold, variant):
    run.set_fold(fold)
    if not (run.CACHE / "val_gt.pkl").exists():
        run.prepare(False, with_train=False)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    tr_dir = run.find_data()[1]
    by = {}
    for i, g in enumerate(gt):
        by.setdefault(g["frame"], []).append((i, g["crops"]))
    return [(variant, "val", Path(meta["val_files"][f]).stem, str(tr_dir / meta["val_files"][f]), sets)
            for f, sets in sorted(by.items())]


def build_sets(procs=4):
    path = OUT / "sets.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    print("models:", P.extract(Path("/kaggle/input") if run.ON_KAGGLE else HERE.parent, MAPS), flush=True)
    te_dir = run.find_data()[2]
    jobs = {f"train{f}": fold_jobs(f, f"{FOLD_MODEL[f]}/d4") for f in (1, 2, 3, 4)}
    jobs["fold0"] = fold_jobs(0, f"{FOLD_MODEL[0]}/d4")
    jobs["ens0"] = [(ENS0, *j[1:]) for j in jobs["fold0"]]
    jobs["test"] = [(TEN, "test", p.stem, str(p), None) for p in sorted(te_dir.glob("*.jpeg"))]
    sets = {}
    with mp.Pool(procs) as pool:
        for name, js in jobs.items():
            t0 = time.time()
            sets[name] = list(pool.imap(frame_job, js, chunksize=4))
            print(f"{name}: {len(js)} frames, {sum(len(r['crops']) for r in sets[name])} candidates, "
                  f"{time.time() - t0:.0f}s", flush=True)
    sets["train"] = [r for f in (1, 2, 3, 4) for r in sets.pop(f"train{f}")]
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(sets, open(path, "wb"))
    return sets


# ---------------------------------------------------------------- one arm on one GPU

def prompt_text(proc, rec):
    q = QP.PROMPT if rec is None else QP.PROMPT.replace("Does the red outline", rec + "\nDoes the red outline")
    return proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                                    add_generation_prompt=True, tokenize=False)


def arm_main(arm, limit=0):
    from PIL import Image
    random.seed(0); torch.manual_seed(0)
    sets = build_sets()
    use_rec = arm == "geo"
    items = [(im, m, float((r["Y"][i] > 0).mean()), r["text"][i] if use_rec else None)
             for r in sets["train"] for i, (im, m) in enumerate(r["crops"])]
    if limit:
        random.shuffle(items); items = items[:limit]
    model, proc = QF.load(MODEL)
    yes, no = QF.yes_no(proc)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    accum, steps = 8, max(len(items) // 8, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler, gov = torch.amp.GradScaler(), QF.governor()
    model.train()
    t0, run_loss, n = time.time(), 0.0, 0
    order = list(range(len(items)))
    random.shuffle(order)
    for i, k in enumerate(order):
        im, m, t, rec = items[k]
        with gov.step():
            s = QF.margin(model, proc, prompt_text(proc, rec), Image.fromarray(QF.panel(im, m, aug=True)), yes, no)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(t, device=s.device)) / accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * accum; n += 1
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            torch.cuda.synchronize()
        if (i + 1) % 500 == 0:
            print(f"[{arm}] {i + 1}/{len(items)}  loss {run_loss / n:.4f}  {(time.time() - t0) / (i + 1):.2f} s each", flush=True)
            run_loss, n = 0.0, 0
    model.eval()
    scores = {}
    with torch.no_grad():
        for name in ("fold0", "ens0", "test"):
            rows = sets[name][:3] if limit else sets[name]
            scores[name] = np.array([float(QF.margin(model, proc, prompt_text(proc, r["text"][i] if use_rec else None),
                                                     Image.fromarray(QF.panel(im, m)), yes, no))
                                     for r in rows for i, (im, m) in enumerate(r["crops"])])
            print(f"[{arm}] scored {name}: {scores[name].size}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    np.savez(RES / f"geo_scores_{arm}{'_smoke' if limit else ''}.npz", **scores)


# ---------------------------------------------------------------- evaluation and CSVs

def counts(rows, s, cut):
    c, k = np.zeros((len(rows), 4)), 0
    for n, r in enumerate(rows):
        m = len(r["crops"])
        keep = s[k:k + m] >= cut
        k += m
        y = r["Y"][keep]
        for q, g in enumerate(r["n_gt"]):
            tp = (y[:, q] > 0).sum()
            c[n] += (tp, keep.sum() - tp, g - tp, y[:, q].sum())
    return c


def cuts_of(s):
    return np.unique(np.quantile(s, np.linspace(0.0, 0.95, 96)))


def best_cut(rows, s):
    cs = cuts_of(s)
    pq = [float(P.pq_of(counts(rows, s, c).sum(axis=0))) for c in cs]
    b = int(np.argmax(pq))
    return float(cs[b]), round(pq[b], 4)


def nested(rows, s):
    """Per-photograph counts with the cut chosen on the other four fifths."""
    per = [len(r["crops"]) for r in rows]
    start = np.concatenate([[0], np.cumsum(per)])
    folds = np.arange(len(rows)) % 5
    out = np.zeros((len(rows), 4))
    for f in range(5):
        tr = [n for n in range(len(rows)) if folds[n] != f]
        sub = [rows[n] for n in tr]
        ss = np.concatenate([s[start[n]:start[n + 1]] for n in tr])
        cut, _ = best_cut(sub, ss)
        for n in np.flatnonzero(folds == f):
            out[n] = counts([rows[n]], s[start[n]:start[n + 1]], cut)[0]
    return out


def boot(a, b):
    rng = np.random.default_rng(0)
    d = [float(P.pq_of(a[i].sum(axis=0)) - P.pq_of(b[i].sum(axis=0))) for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    return dict(mean=round(float(np.mean(d)), 4), lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))


def write_csv(rows, s, cut, path):
    lines, per, k = [], [], 0
    for r in rows:
        m = len(r["crops"])
        keep = [i for i in range(m) if s[k + i] >= cut]
        k += m
        for n, i in enumerate(keep, 1):
            y0, x0, mm = r["masks"][i]
            full = np.zeros((run.FULL, run.FULL), np.uint8)
            full[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
            enc = run.mu.encode(np.asfortranarray(full))
            lines.append(f"{r['stem']}_{n},{enc['counts'].decode('ascii')}")
        per.append(len(keep))
    Path(path).write_text("filament_id,segmentation_rle\n" + "\n".join(lines) + "\n")
    return dict(file=Path(path).name, rows=len(lines), per_image=round(float(np.mean(per)), 3), empty=int(sum(p == 0 for p in per)))


def evaluate(sets, q):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([r["X"] for r in sets["train"]])
    y = np.concatenate([r["Y"].mean(axis=1) for r in sets["train"]])
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                        l2_regularization=1.0, random_state=0).fit(X, y)
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    scorers = {}
    for name in ("fold0", "ens0", "test"):
        g = gbm.predict(np.concatenate([r["X"] for r in sets[name]]))
        sc = dict(gbm14=g)
        for arm in q:
            sc[f"qwen_{arm}"] = sig(q[arm][name])
            sc[f"blend_{arm}"] = 0.5 * g + 0.5 * sig(q[arm][name])
        scorers[name] = sc
    hit = np.concatenate([(r["Y"].mean(axis=1) > 0) for r in sets["fold0"]]).astype(int)
    res = dict(train_candidates=int(len(y)), candidates={k: int(sum(len(r["crops"]) for r in v)) for k, v in sets.items()})
    res["fold0_auc"] = {k: round(float(roc_auc_score(hit, v)), 4) for k, v in scorers["fold0"].items()}
    res["fold0_pq_best_cut"] = {k: best_cut(sets["fold0"], v)[1] for k, v in scorers["fold0"].items()}
    nest = {k: nested(sets["ens0"], v) for k, v in scorers["ens0"].items()}
    res["ens0_nested_pq"] = {k: round(float(P.pq_of(v.sum(axis=0))), 4) for k, v in nest.items()}
    res["ens0_best"] = {k: best_cut(sets["ens0"], v) for k, v in scorers["ens0"].items()}
    res["ens0_minus_gbm14"] = {k: boot(v, nest["gbm14"]) for k, v in nest.items() if k != "gbm14"}
    res["files"] = {}
    for k in scorers["test"]:
        cut = res["ens0_best"][k][0]
        res["files"][k] = write_csv(sets["test"], scorers["test"][k], cut, RES / f"submission_qgeo_{k}.csv")
    return res


def kaggle_main():
    import subprocess
    t0 = time.time()
    build_sets(procs=max(2, (run.os.cpu_count() or 4)))
    print(f"candidates ready in {time.time() - t0:.0f}s", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(RES / f"log_geo_{arm}.txt", "w")
        env = dict(run.os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (RES / f"log_geo_{arm}.txt").read_text().splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][:160] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n" + (RES / f"log_geo_{arm}.txt").read_text()[-2000:], flush=True)
    q = {arm: dict(np.load(RES / f"geo_scores_{arm}.npz")) for arm, p in procs if (RES / f"geo_scores_{arm}.npz").exists()}
    res = evaluate(build_sets(), q)
    json.dump(res, open(RES / "qwen_geo.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ARMS:
        arm_main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        kaggle_main()
