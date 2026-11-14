"""A second model family for the keep-rule: YOLO11-seg, and whether it sees each candidate.

Every model in our submissions is a U-Net (5 R50 + 5 R34), so their mistakes are
alike. The public notebook at 0.40 (prvsiyan, 2026-11-10) trains nothing: it
keeps a filament only when three of five *different* public model families drew
it. Here the second family is our own, trained per fold like the U-Nets:

  GPU 0   YOLO11l-seg on fold 0's 566 training photographs -> fold 0 val + test
  GPU 1   the same on fold 1's                              -> fold 1 val + test

Each photograph is one training image whose targets are the annotators'
filaments merged across annotators (mask IoU >= 0.3 is one filament; the outline
is the union), so the detector never sees the same frame twice with conflicting
labels. For each U-Net candidate, four numbers say whether YOLO drew it too:

  y_iou    best IoU with a YOLO instance (conf >= 0.05)
  y_conf   that instance's confidence
  y_near   the highest confidence among instances overlapping it at IoU >= 0.3
  y_cover  the share of the candidate inside YOLO's confident (>= 0.25) masks

and they are tested three ways, on fold 0's five-model ensemble (ens0) with the
cut chosen nested, as in kfold.py:

  A14 / B18   GBMs trained on the same fold-1 candidates (r50f1 maps): the 14
              numbers of qwen_geo.py, and those plus the four. B18 - A14 is the
              clean answer to "does YOLO add information?"
  gbm14+vote  the 4-fold GBM14 (the current rule) plus lam * y_near, with lam
              and the cut chosen together, nested. Needs YOLO only on fold 0 and
              test, so it is the cheap way into a submission.
  YOLO alone  its own instances (made disjoint, highest confidence first)
              scored on fold 0 at each confidence cut: how good the family is.

Candidates come from qwen_geo.build_sets (the same frames, maps and records as
the Qwen test), so every number here is comparable with qwen_geo.json.
"""
from __future__ import annotations

import collections, json, os, pickle, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen_geo as QG  # noqa: E402  -- imports run (torch before cv2), post_local
from qwen_geo import run, np, cv2, P  # noqa: E402

FULL = run.FULL
SMOKE = os.environ.get("FIL_YOLO_SMOKE") == "1"
# v2 trained folds 0 and 1 and evaluated them; v3 trains folds 2-4 only, so that every
# photograph has an out-of-fold YOLO and kfold.py can give the keep-rule its four numbers
FOLDS = tuple(int(f) for f in os.environ.get("FIL_YOLO_FOLDS", "0,1").split(","))
EVALUATE = set(FOLDS) == {0, 1}
WEIGHTS = "yolo11l-seg.pt"
IMGSZ = 1280
EPOCHS = 1 if SMOKE else 60
BATCH = 4
CONF_MIN, NEAR_IOU, STRONG = 0.05, 0.3, 0.25
MERGE_IOU = 0.3
YDIR = Path("/tmp/yolo") if run.ON_KAGGLE else HERE / "out" / "yolo"
RES = run.WORK / "yolo" if run.ON_KAGGLE else YDIR / "res"
YCOLS = ["y_iou", "y_conf", "y_near", "y_cover"]


# ---------------------------------------------------------------- labels: one target set per photograph

def annotations():
    ann_path, tr_dir, te_dir = run.find_data()
    coco = json.load(open(ann_path))
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    return fname, by_id, tr_dir, te_dir


def merged_polygons(sids, by_id):
    """Every annotator's filaments of one photograph, merged across annotators.

    Two outlines by different annotators at mask IoU >= MERGE_IOU are one filament
    (union-find, strongest pairs first, never joining two outlines of the same
    annotator); its target is the union, as one polygon in [0, 1] coordinates.
    """
    rec = [(k, a) for k, sid in enumerate(sids) for a in by_id[sid] if a["segmentation"]]
    if not rec:
        return []
    rles = [run.mu.merge(run.mu.frPyObjects(a["segmentation"], FULL, FULL)) for _, a in rec]
    iou = np.asarray(run.mu.iou(rles, rles, [0] * len(rles)))
    parent, who = list(range(len(rec))), [{rec[i][0]} for i in range(len(rec))]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = sorted(((iou[i, j], i, j) for i in range(len(rec)) for j in range(i + 1, len(rec))
                    if rec[i][0] != rec[j][0] and iou[i, j] >= MERGE_IOU), reverse=True)
    for _, i, j in pairs:
        a, b = find(i), find(j)
        if a != b and who[a].isdisjoint(who[b]):
            parent[b] = a
            who[a] |= who[b]
    groups = collections.defaultdict(list)
    for i in range(len(rec)):
        groups[find(i)].append(i)
    polys = []
    for idx in groups.values():
        m = run.mu.decode(run.mu.merge([rles[i] for i in idx])).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)[:, 0, :]
        if len(c) >= 3:
            polys.append(c.astype(np.float64) / FULL)
    return polys


def yolo_data(fold):
    """The fold's training photographs as a YOLO-seg dataset; 40 of them double as its val."""
    fname, by_id, tr_dir, _ = annotations()
    train_files, _ = run.photograph_split(fname, fold)
    if SMOKE:
        train_files = train_files[:12]
    sets_of = collections.defaultdict(list)
    for sid, f in fname.items():
        sets_of[f].append(sid)
    root = YDIR / f"data_f{fold}"
    mon = sorted(random.Random(fold).sample(train_files, min(40, len(train_files) // 3)))
    for split, files in (("train", train_files), ("val", mon)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = root / "images" / split / f
            if not dst.exists():
                try:
                    os.symlink(tr_dir / f, dst)
                except OSError:
                    shutil.copy2(tr_dir / f, dst)
            lines = ["0 " + " ".join(f"{v:.5f}" for v in p.ravel()) for p in merged_polygons(sets_of[f], by_id)]
            (root / "labels" / split / (Path(f).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return root


# ---------------------------------------------------------------- one fold on one GPU

def to_rle(m):
    return run.mu.encode(np.asfortranarray(m.astype(np.uint8)))


def worker(fold):
    import ultralytics
    from ultralytics import YOLO
    t0 = time.time()
    print(f"[f{fold}] ultralytics {ultralytics.__version__}, {WEIGHTS}, imgsz {IMGSZ}, {EPOCHS} epochs", flush=True)
    root = yolo_data(fold)
    print(f"[f{fold}] data ready in {time.time() - t0:.0f}s", flush=True)
    model = YOLO(WEIGHTS)
    model.train(data=str(root / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, workers=3,
                cache="ram", mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, degrees=180.0,
                translate=0.05, scale=0.25, shear=0.0, perspective=0.0, flipud=0.5, fliplr=0.5,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.15, cos_lr=True, patience=1000,
                project=str(YDIR / "runs"), name=f"f{fold}", exist_ok=True, plots=False, seed=fold, amp=True)
    w = YDIR / "runs" / f"f{fold}" / "weights" / "last.pt"
    RES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(w, RES / f"yolo_f{fold}.pt")
    print(f"[f{fold}] trained in {(time.time() - t0) / 60:.0f} min", flush=True)

    model = YOLO(str(w))
    fname, _, tr_dir, te_dir = annotations()
    _, val_files = run.photograph_split(fname, fold)
    test = sorted(te_dir.glob("*.jpeg"))
    if SMOKE:
        val_files, test = val_files[:4], test[:3]
    preds = {}
    for split, paths in (("val", [tr_dir / f for f in val_files]), ("test", test)):
        for path in paths:
            r = model.predict(source=str(path), imgsz=IMGSZ, conf=CONF_MIN, iou=0.6, max_det=100,
                              retina_masks=True, verbose=False)[0]
            inst = []
            if r.masks is not None:
                ms = (r.masks.data > 0.5).cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                for mk, c in zip(ms, cf):
                    if mk.shape != (FULL, FULL):
                        mk = cv2.resize(mk.astype(np.uint8), (FULL, FULL), interpolation=cv2.INTER_NEAREST) > 0
                    if mk.any():
                        inst.append((float(c), to_rle(mk)))
            preds[(split, Path(path).stem)] = inst
    pickle.dump(preds, open(RES / f"yolo_preds_f{fold}.pkl", "wb"))
    n = [len(v) for v in preds.values()]
    print(f"[f{fold}] predicted {len(n)} frames, {np.mean(n):.1f} instances each, "
          f"{(time.time() - t0) / 60:.0f} min in all", flush=True)


# ---------------------------------------------------------------- features and scores

def full_rle(y0, x0, mm):
    full = np.zeros((FULL, FULL), np.uint8)
    full[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
    return to_rle(full)


def yolo_feats(rows, preds, split):
    """(n_candidates, 4) per row: does the YOLO family draw this U-Net candidate?"""
    out = []
    for r in rows:
        inst = preds.get((split, r["stem"]), [])
        f = np.zeros((len(r["masks"]), 4), np.float32)
        if len(r["masks"]) and inst:
            conf = np.array([c for c, _ in inst])
            yr = [e for _, e in inst]
            cr = [full_rle(*m) for m in r["masks"]]
            iou = np.asarray(run.mu.iou(cr, yr, [0] * len(yr))).reshape(len(cr), len(yr))
            b = iou.argmax(1)
            f[:, 0] = iou.max(1)
            f[:, 1] = conf[b]
            f[:, 2] = np.where(iou >= NEAR_IOU, conf[None, :], 0.0).max(1)
            strong = [e for c, e in inst if c >= STRONG]
            if strong:
                u = run.mu.merge(strong)
                for i, e in enumerate(cr):
                    f[i, 3] = run.mu.area(run.mu.merge([e, u], intersect=1)) / max(run.mu.area(e), 1)
        out.append(f)
    return out


def disjoint(inst, cut):
    """YOLO's own instances above a confidence cut, overlaps given to the more confident."""
    taken, keep = None, []
    for c, e in sorted(inst, key=lambda t: -t[0]):
        if c < cut:
            break
        if taken is not None:
            m = run.mu.decode(e).astype(bool) & ~run.mu.decode(taken).astype(bool)
            if not m.any():
                continue
            e = to_rle(m)
        keep.append(e)
        taken = e if taken is None else run.mu.merge([taken, e])
    return keep


def gt_sets(fold):
    """Fold's validation annotation sets as RLE lists, by photograph stem."""
    run.set_fold(fold)
    if not (run.CACHE / "val_gt.pkl").exists():
        run.prepare(False, with_train=False)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    by = collections.defaultdict(list)
    for g in gt:
        by[Path(meta["val_files"][g["frame"]]).stem].append([full_rle(y0, x0, m) for y0, x0, m in g["crops"]])
    return by


def yolo_alone(preds, sets_by_stem, cuts=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    tot = {c: np.zeros(4) for c in cuts}
    for stem, sets in sets_by_stem.items():
        inst = preds.get(("val", stem), [])
        for c in cuts:
            pr = disjoint(inst, c)
            for g in sets:
                if not pr or not g:
                    tot[c] += (0, len(pr), len(g), 0.0)
                    continue
                iou = np.asarray(run.mu.iou(pr, g, [0] * len(g))).reshape(len(pr), len(g))
                hit = iou > 0.5
                tot[c] += (hit.sum(), (~hit.any(1)).sum(), (~hit.any(0)).sum(), iou[hit].sum())
    return {f"{c:.1f}": round(float(P.pq_of(v)), 4) for c, v in tot.items()}


def nested_choice(rows, scores):
    """Like QG.nested, but the inner fit picks the best (scorer, cut) pair."""
    per = [len(r["crops"]) for r in rows]
    start = np.concatenate([[0], np.cumsum(per)])
    folds = np.arange(len(rows)) % 5
    out, picks = np.zeros((len(rows), 4)), []
    for f in range(5):
        tr = [n for n in range(len(rows)) if folds[n] != f]
        sub = [rows[n] for n in tr]
        best = max(((QG.best_cut(sub, np.concatenate([s[start[n]:start[n + 1]] for n in tr])), k)
                    for k, s in scores.items()), key=lambda t: t[0][1])
        (cut, _), k = best
        picks.append(k)
        for n in np.flatnonzero(folds == f):
            out[n] = QG.counts([rows[n]], scores[k][start[n]:start[n + 1]], cut)[0]
    return out, picks


def evaluate(sets, preds):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score

    def gbm(X, y):
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                             l2_regularization=1.0, random_state=0).fit(X, y)
    fname = annotations()[0]
    f1 = {Path(f).stem for f in run.photograph_split(fname, 1)[1]}
    tr1 = [r for r in sets["train"] if r["stem"] in f1]
    F = {"tr1": yolo_feats(tr1, preds[1], "val"), "fold0": yolo_feats(sets["fold0"], preds[0], "val"),
         "ens0": yolo_feats(sets["ens0"], preds[0], "val"), "test": yolo_feats(sets["test"], preds[0], "test")}
    rows = {"tr1": tr1, "fold0": sets["fold0"], "ens0": sets["ens0"], "test": sets["test"]}
    X = {k: np.concatenate([r["X"] for r in v]) for k, v in rows.items()}
    Xy = {k: np.concatenate([X[k], np.concatenate(F[k])], 1) for k in rows}
    y1 = np.concatenate([r["Y"].mean(axis=1) for r in tr1])
    yall = np.concatenate([r["Y"].mean(axis=1) for r in sets["train"]])
    a14, b18 = gbm(X["tr1"], y1), gbm(Xy["tr1"], y1)
    g_all = gbm(np.concatenate([r["X"] for r in sets["train"]]), yall)

    res = dict(train_rows_fold1=len(tr1), candidates={k: int(len(X[k])) for k in rows})
    hit = np.concatenate([(r["Y"].mean(axis=1) > 0) for r in sets["fold0"]]).astype(int)
    f0 = np.concatenate(F["fold0"])
    res["fold0_auc"] = {c: round(float(roc_auc_score(hit, f0[:, i])), 4) for i, c in enumerate(YCOLS)}
    res["fold0_auc"].update(A14=round(float(roc_auc_score(hit, a14.predict(X["fold0"]))), 4),
                            B18=round(float(roc_auc_score(hit, b18.predict(Xy["fold0"]))), 4),
                            gbm14_all=round(float(roc_auc_score(hit, g_all.predict(X["fold0"]))), 4))
    e = np.concatenate(F["ens0"])
    sc = {"A14": a14.predict(X["ens0"]), "B18": b18.predict(Xy["ens0"]), "gbm14_all": g_all.predict(X["ens0"])}
    nest = {k: QG.nested(sets["ens0"], v) for k, v in sc.items()}
    lams = (0.1, 0.2, 0.3, 0.5)
    votes = {lam: sc["gbm14_all"] + lam * e[:, 2] for lam in lams}
    nest["gbm14+vote"], picks = nested_choice(sets["ens0"], votes)
    res["ens0_nested_pq"] = {k: round(float(P.pq_of(v.sum(axis=0))), 4) for k, v in nest.items()}
    res["vote_lambda_picked"] = picks
    res["ens0_best"] = {k: QG.best_cut(sets["ens0"], v) for k, v in sc.items()}
    res["ens0_best"].update({f"vote{lam}": QG.best_cut(sets["ens0"], v) for lam, v in votes.items()})
    res["B18_minus_A14"] = QG.boot(nest["B18"], nest["A14"])
    res["vote_minus_gbm14_all"] = QG.boot(nest["gbm14+vote"], nest["gbm14_all"])
    res["yolo_alone_fold0_pq"] = yolo_alone(preds[0], gt_sets(0))

    t = np.concatenate(F["test"])
    lam = collections.Counter(picks).most_common(1)[0][0]
    tests = {"A14": a14.predict(X["test"]), "B18": b18.predict(Xy["test"]), "gbm14_all": g_all.predict(X["test"]),
             f"vote{lam}": g_all.predict(X["test"]) + lam * t[:, 2]}
    res["files"] = {}
    RES.mkdir(parents=True, exist_ok=True)
    for k, s in tests.items():
        cut = res["ens0_best"][k][0]
        res["files"][k] = QG.write_csv(sets["test"], s, cut, RES / f"submission_yolo_{k}.csv")
    return res


# ---------------------------------------------------------------- Kaggle entry

def kaggle_main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    import torch
    gpus = list(range(max(torch.cuda.device_count(), 1)))
    pending, procs, on = list(FOLDS), [], {}

    def launch():
        # one fold per free GPU; a third fold waits for the first to finish
        while pending and gpus:
            fold, g = pending.pop(0), gpus.pop(0)
            log = open(RES / f"log_yolo_f{fold}.txt", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", str(fold)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT)
            procs.append((fold, p))
            on[fold] = g
            print(f"  {(time.time() - t0) / 3600:.2f} h  fold {fold} started on GPU {g}", flush=True)
    launch()
    # the U-Net candidates are CPU work: build them while the detectors train
    sets = QG.build_sets(procs=2) if EVALUATE and not SMOKE else None
    print(f"candidates ready in {time.time() - t0:.0f}s", flush=True)
    last, freed = time.time(), set()
    while pending or any(p.poll() is None for _, p in procs):
        time.sleep(20)
        for fold, p in procs:
            if p.poll() is not None and fold not in freed:
                freed.add(fold)
                gpus.append(on[fold])
        launch()
        if time.time() - last > 600:
            last = time.time()
            for fold, p in procs:
                if p.poll() is not None:
                    continue
                lines = [x for x in (RES / f"log_yolo_f{fold}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  f{fold}: {lines[-1][-160:] if lines else ''}", flush=True)
    for fold, p in procs:
        print(f"\n===== fold {fold} (exit {p.returncode}) =====\n"
              + (RES / f"log_yolo_f{fold}.txt").read_text(errors="replace")[-2500:], flush=True)
    preds = {f: pickle.load(open(RES / f"yolo_preds_f{f}.pkl", "rb")) for f in FOLDS
             if (RES / f"yolo_preds_f{f}.pkl").exists()}
    if SMOKE:
        print("smoke: predictions", {f: len(v) for f, v in preds.items()}, flush=True)
        if 0 in preds:
            g = gt_sets(0)
            print("smoke: yolo alone", yolo_alone(preds[0], {s: g[s] for s in list(g)[:4]}), flush=True)
        return
    if not EVALUATE:
        print("trained folds", sorted(preds), "-- the keep-rule test is kfold.py's", flush=True)
        return
    try:
        res = evaluate(sets, preds)
    except Exception:                        # keep the run green: the predictions are saved, rerun on CPU
        import traceback
        traceback.print_exc()
        return
    json.dump(res, open(RES / "yolo.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
    else:
        kaggle_main()
