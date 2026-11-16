"""Which outline setting suits FilNet's maps? (11-16)

The keep-rule's candidates come from one (threshold, join radius) pair, chosen for the ImageNet U-Nets' maps
(0.6, 3). FilNet's own best fixed rules sit at a higher threshold and a wider join, so each of the pool's 32
pairs is tried as the one setting, with the 21-number rule fitted leave-one-fold-out, on the folds that exist.
    python kernels/aug_unet/filnet_setting.py build <fold>     # that fold's rows (one process per fold)
    python kernels/aug_unet/filnet_setting.py                  # the table -> out/filnet_setting.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, pickle, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
import run  # noqa: E402
from run import np, cv2  # noqa: E402
import candidate_pool as C  # noqa: E402
import time_derot as TD  # noqa: E402

K = HERE.parent
DUMPS = {0: K / "filnet" / "out_kaggle" / "dump_filnet_phys.tar",
         1: K / "filnet_b" / "out_kaggle" / "dump_filnet_phys_f1.tar",
         2: K / "filnet_b" / "out_kaggle" / "dump_filnet_phys_f2.tar",
         3: K / "filnet_c" / "out_kaggle" / "dump_filnet_phys_f3.tar",
         4: K / "filnet_c" / "out_kaggle" / "dump_filnet_phys_f4.tar"}
OUT = HERE / "out"


def rows_path(k):
    return OUT / f"filnet_phys_d4_pool_f{k}.pkl"


def build(k):
    f = TD.folds()[k]
    path_of = {p.stem: p for p in run.find_data()[1].iterdir()}
    maps = {}
    with tarfile.open(DUMPS[k]) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == "val_d4":
                maps[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    out, t0 = [], time.time()
    for i, s in enumerate(f["stems"]):
        sets = f["sets"].get(i, [])
        cands, X = C.candidates(maps[s], cv2.imread(str(path_of[s]), cv2.IMREAD_GRAYSCALE), run.b0_deg(s))
        Y, _ = C.targets(cands, sets)
        out.append(dict(stem=s, fold=k, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                        n_gt=[len(x) for x in sets], conflict=C.conflicts(cands),
                        sett=[frozenset(c["from"]) for c in cands], area=np.array([c["area"] for c in cands], np.int32)))
    pickle.dump(out, open(rows_path(k), "wb"))
    print(f"fold {k}: {len(out)} photographs, {time.time() - t0:.0f}s", flush=True)


def evaluate(frames, folds, setting, min_area=C.POOL_MIN_AREA):
    for f in frames:
        f["one"] = np.array([setting in s for s in f["sett"]], bool) & (f["area"] >= min_area)
    scores = [None] * len(frames)
    for k in folds:
        fit = [f for f in frames if f["fold"] != k]
        idx = [n for n, f in enumerate(frames) if f["fold"] == k]
        for n, s in zip(idx, S._scores(S._fit(fit, "X", True, 0), [frames[n] for n in idx], "X")):
            scores[n] = s
    cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
    pqs = [S.summary(S.tally(frames, scores, c, True)) for c in cuts]
    b = int(np.argmax([r["PQ"] for r in pqs]))
    per = {k: S.summary(S.tally([f for f in frames if f["fold"] == k],
                                [s for f, s in zip(frames, scores) if f["fold"] == k], cuts[b], True))["PQ"] for k in folds}
    fixed = S.summary(S.tally(frames, [np.ones(len(f["one"])) for f in frames], 0.5, True))["PQ"]
    return dict(pqs[b], cut=float(cuts[b]), per_fold=per, fixed=fixed)


def main():
    folds = [k for k in sorted(DUMPS) if rows_path(k).exists()]
    frames = [f for k in folds for f in pickle.load(open(rows_path(k), "rb"))]
    print(f"folds {folds}, {len(frames)} photographs", flush=True)
    res = {}
    for t in C.POOL_T:
        for k in C.POOL_K:
            r = evaluate(frames, folds, (t, k))
            res[f"t {t} join {k}"] = r
            print(f"t {t} join {k:2d}: rule PQ {r['PQ']:.4f} (fixed {r['fixed']:.4f}) cut {r['cut']:.2f}  per fold "
                  + " ".join(f"{v:.4f}" for v in r["per_fold"].values()) + f"  TP {r['TP']} FP {r['FP']} FN {r['FN']}", flush=True)
            json.dump(res, open(OUT / "filnet_setting.json", "w"), indent=1, default=float)
    top = sorted(res, key=lambda n: -res[n]["PQ"])[:3]
    for name in top + ["t 0.6 join 3"]:
        t, k = float(name.split()[1]), int(name.split()[3])
        for a in (150, 200, 300, 400):
            r = evaluate(frames, folds, (t, k), a)
            res[f"{name} area {a}"] = r
            print(f"{name} area >= {a}: rule PQ {r['PQ']:.4f} cut {r['cut']:.2f}  per fold "
                  + " ".join(f"{v:.4f}" for v in r["per_fold"].values()), flush=True)
    json.dump(res, open(OUT / "filnet_setting.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "build":
        build(int(sys.argv[2]))
    else:
        main()
