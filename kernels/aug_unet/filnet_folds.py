"""FilNet (from scratch) end to end: its fold models' maps -> the 21-number keep-rule -> a verified CSV (11-16).

Each fold model's D4 maps of the photographs it never saw give that fold's rows (candidates, 21 numbers, IoU
per expert set). The rule is checked leave-one-fold-out over the folds that exist, its cut chosen there, then
fitted on all rows and applied to the mean of the fold models' test D4 maps.
For a like-for-like comparison the single ImageNet R50's rows (single_r50_707.py) are scored on the same folds.
    python kernels/aug_unet/filnet_folds.py [name]      name: phys (default) | raw | physraw (fold 0 only)
    FILNET_FOLDS_SMOKE=1 ...                            a few photographs, to test the path
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import json, pickle, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
import run  # noqa: E402
from run import np, cv2  # noqa: E402
import candidate_pool as C  # noqa: E402
import time_derot as TD  # noqa: E402

SMOKE = os.environ.get("FILNET_FOLDS_SMOKE") == "1"
NAME = sys.argv[1] if len(sys.argv) > 1 else "phys"
K = HERE.parent
DUMPS = {   # fold -> the dump(s) whose maps are averaged
    "phys": {0: [K / "filnet" / "out_kaggle" / "dump_filnet_phys.tar"],
             1: [K / "filnet_b" / "out_kaggle" / "dump_filnet_phys_f1.tar"],
             2: [K / "filnet_b" / "out_kaggle" / "dump_filnet_phys_f2.tar"],
             3: [K / "filnet_c" / "out_kaggle" / "dump_filnet_phys_f3.tar"],
             4: [K / "filnet_c" / "out_kaggle" / "dump_filnet_phys_f4.tar"]},
    "raw": {0: [K / "filnet" / "out_kaggle" / "dump_filnet_raw.tar"]},
    "physraw": {0: [K / "filnet" / "out_kaggle" / "dump_filnet_phys.tar",
                    K / "filnet" / "out_kaggle" / "dump_filnet_raw.tar"]},
}[NAME]
if os.environ.get("FILNET_FOLDS_FAKE"):          # path test only: the R50 U-Nets' dumps in FilNet's place
    DUMPS = {0: [K / "unet_r2" / "out_kaggle" / "dump_r50.tar"], 1: [K / "unet_r5" / "out_kaggle" / "dump_r50f1.tar"]}
VIEW = os.environ.get("FILNET_VIEW", "d4")
ONLY = os.environ.get("FILNET_ONLY_FOLDS")     # e.g. "0": the same rows, fewer folds (the halves protocol on one)
FRAMES_TAG = f"filnet_{NAME}_{VIEW}" + ("_smoke" if SMOKE else "")
TAG = FRAMES_TAG + (f"_only{ONLY.replace(',', '')}" if ONLY else "")
NO_CSV = os.environ.get("FILNET_NO_CSV") == "1"
OUT = HERE / "out"


def tar_maps(path, sub, stems=None):
    out = {}
    with tarfile.open(path) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == sub and (stems is None or Path(m.name).stem in stems):
                out[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8),
                                                      cv2.IMREAD_GRAYSCALE)
    return out


def mean_maps(paths, sub, stems=None):
    ms = [tar_maps(p, sub, stems) for p in paths]
    return {s: (np.mean([m[s] for m in ms], 0).round().astype(np.uint8) if len(ms) > 1 else ms[0][s]) for s in ms[0]}


def fold_frames(k, f, path_of):
    cache = OUT / f"{FRAMES_TAG}_frames_f{k}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    stems = f["stems"][:6] if SMOKE else f["stems"]
    maps = mean_maps(DUMPS[k], f"val_{VIEW}", set(stems))
    assert len(maps) == len(stems), (k, len(maps), len(stems))
    out, t0 = [], time.time()
    for i, s in enumerate(stems):
        sets = f["sets"].get(i, [])
        cands, X = C.candidates(maps[s], cv2.imread(str(path_of[s]), cv2.IMREAD_GRAYSCALE), run.b0_deg(s))
        Y, _ = C.targets(cands, sets)
        out.append(dict(stem=s, fold=k, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                        n_gt=[len(x) for x in sets], conflict=C.conflicts(cands),
                        one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool)))
    pickle.dump(out, open(cache, "wb"))
    print(f"fold {k}: {len(out)} photographs, {time.time() - t0:.0f}s", flush=True)
    return out


def oof(frames, folds, pool=False):
    """Leave-one-fold-out scores; with a single fold, the halves protocol of solar_unet.stage2_eval instead."""
    scores = [None] * len(frames)
    for k in folds:
        fit = [f for f in frames if f["fold"] != k]
        idx = [n for n, f in enumerate(frames) if f["fold"] == k]
        sc = S._scores(S._fit(fit, "X", not pool, 0), [frames[n] for n in idx], "X")
        for n, s in zip(idx, sc):
            scores[n] = s
    cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
    pqs = [S.summary(S.tally(frames, scores, c, not pool)) for c in cuts]
    b = int(np.argmax([r["PQ"] for r in pqs]))
    return scores, float(cuts[b]), pqs[b]


def main():
    t0 = time.time()
    fs = TD.folds()
    path_of = {p.stem: p for p in run.find_data()[1].iterdir()}
    folds = [k for k in sorted(DUMPS) if all(p.exists() for p in DUMPS[k])]
    if ONLY:
        folds = [k for k in folds if str(k) in ONLY.split(",")]
    print(f"{NAME}, {VIEW}: folds with maps {folds}", flush=True)
    frames = []
    for k in folds:
        frames += fold_frames(k, fs[k], path_of)
    res = dict(name=NAME, view=VIEW, folds=folds, photographs=len(frames))
    fixed = S.summary(S.tally(frames, [np.ones(len(f["one"])) for f in frames], 0.5, True))
    res["fixed"] = fixed
    print(f"{len(frames)} photographs out of fold. Fixed rules (t 0.6, join 3, area 100): {fixed}", flush=True)
    for k in folds:
        fr = [f for f in frames if f["fold"] == k]
        r = S.summary(S.tally(fr, [np.ones(len(f["one"])) for f in fr], 0.5, True))
        res[f"fixed_f{k}"] = r
        print(f"   fold {k}: {r}", flush=True)
    if len(folds) >= 2:
        scores, cut, best = oof(frames, folds)
        res.update(rule21=best, cut=cut)
        print(f"21-number rule, leave-one-fold-out, cut {cut}: {best}", flush=True)
        for k in folds:
            idx = [n for n, f in enumerate(frames) if f["fold"] == k]
            r = S.summary(S.tally([frames[n] for n in idx], [scores[n] for n in idx], cut, True))
            res[f"rule21_f{k}"] = r
            print(f"   fold {k}: {r}", flush=True)
        _, cut_p, best_p = oof(frames, folds, pool=True)
        res.update(pool21=best_p, cut_pool=cut_p)
        print(f"whole pool, 21 numbers, cut {cut_p}: {best_p}", flush=True)
        # the single ImageNet R50 on the same photographs, the same way
        r50 = OUT / "r50_707_frames.pkl"
        if r50.exists() and not SMOKE:
            keep = {f["stem"] for f in frames}
            rf = [f for f in pickle.load(open(r50, "rb")) if f["stem"] in keep]
            _, rc, rb = oof(rf, folds)
            res.update(r50_same_photographs=rb, r50_fixed=S.summary(S.tally(rf, [np.ones(len(f["one"])) for f in rf], 0.5, True)))
            print(f"ImageNet R50 on the same {len(rf)} photographs: fixed {res['r50_fixed']['PQ']}, 21-number rule {rb}", flush=True)
    else:
        S.VARIANTS = {k: v for k, v in S.VARIANTS.items() if v[1] in (None, "X")}   # no network features here
        res["halves"] = {f"seed {sd}": S.stage2_eval(frames, sd) for sd in (0, 1, 2)}
        # the cut for the CSV: the best on fold 0's own out-of-fold halves
        idx = np.random.default_rng(0).permutation(len(frames))
        h = [[frames[i] for i in sorted(idx[::2])], [frames[i] for i in sorted(idx[1::2])]]
        sc = S._scores(S._fit(h[1], "X", True, 0), h[0], "X") + S._scores(S._fit(h[0], "X", True, 0), h[1], "X")
        cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
        pq = [S.summary(S.tally(h[0] + h[1], sc, c, True))["PQ"] for c in cuts]
        cut = float(cuts[int(np.argmax(pq))])
        res["cut"] = cut
    json.dump(res, open(OUT / f"{TAG}.json", "w"), indent=1, default=float)
    if NO_CSV:
        return

    # test: the mean of the fold models' maps, the rule fitted on every row
    model = S._fit(frames, "X", True, 0)
    te_dir = run.find_data()[2]
    test_files = sorted(p.name for p in te_dir.glob("*.jpeg"))
    if SMOKE:
        test_files = test_files[:4]
    tm = mean_maps([p for k in folds for p in DUMPS[k]], f"test_{VIEW}", {Path(n).stem for n in test_files})
    assert len(tm) == len(test_files), (len(tm), len(test_files))
    rows, per = [], []
    for name in test_files:
        s = Path(name).stem
        cands, X = C.candidates(tm[s], cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE), run.b0_deg(s))
        k = [n for n, c in enumerate(cands) if C.ONE_SETTING in c["from"]]
        sc = model.predict(X[k]) if k else []
        n_kept = 0
        for n, q in zip(k, sc):
            if q >= cut:
                c = cands[n]
                m = np.zeros((run.FULL, run.FULL), np.uint8)
                m[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
                n_kept += 1
                rows.append(f"{s}_{n_kept},{run.mu.encode(np.asfortranarray(m))['counts'].decode('ascii')}")
        per.append(n_kept)
    csv = OUT / f"submission_{TAG}_rule21.csv"
    csv.write_text("filament_id,segmentation_rle\n" + "\n".join(rows) + "\n")
    v = subprocess.run([sys.executable, str(HERE.parents[1] / "src" / "verify_submission.py"), str(csv),
                        "--test-dir", str(te_dir)], capture_output=True, text=True)
    res.update(file=str(csv), rows=len(rows), per_image=round(float(np.mean(per)), 2),
               empty=int(sum(x == 0 for x in per)), verify_exit=v.returncode, verify_tail=v.stdout[-400:],
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(OUT / f"{TAG}.json", "w"), indent=1, default=float)
    print(f"{csv.name}: {len(rows)} rows, {np.mean(per):.2f} per photograph, cut {cut}; verify exit {v.returncode}\n"
          f"{v.stdout[-300:]}", flush=True)


if __name__ == "__main__":
    main()
