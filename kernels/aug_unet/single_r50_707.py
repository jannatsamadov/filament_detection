"""One U-Net and the 21-number keep-rule, the rule fitted on all 707 photographs (11-16).

time_selector.py showed the R50 U-Net's out-of-fold D4 maps with the 21-number rule at 0.445 on held-out halves
of all 707 photographs, near the final 16-model file's 0.4487. The earlier single-R50 file fitted its rule on
fold 0's 141 photographs only. Here:
  rows   every photograph's candidates from the R50 that did not see it (five fold models, D4)
  check  five-fold out-of-fold PQ of the rule, the folds being the U-Nets' own (the final file's protocol)
  cut    the one that maximises that out-of-fold PQ
  test   r50all (the R50 trained on all 707) D4 maps, the rule fitted on all 707 rows, written and verified
    python kernels/aug_unet/single_r50_707.py   -> out/submission_r50all_rule21_707.csv, out/single_r50_707.json
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

TEST_TAR = HERE.parent / "unet_r3" / "out_kaggle" / "dump_r50all.tar"
ROWS = HERE / "out" / "r50_707_frames.pkl"
OUT_CSV = HERE / "out" / "submission_r50all_rule21_707.csv"


def frames_707():
    if ROWS.exists():
        return pickle.load(open(ROWS, "rb"))
    fs = TD.folds()
    path_of = {p.stem: p for p in run.find_data()[1].iterdir()}
    out, t0 = [], time.time()
    for k, f in enumerate(fs):
        maps = TD.load_maps(k, set(f["stems"]))
        for i, s in enumerate(f["stems"]):
            sets = f["sets"].get(i, [])
            cands, X = C.candidates(maps[s], cv2.imread(str(path_of[s]), cv2.IMREAD_GRAYSCALE), run.b0_deg(s))
            Y, _ = C.targets(cands, sets)
            out.append(dict(stem=s, fold=k, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                            n_gt=[len(x) for x in sets], conflict=C.conflicts(cands),
                            one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool)))
        print(f"fold {k}: {time.time() - t0:.0f}s", flush=True)
    pickle.dump(out, open(ROWS, "wb"))
    return out


def main():
    t0 = time.time()
    frames = frames_707()
    # five-fold out-of-fold scores, the folds being the U-Nets' own
    scores = [None] * len(frames)
    for k in range(5):
        fit = [f for f in frames if f["fold"] != k]
        idx = [n for n, f in enumerate(frames) if f["fold"] == k]
        sc = S._scores(S._fit(fit, "X", True, 0), [frames[n] for n in idx], "X")
        for n, s in zip(idx, sc):
            scores[n] = s
    cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
    pqs = [S.summary(S.tally(frames, scores, c, True)) for c in cuts]
    best = int(np.argmax([r["PQ"] for r in pqs]))
    cut = float(cuts[best])
    fixed = S.summary(S.tally(frames, [np.ones(len(f["one"])) for f in frames], 0.5, True))
    f0 = [n for n, f in enumerate(frames) if f["fold"] == 0]
    fold0 = S.summary(S.tally([frames[n] for n in f0], [scores[n] for n in f0], cut, True))
    print(f"707 photographs, out of fold: fixed rules {fixed}\n  21-number rule at cut {cut}: {pqs[best]}\n"
          f"  its fold-0 part: {fold0}", flush=True)

    model = S._fit(frames, "X", True, 0)
    tmaps = {}
    with tarfile.open(TEST_TAR) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == "test_d4":
                tmaps[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8),
                                                        cv2.IMREAD_GRAYSCALE)
    te_dir = run.find_data()[2]
    test_files = sorted(p.name for p in te_dir.glob("*.jpeg"))
    assert len(tmaps) == len(test_files) == 180, (len(tmaps), len(test_files))
    rows, per = [], []
    for name in test_files:
        s = Path(name).stem
        cands, X = C.candidates(tmaps[s], cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE), run.b0_deg(s))
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
    OUT_CSV.write_text("filament_id,segmentation_rle\n" + "\n".join(rows) + "\n")
    v = subprocess.run([sys.executable, str(HERE.parents[1] / "src" / "verify_submission.py"), str(OUT_CSV),
                        "--test-dir", str(te_dir)], capture_output=True, text=True)
    res = dict(oof_707_fixed=fixed, oof_707_rule21=pqs[best], fold0_part=fold0, cut=cut, file=str(OUT_CSV),
               rows=len(rows), per_image=round(float(np.mean(per)), 2), empty=int(sum(x == 0 for x in per)),
               verify_exit=v.returncode, verify_tail=v.stdout[-400:], minutes=round((time.time() - t0) / 60, 1))
    json.dump(res, open(HERE / "out" / "single_r50_707.json", "w"), indent=1, default=float)
    print(f"{OUT_CSV.name}: {len(rows)} rows, {np.mean(per):.2f} per photograph; verify exit {v.returncode}\n"
          f"{v.stdout[-300:]}", flush=True)


if __name__ == "__main__":
    main()
