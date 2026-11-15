"""One U-Net (R50, fold 0's model) and the 21-number selector: the cheap submission the user asked for (11-15).

For each view (plain = one pass, d4 = eight): the selector's fold-0 score on held-out halves, then the selector
fitted on all 141 fold-0 photographs (its cut from two inner halves), applied to the 180 test maps, written as a
CSV and checked with src/verify_submission.py.
    python kernels/aug_unet/single_r50_csv.py     -> out/submission_r50_<view>_rule21.csv, out/single_r50.json
"""
import collections, json, pickle, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402  -- also points run.CACHE at the full fold-0 cache
import run  # noqa: E402
from run import np, cv2  # noqa: E402
import candidate_pool as C  # noqa: E402

TAR = HERE.parent / "unet_r2" / "out_kaggle" / "dump_r50.tar"


def maps(view, split):
    out = {}
    with tarfile.open(TAR) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == f"{split}_{view}":
                out[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return out


def main():
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    vimg = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    te_dir = run.find_data()[2]
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    S.VARIANTS = {k: v for k, v in S.VARIANTS.items() if v[1] in (None, "X")}
    res = {}
    for view in ("plain", "d4"):
        t0 = time.time()
        vm, tm = maps(view, "val"), maps(view, "test")
        frames = []
        for i, f in enumerate(meta["val_files"]):
            s = Path(f).stem
            cands, X = C.candidates(vm[s], np.array(vimg[i]), run.b0_deg(s))
            Y, _ = C.targets(cands, by_frame[i])
            frames.append(dict(stem=s, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                               n_gt=[len(x) for x in by_frame[i]], conflict=C.conflicts(cands),
                               one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool)))
        print(f"\nR50, {view}: fold 0 on held-out halves", flush=True)
        r = S.stage2_eval(frames)
        # the final selector: all 141 photographs; its cut from the two inner halves' out-of-fold scores
        inner = [frames[0::2], frames[1::2]]
        oof = [S._scores(S._fit(inner[1 - j], "X", True, 0), inner[j], "X") for j in range(2)]
        cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
        pq = [S.summary(S.tally(inner[0] + inner[1], oof[0] + oof[1], c, True))["PQ"] for c in cuts]
        cut = float(cuts[int(np.argmax(pq))])
        model = S._fit(frames, "X", True, 0)
        rows, per = [], []
        for name in meta["test_files"]:
            s = Path(name).stem
            img = cv2.imread(str(te_dir / name), cv2.IMREAD_GRAYSCALE)
            cands, X = C.candidates(tm[s], img, run.b0_deg(s))
            k = [n for n, c in enumerate(cands) if C.ONE_SETTING in c["from"]]
            sc = model.predict(X[k]) if k else []
            n_kept = 0
            for n, q in zip(k, sc):
                if q >= cut:
                    c = cands[n]
                    m = np.zeros((run.FULL, run.FULL), np.uint8)
                    m[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
                    n_kept += 1
                    rle = run.mu.encode(np.asfortranarray(m))
                    rows.append(f"{s}_{n_kept},{rle['counts'].decode('ascii')}")
            per.append(n_kept)
        out = HERE / "out" / f"submission_r50_{view}_rule21.csv"
        out.write_text("filament_id,segmentation_rle\n" + "\n".join(rows) + "\n")
        v = subprocess.run([sys.executable, str(HERE.parents[1] / "src" / "verify_submission.py"), str(out),
                            "--test-dir", str(te_dir)], capture_output=True, text=True)
        res[view] = dict(fold0_halves=r, cut=cut, file=str(out), rows=len(rows), per_image=round(float(np.mean(per)), 2),
                         empty=int(sum(x == 0 for x in per)), verify_exit=v.returncode, verify_tail=v.stdout[-400:],
                         minutes=round((time.time() - t0) / 60, 1))
        print(f"R50 {view}: cut {cut}, {len(rows)} rows, {np.mean(per):.2f} per photograph; verify exit {v.returncode}\n"
              f"{v.stdout[-300:]}", flush=True)
        json.dump(res, open(HERE / "out" / "single_r50.json", "w"), indent=1)


if __name__ == "__main__":
    main()
