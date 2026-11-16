"""Numbers of the technical report that concern FilNet, with bootstrap intervals over photographs (11-16).

From the rows filnet_folds.py cached (FilNet phys per fold, the raw control on fold 0) and single_r50_707.py's:
  1. the selection rule on FilNet's candidates: keep-all against the rule, leave-one-fold-out
  2. FilNet against the ImageNet R50 on the same photographs, both with the rule
  3. the physical channels: fold 0, phys against raw, the rule fitted on the other folds' FilNet rows
  4. pixel level on fold 0: FilNet's D4 map against the union of the experts
    python kernels/aug_unet/filnet_report.py   -> out/filnet_report.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import json, pickle, sys, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
import run  # noqa: E402
from run import np, cv2  # noqa: E402
import candidate_pool as C  # noqa: E402

OUT = HERE / "out"
CUTS = np.round(np.arange(0.02, 0.6, 0.01), 2)


def counts(frames, scores, cut):
    """Per photograph (TP, FP, FN, sum of matched IoU), the one setting."""
    out = np.zeros((len(frames), 4))
    for n, (f, s) in enumerate(zip(frames, scores)):
        out[n] = S.tally([f], [s], cut, True)
    return out


def pq(c):
    tp, fp, fn, iou = c.sum(axis=0)
    return iou / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)


def boot(a, b, n=2000):
    rng = np.random.default_rng(0)
    d = []
    for _ in range(n):
        s = rng.integers(0, len(a), len(a))
        d.append(pq(a[s]) - pq(b[s]))
    return dict(a=round(pq(a), 4), b=round(pq(b), 4), diff=round(pq(a) - pq(b), 4),
                lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4))


def lofo(frames, folds):
    scores = [None] * len(frames)
    for k in folds:
        fit = [f for f in frames if f["fold"] != k]
        idx = [n for n, f in enumerate(frames) if f["fold"] == k]
        for n, s in zip(idx, S._scores(S._fit(fit, "X", True, 0), [frames[n] for n in idx], "X")):
            scores[n] = s
    best = int(np.argmax([S.summary(S.tally(frames, scores, c, True))["PQ"] for c in CUTS]))
    return scores, float(CUTS[best])


def main():
    folds = [k for k in range(5) if (OUT / f"filnet_phys_d4_frames_f{k}.pkl").exists()]
    fil = [f for k in folds for f in pickle.load(open(OUT / f"filnet_phys_d4_frames_f{k}.pkl", "rb"))]
    res = dict(folds=folds, photographs=len(fil))
    sc, cut = lofo(fil, folds)
    rule = counts(fil, sc, cut)
    keep_all = counts(fil, [np.ones(len(f["one"])) for f in fil], 0.5)
    res["rule_minus_keep_all"] = dict(boot(rule, keep_all), cut=cut)
    print("1. rule against keeping every candidate:", res["rule_minus_keep_all"], flush=True)

    stems = [f["stem"] for f in fil]
    r50 = {f["stem"]: f for f in pickle.load(open(OUT / "r50_707_frames.pkl", "rb"))}
    r50 = [r50[s] for s in stems]
    sc_r, cut_r = lofo(r50, folds)
    res["filnet_minus_r50"] = dict(boot(rule, counts(r50, sc_r, cut_r)), cut_r50=cut_r)
    print("2. FilNet against the ImageNet R50, same photographs, both with the rule:", res["filnet_minus_r50"], flush=True)

    raw_p = OUT / "filnet_raw_d4_frames_f0.pkl"
    if raw_p.exists() and 0 in folds and len(folds) > 1:
        other = [f for f in fil if f["fold"] != 0]
        m = S._fit(other, "X", True, 0)
        sc_o, cut_o = lofo(other, [k for k in folds if k != 0]) if len(folds) > 2 else (None, cut)
        phys0 = [f for f in fil if f["fold"] == 0]
        raw0 = pickle.load(open(raw_p, "rb"))
        assert [f["stem"] for f in phys0] == [f["stem"] for f in raw0]
        a = counts(phys0, S._scores(m, phys0, "X"), cut_o)
        b = counts(raw0, S._scores(m, raw0, "X"), cut_o)
        res["phys_minus_raw_rule"] = dict(boot(a, b), cut=cut_o)
        ka = counts(phys0, [np.ones(len(f["one"])) for f in phys0], 0.5)
        kb = counts(raw0, [np.ones(len(f["one"])) for f in raw0], 0.5)
        res["phys_minus_raw_keep_all"] = boot(ka, kb)
        print("3. physical channels, fold 0, rule from the other folds:", res["phys_minus_raw_rule"],
              "\n   keep all:", res["phys_minus_raw_keep_all"], flush=True)
    json.dump(res, open(OUT / "filnet_report.json", "w"), indent=1, default=float)

    # pixel level, fold 0, against the union of the experts, on the disk
    tar = HERE.parent / "filnet" / "out_kaggle" / "dump_filnet_phys.tar"
    cache = HERE / "out" / "cache_full"
    meta = json.load(open(cache / "meta.json"))
    msk = np.load(cache / "val_msk.npy", mmap_mode="r")
    yy, xx = np.mgrid[:run.FULL, :run.FULL]
    disk = ((xx - run.CX) ** 2 + (yy - run.CY) ** 2) <= (run.R_DISK * 0.995) ** 2
    idx = {Path(f).stem: i for i, f in enumerate(meta["val_files"])}
    tp = fp = fn = 0
    with tarfile.open(tar) as t:
        for mem in t.getmembers():
            if mem.isfile() and Path(mem.name).parent.name == "val_d4":
                p = cv2.imdecode(np.frombuffer(t.extractfile(mem).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                g = np.asarray(msk[idx[Path(mem.name).stem]]) > 0
                pr = (p >= 128) & disk
                tp += int((pr & g).sum()); fp += int((pr & ~g).sum()); fn += int((~pr & g & disk).sum())
    res["pixel_fold0"] = dict(precision=round(tp / (tp + fp), 4), recall=round(tp / (tp + fn), 4),
                              f1=round(2 * tp / (2 * tp + fp + fn), 4))
    print("4. pixel level, fold 0, D4 map at 0.5 against the union of the experts:", res["pixel_fold0"], flush=True)
    json.dump(res, open(OUT / "filnet_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
