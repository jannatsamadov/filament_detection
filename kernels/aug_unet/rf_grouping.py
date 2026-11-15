"""The user's pixel forests with a better pixels-to-filaments step (11-15).

The forests were scored with the fixed chain tuned for the U-Nets (threshold, join, area). With the join the
forest's own map needs and the learned selector, the table + U-Net forest went 0.346 -> 0.441 (fold 0, held-out
halves). Here every forest map gets the same treatment:
  1. the fixed chain swept over threshold 0.2-0.7 and joins up to 24 px: its best setting
  2. at that setting: the selector on the 21 numbers, and the whole 32-outline pool, on held-out halves
    python kernels/aug_unet/rf_grouping.py ["<model name as in pixel_cls.MODELS>" ...]   -> out/rf_grouping.json
"""
import collections, json, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402  -- also points run.CACHE at the full fold-0 cache
import pixel_cls as PC  # noqa: E402
from run import np  # noqa: E402
import run  # noqa: E402
import candidate_pool as C  # noqa: E402

MODELS = tuple(sys.argv[1:]) or (
    "RF, pixel table only",
    "RF: + neighbours' predictions (auto-context)",
    "RF cls: + 12 px neighbours (1358 col.)",
    "RF cls: neighbours, no positions (478 col.)",
    "RF cls: neighbours + rotation (2056 col.)",
    "RF reg: + 12 px neighbours (1358 col.)",
)
OUT = HERE / "out" / ("rf_grouping.json" if len(sys.argv) < 2 else "rf_grouping_extra.json")


def main():
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    vimg = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    D = PC.Dumps()
    variants = {k: v for k, v in S.VARIANTS.items() if v[1] in (None, "X")}
    res = json.load(open(OUT)) if OUT.exists() else {}
    for name in MODELS:
        if name in res:
            continue
        t0 = time.time()
        stems = [Path(f).stem for f in meta["val_files"]]
        probs = [D.model(name, s, {}) for s in stems]
        run.THRESH, run.MERGE, run.MIN_AREA = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7), (0, 3, 7, 12), (100,)
        rows = run.sweep(probs, gt)
        b = rows[0]
        joins = {2 * k: round(max(r["PQ"] for r in rows if r["merge"] == k), 4) for k in run.MERGE}
        print(f"\n{name}: fixed chain best PQ {b['PQ']:.4f} at t {b['thresh']} join {2 * b['merge']} px; "
              f"best per join {joins}", flush=True)
        C.ONE_SETTING = (b["thresh"], b["merge"])
        frames = []
        for i, s in enumerate(stems):
            cands, X = C.candidates(probs[i], np.array(vimg[i]), run.b0_deg(s))
            Y, _ = C.targets(cands, by_frame[i])
            frames.append(dict(stem=s, X=X, Z=np.zeros((len(cands), 0), np.float16), Y=Y.astype(np.float32),
                               n_gt=[len(x) for x in by_frame[i]], conflict=C.conflicts(cands),
                               one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool),
                               area=np.array([c["area"] for c in cands], np.int32)))
        S.VARIANTS = variants
        r = S.stage2_eval(frames)
        res[name] = dict(fixed_best=dict(PQ=b["PQ"], SQ=b["SQ"], RQ=b["RQ"], t=b["thresh"], join_px=2 * b["merge"]),
                         per_join=joins, selector=r, minutes=round((time.time() - t0) / 60, 1))
        json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
