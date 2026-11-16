"""Time as evidence for the keep-rule (11-16): does the filament also show 1-3.5 days earlier or later?

A real filament lives for days and turns with the Sun; a false one is usually one photograph's noise. For each
candidate outline of a photograph, the neighbour photographs' maps (same fold, within 84 h) are carried forward
by the rotation rate the experts' masks show (time_derot.py, part A) and read inside the outline:
    z1 how many neighbours, z2 the highest mean probability inside it, z3 the largest share of it above 0.5,
    z4 how much of it the neighbour sees at all (NaN where there is no neighbour or it is not visible).
The keep-rule is the same gradient-boosted regressor on the 21 numbers, with and without these 4 numbers, scored
on held-out halves of all 707 photographs (the R50 U-Net's out-of-fold D4 maps), three different splits.
    python kernels/aug_unet/time_selector.py      -> out/time_selector.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")        # the laptop: no all-core load
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
import run  # noqa: E402
from run import np, cv2  # noqa: E402
import candidate_pool as C  # noqa: E402
import time_derot as TD  # noqa: E402

OUT = HERE / "out" / "time_selector.json"


def time_numbers(cands, carried):
    Z = np.full((len(cands), 4), np.nan, np.float32)
    Z[:, 0] = len(carried)
    for n, c in enumerate(cands):
        m = c["m"].astype(bool)
        h, w = m.shape
        best = [np.nan, np.nan, np.nan]
        for mp, v in carried:
            vals = mp[c["y0"]:c["y0"] + h, c["x0"]:c["x0"] + w][m]
            vis = v[c["y0"]:c["y0"] + h, c["x0"]:c["x0"] + w][m]
            frac_vis = float(vis.mean()) if vis.size else 0.0
            best[2] = frac_vis if np.isnan(best[2]) else max(best[2], frac_vis)
            if vis.sum() >= 10:
                mean_p = float(vals[vis].mean()) / 255.0
                hi = float((vals[vis] >= 128).mean())
                best[0] = mean_p if np.isnan(best[0]) else max(best[0], mean_p)
                best[1] = hi if np.isnan(best[1]) else max(best[1], hi)
        Z[n, 1:] = best
    return Z


def main():
    t0 = time.time()
    fs = TD.folds()
    rate = json.load(open(TD.OUT))["A"]["peak_rate"] if TD.OUT.exists() else 13.72
    tr_dir = run.find_data()[1]
    path_of = {p.stem: p for p in tr_dir.iterdir()}
    frames = []
    for k, f in enumerate(fs):
        maps = TD.load_maps(k, set(f["stems"]))
        t = np.array([TD.hours(s) for s in f["stems"]])
        for i, s in enumerate(f["stems"]):
            sets = f["sets"].get(i, [])
            img = cv2.imread(str(path_of[s]), cv2.IMREAD_GRAYSCALE)
            cands, X = C.candidates(maps[s], img, run.b0_deg(s))
            Y, _ = C.targets(cands, sets)
            near = [j for j in range(len(f["stems"])) if j != i and 1.0 <= abs(t[i] - t[j]) <= TD.MAX_DT_H]
            carried = [TD.carry(maps[f["stems"][j]].astype(np.float32), run.b0_deg(f["stems"][j]), run.b0_deg(s),
                                (t[i] - t[j]) / 24.0, rate, run.FULL) for j in near]
            frames.append(dict(stem=s, X=X, Z=time_numbers(cands, carried), Y=Y.astype(np.float32),
                               n_gt=[len(x) for x in sets], conflict=C.conflicts(cands),
                               one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool), nb=bool(near)))
        print(f"fold {k}: {len(f['stems'])} photographs, {time.time() - t0:.0f}s", flush=True)
    n_nb = sum(f["nb"] for f in frames)
    print(f"{len(frames)} photographs, {n_nb} with a neighbour; rotation {rate} deg/day (synodic, equator)", flush=True)
    S.VARIANTS = {
        "fixed rules (t 0.6, join 3, area 100), keep all": (True, None),
        "one outline, 21 numbers": (True, "X"),
        "one outline, 21 numbers + 4 time numbers": (True, "XZ"),
        "whole pool, 21 numbers": (False, "X"),
        "whole pool, 21 numbers + 4 time numbers": (False, "XZ"),
    }
    res = dict(photographs=len(frames), with_neighbour=n_nb, rate=rate, splits={})
    for seed in (0, 1, 2):
        print(f"\nsplit {seed}: held-out halves of all {len(frames)} photographs", flush=True)
        res["splits"][seed] = S.stage2_eval(frames, seed=seed)
        json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"\n{(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
