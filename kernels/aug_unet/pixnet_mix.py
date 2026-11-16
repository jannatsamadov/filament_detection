"""PixNet (the user's from-scratch per-pixel network, aug_unet/pixnet.py) as one more member of the
averaged map, and the user's gap filling measured on its own.

PixNet exists for fold 0 only, like the Qwen segmenter, so it enters fold 0's averaged map rather
than the keep-rule:
  1. each PixNet arm alone against the Qwen segmenter and one R50, all plain, on fold 0's photographs
  2. fold 0's ensemble (five U-Nets + the Qwen map, 0.4578) with PixNet averaged in, under the
     25-column rule trained on folds 1-4 with the cut nested inside fold 0
  3. the gap filling: each filament's mask closed (the pixels missed between its pieces filled)
     before matching, at several kernel sizes, on the pair's out-of-fold maps of all 707
     photographs and on fold 0's ensemble -- the same pieces, only their gaps change
  4. test CSVs of the ten U-Nets + Qwen + PixNet (and the 0.39 file again, as the pipeline check)
CPU only.
    kernels/build.py builds kernels/pixnet_mix/run.py (and pixnet_mix_smoke/) from this file
"""
from __future__ import annotations

import json, multiprocessing as mp, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidate_pool as C  # noqa: E402,F401
import kfold as K  # noqa: E402
import post_local as P  # noqa: E402
import run  # noqa: E402

# ---------------------------------------------------------------- pixnet-mix settings
import time  # noqa: E402

XM_ARMS = ("pixnet_scratch", "pixnet_pre")    # dump_pixnet_<arm>.tar from filament-pixnet (val_plain, test_plain)
XM_W = 1.0                                    # one member, as the Qwen map (K.QSEG_W)
XM_CLOSE = (0, 3, 5, 9, 15)                   # closing kernel, px (ellipse diameter); 0 = the pipeline as it is
XM_CLOSE_T = (0.55, 0.6, 0.65, 0.7)           # the small grid the closing is compared on
XM_CLOSE_K = (3, 7)
XM_CLOSE_A = (350, 500)
XM_SHOW = 3                                   # fold-0 photographs whose maps are copied out for the figure
XM_SMOKE = False                              # kernels/build.py sets this in pixnet_mix_smoke/run.py


def xm_mix(d, name, members, extras, split, stems, view=None):
    """d/dump/<name>/<split>_<view>/: the members' mean with each (variant, weight) in extras added."""
    view = view or K.VIEW
    out = Path(d) / "dump" / name / f"{split}_{view}"
    out.mkdir(parents=True, exist_ok=True)
    for s in stems:
        acc = sum(P.load_map(Path(d), f"{m}/{view}", split, s).astype(np.float32) for m in members)
        wsum = float(len(members))
        for v, w in extras:
            acc = acc + w * P.load_map(Path(d), v, split, s).astype(np.float32)
            wsum += w
        cv2.imwrite(str(out / f"{s}.png"), np.round(acc / wsum).astype(np.uint8))
    return f"{name}/{view}"


def xm_components(p, t, k, c):
    """run.components with each filament's gaps filled: the thresholded mask is closed with a c px
    ellipse, and a closed pixel joins the filament whose 2k px neighbourhood it lies in. c = 0
    gives run.components exactly; the pieces and their grouping never change, only the gaps."""
    b = (p >= int(round(255 * t))).astype(np.uint8)
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        lab = cv2.connectedComponents(cv2.dilate(b, ker), connectivity=8)[1]
    else:
        lab = cv2.connectedComponents(b, connectivity=8)[1]
    if c:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c)))
    lab[b == 0] = 0
    return lab, np.bincount(lab.ravel(), minlength=int(lab.max()) + 1)


def xm_close_job(job):
    d, variant, stem, sets = job
    p = P.load_map(Path(d), variant, "val", stem)
    tot = np.zeros((len(XM_CLOSE), len(XM_CLOSE_T), len(XM_CLOSE_K), len(XM_CLOSE_A), 4))
    for ci, c in enumerate(XM_CLOSE):
        for i, t in enumerate(XM_CLOSE_T):
            for j, k in enumerate(XM_CLOSE_K):
                lab, area = xm_components(p, t, k, c)
                for ai, a in enumerate(XM_CLOSE_A):
                    keep = area >= a
                    keep[0] = False
                    for _, crops in sets:
                        tot[ci, i, j, ai] += np.array(run.pq_counts(crops, lab, area, keep), float)
    return tot


def xm_close_compare(g):
    """g: (photos, closings, T, K, A, 4). Each closing at its own best point, against no closing."""
    out, at = {}, {}
    for ci, c in enumerate(XM_CLOSE):
        pq = P.pq_of(g[:, ci].sum(axis=0))
        i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
        at[c] = g[:, ci, i, j, a]
        out[f"close{c}"] = dict(point=[XM_CLOSE_T[i], XM_CLOSE_K[j], XM_CLOSE_A[a]], **K.summary(at[c].sum(axis=0)),
                                minus_none=K.boot(at[c], at[0]) if c else None)
    # and at the point without closing: only the gaps differ
    pq0 = P.pq_of(g[:, 0].sum(axis=0))
    i, j, a = np.unravel_index(np.argmax(pq0), pq0.shape)
    out["same_point"] = {f"close{c}": dict(**K.summary(g[:, ci, i, j, a].sum(axis=0)),
                                           minus_none=K.boot(g[:, ci, i, j, a], g[:, 0, i, j, a]) if c else None)
                         for ci, c in enumerate(XM_CLOSE)}
    return out


def xm_ens0_frames(d, t0, variant, tag, imap, yolo):
    cache = Path(d) / f"xm_pool_ens0_{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    ens = K.one_setting(list(imap(K.pool_job, [(str(d), variant, stem, t0["cache"], f, sets, str(cache), True)
                                               for f, stem, sets in t0["frames"]])))
    ens = K.add_neighbours(ens, Path(d), [variant] * len(ens), "val", np.zeros(len(ens), int), imap)
    return K.add_yolo(ens, [yolo[0]] * len(ens), "val")


def xm_best(d, variants, frames, imap):
    """Each variant alone at its own best grid point on these photographs: summary and per-photo counts."""
    jobs = [(str(d), v, stem, sets) for v in variants.values() for _, stem, sets in frames]
    g = np.stack(list(imap(K.grid_job, jobs))).reshape(len(variants), len(frames), len(P.THRESH), len(P.MERGE),
                                                       len(P.MIN_AREA), 4)
    res, at = {}, {}
    for name, gv in zip(variants, g):
        pq = P.pq_of(gv.sum(axis=0))
        i, j, a = np.unravel_index(np.argmax(pq), pq.shape)
        at[name] = gv[:, i, j, a]
        res[name] = dict(point=[P.THRESH[i], P.MERGE[j], P.MIN_AREA[a]], **K.summary(at[name].sum(axis=0)))
    return res, at


def xm_main(src, d, out, procs, smoke=False):
    d, out = Path(d), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(*a):
        print(f"[{(time.time() - t_start) / 60:6.1f} min]", *a, flush=True)

    log("models:", P.extract(src, d))
    for arm in XM_ARMS:
        assert (d / "dump" / arm).exists(), f"{arm}: attach the filament-pixnet kernel's output"
    assert (d / "dump" / K.QSEG.split("/")[0]).exists(), "attach the filament-qwen-seg kernel's output"
    yolo = K.load_yolo(src)
    if smoke:
        for f in range(5):
            yolo.setdefault(f, {})
    assert sorted(yolo) == [0, 1, 2, 3, 4], "every fold needs its own YOLO"
    truth = K.fold_truth(smoke)
    folds = np.concatenate([[n] * len(t["frames"]) for n, t in enumerate(truth)])
    t0 = truth[0]
    s0 = [s for _, s, _ in t0["frames"]]
    test_files = t0["meta"]["test_files"]
    test_stems = [Path(n).stem for n in test_files]
    pool = mp.Pool(procs) if procs > 1 else None
    imap = pool.imap if pool else map
    res = dict(photographs=len(folds), fold0=len(s0), arms=list(XM_ARMS), weight=XM_W)
    saved = {}

    # 1. each map alone on fold 0, plain
    alone = {a: f"{a}/plain" for a in XM_ARMS}
    alone.update(qseg_pre=K.QSEG, r50_plain="r50/plain")
    res["alone"], at = xm_best(d, alone, t0["frames"], imap)
    res["alone_diff"] = {f"{a}_minus_{b}": K.boot(at[a], at[b]) for a in XM_ARMS for b in ("r50_plain", "qseg_pre")}
    res["alone_diff"]["pixnet_scratch_minus_pixnet_pre"] = K.boot(at["pixnet_scratch"], at["pixnet_pre"])
    log("alone", res["alone"], res["alone_diff"])

    # 2. fold 0's averaged maps with PixNet in, each at its own best grid point (no rule)
    members = K.ENS0.split("+")
    q = (K.QSEG, K.QSEG_W)
    sc, pr = ("pixnet_scratch/plain", XM_W), ("pixnet_pre/plain", XM_W)
    mixes = {"ens0": f"ens:{K.ENS0}/{K.VIEW}",
             "ens0q": xm_mix(d, "xq0", members, [q], "val", s0),
             "ens0x": xm_mix(d, "xx0", members, [sc], "val", s0),
             "ens0qx": xm_mix(d, "xqx0", members, [q, sc], "val", s0),
             "ens0qxp": xm_mix(d, "xqxp0", members, [q, sc, pr], "val", s0)}
    res["ens0_grid"], at_g = xm_best(d, mixes, t0["frames"], imap)
    res["ens0_grid_diff"] = {f"{a}_minus_{b}": K.boot(at_g[a], at_g[b]) for a, b in
                             (("ens0q", "ens0"), ("ens0x", "ens0"), ("ens0qx", "ens0q"), ("ens0qxp", "ens0q"))}
    log("ens0 grid", res["ens0_grid"], res["ens0_grid_diff"])

    # 3. gap filling: the pair out-of-fold on every photograph, and fold 0's ensembles
    g = np.stack(list(imap(xm_close_job, [(str(d), f"{K.PAIR[n]}/{K.VIEW}", stem, sets)
                                          for n, t in enumerate(truth) for _, stem, sets in t["frames"]])))
    res["close_pair_oof"] = xm_close_compare(g)
    saved["close_pair_oof"] = g
    log("closing, pair out-of-fold", res["close_pair_oof"])
    for name in ("ens0q", "ens0qx"):
        g = np.stack(list(imap(xm_close_job, [(str(d), mixes[name], stem, sets) for _, stem, sets in t0["frames"]])))
        res[f"close_{name}"] = xm_close_compare(g)
        saved[f"close_{name}"] = g
        log(f"closing, {name}", res[f"close_{name}"])

    # 4. the 25-column rule: trained on the pair's candidates of folds 1-4, cut nested in fold 0
    GY = K.GEO_YOLO_COLS
    cache = d / f"kfold_pool_rle_{K.VIEW}"
    cache.mkdir(parents=True, exist_ok=True)
    frames = K.one_setting(list(imap(K.pool_job, [(str(d), f"{K.PAIR[n]}/{K.VIEW}", stem, t["cache"], f, sets, str(cache), True)
                                                  for n, t in enumerate(truth) for f, stem, sets in t["frames"]])))
    frames = K.add_neighbours(frames, d, [f"{K.PAIR[g_]}/{K.VIEW}" for g_ in folds], "val", folds, imap)
    frames = K.add_yolo(frames, [yolo[g_] for g_ in folds], "val")
    log("rule frames", int(sum(len(f["X"]) for f in frames)), "candidates")
    tr, nested = {}, {}
    for name, variant in mixes.items():
        e = xm_ens0_frames(d, t0, variant, name, imap, yolo)
        tr[name], nested[name] = K.transfer(frames, folds, e, GY)
        saved[f"nested_{name}"] = nested[name]
        log("transfer", name, tr[name]["on_ens0_nested_cut"], "best cut", tr[name]["on_ens0_best_cut"])
    res["transfer"] = tr
    res["transfer_diff"] = {f"{a}_minus_{b}": K.boot(nested[a], nested[b]) for a, b in
                            (("ens0q", "ens0"), ("ens0x", "ens0"), ("ens0qx", "ens0q"), ("ens0qxp", "ens0q"),
                             ("ens0qxp", "ens0qx"))}
    log("transfer diff", res["transfer_diff"])

    # 5. test CSVs: the ten keep each extra map's share (2 of 14 as 1 of 7); the 0.39 file first, as the check
    ten = K.FAMILIES["r50"] + K.FAMILIES["r34"]
    q2, sc2, pr2 = (K.QSEG, 2 * K.QSEG_W), ("pixnet_scratch/plain", 2 * XM_W), ("pixnet_pre/plain", 2 * XM_W)
    final = K.fit(frames, np.arange(len(frames)), GY)
    files = {"kf10q_rule_geo_yolo": (xm_mix(d, "xq10", ten, [q2], "test", test_stems), "ens0q"),
             "kf10qx_rule_geo_yolo": (xm_mix(d, "xqx10", ten, [q2, sc2], "test", test_stems), "ens0qx"),
             "kf10qxp_rule_geo_yolo": (xm_mix(d, "xqxp10", ten, [q2, sc2, pr2], "test", test_stems), "ens0qxp")}
    res["files"] = {}
    for fname, (variant, key) in files.items():
        cut = tr[key]["on_ens0_best_cut"]["cut"]
        res["files"][fname] = K.write_csv(d, variant, test_files, out / f"submission_{fname}.csv", final, cut, GY, yolo=yolo)
        res["files"][fname]["cut"] = cut
        res["files"][fname]["verify"] = K.verify(out / f"submission_{fname}.csv")

    # maps of a few fold-0 photographs for the figure, and the per-photograph counts
    show = out / "maps"
    show.mkdir(exist_ok=True)
    for s in s0[:XM_SHOW]:
        for name, v in (("pixnet_scratch", "pixnet_scratch/plain"), ("pixnet_pre", "pixnet_pre/plain"),
                        ("qseg", K.QSEG), ("ens0q", mixes["ens0q"]), ("ens0qx", mixes["ens0qx"])):
            cv2.imwrite(str(show / f"{s}_{name}.png"), P.load_map(d, v, "val", s))
    np.savez_compressed(out / "xm_counts.npz", stems0=np.array(s0), **saved)
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    json.dump(res, open(out / "pixnet_mix.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def kaggle_pixnet_mix():
    """filament-pixnet-mix: the dump kernels' tars, both YOLO kernels', the Qwen segmenter's and PixNet's outputs."""
    xm_main(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, run.os.cpu_count() or 4, smoke=XM_SMOKE)


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    """Local check of the gap filling on a drawn map: c = 0 must equal run.components."""
    p = np.zeros((run.FULL, run.FULL), np.uint8)
    p[1000:1012, 400:600] = 230                   # one filament in three pieces, 4 px and 8 px gaps
    p[1000:1012, 604:700] = 230
    p[1000:1012, 708:800] = 230
    p[1500:1510, 1500:1505] = 230                 # a speck far away
    for k in (0, 3, 7):
        lab0, area0 = run.components(p, 0.65, k)
        lab, area = xm_components(p, 0.65, k, 0)
        assert (lab0 == lab).all() and (area0 == area).all(), k
    for c in XM_CLOSE:
        lab, area = xm_components(p, 0.65, 7, c)
        print(f"close {c:2d}: groups {int((area[1:] > 0).sum())}, filament area {int(area[lab[1005, 450]])}, "
              f"gap pixel labelled {bool(lab[1005, 602])}/{bool(lab[1005, 705])}")


if __name__ == "__main__":
    main()
