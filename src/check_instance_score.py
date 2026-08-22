"""Give every candidate filament a confidence, instead of one threshold for all.

The pipeline decides what counts as a filament with two global numbers: how
confident the spine head must be, and how long the component must be. Every
candidate is judged by the same pair, so a long faint filament and a short faint
speck are treated alike -- and calibration, forced to choose, keeps cutting until
the specks are gone and takes real filaments with them. Measured: at the strict
setting the model misses 217 filaments and still emits 186 false ones.

An instance-level score fixes that. It is the one thing a detector like YOLO has
that this pipeline does not, and it does not need a different framework: the
maps already carry the evidence -- how confident the spine is along its whole
length, how long it is, how strongly the semantic head agrees, how far out on the
disk it sits. A small logistic model over those, fitted on one half of the
validation photographs and scored on the other, says whether ranking beats
thresholding.

Candidates are generated at a deliberately loose setting so the scorer has both
the real filaments and the junk to separate.
"""
from __future__ import annotations

import argparse, collections, json, os, sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as cc_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))
from spine_data import load_annotations, make_split, DISK_CX, DISK_CY, DISK_R, FULL  # noqa: E402
from spine_model import build_spine_model                     # noqa: E402
from spine_infer import order_path, fit_spline, sweep         # noqa: E402
from spine_train2 import load_stack, predict_frame            # noqa: E402
from metrics import pq_from_masks, aggregate                  # noqa: E402
from evaluate import gt_masks_full                            # noqa: E402
from gpu_guard import Governor                                # noqa: E402
from gpu_lock import GpuLock                                  # noqa: E402

FEATURES = ["spine_mean", "spine_max", "spine_p10", "length", "sem_mean",
            "area", "radius_mean", "rho", "elongation"]


def candidates(spn, rad, sem, spine_thresh, min_spine_px, sem_thresh=0.5,
               min_area=60):
    """Assembled masks plus the evidence behind each, at a loose threshold."""
    from scipy.ndimage import binary_dilation
    binm = spn >= spine_thresh
    if not binm.any():
        return [], []
    lab, n = cc_label(binm, structure=np.ones((3, 3)))
    sem_bin = binary_dilation(sem >= sem_thresh, iterations=2)
    masks, feats = [], []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < min_spine_px:
            continue
        p = spn[ys, xs]
        path = order_path(np.c_[xs, ys])
        curve = fit_spline(path, smooth_per_point=2.0)
        cx = np.clip(curve[:, 0].astype(int), 0, FULL - 1)
        cy = np.clip(curve[:, 1].astype(int), 0, FULL - 1)
        radii = rad[cy, cx]
        m = sweep(curve, radii, shape=spn.shape) & sem_bin
        a = int(m.sum())
        if a < min_area:
            continue
        d = np.hypot(xs - xs.mean(), ys - ys.mean())
        masks.append(m)
        feats.append([
            float(p.mean()), float(p.max()), float(np.percentile(p, 10)),
            float(len(xs)), float(sem[m].mean()) if a else 0.0, float(a),
            float(radii.mean()),
            float(np.hypot(xs.mean() - DISK_CX, ys.mean() - DISK_CY) / DISK_R),
            float(len(xs)) / max(float(a), 1.0),
        ])
    return masks, feats


def iou(a, b):
    i = np.logical_and(a, b).sum()
    return float(i) / float(np.logical_or(a, b).sum()) if i else 0.0


def fit_logistic(X, y, epochs=3000, lr=0.1):
    """Plain logistic regression; sklearn is not installed and this is nine features."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    Z = np.c_[Z, np.ones(len(Z))]
    w = np.zeros(Z.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y)) / len(Z)
    return w, mu, sd


def score(X, w, mu, sd):
    Z = np.c_[(X - mu) / (sd + 1e-8), np.ones(len(X))]
    return 1.0 / (1.0 + np.exp(-Z @ w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_v4_best.pt"))
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--loose-thresh", type=float, default=0.70)
    ap.add_argument("--loose-min-px", type=int, default=100)
    ap.add_argument("--base-thresh", type=float, default=0.90)
    ap.add_argument("--base-min-px", type=int, default=700)
    args = ap.parse_args()

    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)
    val_ids = val_ids[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = int(ck["model"]["stem.0.weight"].shape[1])
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    gov = Governor(target_util=55, check_every=2)
    lock = GpuLock("filament_detection", "instance score", wait=True).acquire()
    per_set = []
    try:
        for k, iid in enumerate(val_ids, 1):
            stem = Path(fname[iid]).stem
            stack = load_stack(stem, "train", in_ch > 1)
            with gov.step():
                sem, spn, rad = predict_frame(model, stack, device, tta=True)
            sem *= ondisk; spn *= ondisk
            gts = gt_masks_full(by_id[iid])
            masks, feats = candidates(spn, rad, sem, args.loose_thresh,
                                      args.loose_min_px)
            hits = [max((iou(m, g) for g in gts), default=0.0) > 0.5 for m in masks]
            base, _ = candidates(spn, rad, sem, args.base_thresh, args.base_min_px)
            per_set.append(dict(gts=gts, masks=masks, feats=np.array(feats,
                                dtype=np.float64).reshape(-1, len(FEATURES)),
                                hits=np.array(hits, float), base=base))
            if k % 10 == 0:
                lock.progress(f"{k}/{len(val_ids)}")
                print(f"  {k}/{len(val_ids)}", flush=True)
    finally:
        lock.release()

    n_cand = sum(len(p["masks"]) for p in per_set)
    n_hit = sum(int(p["hits"].sum()) for p in per_set)
    print(f"\n{n_cand} candidates at the loose setting, {n_hit} of them real "
          f"({100*n_hit/max(n_cand,1):.1f}%)")

    # fit on one half of the photographs, score on the other
    half = len(per_set) // 2
    A, B = per_set[:half], per_set[half:]
    X = np.vstack([p["feats"] for p in A if len(p["feats"])])
    y = np.concatenate([p["hits"] for p in A if len(p["hits"])])
    w, mu, sd = fit_logistic(X, y)
    print("\nwhat the score keys on (standardised weights):")
    for f, v in sorted(zip(FEATURES, w[:-1]), key=lambda t: -abs(t[1])):
        print(f"   {f:12s} {v:+.3f}")

    base = aggregate([pq_from_masks(p["gts"], p["base"]) for p in B])
    print(f"\nheld-out half ({len(B)} sets)")
    print(f"  global threshold {args.base_thresh}/{args.base_min_px}: "
          f"PQ {base['PQ']:.4f}  TP {base['TP']:3d} FP {base['FP']:3d} FN {base['FN']:3d}")

    print("  instance score, by cutoff:")
    best = None
    for cut in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        res = []
        for p in B:
            if len(p["feats"]):
                s = score(p["feats"], w, mu, sd)
                keep = [m for m, v in zip(p["masks"], s) if v >= cut]
            else:
                keep = []
            res.append(pq_from_masks(p["gts"], keep))
        a = aggregate(res)
        mark = ""
        if best is None or a["PQ"] > best[1]["PQ"]:
            best = (cut, a); mark = ""
        print(f"    cut {cut:.1f}: PQ {a['PQ']:.4f}  TP {a['TP']:3d} "
              f"FP {a['FP']:3d} FN {a['FN']:3d}{mark}")
    cut, a = best
    d = a["PQ"] - base["PQ"]
    print(f"\n  best cutoff {cut:.1f} -> PQ {a['PQ']:.4f} vs {base['PQ']:.4f}  "
          f"({d:+.4f})")
    print("  " + ("ranking beats thresholding" if d > 0 else
                  "no gain -- the global threshold is already doing this job"))

    json.dump({"weights": dict(zip(FEATURES + ["bias"], w.tolist())),
               "mu": mu.tolist(), "sd": sd.tolist(), "best_cut": cut,
               "held_out_baseline": base, "held_out_scored": a},
              open(ROOT / "instance_score.json", "w"), indent=1)
    print(f"\ngovernor: {gov.summary()}")
    print("wrote instance_score.json")


if __name__ == "__main__":
    main()
