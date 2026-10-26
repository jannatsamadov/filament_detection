"""An image-based keep-rule: a small CNN decides which candidate components to keep.

Post v3 (HANDOFF section 0) put the remaining PQ in the choice of components, not
in their outlines: on the 5-model ensemble's candidates (t 0.6, merge 3, >= 100 px)
a perfect keep-rule scores 0.519 on validation, against 0.435 for the grid and
0.440 for the learned keep-rule, which sees only ten hand-made features. This one
sees the photograph around each candidate.

Each candidate becomes a crop around its bounding box, resized to SIDE px, with
three channels -- the photograph, the ensemble probability, the component's own
mask (other candidates at half weight) -- plus the ten hand features and the crop
scale. The target is the candidate's matched IoU averaged over the frame's
annotation sets, as for the learned keep-rule; a candidate is kept where the
prediction exceeds a cut chosen on the out-of-fold predictions.

Stage 1 (this file): only the 141 validation photographs have honest candidates
(no model trained on them), so the area rule, the GBM keep-rule, the CNN and the
two averaged are all scored out-of-fold over the same five photograph-grouped
folds the GBM keep-rule always used, next to the oracle. A paired bootstrap over
photographs says whether a gain over the GBM is more than noise. Then a final CNN
trained on every validation candidate writes the test CSVs.

On Kaggle this is the kernel filament-keep-cnn, built by kernels/build.py; it
runs on the CPU unless a GPU is attached. Locally, on dumped maps:
    python kernels/aug_unet/keep_cnn.py --dir <folder with dump_*.tar> --smoke
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- off Kaggle this hides the GPU and refuses if it is visible
from post_local import (MIN_AREA, comp_features, extract, ground_truth, load_map,  # noqa: E402
                        matched_iou, pq_of, swap)

# ---------------------------------------------------------------- keep-cnn settings

CNN_MODELS = ("flip", "full", "r50", "solar", "union")
CNN_VIEW = "plain"
CNN_T, CNN_MERGE, CNN_AMIN = 0.6, 3, 100   # where post v3's oracle peaked
CNN_SIDE, CNN_PAD, CNN_WIN = 128, 24, (96, 512)   # bbox + PAD a side, clipped to WIN, resized to SIDE
CNN_EPOCHS, CNN_BATCH, CNN_LR, CNN_FOLDS = 20, 32, 3e-4, 5
CNN_CUTS = np.round(np.arange(0.025, 0.60, 0.025), 3)


def crop_window(a, y0, x0, side):
    """a[y0:y0 + side, x0:x0 + side], zero outside the frame."""
    out = np.zeros((side, side), a.dtype)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + side, a.shape[0]), min(x0 + side, a.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def cnn_crops(img, p, lab, ids):
    """(n, 3, SIDE, SIDE) float16 crops, and each crop's log scale."""
    from scipy import ndimage
    boxes = ndimage.find_objects(lab)
    C = np.zeros((len(ids), 3, CNN_SIDE, CNN_SIDE), np.float16)
    scale = np.zeros(len(ids), np.float32)
    for n, j in enumerate(ids):
        sy, sx = boxes[j - 1]
        side = int(np.clip(max(sy.stop - sy.start, sx.stop - sx.start) + 2 * CNN_PAD, *CNN_WIN))
        y0, x0 = (sy.start + sy.stop - side) // 2, (sx.start + sx.stop - side) // 2
        im = crop_window(img, y0, x0, side).astype(np.float32)
        lb = crop_window(lab, y0, x0, side)
        chans = ((im - np.median(im)) / 32.0,                       # filaments are tens of levels deep
                 crop_window(p, y0, x0, side).astype(np.float32) / 255.0,
                 (lb == j) + 0.5 * ((lb > 0) & (lb != j)))
        interp = cv2.INTER_AREA if side > CNN_SIDE else cv2.INTER_LINEAR
        for c, a in enumerate(chans):
            C[n, c] = cv2.resize(np.asarray(a, np.float32), (CNN_SIDE, CNN_SIDE), interpolation=interp)
        scale[n] = np.log(side / CNN_SIDE)
    return C, scale


def cnn_candidates_job(job):
    """One frame's candidates: features (10 hand + scale), crops, matched IoU per set."""
    d, variant, split, path, sets = job
    stem = Path(path).stem
    p = load_map(Path(d), variant, split, stem)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    lab, area = run.components(p, CNN_T, CNN_MERGE)
    ids = [j for j in range(1, len(area)) if area[j] >= CNN_AMIN]
    n_gt = [len(c) for _, c in sets]
    if not ids:
        return (stem, np.zeros((0, 11), np.float32), np.zeros((0, 3, CNN_SIDE, CNN_SIDE), np.float16),
                np.zeros((0, len(sets))), n_gt)
    C, scale = cnn_crops(img, p, lab, ids)
    X = np.concatenate([comp_features(p, img, lab, area, ids), scale[:, None]], 1).astype(np.float32)
    return stem, X, C, matched_iou(lab, area, ids, sets), n_gt


# ---------------------------------------------------------------- the network

_R18 = None


def cnn_net(n_feat):
    """ResNet-18 on the crop, its 512 features joined with the scalar ones."""
    global _R18
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    if _R18 is None:
        try:
            _R18 = ("imagenet", resnet18(weights="IMAGENET1K_V1").state_dict())
        except Exception as e:                   # no internet: learn from scratch
            _R18 = (f"random ({type(e).__name__})", None)
    cnn = resnet18(weights=None)
    if _R18[1] is not None:
        cnn.load_state_dict(_R18[1])
    cnn.fc = nn.Identity()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = cnn
            self.head = nn.Sequential(nn.Linear(512 + n_feat, 128), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(128, 1))

        def forward(self, x, f):
            return self.head(torch.cat([self.cnn(x), f], 1)).squeeze(1)

    return Net(), _R18[0]


def d4_view(x, k):
    """The k-th of the eight D4 views of a (B, C, H, W) batch."""
    if k & 4:
        x = x.transpose(2, 3)
    if k & 1:
        x = x.flip(3)
    if k & 2:
        x = x.flip(2)
    return x


def cnn_fit(C, feats, y, epochs, seed, device):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net, init = cnn_net(feats.shape[1])
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=CNN_LR, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(y) / CNN_BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CNN_LR, total_steps=steps, pct_start=0.15)
    Ct, Ft = torch.from_numpy(C), torch.from_numpy(feats)
    yt = torch.from_numpy(y.astype(np.float32))
    net.train()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for b in range(0, len(y), CNN_BATCH):
            i = torch.from_numpy(order[b:b + CNN_BATCH])
            x = d4_view(Ct[i].float().to(device), int(rng.integers(8)))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(x, Ft[i].to(device)), yt[i].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net, init


def cnn_predict(net, C, feats, device):
    """Mean sigmoid over the eight D4 views."""
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(C), 128):
            x = torch.from_numpy(C[b:b + 128]).float().to(device)
            f = torch.from_numpy(feats[b:b + 128]).to(device)
            out.append(torch.stack([torch.sigmoid(net(d4_view(x, k), f)) for k in range(8)])
                       .mean(0).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def gbm_model():
    """The learned keep-rule's regressor, as post_local.keep_rule fits it."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=0)


def standardise(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [((a - mu) / sd).astype(np.float32) for a in (train,) + others]


# ---------------------------------------------------------------- scoring

def frame_counts(frames, score, cut):
    """Per frame (TP, FP, FN, summed IoU) when components with score > cut are kept."""
    out = np.zeros((len(frames), 4))
    start = 0
    for f_i, (_, Xf, Yf, n_gt) in enumerate(frames):
        keep = score[start:start + len(Xf)] > cut
        start += len(Xf)
        for s_i, ng in enumerate(n_gt):
            iou = Yf[keep, s_i] if len(Xf) else np.zeros(0)
            t_ = int((iou > 0).sum())
            out[f_i] += (t_, int(keep.sum()) - t_, ng - t_, float(iou.sum()))
    return out


def best_cut(frames, score, cuts):
    """The cut with the highest pooled PQ, its PQ and per-frame counts."""
    rows = [(c, frame_counts(frames, score, c)) for c in cuts]
    c, fc = max(rows, key=lambda r: pq_of(r[1].sum(0)))
    tot = fc.sum(0)
    return dict(cut=float(c), PQ=float(pq_of(tot)), SQ=float(tot[3] / max(tot[0], 1)),
                TP=int(tot[0]), FP=int(tot[1]), FN=int(tot[2]),
                curve=[(float(cc), float(pq_of(f.sum(0)))) for cc, f in rows]), fc


def paired_bootstrap(fc_a, fc_b, n=2000, seed=0):
    """PQ(a) - PQ(b) over photographs resampled with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fc_a), (n, len(fc_a)))
    diff = np.array([pq_of(fc_a[i].sum(0)) - pq_of(fc_b[i].sum(0)) for i in idx])
    return dict(mean=float(diff.mean()), lo5=float(np.percentile(diff, 5)),
                hi95=float(np.percentile(diff, 95)), p_gain=float((diff > 0).mean()))


# ---------------------------------------------------------------- stage 1

def cnn_write(d, variant, test_frames, scores, cut, out):
    """Submission CSV: the test candidates whose score exceeds the cut."""
    rows, per, start = [], [], 0
    for stem, n in test_frames:
        sc, start = scores[start:start + n], start + n
        p = load_map(d, variant, "test", stem)
        lab, area = run.components(p, CNN_T, CNN_MERGE)
        ids = [j for j in range(1, len(area)) if area[j] >= CNN_AMIN]
        assert len(ids) == n, (stem, len(ids), n)
        keep = [j for j, s in zip(ids, sc) if s > cut]
        for k, j in enumerate(keep, 1):
            r = run.mu.encode(np.asfortranarray((lab == j).astype(np.uint8)))
            rows.append(f"{stem}_{k},{r['counts'].decode('ascii')}")
        per.append(len(keep))
    out.write_text("filament_id,segmentation_rle\n" + "".join(r + "\n" for r in rows))
    return len(rows), float(np.mean(per))


def keep_cnn_main(src, d, out_dir, procs, smoke=False, epochs=CNN_EPOCHS):
    t0 = time.time()
    dumped = extract(src, d)
    models = [m for m in CNN_MODELS if m in dumped and (d / "dump" / m / "val_plain").exists()]
    variant = (f"ens:{'+'.join(models)}" if len(models) > 1 else models[0]) + f"/{CNN_VIEW}"
    # on test, r50all (every photograph) also stands in for r50, as in post v3
    variants = [variant] + ([swap(variant, "r50", "r50all")]
                            if "r50all" in dumped and swap(variant, "r50", "r50all") != variant else [])
    meta, gt = ground_truth(smoke)
    _, tr_dir, te_dir = run.find_data()
    print(f"candidates at t {CNN_T} merge {CNN_MERGE} area >= {CNN_AMIN}: val {variant}, test {variants}",
          flush=True)

    # every candidate first, before torch starts its threads in this process
    by_frame = {}
    for i, g in enumerate(gt):
        by_frame.setdefault(g["frame"], []).append((i, g["crops"]))
    with mp.Pool(procs) as pool:
        val = pool.map(cnn_candidates_job, [(str(d), variant, "val", str(tr_dir / meta["val_files"][f]), sets)
                                            for f, sets in by_frame.items()], chunksize=2)
        test = {v: pool.map(cnn_candidates_job, [(str(d), v, "test", str(te_dir / f), [])
                                                 for f in meta["test_files"]], chunksize=2)
                for v in variants}
    frames = [(s, X, Y, n) for s, X, _, Y, n in val]
    X = np.concatenate([v[1] for v in val])
    C = np.concatenate([v[2] for v in val])
    y = np.concatenate([v[3].mean(axis=1) for v in val])
    # the folds the GBM keep-rule has always used: photograph n goes to fold n % 5
    fold = np.concatenate([[n % CNN_FOLDS] * len(v[1]) for n, v in enumerate(val)])
    print(f"{len(frames)} val photographs, {len(y)} candidates, {int((y > 0).sum())} match someone; "
          f"test {[sum(len(t[1]) for t in test[v]) for v in variants]} ({time.time() - t0:.0f}s)", flush=True)

    import torch
    torch.set_num_threads(procs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = {"gbm": np.zeros(len(y)), "cnn": np.zeros(len(y))}
    init = None
    for k in range(CNN_FOLDS):
        te = fold == k
        oof["gbm"][te] = gbm_model().fit(X[~te, :10], y[~te]).predict(X[te, :10])
        f_tr, f_te = standardise(X[~te], X[te])
        net, init = cnn_fit(C[~te], f_tr, y[~te], epochs, seed=k, device=device)
        oof["cnn"][te] = cnn_predict(net, C[te], f_te, device)
        print(f"  fold {k}: {int(te.sum())} candidates, CNN init {init}, {device} "
              f"({time.time() - t0:.0f}s)", flush=True)
    oof["cnn+gbm"] = (oof["cnn"] + oof["gbm"]) / 2

    # every rule picks its own cut on the same out-of-fold candidates
    scores = {"area": (X[:, 0], np.log(np.array(MIN_AREA, float)) - 1e-6), "oracle": (y, CNN_CUTS)}
    scores.update({m: (s, CNN_CUTS) for m, s in oof.items()})
    res, fcs = {}, {}
    for m, (s, cuts) in scores.items():
        res[m], fcs[m] = best_cut(frames, s, cuts)
        print(f"  {m:8s} PQ {res[m]['PQ']:.4f}  SQ {res[m]['SQ']:.3f}  TP {res[m]['TP']} FP {res[m]['FP']} "
              f"FN {res[m]['FN']}  at cut {res[m]['cut']:.3f}", flush=True)
    boot = {f"{a} - {b}": paired_bootstrap(fcs[a], fcs[b])
            for a, b in (("cnn", "gbm"), ("cnn+gbm", "gbm"), ("cnn", "area"), ("gbm", "area"))}
    for k, v in boot.items():
        print(f"  {k:14s} {v['mean']:+.4f}  90% [{v['lo5']:+.4f}, {v['hi95']:+.4f}]  "
              f"P(gain) {v['p_gain']:.2f}", flush=True)
    np.savez_compressed(out_dir / "keep_cnn_val.npz", X=X, y=y, fold=fold,
                        **{f"oof_{m.replace('+', '_')}": s for m, s in oof.items()})

    # final models on every validation candidate, then the test photographs
    f_all, = standardise(X)
    net, init = cnn_fit(C, f_all, y, epochs, seed=99, device=device)
    final_gbm = gbm_model().fit(X[:, :10], y)
    del C
    files = {}
    for v in variants:
        Xt = np.concatenate([t[1] for t in test[v]])
        _, ft = standardise(X, Xt)
        pred = {"cnn": cnn_predict(net, np.concatenate([t[2] for t in test[v]]), ft, device),
                "gbm": final_gbm.predict(Xt[:, :10])}
        pred["cnn+gbm"] = (pred["cnn"] + pred["gbm"]) / 2
        test_frames = [(t[0], len(t[1])) for t in test[v]]
        for m in ("cnn", "cnn+gbm"):
            name = (f"submission_keepcnn_{v.replace('/', '_').replace(':', '-').replace('+', '-')}"
                    f"_{m.replace('+', '-')}.csv")
            n, mean = cnn_write(d, v, test_frames, pred[m], res[m]["cut"], out_dir / name)
            files[f"{v} {m}"] = dict(file=name, rows=n, per_image=mean, cut=res[m]["cut"],
                                     val_PQ=res[m]["PQ"])
            print(f"  {v} {m} -> {name}: {n} rows, {mean:.1f} per image", flush=True)
    json.dump(dict(variant=variant, point=dict(thresh=CNN_T, merge=CNN_MERGE, min_area=CNN_AMIN),
                   crop=dict(side=CNN_SIDE, pad=CNN_PAD, window=CNN_WIN), epochs=epochs, init=init,
                   n_candidates=int(len(y)), results=res, bootstrap=boot, files=files),
              open(out_dir / "keep_cnn.json", "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s", flush=True)
    return files


def kaggle_keep_cnn():
    """The kernel's entry point: the dump kernels' tars arrive under /kaggle/input."""
    keep_cnn_main(Path("/kaggle/input"), Path("/tmp/fil_post"), run.WORK, procs=run.os.cpu_count() or 4)


# ---------------------------------------------------------------- CLI
# Everything below is local only; kernels/build.py stops copying here.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "out_kaggle"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="the local 10-frame validation")
    ap.add_argument("--epochs", type=int, default=CNN_EPOCHS)
    args = ap.parse_args()
    d = Path(args.dir)
    keep_cnn_main(d, d, d, args.procs, args.smoke, args.epochs)


if __name__ == "__main__":
    main()
