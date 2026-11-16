"""Figures of the FilNet report, written into report/overleaf/:
  fig_pipeline.pdf  the submitted pipeline (drawn here)
  fig_inputs.jpg    four of the seven input channels on a validation photograph
  fig_test.jpg      the submitted file's masks on a test photograph
  fig_morph.jpg     validation photographs: experts' outlines and FilNet's kept filaments

    python report/make_figs_filnet.py <submission.csv> [folds, e.g. 3]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import collections, csv, json, pickle, sys, tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "overleaf"
CODE = ROOT / "kernels" / "aug_unet"
sys.path.insert(0, str(CODE))
NET, PHYS, POST = "#DCEAF6", "#E3F2DC", "#EEEEEE"


def box(ax, x, y, w, h, text, color, size=5.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=color, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"))


def pipeline(n_folds, cut):
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.00, 0.62, 0.14, 0.30, "GONG\nH$\\alpha$\n2048$^2$", POST)
    box(ax, 0.21, 0.56, 0.33, 0.42, "7 physical channels\n(no weights):\nbrightness; contrast\nto the quiet Sun;\nlocal darkness, 2 scales;\nlatitude, longitude, $\\mu$", PHYS, 4.9)
    box(ax, 0.61, 0.56, 0.39, 0.42, f"FilNet $\\times$ {n_folds} folds\nresidual U-Net,\n17.6 M weights, trained\nfrom scratch on 512 px\nwindows; inference on the\nfull frame with D4", NET, 4.9)
    box(ax, 0.61, 0.04, 0.39, 0.40, "mean map\n$\\downarrow$ candidates:\nthreshold 0.6, join 3 px,\narea $\\geq$ 100 px", POST, 4.9)
    box(ax, 0.21, 0.04, 0.33, 0.40, f"selection rule\nGBM, 21 features:\nappearance,\nheliographic lat. & long.,\naxis orientation;\ncut {cut}", POST, 4.9)
    box(ax, 0.00, 0.12, 0.14, 0.24, "masks\n(RLE)", POST)
    arrow(ax, 0.14, 0.77, 0.21, 0.77)
    arrow(ax, 0.54, 0.77, 0.61, 0.77)
    arrow(ax, 0.805, 0.56, 0.805, 0.44)
    arrow(ax, 0.61, 0.24, 0.54, 0.24)
    arrow(ax, 0.21, 0.24, 0.14, 0.24)
    fig.savefig(OUT / "fig_pipeline.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def inputs():
    import run
    from run import torch, np, cv2
    import filnet as FN
    cache = CODE / "out" / "cache_full"
    meta = json.load(open(cache / "meta.json"))
    i = 5
    stem = Path(meta["val_files"][i]).stem
    img = np.array(np.load(cache / "val_img.npy", mmap_mode="r")[i])
    msk = np.array(np.load(cache / "val_msk.npy", mmap_mode="r")[i]) > 0
    geo = run.geo_channels(torch.tensor([[1.0, 0.0, 0.0, 1.0, run.b0_deg(stem)]]), torch.device("cpu"))
    ch = FN.front(run.to_input(img)[None, None], geo, "phys")[0].numpy()
    y0, x0, s = 420, 820, 720
    names = ["brightness", "contrast to the quiet Sun", "darker than $\\sim$30 px around", "darker than $\\sim$130 px around"]
    fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.5), gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for k, ax in enumerate(axs.ravel()):
        ax.imshow(ch[k][y0:y0 + s, x0:x0 + s], cmap="gray", vmin=-1.6 if k else -0.6, vmax=1.6 if k else 0.6)
        if k == 0:
            ax.contour(msk[y0:y0 + s, x0:x0 + s], [0.5], colors="#3CDC3C", linewidths=0.5)
        ax.set_title(names[k], fontsize=6.5, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "fig_inputs.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02, pil_kwargs=dict(quality=88))
    plt.close(fig)
    print("inputs figure:", stem)


def test_morphology(sub):
    import cv2
    import numpy as np
    from pycocotools import mask as mu
    test_dir = next(p for p in (ROOT / "MAGFiLO_1.0_Kaggle_2026").rglob("test_images") if p.is_dir())
    by = collections.defaultdict(list)
    with open(sub) as f:
        for row in csv.DictReader(f):
            by[row["filament_id"].rsplit("_", 1)[0]].append(row["segmentation_rle"])
    stem = "20130530133654Bh" if "20130530133654Bh" in by else max(by, key=lambda s: len(by[s]))
    img = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    masks = [mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(np.uint8) for r in by[stem]]
    for m in masks:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cs, -1, (255, 140, 0), 2)
    areas = [int(m.sum()) for m in masks]
    big = [masks[i] for i in np.argsort(areas)[::-1][:2]]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.04))
    axs[0].imshow(rgb[::2, ::2])
    axs[0].set_title(f"{stem}: {len(masks)} filaments", fontsize=7)
    for ax, m in zip(axs[1:], big):
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = max(150, int(max(np.ptp(ys), np.ptp(xs)) * 0.65))
        y0, x0 = max(cy - h, 0), max(cx - h, 0)
        ax.imshow(rgb[y0:cy + h, x0:cx + h])
        axs[0].add_patch(plt.Rectangle((x0 / 2, y0 / 2), h, h, fill=False, ec="#56B4E9", lw=0.8))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "fig_test.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02, pil_kwargs=dict(quality=88))
    plt.close(fig)
    print("test photograph", stem, len(masks), "filaments")


def morph(cut):
    """Two fold-0 validation photographs: the experts' outlines and FilNet's candidates kept by the rule fitted
    on the other folds' rows."""
    import run
    from run import np, cv2
    import candidate_pool as C
    import solar_unet as S
    import time_derot as TD
    o = CODE / "out"
    other = [f for k in (1, 2, 3, 4) if (o / f"filnet_phys_d4_frames_f{k}.pkl").exists()
             for f in pickle.load(open(o / f"filnet_phys_d4_frames_f{k}.pkl", "rb"))]
    rule = S._fit(other, "X", True, 0)
    f0 = TD.folds()[0]
    # the two fold-0 photographs with three experts and the most outlines
    order = sorted(range(len(f0["stems"])), key=lambda i: (-len(f0["sets"].get(i, [])),
                                                           -sum(len(c) for c in f0["sets"].get(i, []))))
    want = [f0["stems"][i] for i in order[:2]]
    maps = {}
    with tarfile.open(ROOT / "kernels" / "filnet" / "out_kaggle" / "dump_filnet_phys.tar") as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).parent.name == "val_d4" and Path(m.name).stem in want:
                maps[Path(m.name).stem] = cv2.imdecode(np.frombuffer(t.extractfile(m).read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    tr_dir = run.find_data()[1]
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.78), gridspec_kw=dict(wspace=0.03))
    for ax, stem in zip(axs, want):
        img = cv2.imread(str(tr_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        sets = f0["sets"].get(f0["stems"].index(stem), [])
        union = np.zeros(img.shape, np.uint8)
        for crops in sets:
            for y0, x0, g in crops:
                full = np.zeros(img.shape, np.uint8)
                full[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
                union |= full
                cs, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cs, -1, (60, 220, 60), 2)
        cands, X = C.candidates(maps[stem], img, run.b0_deg(stem))
        k = [n for n, c in enumerate(cands) if C.ONE_SETTING in c["from"]]
        sc = rule.predict(X[k]) if k else []
        for n, q in zip(k, sc):
            if q >= cut:
                c = cands[n]
                full = np.zeros(img.shape, np.uint8)
                full[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
                cs, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cs, -1, (255, 140, 0), 2)
        n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(union, np.ones((61, 61), np.uint8)))
        cx, cy = cen[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))].astype(int)
        s = 620
        y0, x0 = int(np.clip(cy - s // 2, 0, 2048 - s)), int(np.clip(cx - s // 2, 0, 2048 - s))
        ax.imshow(rgb[y0:y0 + s, x0:x0 + s])
        ax.set_xticks([]); ax.set_yticks([])
        print("morph:", stem, len(sets), "expert sets")
    fig.savefig(OUT / "fig_morph.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02, pil_kwargs=dict(quality=88))
    plt.close(fig)


if __name__ == "__main__":
    sub = Path(sys.argv[1])
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.24
    pipeline(n_folds, cut)
    test_morphology(sub)
    inputs()
    morph(cut)
    print("wrote the figures into", OUT)
