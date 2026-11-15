"""Three models on the same three H-alpha photographs, two figures each (user, 11-15).

  <model>_pixels.png     the pixels the model calls filament (probability >= 0.5, as in the pixel confusion matrix)
  <model>_filaments.png  its filaments after grouping and the learned selector: TP blue (matches an expert's,
                         IoU > 0.5), FP orange (no expert filament there), FN magenta (an expert's filament it missed)

Photographs: fold-0 validation photographs drawn by one expert only (so TP/FP/FN are unambiguous), at the 25th,
50th and 90th percentile of the expert's filament count. The selector deciding a photograph was fitted on the
other half of fold 0 (stage2_baselines.py / rf_grouping.py, the same held-out halves), so these are the decisions
the confusion tables count.
    python kernels/aug_unet/compare_fig.py     -> figures/compare/*.png
"""
import json, pickle, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402  -- also points run.CACHE at the full fold-0 cache
import pixel_cls as PC  # noqa: E402
import run  # noqa: E402
from run import np  # noqa: E402
import candidate_pool as C  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = HERE.parents[1] / "figures" / "compare"
# tag -> (pixel_cls model, frames pickle, its one setting (t, dilation radius), shown name)
MODELS = {
    "rf": ("RF, table + U-Net pair", "stage2_rf,_table__u-net_pair.pkl", (0.5, 12),
           "RF (35 columns + U-Net pair), join 24 px + selector"),
    "pixnet": ("PixNet, 30 epochs", "stage2_pixnet,_30_epochs.pkl", (0.6, 3),
               "PixNet, join 6 px + selector"),
    "top": ("5 U-Nets + Qwen (submitted mix)", "stage2_5_u-nets__qwen.pkl", (0.7, 7),
            "5 U-Nets + Qwen (the submitted map), join 14 px + selector"),
}
COL = dict(TP="#1f6fd1", FP="#ff9f1c", FN="#d81b60", PIX="#ff3b30")
CROP = slice(110, 1938)            # the disk (radius 904 px about 1024)


def selected(frames, pos, cands, cut_cache):
    """The held-out-halves decision for frames[pos], exactly as solar_unet.stage2_eval makes it."""
    idx = np.random.default_rng(0).permutation(len(frames))
    halves = [sorted(idx[::2]), sorted(idx[1::2])]
    h = 0 if pos in halves[0] else 1
    if h not in cut_cache:
        fit = [frames[i] for i in halves[1 - h]]
        inner = [fit[0::2], fit[1::2]]
        oof = [S._scores(S._fit(inner[1 - j], "X", True, 0), inner[j], "X") for j in range(2)]
        cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
        pq = [S.summary(S.tally(inner[0] + inner[1], oof[0] + oof[1], c, True))["PQ"] for c in cuts]
        cut_cache[h] = (S._fit(fit, "X", True, 0), float(cuts[int(np.argmax(pq))]))
    model, cut = cut_cache[h]
    f = frames[pos]
    s = model.predict(f["X"])
    return [n for n in C.select(s, f["conflict"], f["one"]) if s[n] >= cut], cut


def full(c):
    m = np.zeros((run.FULL, run.FULL), bool)
    m[c["y0"]:c["y0"] + c["m"].shape[0], c["x0"]:c["x0"] + c["m"].shape[1]] = c["m"]
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    vimg = np.load(run.CACHE / "val_img.npy", mmap_mode="r")
    sets = {}
    for g in gt:
        sets.setdefault(g["frame"], []).append(g["crops"])
    single = sorted((len(v[0]), i) for i, v in sets.items() if len(v) == 1)
    pick = [single[int(q * (len(single) - 1))][1] for q in (0.25, 0.5, 0.9)]
    stems = [Path(meta["val_files"][i]).stem for i in pick]
    print("photographs:", stems, [len(sets[i][0]) for i in pick], flush=True)
    D = PC.Dumps()
    summary = {}
    for tag, (name, pkl, one, shown) in MODELS.items():
        frames = pickle.load(open(HERE / "out" / pkl, "rb"))
        pos_of = {f["stem"]: n for n, f in enumerate(frames)}
        C.ONE_SETTING = one
        figs = {k: plt.subplots(1, 3, figsize=(21, 7.6)) for k in ("pixels", "filaments")}
        cut_cache, summary[tag] = {}, {}
        for col, (i, stem) in enumerate(zip(pick, stems)):
            img = np.array(vimg[i])
            p = D.model(name, stem, {})
            truth = np.zeros((run.FULL, run.FULL), bool)
            gmasks = []
            for y0, x0, g in sets[i][0]:
                m = np.zeros((run.FULL, run.FULL), bool)
                m[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
                gmasks.append(m)
                truth |= m
            # 1. pixels
            pred = p >= 128
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            ax = figs["pixels"][1][col]
            rgb = np.repeat(img[..., None], 3, 2).astype(np.float32) / 255.0
            rgb[pred] = np.array(matplotlib.colors.to_rgb(COL["PIX"]))
            ax.imshow(rgb[CROP, CROP])
            ax.set_title(f"{stem}\nfilament pixels: {pred.sum():,}  (expert: {truth.sum():,})\n"
                         f"precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn, 1):.2f}  "
                         f"F1 {2 * tp / max(2 * tp + fp + fn, 1):.2f}", fontsize=11)
            ax.axis("off")
            # 2. filaments
            cands, X = C.candidates(p, img, run.b0_deg(stem))
            f = frames[pos_of[stem]]
            assert len(cands) == len(f["one"]) and np.array_equal(
                np.array([C.ONE_SETTING in c["from"] for c in cands]), f["one"]), f"{tag} {stem}: pool differs"
            chosen, cut = selected(frames, pos_of[stem], cands, cut_cache)
            pm = [full(cands[n]) for n in chosen]
            matched_g, kinds = set(), []
            for m in pm:
                best, bj = 0.0, None
                for j, g in enumerate(gmasks):
                    inter = (m & g).sum()
                    if inter:
                        iou = inter / (m | g).sum()
                        if iou > best:
                            best, bj = iou, j
                if best > 0.5 and bj not in matched_g:
                    matched_g.add(bj)
                    kinds.append("TP")
                else:
                    kinds.append("FP")
            rgb = np.repeat(img[..., None], 3, 2).astype(np.float32) / 255.0
            for j, g in enumerate(gmasks):
                if j not in matched_g:
                    rgb[g] = 0.35 * rgb[g] + 0.65 * np.array(matplotlib.colors.to_rgb(COL["FN"]))
            for m, k in zip(pm, kinds):
                rgb[m] = 0.35 * rgb[m] + 0.65 * np.array(matplotlib.colors.to_rgb(COL[k]))
            # a missed filament is also outlined on top: where a prediction covers it with the wrong shape
            # (IoU <= 0.5, an orange FP over it), the fill alone would hide it
            from run import cv2
            for j, g in enumerate(gmasks):
                if j not in matched_g:
                    cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(rgb, cs, -1, matplotlib.colors.to_rgb(COL["FN"]), 4)
            n_tp = kinds.count("TP")
            n_fp, n_fn = kinds.count("FP"), len(gmasks) - n_tp
            ax = figs["filaments"][1][col]
            ax.imshow(rgb[CROP, CROP])
            ax.set_title(f"{stem}\nexpert filaments {len(gmasks)}, predicted {len(pm)}\n"
                         f"TP {n_tp}   FP {n_fp}   FN {n_fn}", fontsize=11)
            ax.axis("off")
            summary[tag][stem] = dict(pixels=dict(TP=tp, FP=fp, FN=fn), filaments=dict(TP=n_tp, FP=n_fp, FN=n_fn), cut=cut)
        for kind, (fig, _) in figs.items():
            if kind == "pixels":
                fig.suptitle(f"{shown}\npixels it calls filament (probability >= 0.5), in red", fontsize=13)
                fig.legend(handles=[Patch(color=COL["PIX"], label="predicted filament pixel")], loc="lower center")
            else:
                fig.suptitle(f"{shown}\nfilaments after grouping and the selector (selector fitted without this photograph)",
                             fontsize=13)
                fig.legend(handles=[Patch(color=COL["TP"], label="TP: matches an expert filament (IoU > 0.5)"),
                                    Patch(color=COL["FP"], label="FP: predicted, no expert filament"),
                                    Patch(color=COL["FN"], label="FN: expert filament the model missed (filled + outlined)")],
                           loc="lower center", ncol=3, fontsize=11)
            fig.tight_layout(rect=(0, 0.05, 1, 0.93))
            fig.savefig(OUT / f"{tag}_{kind}.png", dpi=110)
            plt.close(fig)
        print(tag, json.dumps(summary[tag]), flush=True)
    json.dump(dict(photographs=stems, models={k: v[3] for k, v in MODELS.items()}, counts=summary),
              open(OUT / "compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
