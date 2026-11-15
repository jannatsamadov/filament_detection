"""Pixel classification: how well each model labels single pixels filament / not filament.

Fold-0 validation, 141 photographs at 2048 px; no model saw these labels. Truth for a pixel is
"at least one expert marked it" (the union); a second F1 is against each expert on his own, pooled.
A human row scores each expert against the union of the other experts on the same photograph.
Every metric is computed from per-model histograms of the 256 probability levels, so the ROC and
precision-recall curves are exact at the map's own resolution.

    python kernels/aug_unet/pixel_cls.py          -> kernels/aug_unet/out/pixel_cls.json, figures/pixel_cls.png
"""
import collections, json, sys, tarfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402

KER = HERE.parent
OUT_JSON = HERE / "out" / "pixel_cls.json"
OUT_FIG = HERE.parents[1] / "figures" / "pixel_cls.png"

# name -> list of (tar, folder inside it); several entries are averaged
DUMPS = {
    "flip": (KER / "aug_dump/out_kaggle/dump_flip.tar", "val_d4"),
    "full": (KER / "aug_dump/out_kaggle/dump_full.tar", "val_d4"),
    "r50": (KER / "unet_r2/out_kaggle/dump_r50.tar", "val_d4"),
    "union": (KER / "unet_r2/out_kaggle/dump_union.tar", "val_d4"),
    "solar": (KER / "unet_r3/out_kaggle/dump_solar.tar", "val_d4"),
    "qseg": (KER / "qwen_seg/out_kaggle/dump_qwenseg_pre.tar", "val_plain"),
    "pixnet": (KER / "pixnet/out_kaggle/dump_pixnet_scratch.tar", "val_plain"),
    "pixnet42": (HERE / "out/dump_pixnet_cont.tar", "val_plain"),
}
RF0 = KER / "pixel_rf0/out_kaggle/dump_rf0.tar"      # filament-pixel-rf0: the forest's fold-0 maps
U5 = ("flip", "full", "r50", "union", "solar")
MODELS = {                                   # shown name -> (members, weights)
    "U-Net R50 (one model)": (("r50",), (1,)),
    "5 U-Nets (mean)": (U5, (1,) * 5),
    "Qwen-seg (ViT)": (("qseg",), (1,)),
    "5 U-Nets + Qwen (submitted mix)": (U5 + ("qseg",), (1,) * 6),
    "PixNet, 30 epochs": (("pixnet",), (1,)),
    "PixNet, 42 epochs": (("pixnet42",), (1,)),
}
if RF0.exists():
    DUMPS.update({"pair": (RF0, "pair"), "rfA": (RF0, "pixA"), "rfB": (RF0, "pixB")})
    MODELS.update({"R50 + R34 pair (RF's input)": (("pair",), (1,)),
                   "RF, pixel table only": (("rfA",), (1,)),
                   "RF, table + U-Net pair": (("rfB",), (1,))})
IDEAS = KER / "pixel_ideas/out_kaggle/dump_ideas.tar"   # filament-pixel-ideas: the user's questions, one arm each (11-14)
if IDEAS.exists():
    for arm, shown in (("base", "RF: as it ran (target = share of experts)"),
                       ("onehot", "RF: station as 0/1 columns"),
                       ("photo", "RF: + brightness/contrast/blur copies"),
                       ("count", "RF: target by expert count (0.45/0.65/0.95)"),
                       ("count1", "RF: expert count, single expert = 0.95"),
                       ("stage1", "RF: first forest (half the photographs)"),
                       ("context", "RF: + neighbours' predictions (auto-context)")):
        DUMPS[f"i_{arm}"] = (IDEAS, arm)
        MODELS[shown] = ((f"i_{arm}",), (1,))
# the user's neighbour table (aug_unet/pixel_nb.py, 11-14): regression, classifier, classifier + rotation
for tar, arms in ((KER / "pixel_nb/out_kaggle/dump_nb.tar",
                   (("ctrl", "RF reg: 35 columns, same rows (control)"), ("nb", "RF reg: + 12 px neighbours (1358 col.)"),
                    ("nb_count", "RF reg: neighbours, target 0.45/0.65/0.95"))),
                  (KER / "pixel_nbc/out_kaggle/dump_nbc.tar",
                   (("ctrlc", "RF cls: 35 columns, same rows (control)"), ("nbc", "RF cls: + 12 px neighbours (1358 col.)"))),
                  (KER / "pixel_nbr/out_kaggle/dump_nbcr.tar",
                   (("nbcr", "RF cls: neighbours + rotation (2056 col.)"),)),
                  (KER / "pixel_nbcc/out_kaggle/dump_nbcc.tar",
                   (("nbcc", "RF cls: neighbours, no positions (478 col.)"),))):
    if tar.exists():
        for arm, shown in arms:
            DUMPS[f"n_{arm}"] = (tar, arm)
            MODELS[shown] = ((f"n_{arm}",), (1,))
T_FIXED = 0.5
BAND = 16                                   # px at 2048: the "hard" pixels are expert pixels and those this close to them


class Dumps:
    def __init__(self):
        self.tars, self.idx = {}, {}
        for k, (path, folder) in DUMPS.items():
            t = tarfile.open(path)
            self.tars[k] = t
            self.idx[k] = {Path(m.name).stem: m for m in t.getmembers()
                           if m.isfile() and Path(m.name).parent.name == folder}

    def stems(self):
        return sorted(set.intersection(*(set(v) for v in self.idx.values())))

    def load(self, k, stem):
        b = self.tars[k].extractfile(self.idx[k][stem]).read()
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)

    def model(self, name, stem, cache):
        members, w = MODELS[name]
        for m in members:
            if m not in cache:
                cache[m] = self.load(m, stem)
        if len(members) == 1:
            return cache[members[0]]
        acc = sum(wi * cache[m].astype(np.float32) for m, wi in zip(members, w))
        return np.round(acc / sum(w)).astype(np.uint8)


def experts(stems):
    ann, _, _ = run.find_data()
    coco = json.load(open(ann))
    fname = {i["id"]: Path(i["file_name"]).stem for i in coco["images"]}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in coco["annotations"]:
        if fname[a["image_id"]] in stems:
            by[fname[a["image_id"]]][a["image_id"]].append(a)
    return by


def expert_masks(anns_by_set):
    out = []
    for anns in anns_by_set.values():
        m = np.zeros((run.FULL, run.FULL), bool)
        for a in anns:
            m |= run.ann_mask(a)
        out.append(m)
    return out


def metrics(pos, neg, t=T_FIXED):
    """pos / neg: histograms over the 256 levels. Prediction is filament where level >= round(255 t)."""
    c = int(round(255 * t))
    tp, fn = pos[c:].sum(), pos[:c].sum()
    fp, tn = neg[c:].sum(), neg[:c].sum()
    n = tp + fn + fp + tn
    # ROC and PR curves over all cut levels, from strict to lenient
    ctp, cfp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    tpr, fpr = np.r_[0, ctp / pos.sum()], np.r_[0, cfp / neg.sum()]
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / pos.sum()
    f1c = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best = int(np.argmax(f1c))
    return dict(
        AUC=float(np.trapezoid(tpr, fpr)),
        AP=float(np.sum(np.diff(np.r_[0, rec]) * prec)),
        precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
        F1=float(2 * tp / max(2 * tp + fp + fn, 1)), IoU=float(tp / max(tp + fp + fn, 1)),
        accuracy=float((tp + tn) / n), specificity=float(tn / max(tn + fp, 1)),
        balanced_accuracy=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
        best_F1=float(f1c[best]), best_t=round((255 - best) / 255, 3),
        all_background_accuracy=float(neg.sum() / n),
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn)), (rec, prec)


def main():
    D = Dumps()
    stems = D.stems()
    by = experts(set(stems))
    stems = [s for s in stems if s in by]
    print(f"{len(stems)} fold-0 validation photographs, {sum(len(by[s]) for s in stems)} expert sets")
    H = {m: dict(pos=np.zeros(256, np.int64), neg=np.zeros(256, np.int64),
                 pos1=np.zeros(256, np.int64), neg1=np.zeros(256, np.int64),
                 posb=np.zeros(256, np.int64), negb=np.zeros(256, np.int64)) for m in MODELS}
    hum = np.zeros(4, np.int64)             # TP FP FN TN, one expert against the union of the others
    per = {m: {} for m in MODELS}           # per photograph at T_FIXED: TP FP FN
    c = int(round(255 * T_FIXED))
    for i, s in enumerate(stems):
        ms = expert_masks(by[s])
        gt = np.logical_or.reduce(ms)
        ring = cv2.dilate(gt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1,) * 2)).astype(bool) & ~gt
        cache = {}
        for m in MODELS:
            p = D.model(m, s, cache)
            allh = np.bincount(p.ravel(), minlength=256)
            ph = np.bincount(p[gt], minlength=256)
            tp, fp = int(ph[c:].sum()), int((allh - ph)[c:].sum())
            per[m][s] = (tp, fp, int(gt.sum()) - tp)
            H[m]["pos"] += ph
            H[m]["neg"] += allh - ph
            H[m]["posb"] += ph
            H[m]["negb"] += np.bincount(p[ring], minlength=256)
            for e in ms:                      # each expert on his own
                pe = np.bincount(p[e], minlength=256)
                H[m]["pos1"] += pe
                H[m]["neg1"] += allh - pe
        if len(ms) > 1:
            for j, e in enumerate(ms):
                rest = np.logical_or.reduce([x for k, x in enumerate(ms) if k != j])
                tp = np.count_nonzero(e & rest)
                fp, fn = np.count_nonzero(e & ~rest), np.count_nonzero(~e & rest)
                hum += [tp, fp, fn, e.size - tp - fp - fn]
        if i % 20 == 0:
            print(f"  {i + 1}/{len(stems)} {s}", flush=True)

    np.savez(OUT_JSON.with_suffix(".npz"), hum=hum, **{f"{m}|{k}": v for m in H for k, v in H[m].items()})
    res = dict(photographs=len(stems), truth="union of experts; F1_single_expert pools each expert on his own",
               threshold=T_FIXED, band_px=BAND, models={}, curves={}, per_photo=per,
               experts_per_photo={s: len(by[s]) for s in stems},
               filaments_per_photo={s: max(len(v) for v in by[s].values()) for s in stems})
    for m in MODELS:
        r, (rec, prec) = metrics(H[m]["pos"], H[m]["neg"])
        r1, _ = metrics(H[m]["pos1"], H[m]["neg1"])
        r["F1_single_expert"] = r1["F1"]
        rb, _ = metrics(H[m]["posb"], H[m]["negb"])
        r["band"] = {q: rb[q] for q in ("AUC", "AP", "precision", "recall", "F1", "accuracy", "all_background_accuracy")}
        res["models"][m] = r
        keep = np.unique(np.linspace(0, len(rec) - 1, 120).astype(int))
        res["curves"][m] = dict(recall=rec[keep].round(4).tolist(), precision=prec[keep].round(4).tolist())
    tp, fp, fn, tn = hum
    res["human_one_expert_vs_union_of_others"] = dict(
        precision=float(tp / (tp + fp)), recall=float(tp / (tp + fn)), F1=float(2 * tp / (2 * tp + fp + fn)),
        IoU=float(tp / (tp + fp + fn)), accuracy=float((tp + tn) / hum.sum()),
        specificity=float(tn / (tn + fp)))
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print("wrote", OUT_JSON)
    for m, r in res["models"].items():
        print(f"{m:34s} AUC {r['AUC']:.4f} AP {r['AP']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} "
              f"F1 {r['F1']:.3f} IoU {r['IoU']:.3f} acc {r['accuracy']:.4f} bestF1 {r['best_F1']:.3f}@{r['best_t']}")
    h = res["human_one_expert_vs_union_of_others"]
    print(f"{'human (one expert vs others)':34s} P {h['precision']:.3f} R {h['recall']:.3f} F1 {h['F1']:.3f} "
          f"IoU {h['IoU']:.3f} acc {h['accuracy']:.4f}")


if __name__ == "__main__":
    main()
