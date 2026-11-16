"""Where the validation PQ goes, and how much of it a model could ever recover.

Works on the dumped probability maps of the 5-model ensemble at its tuned point,
one process, no GPU:

  1. reachable PQ: the other annotators' consensus scored against each held-out
     annotator, next to the ensemble on exactly the same annotation sets
  2. every ground-truth filament and every prediction sorted by what happened to it
  3. PQ by year and by station, reweighted to the test set's mix of both
  4. the no-training dark-pixel notebook (titled 0.46, from the Dice era) under PQ

    python kernels/aug_unet/error_analysis.py --dir kernels/aug_post/out
"""
from __future__ import annotations

import argparse, collections, json, sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import post_local as P  # noqa: E402
import run  # noqa: E402

VARIANT = "ens:flip+full+r50+solar+union/plain"
T, K, MIN_AREA = 0.65, 3, 350
T_LOW = 0.30                # what the model still sees below the operating threshold
DARK_PERCENTILE = 90.0      # the notebook's own settings
DARK_MAX_KEEP = 120


def to_full(crops):
    """Annotation set as a list of (y0, x0, boolean crop); painted onto a full frame."""
    m = np.zeros((run.FULL, run.FULL), np.uint8)
    for y0, x0, c in crops:
        m[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] |= c.astype(np.uint8)
    return m


def overlaps(crop, lab, area):
    """{label: intersection} of one GT crop with a label image."""
    y0, x0, c = crop
    lv = lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c]
    lv = lv[lv > 0]
    if lv.size == 0:
        return {}
    ids, inter = np.unique(lv, return_counts=True)
    return dict(zip(ids.tolist(), inter.tolist()))


def iou_to_crops(comp_mask, crops):
    """Best IoU of one predicted component with a list of GT crops."""
    best, a = 0.0, int(comp_mask.sum())
    for y0, x0, c in crops:
        it = int(comp_mask[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c].sum())
        if it:
            best = max(best, it / (int(c.sum()) + a - it))
    return best


def classify_gt(crop, lab, area, keep, lab_low):
    """What happened to one ground-truth filament at the operating point."""
    ga = int(crop[2].sum())
    inter = {j: it for j, it in overlaps(crop, lab, area).items() if keep[j]}
    ious = {j: it / (ga + int(area[j]) - it) for j, it in inter.items()}
    if ious and max(ious.values()) > 0.5:
        return "matched"
    if inter:
        cov = sum(inter.values()) / ga
        u = sum(inter.values())
        union_iou = u / (ga + sum(int(area[j]) for j in inter) - u)
        best = max(inter, key=inter.get)
        if len(inter) > 1 and union_iou > 0.5:
            return "fragmented"
        if inter[best] / ga >= 0.5:
            return "prediction too large"
        if cov >= 0.3:
            return "prediction too small"
    if sum(overlaps(crop, lab_low, None).values()) / ga >= 0.3:
        return "faint (seen below threshold)"
    return "not seen"


def consensus(masks, n):
    """Pixels at least half of the n annotators drew, as filament instances."""
    lab, area = run.components((sum(masks).astype(np.float32) / n * 255).astype(np.uint8), 0.5, 0)
    keep = area >= 100
    keep[0] = False
    return lab, area, keep


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scores(c):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in run.summarise(c).items()}


def dark_pixels(img):
    """The public no-training notebook's rule, unchanged."""
    im = img.astype(np.float32)
    h, w = im.shape
    y, x = np.ogrid[:h, :w]
    disk = (im > 5) & (((y - (h - 1) / 2) ** 2 + (x - (w - 1) / 2) ** 2) < 1000.0 ** 2)
    resid = cv2.GaussianBlur(im, (0, 0), 40.0, borderType=cv2.BORDER_REFLECT) - im
    b = (resid > max(1.0, float(np.percentile(resid[disk], DARK_PERCENTILE)))) & disk
    b = cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    good = [i for i in range(1, n) if 80 <= st[i, cv2.CC_STAT_AREA] <= 150_000
            and st[i, cv2.CC_STAT_AREA] / max(1, st[i, cv2.CC_STAT_WIDTH] * st[i, cv2.CC_STAT_HEIGHT]) <= 0.99]
    good = sorted(good, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:DARK_MAX_KEEP]
    area = np.bincount(lab.ravel(), minlength=n)
    keep = np.zeros(n, bool)
    keep[good] = True
    return lab, area, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="kernels/aug_post/out")
    ap.add_argument("--out", default="kernels/aug_post/out/error_analysis.json")
    a = ap.parse_args()
    d = Path(a.dir)
    meta, gt = P.ground_truth()
    stems = [Path(f).stem for f in meta["val_files"]]
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    val_img = np.load(run.CACHE / "val_img.npy", mmap_mode="r")

    gt_kind, fp_kind = collections.Counter(), collections.Counter()
    gt_kind_by_n = collections.defaultdict(collections.Counter)
    fn_only_one = collections.Counter()
    model_all = [0, 0, 0, 0.0]
    model_same, cons_same, pair_same = [0, 0, 0, 0.0], [0, 0, 0, 0.0], [0, 0, 0, 0.0]
    by_n = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    by_year, by_station = collections.defaultdict(lambda: [0, 0, 0, 0.0]), collections.defaultdict(lambda: [0, 0, 0, 0.0])
    frames_year, frames_station = collections.Counter(), collections.Counter()
    dark = [0, 0, 0, 0.0]

    for f, sets in sorted(by_frame.items()):
        stem = stems[f]
        year, station = stem[:4], stem[-2]
        frames_year[year] += 1
        frames_station[station] += 1
        p = P.load_map(d, VARIANT, "val", stem)
        lab, area = run.components(p, T, K)
        keep = area >= MIN_AREA
        keep[0] = False
        lab_low, _ = run.components(p, T_LOW, K)
        dlab, darea, dkeep = dark_pixels(np.asarray(val_img[f]))
        masks = [to_full(s) for s in sets]
        n = len(sets)

        for i, crops in enumerate(sets):
            c = run.pq_counts(crops, lab, area, keep)
            model_all = add(model_all, c)
            by_n[n] = add(by_n[n], c)
            by_year[year] = add(by_year[year], c)
            by_station[station] = add(by_station[station], c)
            dark = add(dark, run.pq_counts(crops, dlab, darea, dkeep))

            others = [s for j, s in enumerate(sets) if j != i]
            matched = set()
            for crop in crops:
                kind = classify_gt(crop, lab, area, keep, lab_low)
                gt_kind[kind] += 1
                gt_kind_by_n[n][kind] += 1
                if kind != "matched" and others:
                    m = np.zeros((run.FULL, run.FULL), bool)
                    m[crop[0]:crop[0] + crop[2].shape[0], crop[1]:crop[1] + crop[2].shape[1]] = crop[2]
                    agree = sum(iou_to_crops(m, o) > 0.3 for o in others)
                    fn_only_one["others drew it" if agree else "no other annotator drew it"] += 1
                if kind == "matched":
                    ga = int(crop[2].sum())
                    inter = overlaps(crop, lab, area)
                    matched.update(j for j, it in inter.items() if keep[j] and it / (ga + int(area[j]) - it) > 0.5)

            for j in np.flatnonzero(keep):
                if int(j) in matched:
                    continue
                comp = lab == j
                if iou_to_crops(comp, crops) >= 0.2:
                    fp_kind["overlaps this annotator's filament, IoU 0.2-0.5"] += 1
                elif others and any(iou_to_crops(comp, o) > 0.5 for o in others):
                    fp_kind["another annotator drew it"] += 1
                elif others:
                    fp_kind["no annotator of this frame drew it"] += 1
                else:
                    fp_kind["single-annotator frame, not drawn"] += 1

            if n >= 2:
                model_same = add(model_same, c)
                ol, oa, ok = consensus([m for j, m in enumerate(masks) if j != i], n - 1)
                cons_same = add(cons_same, run.pq_counts(crops, ol, oa, ok))
                for j in range(n):
                    if j != i:
                        pl, pa, pk = consensus([masks[j]], 1)
                        pair_same = add(pair_same, run.pq_counts(crops, pl, pa, pk))
        print(f"frame {f + 1}/{len(stems)}", flush=True) if (f + 1) % 20 == 0 else None

    test = sorted(Path(p).stem for p in run.data_dirs()[2].glob("*.jpeg")) if hasattr(run, "data_dirs") else []
    if not test:
        root = Path(__file__).resolve().parents[2] / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
        test = sorted(p.stem for p in root.glob("*.jpeg"))
    test_year, test_station = collections.Counter(s[:4] for s in test), collections.Counter(s[-2] for s in test)

    def reweighted(groups, val_n, test_n):
        c = [0.0, 0.0, 0.0, 0.0]
        for k, v in groups.items():
            w = (test_n.get(k, 0) / len(test)) / (val_n[k] / len(stems))
            c = add(c, [w * x for x in v])
        den = c[0] + 0.5 * c[1] + 0.5 * c[2]
        return round(c[3] / den, 4)

    res = dict(
        variant=VARIANT, point=[T, K, MIN_AREA], overall=scores(model_all),
        reach=dict(sets_with_other_annotators=int(sum(len(v) for v in by_frame.values() if len(v) >= 2)),
                   ensemble=scores(model_same), consensus_of_others=scores(cons_same), one_other_annotator=scores(pair_same)),
        by_annotators={k: scores(v) for k, v in sorted(by_n.items())},
        gt=dict(gt_kind), gt_by_annotators={k: dict(v) for k, v in sorted(gt_kind_by_n.items())},
        fn_agreement=dict(fn_only_one), fp=dict(fp_kind),
        by_year={k: {**scores(v), "val_frames": frames_year[k], "test_frames": test_year.get(k, 0)} for k, v in sorted(by_year.items())},
        by_station={k: {**scores(v), "val_frames": frames_station[k], "test_frames": test_station.get(k, 0)} for k, v in sorted(by_station.items())},
        test_mix_year=dict(sorted(test_year.items())), test_mix_station=dict(sorted(test_station.items())),
        pq_reweighted_to_test_year=reweighted(by_year, frames_year, test_year),
        pq_reweighted_to_test_station=reweighted(by_station, frames_station, test_station),
        dark_pixel_notebook=scores(dark),
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
