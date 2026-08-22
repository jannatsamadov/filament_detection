"""Rebuild the training targets with evidence gathered across days.

The consensus targets score a pixel by the fraction of that photograph's own
annotators who covered it. That undercounts badly: 411 of the 707 photographs
have a single annotator, so everything in them scores 1.0 or 0, and where three
people worked, a filament only one of them drew scores 0.33 and the model learns
it as mostly background. It recovers 29% of those.

Rotation supplies evidence the within-photograph count cannot see. Linking every
annotation to whatever it lands on in a photograph taken within three and a half
days -- three times more often than a curve dropped where it does not belong --
finds 1320 filaments that a second, independent photograph also shows. A
filament one person drew and another photograph confirms is not doubtful.

The weights follow what was measured rather than what seems reasonable:

  one annotator, no other evidence      0.40   (40.4% of these are confirmed
                                                by another photograph, measured)
  one of several annotators, alone      0.29   (28.6%, measured)
  confirmed by another photograph      +0.30 per independent confirmation
  every annotator of the photograph     1.00   (nothing to add)

Human marks are a floor. Nothing here lowers a pixel a person drew; the
adjustment only raises what the evidence supports.

Written per photograph, same layout as consensus/, so the training code needs
only a different --consensus-dir.
"""
from __future__ import annotations

import argparse, collections, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
OUT = ROOT / "consensus_v5"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles                             # noqa: E402
from propagate_labels import obs_time, summarise, project            # noqa: E402
from spine_data import (rasterize, spine_heatmap, DISK_CX, DISK_CY,  # noqa: E402
                        DISK_R, FULL)

MAX_GAP = 3.5
TOL_PX = 25
HIT_FRAC = 0.30
SIGMA = 3.0

W_ALONE_MULTI = 0.29      # one of several annotators drew it, measured
W_ALONE_SINGLE = 0.40     # sole annotator of the photograph, measured
W_CONFIRM = 0.30          # each independent confirmation from another photograph


def label_map(anns):
    lab = np.zeros((FULL, FULL), np.int32)
    for k, a in enumerate(anns, 1):
        im = Image.new("L", (FULL, FULL), 0)
        dr = ImageDraw.Draw(im)
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        lab[np.asarray(im, bool)] = k
    dist, idx = distance_transform_edt(lab == 0, return_indices=True)
    return lab, dist, idx


def hit(lab, dist, idx, px, py):
    xi = np.clip(np.round(px).astype(int), 0, FULL - 1)
    yi = np.clip(np.round(py).astype(int), 0, FULL - 1)
    near = dist[yi, xi] <= TOL_PX
    if near.mean() < HIT_FRAC:
        return 0
    v = lab[idx[0][yi[near], xi[near]], idx[1][yi[near], xi[near]]]
    v = v[v > 0]
    return int(np.bincount(v).argmax()) if len(v) else 0


def gather(photos, flat, times, owner, photo_recs):
    """How many other photographs independently show each annotation."""
    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    by_target = collections.defaultdict(list)
    for sa, sb in pairs:
        by_target[sb].append(sa)
        by_target[sa].append(sb)
    print(f"{len(pairs)} pairs within {MAX_GAP} days over {len(by_target)} photographs")

    confirm = collections.defaultdict(set)   # (photo, index) -> other photographs
    t0 = time.time()
    for k, dst in enumerate(sorted(by_target), 1):
        lab, dist, idx = label_map(flat[dst])
        for src in by_target[dst]:
            ang = solar_angles(times[src])
            for j, a in enumerate(flat[src]):
                su = summarise(a, ang)
                if su is None:
                    continue
                p = project(su, times[src], times[dst])
                if p is None:
                    continue
                h = hit(lab, dist, idx, p["px"], p["py"])
                if h:
                    # each side gains a confirmation from the other photograph
                    confirm[(dst, h - 1)].add(src)
                    confirm[(src, j)].add(dst)
        del lab, dist, idx
        if k % 50 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(by_target)}  {el/60:.0f} min, "
                  f"{el/k*(len(by_target)-k)/60:.0f} min left", flush=True)
    return confirm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    photo_of = {im["id"]: im["id"].split("-")[1] for im in d["images"]}
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)
    photo_recs = collections.defaultdict(list)
    for rid, st in photo_of.items():
        photo_recs[st].append(rid)
    photos = sorted(photo_recs)
    times = {s: obs_time(s) for s in photos}

    flat, owner = {}, {}
    for s in photos:
        lst = []
        for rid in sorted(photo_recs[s]):
            for a in ann_of[rid]:
                lst.append(a)
                owner[id(a)] = rid
        flat[s] = lst
    print(f"{len(photos)} photographs, {sum(len(v) for v in flat.values())} annotations")

    confirm = gather(photos, flat, times, owner, photo_recs)
    n_conf = sum(1 for v in confirm.values() if v)
    print(f"{n_conf} annotations confirmed by at least one other photograph\n")

    # per-annotation weight
    stats = collections.Counter()
    for s in photos:
        n_ann = len(photo_recs[s])
        lab, dist, idx = label_map(flat[s])
        drew = collections.defaultdict(set)
        which = []
        for a in flat[s]:
            sx = np.asarray(a["spine"][0::2], float)
            sy = np.asarray(a["spine"][1::2], float)
            h = hit(lab, dist, idx, sx, sy)
            which.append(h)
            drew[h].add(owner[id(a)])

        weights = []
        for j, (a, h) in enumerate(zip(flat[s], which)):
            v = len(drew[h])
            if n_ann == 1:
                w = W_ALONE_SINGLE
            elif v >= n_ann:
                w = 1.0
            elif v == 1:
                w = W_ALONE_MULTI
            else:
                w = v / n_ann
            base = w
            w = min(1.0, w + W_CONFIRM * len(confirm.get((s, j), ())))
            weights.append(w)
            stats["raised" if w > base + 1e-6 else "unchanged"] += 1

        # rasterise, weighted
        votes = np.zeros((FULL, FULL), np.float32)
        spine_v = np.zeros((FULL, FULL), np.float32)
        rad_sum = np.zeros((FULL, FULL), np.float32)
        rad_cnt = np.zeros((FULL, FULL), np.float32)
        for a, w in zip(flat[s], weights):
            mask, spine, radius = rasterize([a])
            votes = np.maximum(votes, mask.astype(np.float32) * w)
            spine_v = np.maximum(spine_v, spine_heatmap(spine, SIGMA) * w)
            rad_sum += radius * mask
            rad_cnt += mask
        radius = np.divide(rad_sum, np.maximum(rad_cnt, 1.0))

        dd = out / s
        dd.mkdir(exist_ok=True)
        Image.fromarray((np.clip(votes, 0, 1) * 255).astype(np.uint8)).save(
            dd / "mask_vote.png", optimize=True)
        Image.fromarray((np.clip(spine_v, 0, 1) * 255).astype(np.uint8)).save(
            dd / "spine_vote.png", optimize=True)
        Image.fromarray(np.clip(radius, 0, 255).astype(np.uint8)).save(
            dd / "radius.png", optimize=True)
        json.dump({"n_annotators": n_ann, "n_filaments": len(flat[s]),
                   "weights": [round(w, 3) for w in weights]},
                  open(dd / "meta.json", "w"))
        del lab, dist, idx

    print(f"weights raised by cross-day evidence: {stats['raised']}, "
          f"left alone: {stats['unchanged']}")
    print(f"wrote {len(list(out.iterdir()))} photographs to {out}")


if __name__ == "__main__":
    main()
