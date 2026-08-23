"""Build the v5 label store: one record per filament, with a counter behind it.

The consensus targets froze a filament's confidence at the fraction of a
photograph's annotators who drew it. That is only part of the evidence, and it
cannot change once written. This writes something the training loop can keep
updating instead: an instance map, and beside it a count of what speaks for each
filament and what speaks against.

The score is a / (a + b), and a filament counts as real above 0.75. Human
agreement opens the count:

    3 annotators, 1 drew it     0.40      a 2.0   b 3.0
    3 annotators, 2 drew it     0.80      a 4.0   b 1.0
    2 annotators, 1 drew it     0.50      a 2.5   b 2.5
    sole annotator              0.66      a 3.3   b 1.7
    all annotators              1.00      immovable

The opening weight is 5 in every case, so later evidence can move it -- except
where everyone agreed, which nothing may lower.

Rotation supplies the rest. A filament drawn in one photograph is carried to
another taken within three and a half days; the projection misses by a median
14 px, measured against filaments annotated in both. Where it lands on an
annotation, who drew that annotation decides what it is worth:

    a different annotator      counts as another annotator's vote -- an
                               independent pair of eyes on a different image
    the same annotator         a += 2.5, roughly two epochs of model agreement:
                               a different image, but the same person's habits

Nothing reaches 0.75 on geometry alone. A singly-drawn filament with one
same-annotator confirmation sits at 0.54; the model has to supply the rest
during training.

Written per photograph:
  inst.png      uint16 instance ids over the annotation outlines
  spine_id.png  uint16 instance ids along the spines
  radius.png    half-width
  meta.json     per-instance a, b, chirality, and where the evidence came from
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
OUT = ROOT / "labels_v5"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles                             # noqa: E402
from propagate_labels import obs_time, summarise, project            # noqa: E402
from spine_data import FULL                                          # noqa: E402

MAX_GAP = 3.5
TOL_PX = 25
HIT_FRAC = 0.30
MERGE_IOU = 0.30           # two annotators' outlines this close are one filament
PRIOR_WEIGHT = 5.0
SAME_AUTHOR_A = 2.5        # a same-annotator confirmation from another day

# what a filament opens at, by (how many drew it, how many looked)
OPENING = {(1, 1): 0.66, (1, 2): 0.50, (2, 2): 1.00,
           (1, 3): 0.40, (2, 3): 0.80, (3, 3): 1.00}


CROSSDAY_CAP = 0.90        # evidence from another day never equals same-frame
                           # unanimity: about a third of the links are chance


def opening_score(v, n):
    """The table first. A sole annotator is 0.66, not 1.0 -- nobody checked."""
    if (v, n) in OPENING:
        return OPENING[(v, n)]
    if v >= n:
        return 1.0
    return float(min(1.0, 1.2 * v / n))


def poly_mask(anns):
    im = Image.new("L", (FULL, FULL), 0)
    dr = ImageDraw.Draw(im)
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.asarray(im, bool)


def bbox_of(a):
    x, y, w, h = a["bbox"]
    return x, y, x + w, y + h


def overlaps(a, b):
    ax0, ay0, ax1, ay1 = bbox_of(a)
    bx0, by0, bx1, by1 = bbox_of(b)
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def group_filaments(anns):
    """Annotations of one photograph, grouped so each group is one filament."""
    n = len(anns)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    masks = {}
    for i, j in combinations(range(n), 2):
        if not overlaps(anns[i], anns[j]):
            continue
        for k in (i, j):
            if k not in masks:
                masks[k] = poly_mask([anns[k]])
        inter = np.logical_and(masks[i], masks[j]).sum()
        if not inter:
            continue
        union = np.logical_or(masks[i], masks[j]).sum()
        if inter / union >= MERGE_IOU:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def rasterise(anns, groups, spine_width=3, pad=6):
    """Instance id maps for the outlines and for the spines, plus half-widths.

    Everything is drawn inside each filament's own bounding box. A filament
    occupies a few thousand of the four million pixels in a frame, and running
    the distance transform over the whole frame for each of them cost ten
    seconds a photograph -- about two hours over the dataset, for nothing.
    """
    inst = np.zeros((FULL, FULL), np.uint16)
    spid = np.zeros((FULL, FULL), np.uint16)
    radius = np.zeros((FULL, FULL), np.float32)
    for gi, members in enumerate(groups, 1):
        xs0 = ys0 = FULL
        xs1 = ys1 = 0
        for k in members:
            x0, y0, x1, y1 = bbox_of(anns[k])
            xs0, ys0 = min(xs0, x0), min(ys0, y0)
            xs1, ys1 = max(xs1, x1), max(ys1, y1)
        x0 = int(max(0, xs0 - pad)); y0 = int(max(0, ys0 - pad))
        x1 = int(min(FULL, xs1 + pad + 1)); y1 = int(min(FULL, ys1 + pad + 1))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        im = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(im)
        sp = Image.new("L", (w, h), 0)
        ds = ImageDraw.Draw(sp)
        for k in members:
            a = anns[k]
            for poly in a["segmentation"]:
                if len(poly) >= 6:
                    dr.polygon([(x - x0, y - y0)
                                for x, y in zip(poly[0::2], poly[1::2])], fill=1)
            s = a["spine"]
            pts = [(x - x0, y - y0) for x, y in zip(s[0::2], s[1::2])]
            if len(pts) >= 2:
                ds.line(pts, fill=1, width=spine_width)
        m = np.asarray(im, bool)
        if not m.any():
            continue
        sub_inst = inst[y0:y1, x0:x1]
        sub_inst[m] = gi
        sub_spid = spid[y0:y1, x0:x1]
        sub_spid[np.asarray(sp, bool)] = gi
        d = distance_transform_edt(m)
        sub_rad = radius[y0:y1, x0:x1]
        sub_rad[m] = np.maximum(sub_rad[m], d[m])
    return inst, spid, radius


def label_lookup(inst):
    dist, idx = distance_transform_edt(inst == 0, return_indices=True)
    return dist, idx


def hit_id(inst, dist, idx, px, py):
    xi = np.clip(np.round(px).astype(int), 0, FULL - 1)
    yi = np.clip(np.round(py).astype(int), 0, FULL - 1)
    near = dist[yi, xi] <= TOL_PX
    if near.mean() < HIT_FRAC:
        return 0
    v = inst[idx[0][yi[near], xi[near]], idx[1][yi[near], xi[near]]]
    v = v[v > 0]
    return int(np.bincount(v).argmax()) if len(v) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    # image record id is "<group><annotator>-<photograph>"; the six-digit prefix
    # identifies the person, and it is stable across the photographs they did
    author_of, photo_of = {}, {}
    for im in d["images"]:
        pref, st = im["id"].split("-")
        author_of[im["id"]] = pref
        photo_of[im["id"]] = st
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)
    photo_recs = collections.defaultdict(list)
    for rid, st in photo_of.items():
        photo_recs[st].append(rid)
    photos = sorted(photo_recs)
    times = {s: obs_time(s) for s in photos}

    print(f"{len(photos)} photographs, {len(d['annotations'])} annotations, "
          f"{len(set(author_of.values()))} annotator identities")

    # ---- 1. group each photograph's annotations into filaments -------------
    t0 = time.time()
    store = {}
    for k, s in enumerate(photos, 1):
        anns, who = [], []
        for rid in sorted(photo_recs[s]):
            for a in ann_of[rid]:
                anns.append(a); who.append(author_of[rid])
        groups = group_filaments(anns)
        inst, spid, radius = rasterise(anns, groups)
        n_ann = len(photo_recs[s])
        recs = []
        for gi, members in enumerate(groups, 1):
            drew = {who[i] for i in members}
            cats = collections.Counter(anns[i]["category_id"] for i in members)
            recs.append(dict(id=gi, n_drew=len(drew), n_annotators=n_ann,
                             authors=sorted(drew),
                             chirality=int(cats.most_common(1)[0][0]),
                             cd_diff=[], cd_same=0))
        store[s] = dict(anns=anns, groups=groups, recs=recs,
                        inst=inst, spid=spid, radius=radius)
        if k % 100 == 0:
            el = time.time() - t0
            print(f"  grouped {k}/{len(photos)}  {el/60:.0f} min, "
                  f"{el/k*(len(photos)-k)/60:.0f} min left", flush=True)
    print(f"grouping done: {sum(len(v['recs']) for v in store.values())} filaments")

    # ---- 2. carry every filament to the neighbouring photographs -----------
    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    by_target = collections.defaultdict(list)
    for sa, sb in pairs:
        by_target[sb].append(sa); by_target[sa].append(sb)
    print(f"\n{len(pairs)} pairs within {MAX_GAP} days over {len(by_target)} photographs")

    t0 = time.time()
    for k, dst in enumerate(sorted(by_target), 1):
        S = store[dst]
        dist, idx = label_lookup(S["inst"])
        for src in by_target[dst]:
            T = store[src]
            ang = solar_angles(times[src])
            for gi, members in enumerate(T["groups"], 1):
                a = T["anns"][members[0]]
                su = summarise(a, ang)
                if su is None:
                    continue
                p = project(su, times[src], times[dst])
                if p is None:
                    continue
                h = hit_id(S["inst"], dist, idx, p["px"], p["py"])
                if not h:
                    continue
                # who confirms whom: the source filament's authors against the
                # target filament's authors
                src_auth = set(T["recs"][gi - 1]["authors"])
                dst_auth = set(S["recs"][h - 1]["authors"])
                for rec, other in ((S["recs"][h - 1], src_auth),
                                   (T["recs"][gi - 1], dst_auth)):
                    new = other - set(rec["authors"])
                    if new:
                        rec["cd_diff"] = sorted(set(rec["cd_diff"]) | new)
                    else:
                        rec["cd_same"] += 1
        del dist, idx
        if k % 50 == 0:
            el = time.time() - t0
            print(f"  linked {k}/{len(by_target)}  {el/60:.0f} min, "
                  f"{el/k*(len(by_target)-k)/60:.0f} min left", flush=True)

    # ---- 3. opening counts, and write ------------------------------------
    stats = collections.Counter()
    for s in photos:
        S = store[s]
        for rec in S["recs"]:
            n = rec["n_annotators"]
            v = rec["n_drew"]
            # a different annotator on another day is another pair of eyes,
            # but capped: it is a different image, and a share of the links are
            # chance rather than the same filament
            v_eff = v + len(rec["cd_diff"])
            n_eff = max(n, v_eff)
            score = opening_score(v, n)
            if rec["cd_diff"]:
                # evidence may only raise: a filament everyone drew stays at 1.0
                score = max(score, min(opening_score(v_eff, n_eff), CROSSDAY_CAP))
            frozen = v >= n and n >= 2          # everyone who looked drew it
            a = PRIOR_WEIGHT * score
            b = PRIOR_WEIGHT * (1.0 - score)
            a += SAME_AUTHOR_A * rec["cd_same"]
            rec.update(a=round(float(a), 3), b=round(float(b), 3),
                       frozen=bool(frozen), opening=round(float(score), 3),
                       n_drew_eff=v_eff)
            stats["frozen" if frozen else "movable"] += 1
            if rec["cd_diff"]:
                stats["confirmed_by_other_author"] += 1
            if rec["cd_same"]:
                stats["confirmed_by_same_author"] += 1
            if not frozen and a / (a + b) >= 0.75:
                stats["already_above_threshold"] += 1

        dd = out / s
        dd.mkdir(exist_ok=True)
        Image.fromarray(S["inst"]).save(dd / "inst.png", optimize=True)
        Image.fromarray(S["spid"]).save(dd / "spine_id.png", optimize=True)
        Image.fromarray(np.clip(S["radius"], 0, 255).astype(np.uint8)).save(
            dd / "radius.png", optimize=True)
        json.dump({"photograph": s, "n_annotators": len(photo_recs[s]),
                   "filaments": S["recs"]}, open(dd / "meta.json", "w"), indent=1)

    print(f"\n{dict(stats)}")
    print(f"wrote {len(list(out.iterdir()))} photographs to {out}")


if __name__ == "__main__":
    main()
