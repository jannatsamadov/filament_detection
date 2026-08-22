"""Count the votes a filament gets across days, not just within one photograph.

The earlier count asked how many of a photograph's own annotators drew each
filament. That misses the case this dataset is full of: the same filament
photographed on three days, drawn by a different person each time. Under the old
count it scores 1/3 three times over and looks doubtful three times over; in
truth three independent people saw it.

Photographs come in groups, and a group is one annotator team. Within a group
only 32 pairs fall within three and a half days. Across groups there are 582,
and those are the valuable ones -- a different team entirely, which is as
independent as this dataset gets.

So: link every annotation to every other annotation it lands on after rotation,
across all pairs, and let the connected pieces be the filament. Its support is
how many separate photographs it appears in. A filament one annotator drew and
nobody else in that photograph did, but which three other photographs also show,
is not doubtful at all.

The control is the same projection dropped where it does not belong. Annotations
cover a few percent of the disk, so chance hits are not negligible and the raw
count means nothing without them.
"""
from __future__ import annotations

import collections, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles                          # noqa: E402
from propagate_labels import obs_time, summarise, project         # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL             # noqa: E402

MAX_GAP = 3.5
TOL_PX = 25            # the projection misses by a median 14 px; filaments are 8 px wide
HIT_FRAC = 0.30


class Union:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def label_map(anns):
    """Per-pixel index into `anns`, plus the nearest-annotation lookup around it."""
    lab = np.zeros((FULL, FULL), np.int32)
    for k, a in enumerate(anns, 1):
        im = Image.new("L", (FULL, FULL), 0)
        dr = ImageDraw.Draw(im)
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        m = np.asarray(im, bool)
        lab[m] = k
    dist, idx = distance_transform_edt(lab == 0, return_indices=True)
    return lab, dist, idx


def hit(lab, dist, idx, px, py):
    """Which annotation the curve falls on, allowing TOL_PX of slack. 0 = none."""
    xi = np.clip(np.round(px).astype(int), 0, FULL - 1)
    yi = np.clip(np.round(py).astype(int), 0, FULL - 1)
    near = dist[yi, xi] <= TOL_PX
    if near.mean() < HIT_FRAC:
        return 0
    ys, xs = idx[0][yi[near], xi[near]], idx[1][yi[near], xi[near]]
    vals = lab[ys, xs]
    vals = vals[vals > 0]
    if not len(vals):
        return 0
    return int(np.bincount(vals).argmax())


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    recs = {}          # image record id -> (photo stem, group, annotator slot)
    for im in d["images"]:
        pref, st = im["id"].split("-")
        recs[im["id"]] = (st, pref[:4], pref[4:])
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)

    photo_recs = collections.defaultdict(list)
    for rid, (st, g, s) in recs.items():
        photo_recs[st].append(rid)
    photos = sorted(photo_recs)
    times = {s: obs_time(s) for s in photos}

    # one flat list of annotations per photograph, remembering who drew each
    flat, owner = {}, {}
    for s in photos:
        lst = []
        for rid in sorted(photo_recs[s]):
            for a in ann_of[rid]:
                lst.append(a)
                owner[id(a)] = rid
        flat[s] = lst

    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    used = sorted({x for p in pairs for x in p})
    print(f"{len(photos)} photographs, {len(pairs)} pairs within {MAX_GAP} days, "
          f"{len(used)} photographs involved")

    # One outline map per photograph is 80 MB, and there are 575 of them -- so
    # they are built per target and thrown away, not cached. Sources are grouped
    # by target so each map is built exactly once.
    by_target = collections.defaultdict(list)
    for sa, sb in pairs:
        by_target[sb].append(sa)
        by_target[sa].append(sb)

    uf = Union()
    rng = np.random.default_rng(0)
    vote_cache = {}
    n_proj = n_hit = n_ctl = n_ctl_hit = 0
    for k, dst in enumerate(sorted(by_target), 1):
        lab, dist, idx = label_map(flat[dst])

        # while the map is here, record how many of this photograph's own
        # annotators drew each of its filaments
        n_annotators = len(photo_recs[dst])
        acc = collections.defaultdict(set)
        which = []
        for j, a in enumerate(flat[dst]):
            sx = np.asarray(a["spine"][0::2], float)
            sy = np.asarray(a["spine"][1::2], float)
            h = hit(lab, dist, idx, sx, sy)
            which.append(h)
            acc[h].add(owner[id(a)])
        for j, h in enumerate(which):
            vote_cache[(dst, j)] = (len(acc[h]), n_annotators)

        for src in by_target[dst]:
            ang = solar_angles(times[src])
            for j, a in enumerate(flat[src]):
                su = summarise(a, ang)
                if su is None:
                    continue
                p = project(su, times[src], times[dst])
                if p is None:
                    continue
                n_proj += 1
                h = hit(lab, dist, idx, p["px"], p["py"])
                if h:
                    n_hit += 1
                    uf.join((src, j), (dst, h - 1))
                th = rng.uniform(0, 2 * np.pi); dd = rng.uniform(220, 420)
                cx, cy = p["px"] + dd * np.cos(th), p["py"] + dd * np.sin(th)
                if (np.hypot(cx - DISK_CX, cy - DISK_CY) / DISK_R).max() < 0.90:
                    n_ctl += 1
                    if hit(lab, dist, idx, cx, cy):
                        n_ctl_hit += 1
        del lab, dist, idx
        if k % 50 == 0:
            print(f"  {k}/{len(by_target)} target frames, "
                  f"{n_hit}/{n_proj} linked", flush=True)

    print(f"\n{n_proj} projections, {n_hit} landed on an annotation "
          f"({100*n_hit/max(n_proj,1):.1f}%)")
    print(f"control: {n_ctl_hit}/{n_ctl} ({100*n_ctl_hit/max(n_ctl,1):.1f}%) "
          f"-> ratio {(n_hit/max(n_proj,1))/max(n_ctl_hit/max(n_ctl,1),1e-9):.1f}x")

    # components
    comp = collections.defaultdict(list)
    for s in sorted(by_target):
        for j in range(len(flat[s])):
            comp[uf.find((s, j))].append((s, j))
    sizes = collections.Counter(len(v) for v in comp.values())
    linked = {k: v for k, v in comp.items() if len(v) > 1}
    print(f"\n{len(comp)} filament groups; {len(linked)} of them span more than one "
          f"annotation ({sum(len(v) for v in linked.values())} annotations)")
    print("  group size:", dict(sorted(sizes.items())[:8]))

    # the number that matters: annotations only one of a photograph's annotators
    # drew, but which another photograph independently shows
    weak = strong = weak_supported = 0
    for cid, members in comp.items():
        others = {s for s, _ in members}
        for s, j in members:
            v, n = vote_cache[(s, j)]
            if n >= 2 and v == 1:
                weak += 1
                if len(others - {s}) >= 1:
                    weak_supported += 1
            elif n >= 2 and v >= 2:
                strong += 1
    print(f"\nfilaments only one of several annotators drew: {weak}")
    print(f"   of those, another photograph independently shows: {weak_supported} "
          f"({100*weak_supported/max(weak,1):.1f}%)")
    print(f"filaments two or more annotators drew: {strong}")

    single = sum(1 for s in by_target if len(photo_recs[s]) == 1 for _ in flat[s])
    single_sup = 0
    for cid, members in comp.items():
        srcs = {s for s, _ in members}
        for s, j in members:
            if len(photo_recs[s]) == 1 and len(srcs - {s}) >= 1:
                single_sup += 1
    print(f"\nfilaments in photographs only one person annotated: {single}")
    print(f"   of those, another photograph independently shows: {single_sup} "
          f"({100*single_sup/max(single,1):.1f}%)   <- new evidence, "
          f"unavailable to a within-photograph count")

    json.dump({"n_proj": n_proj, "n_hit": n_hit, "n_ctl": n_ctl,
               "n_ctl_hit": n_ctl_hit, "weak": weak,
               "weak_supported": weak_supported, "single": single,
               "single_supported": single_sup},
              open(ROOT / "crossday_votes.json", "w"), indent=1)
    print("\nwrote crossday_votes.json")


if __name__ == "__main__":
    main()
