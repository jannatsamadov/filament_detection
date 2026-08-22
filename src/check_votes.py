"""Does yesterday's certain filament land on today's doubtful one?

151 photographs were annotated by three people, so every filament there carries
a vote count: 3/3 means all three drew it, 1/3 means only one did. A filament
with one vote is the problem case -- it is real enough that an expert marked it,
but the model is taught it as background wherever the other two left it out, and
the model finds only 29% of them.

The proposal is to let time settle those cases. If a filament all three
annotators drew in one photograph lands, after rotation, on a one-vote filament
in another, that one vote gains independent support and can be promoted.

This needs no model and no GPU. It also needs none of the filament matching that
failed earlier: the target is not "find the same filament", it is "does the
projected curve fall inside a polygon a human drew", which the annotation
outlines answer directly.

The control is the same curve dropped somewhere it does not belong. Filaments
cover a few percent of the disk, so landing on one by chance is not rare enough
to ignore.
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
from solar_ephemeris import solar_angles                             # noqa: E402
from propagate_labels import obs_time, summarise, project            # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL                # noqa: E402

MAX_GAP = 3.5
HIT_FRAC = 0.30        # this much of the projected spine near a polygon = a hit
TOL_PX = 25            # the projection misses by a median 14 px and filaments are
                       # 8 px wide, so demanding it land *inside* the outline is
                       # asking for an accuracy the geometry does not have. This
                       # is the "look here" tolerance, not a claim of overlap.


def vote_map(anns_by_annotator):
    """0-3 per pixel: how many annotators drew a filament there."""
    acc = np.zeros((FULL, FULL), np.uint8)
    for anns in anns_by_annotator:
        im = Image.new("L", (FULL, FULL), 0)
        dr = ImageDraw.Draw(im)
        for a in anns:
            for poly in a["segmentation"]:
                if len(poly) >= 6:
                    dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        acc += np.asarray(im, np.uint8)
    return acc


def near_maps(vmap):
    """Distance to the nearest pixel of each vote level."""
    return {k: distance_transform_edt(vmap != k) for k in (1, 2, 3)}


def votes_at(near, px, py):
    """Best vote level within TOL_PX of the curve; 0 if nothing is near."""
    xi = np.clip(np.round(px).astype(int), 0, FULL - 1)
    yi = np.clip(np.round(py).astype(int), 0, FULL - 1)
    for k in (3, 2, 1):
        if (near[k][yi, xi] <= TOL_PX).mean() >= HIT_FRAC:
            return k
    return 0


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)
    photo_ids = collections.defaultdict(list)
    for iid in ann_of:
        photo_ids[stem[iid]].append(iid)

    three = sorted(s for s in photo_ids if len(photo_ids[s]) == 3)
    times = {s: obs_time(s) for s in three}
    print(f"{len(three)} photographs annotated by three people")

    pairs = [(a, b) for a, b in combinations(three, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    print(f"{len(pairs)} pairs of them within {MAX_GAP} days\n")
    if not pairs:
        print("no pairs -- the three-annotator photographs are too far apart")
        return

    vmaps = {}
    for s in set([x for p in pairs for x in p]):
        vmaps[s] = vote_map([ann_of[i] for i in sorted(photo_ids[s])])
    nearm = {s: near_maps(v) for s, v in vmaps.items()}

    # what fraction of the disk each vote level covers, for the chance baseline
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.95) ** 2
    cover = collections.Counter()
    for s, v in vmaps.items():
        for k in (1, 2, 3):
            cover[k] += float((nearm[s][k][ondisk] <= TOL_PX).mean())
    for k in (1, 2, 3):
        cover[k] /= len(vmaps)
    print("share of the disk within tolerance of each vote level: "
          + ", ".join(f"{k}/3 {100*cover[k]:.2f}%" for k in (1, 2, 3)))

    rng = np.random.default_rng(0)
    got = collections.Counter()
    ctl = collections.Counter()
    n = 0
    for k, (sa, sb) in enumerate(pairs, 1):
        for src_s, dst_s in ((sa, sb), (sb, sa)):
            t_src, t_dst = times[src_s], times[dst_s]
            ang = solar_angles(t_src)
            vmap = nearm[dst_s]
            src_vote = near_maps(vote_map([ann_of[i]
                                           for i in sorted(photo_ids[src_s])]))
            for a in ann_of[sorted(photo_ids[src_s])[0]]:
                s = summarise(a, ang)
                if s is None:
                    continue
                # only filaments all three drew: those are the certain ones
                sx = np.asarray(a["spine"][0::2], float)
                sy = np.asarray(a["spine"][1::2], float)
                if votes_at(src_vote, sx, sy) < 3:
                    continue
                p = project(s, t_src, t_dst)
                if p is None:
                    continue
                n += 1
                got[votes_at(vmap, p["px"], p["py"])] += 1

                ang_r = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(220, 420)
                cx = p["px"] + dist * np.cos(ang_r)
                cy = p["py"] + dist * np.sin(ang_r)
                if (np.hypot(cx - DISK_CX, cy - DISK_CY) / DISK_R).max() < 0.90:
                    ctl[votes_at(vmap, cx, cy)] += 1
        if k % 100 == 0:
            print(f"  {k}/{len(pairs)} pairs, {n} projections", flush=True)

    nc = sum(ctl.values())
    print(f"\n{n} projections of filaments all three annotators drew\n")
    print("where they land in the other photograph:")
    print("  votes there   projected      random curve   ratio")
    for v in (3, 2, 1, 0):
        a = 100 * got[v] / max(n, 1)
        b = 100 * ctl[v] / max(nc, 1)
        lab = {3: "3/3 certain", 2: "2/3", 1: "1/3 doubtful", 0: "nothing"}[v]
        r = f"{a/b:.1f}x" if b > 0.05 else "  -"
        print(f"  {lab:14s} {a:6.1f}%        {b:6.1f}%       {r}")

    promote = 100 * (got[1] + got[2]) / max(n, 1)
    promote_c = 100 * (ctl[1] + ctl[2]) / max(nc, 1)
    print(f"\nlands on a weakly marked filament (1/3 or 2/3): "
          f"{promote:.1f}%  vs  {promote_c:.1f}% by chance")
    print(f"that is {got[1]+got[2]} labels that could be promoted, "
          f"of which about {int((got[1]+got[2])*promote_c/max(promote,1e-9))} "
          f"would be chance hits")

    json.dump({"n": n, "hit": dict(got), "control": dict(ctl),
               "promote_pct": promote, "chance_pct": promote_c},
              open(ROOT / "votes_check.json", "w"), indent=1)
    print("\nwrote votes_check.json")


if __name__ == "__main__":
    main()
