"""Find the rotation rate by scanning candidates, not by averaging drifts.

The first attempt matched each filament to its nearest neighbour in the next
photograph and averaged the leftover longitude drift. That gave rotation
*increasing* with latitude -- physically backwards -- because nearest-neighbour
matching pulls the drift toward zero and, where the population is uneven, can
invert its sign. The measurement was of the matching, not of the Sun.

This scans instead. For each candidate rate, every filament of the earlier
photograph is carried forward by that rate and counted as recovered if a filament
of the later photograph sits within a tight window. The rate that recovers the
most filaments is the answer, and no averaging of a biased quantity is involved.

Run per latitude band, which is where the laws differ.
"""
from __future__ import annotations

import collections, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles, pixel_to_heliographic, rotation_rate  # noqa
from spine_data import DISK_CX, DISK_CY, DISK_R                                 # noqa

CARRINGTON = 14.184          # sidereal deg/day, the frame Carrington longitude uses


def obs_time(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def filaments_heliographic(anns, angles, max_rho=0.85):
    """(lat, lon, area) per filament, dropping anything near the limb.

    Near the limb a small error in the projected position is a large error in
    longitude, so those filaments would only add noise.
    """
    P, B0, L0 = angles
    out = []
    for a in anns:
        s = a["spine"]
        if len(s) < 4:
            continue
        xs = np.asarray(s[0::2], float); ys = np.asarray(s[1::2], float)
        rho = np.hypot(xs - DISK_CX, ys - DISK_CY) / DISK_R
        if rho.mean() > max_rho:
            continue
        lat, lon = pixel_to_heliographic(xs, ys, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
        ok = np.isfinite(lat)
        if ok.sum() < 2:
            continue
        ang = np.radians(lon[ok])
        out.append((float(np.mean(lat[ok])),
                    float(np.degrees(np.arctan2(np.sin(ang).mean(),
                                                np.cos(ang).mean())) % 360.0),
                    float(a["area"])))
    return out


def build():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    per_photo = {}
    for iid, f in fname.items():
        stem = Path(f).stem
        if stem not in per_photo or len(by_id[iid]) > len(by_id[per_photo[stem]]):
            per_photo[stem] = iid

    order = sorted(per_photo, key=obs_time)
    times = {s: obs_time(s) for s in order}
    feats = {s: filaments_heliographic(by_id[per_photo[s]], solar_angles(times[s]))
             for s in order}

    pairs = []
    for i, s1 in enumerate(order):
        for s2 in order[i + 1:i + 6]:
            gap = (times[s2] - times[s1]).total_seconds() / 86400.0
            if 0.25 <= gap <= 2.5:
                pairs.append((s1, s2, gap))
    return feats, pairs


def recovered(feats, pairs, rate, lat_lo, lat_hi, lat_tol=5.0, lon_tol=4.0,
              min_area=800):
    """How many filaments are found again after carrying them at `rate`."""
    hit = tot = 0
    excess = CARRINGTON - rate        # drift of a feature in the Carrington frame
    for s1, s2, gap in pairs:
        a = [p for p in feats[s1] if p[2] >= min_area and lat_lo <= abs(p[0]) < lat_hi]
        b = [p for p in feats[s2] if p[2] >= min_area]
        if not a or not b:
            continue
        blat = np.array([p[0] for p in b]); blon = np.array([p[1] for p in b])
        for la, lo, _ in a:
            tot += 1
            pred = (lo - excess * gap) % 360.0
            dlat = np.abs(blat - la)
            dlon = np.abs((blon - pred + 180.0) % 360.0 - 180.0)
            if np.any((dlat <= lat_tol) & (dlon <= lon_tol)):
                hit += 1
    return hit, tot


def main():
    print("building heliographic catalogue...", flush=True)
    feats, pairs = build()
    n = sum(len(v) for v in feats.values())
    print(f"{len(feats)} photographs, {n} filaments away from the limb, "
          f"{len(pairs)} pairs between 0.25 and 2.5 days\n")

    rates = np.arange(11.0, 16.01, 0.25)
    bands = [(0, 15), (15, 30), (30, 45), (45, 90)]

    print(f"{'|lat| band':>12} {'n':>6} {'best rate':>10} {'recovered':>10} "
          f"{'Snodgrass':>10} {'Howard':>8} {'rigid':>7}")
    for lo, hi in bands:
        scores = [(recovered(feats, pairs, r, lo, hi), r) for r in rates]
        (hit, tot), best = max(scores, key=lambda t: t[0][0])
        if tot < 30:
            print(f"{f'{lo}-{hi}':>12} {tot:>6}   too few matches")
            continue
        mid = 0.5 * (lo + hi)
        print(f"{f'{lo}-{hi}':>12} {tot:>6} {best:10.2f} "
              f"{100*hit/tot:9.1f}% {float(rotation_rate(mid,'snodgrass',False)):10.2f} "
              f"{float(rotation_rate(mid,'howard',False)):8.2f} {CARRINGTON:7.2f}")

    print("\nRecovery curve for the equatorial band, to show the peak is real:")
    for r in np.arange(12.5, 15.51, 0.25):
        hit, tot = recovered(feats, pairs, r, 0, 20)
        bar = "#" * int(60 * hit / max(tot, 1))
        print(f"  {r:5.2f} deg/day  {100*hit/max(tot,1):5.1f}%  {bar}")


if __name__ == "__main__":
    main()
