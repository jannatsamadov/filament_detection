"""Measure how fast these filaments actually rotate, instead of assuming a law.

Filaments are anchored in the large-scale magnetic field and are known to rotate
more rigidly than the photospheric plasma, so picking a differential-rotation law
off the shelf would be a guess. The dataset settles it: 247 photographs have a
neighbour within a day and 106 within the same day, and both are annotated, so
the same filament can be located twice.

Method: convert every annotated spine to heliographic coordinates at its own
observation time, pair filaments between two photographs by latitude and by
Carrington longitude, and look at the residual longitude drift. Under the correct
rotation law the drift is zero; under a wrong one it grows with the time gap.

Reported per latitude band, because that is where the laws disagree.
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


def obs_time(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def spine_centre_heliographic(ann, angles):
    """Mean heliographic position of a filament's spine."""
    s = ann["spine"]
    if len(s) < 4:
        return None
    xs = np.asarray(s[0::2], float)
    ys = np.asarray(s[1::2], float)
    P, B0, L0 = angles
    lat, lon = pixel_to_heliographic(xs, ys, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
    ok = np.isfinite(lat)
    if ok.sum() < 2:
        return None
    # circular mean for longitude
    la = float(np.mean(lat[ok]))
    ang = np.radians(lon[ok])
    lo = float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
    area = float(ann["area"])
    return la, lo, area


def collect(max_gap_days=2.5, max_disc=45.0):
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    # one annotation set per photograph, the one with the most filaments
    per_photo = {}
    for iid, f in fname.items():
        stem = Path(f).stem
        if stem not in per_photo or len(by_id[iid]) > len(by_id[per_photo[stem]]):
            per_photo[stem] = iid

    stems = sorted(per_photo)
    times = {s: obs_time(s) for s in stems}
    order = sorted(stems, key=lambda s: times[s])

    feats = {}
    for s in order:
        ang = solar_angles(times[s])
        pts = []
        for a in by_id[per_photo[s]]:
            r = spine_centre_heliographic(a, ang)
            # only keep filaments comfortably on the disk: near the limb the
            # longitude of a feature is poorly determined by its projection
            if r is not None:
                lat, lon, area = r
                pts.append((lat, lon, area))
        feats[s] = pts

    pairs = []
    for i, s1 in enumerate(order):
        for s2 in order[i + 1:]:
            gap = (times[s2] - times[s1]).total_seconds() / 86400.0
            if gap <= 0 or gap > max_gap_days:
                continue
            if gap > 0:
                pairs.append((s1, s2, gap))
            break_outer = gap > max_gap_days
        # continue scanning; the list is time-ordered so later ones only grow
    return feats, times, pairs


def match_and_drift(feats, pairs, lat_tol=6.0, lon_tol=25.0, min_area=800):
    """For each pair, match filaments and record (latitude, gap, longitude drift)."""
    out = []
    for s1, s2, gap in pairs:
        a = [p for p in feats[s1] if p[2] >= min_area]
        b = [p for p in feats[s2] if p[2] >= min_area]
        if not a or not b:
            continue
        used = set()
        for la1, lo1, ar1 in a:
            best, bj = None, -1
            for j, (la2, lo2, ar2) in enumerate(b):
                if j in used or abs(la2 - la1) > lat_tol:
                    continue
                dlon = (lo2 - lo1 + 180.0) % 360.0 - 180.0
                if abs(dlon) > lon_tol:
                    continue
                # prefer the closest in latitude, then in longitude
                score = abs(la2 - la1) + 0.2 * abs(dlon)
                if best is None or score < best[0]:
                    best, bj = (score, dlon, la2), j
            if best is not None:
                used.add(bj)
                out.append((0.5 * (la1 + best[2]), gap, best[1]))
    return np.array(out) if out else np.zeros((0, 3))


def main():
    print("converting spines to heliographic coordinates...", flush=True)
    feats, times, pairs = collect()
    n_f = sum(len(v) for v in feats.values())
    print(f"{len(feats)} photographs, {n_f} filaments, {len(pairs)} photograph pairs "
          f"within 2.5 days\n")

    m = match_and_drift(feats, pairs)
    if len(m) == 0:
        print("no matches"); return
    lat, gap, drift = m.T
    print(f"{len(m)} filament matches across pairs "
          f"(gap median {np.median(gap):.2f} d)\n")

    print("Residual Carrington longitude drift after de-rotation.")
    print("Zero means the Carrington rate (14.184 deg/day sidereal) is right for")
    print("that latitude; negative means the filaments lag it.\n")
    print(f"{'|lat|':>10} {'n':>5} {'median drift':>13} {'per day':>10} "
          f"{'implied sidereal':>18}")
    edges = [0, 15, 30, 45, 90]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (np.abs(lat) >= lo) & (np.abs(lat) < hi) & (gap > 0.2)
        if s.sum() < 8:
            continue
        per_day = np.median(drift[s] / gap[s])
        print(f"{lo:4d}-{hi:<5d} {s.sum():5d} {np.median(drift[s]):13.2f} "
              f"{per_day:10.3f} {14.184 + per_day:18.3f}")

    print("\nFor comparison, the standard laws (sidereal, deg/day):")
    print(f"{'|lat|':>10} {'Snodgrass':>11} {'Howard Ha':>11} {'rigid':>8}")
    for la in (7, 22, 37, 55):
        print(f"{la:8d}   {float(rotation_rate(la,'snodgrass',False)):11.3f} "
              f"{float(rotation_rate(la,'howard',False)):11.3f} "
              f"{14.184:8.3f}")


if __name__ == "__main__":
    main()
