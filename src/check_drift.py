"""Measure the Sun's rotation from the annotations themselves.

Every rotation number used so far came from a textbook law -- Snodgrass, applied
to the latitude of each spine point. Nothing checked that law against this
dataset. This does, using only what the annotators drew: take a filament marked
in one photograph, find the same filament in another taken hours or days later,
and measure how far it actually moved.

Filaments carry no identity across photographs, so identity has to be inferred.
The match is deliberately loose in longitude -- wide enough to contain any
plausible rotation rate, so the measurement is not assuming the answer -- and
tight in latitude, which rotation does not change. A filament also has to keep a
comparable size to be called the same filament; they grow, but they do not
triple in a day.

Two things come out of it. The observed drift against latitude is the rotation
law this data actually shows. And where a filament marked by all three
annotators lands on one marked by only one, that is a missing label found by
geometry alone -- the check that decides whether time is worth pursuing at all.
"""
from __future__ import annotations

import collections, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
sys.path.insert(0, str(HERE))
from solar_ephemeris import (solar_angles, pixel_to_heliographic,      # noqa: E402
                             rotation_rate)
from propagate_labels import obs_time, spine_arrays                    # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R                        # noqa: E402

CARRINGTON = 14.184          # deg/day, the rigid reference frame
MAX_GAP = 3.5                # days
LAT_TOL = 8.0                # rotation does not move a filament in latitude
LON_WINDOW = 40.0            # wide enough to hold any plausible rotation rate
AREA_RATIO = 3.0
MAX_RHO = 0.85               # stay away from the limb, where projection blows up


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def describe(ann, angles):
    """Heliographic centroid, extent and size of one annotation."""
    sx, sy = spine_arrays(ann)
    if len(sx) < 2:
        return None
    rho = np.hypot(sx - DISK_CX, sy - DISK_CY) / DISK_R
    if rho.mean() > MAX_RHO:
        return None
    P, B0, L0 = angles
    lat, lon = pixel_to_heliographic(sx, sy, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return None
    a = np.radians(lon[ok])
    mlon = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0
    return dict(mlat=float(lat[ok].mean()), mlon=float(mlon),
                area=float(ann["area"]), n_votes=None,
                lat_span=float(lat[ok].max() - lat[ok].min()),
                lon_span=float(np.degrees(np.ptp(np.unwrap(a)))))


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)

    # How many annotators worked on each photograph, and their filaments pooled.
    # A photograph appears once per annotator, under different image_ids.
    photo_ids = collections.defaultdict(list)
    for iid in ann_of:
        photo_ids[stem[iid]].append(iid)
    photos = sorted(photo_ids)
    times = {s: obs_time(s) for s in photos}
    n_ann = {s: len(photo_ids[s]) for s in photos}
    print(f"{len(photos)} photographs; annotator counts: "
          f"{dict(collections.Counter(n_ann.values()))}")

    # Per photograph: every filament, tagged with which annotator drew it.
    feats = {}
    for s in photos:
        ang = solar_angles(times[s])
        out = []
        for k, iid in enumerate(sorted(photo_ids[s])):
            for a in ann_of[iid]:
                f = describe(a, ang)
                if f is not None:
                    f["who"] = k
                    out.append(f)
        feats[s] = out
    print(f"{sum(len(v) for v in feats.values())} usable filaments "
          f"(spine on disk, rho < {MAX_RHO})\n")

    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    print(f"{len(pairs)} photograph pairs up to {MAX_GAP} days apart")

    rows = []
    for sa, sb in pairs:
        ta, tb = times[sa], times[sb]
        if tb < ta:
            sa, sb, ta, tb = sb, sa, tb, ta
        gap = (tb - ta).total_seconds() / 86400.0
        A = [f for f in feats[sa] if f["who"] == 0]      # one annotator's view
        B = [f for f in feats[sb] if f["who"] == 0]
        if not A or not B:
            continue
        for fa in A:
            # Longitude is what is being measured, so it must not decide the
            # match -- picking the nearest candidate in longitude would drag
            # the answer toward zero drift. Latitude and size decide instead,
            # and the match is taken only when it is unambiguous: exactly one
            # candidate in the band. Everything else is thrown away.
            cand = []
            for fb in B:
                if abs(fb["mlat"] - fa["mlat"]) > LAT_TOL:
                    continue
                r = fb["area"] / max(fa["area"], 1.0)
                if not (1 / AREA_RATIO <= r <= AREA_RATIO):
                    continue
                if abs(wrap(fb["mlon"] - fa["mlon"])) > LON_WINDOW:
                    continue          # wide guard only, not a selector
                cand.append(fb)
            if len(cand) != 1:
                continue
            fb = cand[0]
            dlon = wrap(fb["mlon"] - fa["mlon"])
            rows.append(dict(gap=gap, lat=fa["mlat"], dlon=dlon,
                             dlat=fb["mlat"] - fa["mlat"],
                             rate=dlon / gap,
                             area_ratio=fb["area"] / max(fa["area"], 1.0)))

    r = rows
    print(f"{len(r)} filament matches\n")
    if len(r) < 50:
        print("too few matches to measure anything"); return

    gap = np.array([x["gap"] for x in r]); lat = np.array([x["lat"] for x in r])
    rate = np.array([x["rate"] for x in r]); dlat = np.array([x["dlat"] for x in r])
    ar = np.array([x["area_ratio"] for x in r])

    print("1. HOW FAST FILAMENTS ACTUALLY DRIFT")
    print("   (relative to the Carrington frame; negative = slower than Carrington)")
    print(f"   overall median {np.median(rate):+.3f} deg/day  "
          f"(IQR {np.percentile(rate,25):+.3f} to {np.percentile(rate,75):+.3f})")
    print("\n   by latitude band:")
    print("   |lat|      n   observed drift   Snodgrass predicts   difference")
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 45), (45, 90)):
        m = (np.abs(lat) >= lo) & (np.abs(lat) < hi) & (gap > 0.15)
        if m.sum() < 15:
            continue
        obs = float(np.median(rate[m]))
        pred = CARRINGTON - float(np.median(
            [rotation_rate(x, "snodgrass", synodic=False) for x in lat[m]]))
        print(f"   {lo:2d}-{hi:2d}   {m.sum():4d}   {obs:+8.3f}        "
              f"{-pred:+8.3f}          {obs + pred:+7.3f}")

    print("\n2. WHICH LAW FITS THIS DATA BEST")
    print("   (median absolute longitude error after removing each law, deg)")
    long_gap = gap > 0.15
    for law in ("carrington", "snodgrass", "howard"):
        if law == "carrington":
            pred_rate = np.zeros_like(lat)
        else:
            pred_rate = np.array([CARRINGTON - rotation_rate(x, law, synodic=False)
                                  for x in lat])
        resid = (rate - pred_rate) * gap
        print(f"   {law:11s} {np.median(np.abs(resid[long_gap])):.3f} deg "
              f"= {np.median(np.abs(resid[long_gap]))*15.8:.1f} px")

    print("\n3. WHAT ELSE CHANGES BETWEEN THE TWO PHOTOGRAPHS")
    print(f"   latitude shift    median {np.median(np.abs(dlat)):.2f} deg "
          f"(rotation should not move it at all)")
    print(f"   area ratio        median {np.median(ar):.2f}, "
          f"10th {np.percentile(ar,10):.2f}, 90th {np.percentile(ar,90):.2f}")
    for lo, hi in ((0, 0.3), (0.3, 1.2), (1.2, 2.5), (2.5, 3.5)):
        m = (gap >= lo) & (gap < hi)
        if m.sum() > 15:
            print(f"     gap {lo}-{hi} d: n={m.sum():4d}  area ratio median "
                  f"{np.median(ar[m]):.2f}  |dlat| {np.median(np.abs(dlat[m])):.2f} deg")

    json.dump({"n_matches": len(r),
               "median_rate": float(np.median(rate)),
               "median_abs_dlat": float(np.median(np.abs(dlat))),
               "median_area_ratio": float(np.median(ar))},
              open(ROOT / "drift_check.json", "w"), indent=1)
    print("\nwrote drift_check.json")


if __name__ == "__main__":
    main()
