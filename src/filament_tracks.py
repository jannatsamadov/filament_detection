"""Give every filament a persistent identity across the observation series.

A filament lives for days to weeks while the Sun turns underneath it, so the
same object appears in several photographs at the same heliographic position.
Linking those appearances into tracks buys two things that nothing else here
provides:

  propagation   a filament annotated on day 1 and day 3 but missed on day 2 can
                be restored to day 2, which attacks the recall problem at the
                label level rather than by lowering a threshold

  persistence   a second, independent confidence signal. Annotator agreement
                says how many experts saw a filament; track length says how many
                days it survived. A feature seen once and never again is
                suspect in a way a single annotator's vote cannot express.

Linking happens in Carrington longitude, where a filament is nearly stationary:
it drifts only by (14.184 - omega(lat)) deg/day, the difference between the
Carrington frame and its true rate. Which differential-rotation law is used
barely matters over the gaps here -- measured, the choice moves a feature by
about 3 px at 30 deg latitude and 18 px at 60 deg over a day, against a filament
length of 125 px.
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

CARRINGTON = 14.184


def obs_time(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


class Detection:
    __slots__ = ("stem", "time", "ann_id", "image_id", "lat", "lon", "area",
                 "length", "rho", "track")

    def __init__(self, stem, time, ann_id, image_id, lat, lon, area, length, rho):
        self.stem, self.time = stem, time
        self.ann_id, self.image_id = ann_id, image_id
        self.lat, self.lon, self.area = lat, lon, area
        self.length, self.rho = length, rho
        self.track = None


def to_heliographic(anns, stem, image_id, max_rho=0.90):
    t = obs_time(stem)
    P, B0, L0 = solar_angles(t)
    out = []
    for a in anns:
        s = a["spine"]
        if len(s) < 4:
            continue
        xs = np.asarray(s[0::2], float); ys = np.asarray(s[1::2], float)
        rho = float(np.hypot(xs - DISK_CX, ys - DISK_CY).mean() / DISK_R)
        if rho > max_rho:
            continue
        lat, lon = pixel_to_heliographic(xs, ys, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
        ok = np.isfinite(lat)
        if ok.sum() < 2:
            continue
        ang = np.radians(lon[ok])
        out.append(Detection(
            stem, t, a["id"], image_id,
            float(np.mean(lat[ok])),
            float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0),
            float(a["area"]),
            float(np.hypot(np.diff(xs), np.diff(ys)).sum()),
            rho))
    return out


def predicted_longitude(det, t2):
    """Where a detection's Carrington longitude should be at a later time."""
    gap = (t2 - det.time).total_seconds() / 86400.0
    drift = CARRINGTON - float(rotation_rate(det.lat, "snodgrass", synodic=False))
    return (det.lon - drift * gap) % 360.0


def link(detections, max_gap_days=3.0, lat_tol=6.0, lon_tol=6.0):
    """Greedy chaining in time; each detection joins the nearest open track."""
    by_stem = collections.defaultdict(list)
    for d in detections:
        by_stem[d.stem].append(d)
    stems = sorted(by_stem, key=obs_time)

    tracks = []          # each track: list of detections, time-ordered
    for stem in stems:
        t = obs_time(stem)
        open_tracks = [tr for tr in tracks
                       if (t - tr[-1].time).total_seconds() / 86400.0 <= max_gap_days]
        taken = set()
        for det in sorted(by_stem[stem], key=lambda d: -d.area):
            best, best_i = None, -1
            for i, tr in enumerate(open_tracks):
                if i in taken or tr[-1].stem == stem:
                    continue
                last = tr[-1]
                pred = predicted_longitude(last, t)
                dlat = abs(last.lat - det.lat)
                dlon = abs((det.lon - pred + 180.0) % 360.0 - 180.0)
                if dlat > lat_tol or dlon > lon_tol:
                    continue
                score = dlat / lat_tol + dlon / lon_tol
                if best is None or score < best:
                    best, best_i = score, i
            if best is not None:
                taken.add(best_i)
                open_tracks[best_i].append(det)
                det.track = open_tracks[best_i]
            else:
                tr = [det]
                det.track = tr
                tracks.append(tr)
    return tracks


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    # one annotation set per photograph: the annotator who marked the most
    per_photo = {}
    for iid, f in fname.items():
        stem = Path(f).stem
        if stem not in per_photo or len(by_id[iid]) > len(by_id[per_photo[stem]]):
            per_photo[stem] = iid

    dets = []
    for stem, iid in per_photo.items():
        dets += to_heliographic(by_id[iid], stem, iid)
    print(f"{len(per_photo)} photographs, {len(dets)} filaments on the disk\n")

    tracks = link(dets)
    lens = np.array([len(t) for t in tracks])
    span = np.array([(t[-1].time - t[0].time).total_seconds() / 86400.0 for t in tracks])

    print(f"{len(tracks)} tracks")
    print(f"{'appearances':>12} {'tracks':>8} {'filaments':>10} {'share':>7}")
    for k in (1, 2, 3, 4):
        s = lens == k if k < 4 else lens >= 4
        lab = f"{k}" if k < 4 else "4+"
        print(f"{lab:>12} {int(s.sum()):8d} {int(lens[s].sum()):10d} "
              f"{100*lens[s].sum()/lens.sum():6.1f}%")
    print()
    multi = lens >= 2
    print(f"filaments in a multi-day track : {int(lens[multi].sum())} "
          f"({100*lens[multi].sum()/lens.sum():.1f}%)")
    print(f"track span, multi-day tracks   : median {np.median(span[multi]):.1f} d, "
          f"max {span.max():.1f} d")
    print()

    # Persistence against size: a long-lived filament should be a large one
    area = np.array([np.mean([x.area for x in t]) for t in tracks])
    print(f"{'appearances':>12} {'median area':>12} {'median |lat|':>13}")
    for k in (1, 2, 3):
        s = lens == k if k < 3 else lens >= 3
        lab = f"{k}" if k < 3 else "3+"
        la = np.array([np.mean([abs(x.lat) for x in t]) for t in tracks])
        print(f"{lab:>12} {np.median(area[s]):12.0f} {np.median(la[s]):13.1f}")

    out = ROOT / "filament_tracks.json"
    json.dump([{"track": i,
                "appearances": len(t),
                "span_days": (t[-1].time - t[0].time).total_seconds() / 86400.0,
                "members": [{"stem": x.stem, "image_id": x.image_id,
                             "ann_id": x.ann_id, "lat": x.lat, "lon": x.lon,
                             "area": x.area} for x in t]}
               for i, t in enumerate(tracks)], open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
