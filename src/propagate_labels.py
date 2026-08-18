"""Restore filaments an annotator missed, using the neighbouring day's labels.

The model finds 71% of filaments two experts agree on but only 29% of those a
single expert marked, and at the operating point it now misses half of them
outright. The cause is in the labels: a filament one expert skipped is taught to
the model as background.

Time gives an independent way to catch those. Measured on this dataset, of the
filaments annotated in one photograph and still on the visible disk in another
taken the *same day* -- a different GONG station, so the Sun has turned by only
two or three degrees -- 29% carry no annotation in the second one. That is not
evolution; a filament does not vanish in three hours. At a one-day gap the figure
is 46%. At two to three days it is 74%, which is mostly real evolution, so
nothing beyond 1.5 days is propagated here.

A propagated filament is written with a reduced vote, because it is inferred
rather than observed: it should pull the model toward "something is here" without
claiming the confidence of a filament an expert actually drew.

The mask is carried across by warping the spine through heliographic coordinates
and sweeping the recorded half-width along it, rather than warping pixels. That
matches how the model represents a filament anyway, and sweeping the annotated
spine reproduces the annotated mask at median IoU 0.758.
"""
from __future__ import annotations

import collections, json, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
CONSENSUS = ROOT / "consensus"
OUT = ROOT / "consensus_prop"
sys.path.insert(0, str(HERE))
from solar_ephemeris import (solar_angles, pixel_to_heliographic,
                             heliographic_to_pixel, rotation_rate)   # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL, spine_heatmap  # noqa: E402

CARRINGTON = 14.184
MAX_GAP_DAYS = 1.5
PROP_VOTE = 0.4            # weight of an inferred filament, against 1.0 observed
MATCH_LAT, MATCH_LON = 6.0, 6.0
SIGMA = 3.0


def obs_time(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def spine_arrays(ann):
    s = ann["spine"]
    return np.asarray(s[0::2], float), np.asarray(s[1::2], float)


def local_halfwidth(ann):
    """Half-width at each spine point, from the annotated polygon."""
    p = ann["segmentation"][0]
    xs, ys = np.asarray(p[0::2], float), np.asarray(p[1::2], float)
    pad = 25
    x0, y0 = xs.min() - pad, ys.min() - pad
    w = int(xs.max() - x0 + pad) + 1
    h = int(ys.max() - y0 + pad) + 1
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon(list(zip(xs - x0, ys - y0)), fill=1)
    m = np.asarray(im, bool)
    if m.sum() < 10:
        return None
    dt = distance_transform_edt(m)
    sx, sy = spine_arrays(ann)
    xi = np.clip((sx - x0).astype(int), 0, w - 1)
    yi = np.clip((sy - y0).astype(int), 0, h - 1)
    return dt[yi, xi]


def summarise(ann, angles):
    """Heliographic spine, half-widths and mean position for one annotation."""
    sx, sy = spine_arrays(ann)
    if len(sx) < 2:
        return None
    P, B0, L0 = angles
    lat, lon = pixel_to_heliographic(sx, sy, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
    ok = np.isfinite(lat)
    if ok.sum() < 2:
        return None
    r = local_halfwidth(ann)
    if r is None:
        return None
    ang = np.radians(lon[ok])
    return dict(lat=lat[ok], lon=lon[ok], r=r[ok], area=float(ann["area"]),
                mlat=float(lat[ok].mean()),
                mlon=float(np.degrees(np.arctan2(np.sin(ang).mean(),
                                                 np.cos(ang).mean())) % 360.0))


def project(src, t_src, t_dst):
    """Carry a summarised filament forward to another observation time."""
    gap = (t_dst - t_src).total_seconds() / 86400.0
    drift = CARRINGTON - rotation_rate(src["lat"], "snodgrass", synodic=False)
    lon = (src["lon"] - drift * gap) % 360.0
    P, B0, L0 = solar_angles(t_dst)
    px, py, vis = heliographic_to_pixel(src["lat"], lon, DISK_CX, DISK_CY, DISK_R,
                                        P, B0, L0)
    if not np.all(vis):
        return None
    rho = np.hypot(px - DISK_CX, py - DISK_CY) / DISK_R
    if rho.mean() > 0.85:
        return None
    mdrift = CARRINGTON - float(rotation_rate(src["mlat"], "snodgrass", synodic=False))
    return dict(px=px, py=py, r=src["r"],
                mlat=src["mlat"], mlon=(src["mlon"] - mdrift * gap) % 360.0)


def sweep(px, py, r, canvas):
    dr = ImageDraw.Draw(canvas)
    for x, y, rr in zip(px, py, r):
        rr = max(float(rr), 1.0)
        dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=1)


def main():
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}

    by_stem = collections.defaultdict(list)
    for iid, f in fname.items():
        by_stem[Path(f).stem].append(iid)
    stems = sorted(by_stem, key=obs_time)
    times = {s: obs_time(s) for s in stems}

    # union of every annotator's filaments per photograph, summarised once
    print("summarising filaments in heliographic coordinates...", flush=True)
    summary, present = {}, {}
    for s in stems:
        ang = solar_angles(times[s])
        items, pos = [], []
        for iid in by_stem[s]:
            for a in by_id[iid]:
                r = summarise(a, ang)
                if r is not None:
                    items.append(r)
                    pos.append((r["mlat"], r["mlon"]))
        summary[s] = items
        present[s] = np.array(pos) if pos else np.zeros((0, 2))

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    n_added = n_photo = 0
    for k, dst in enumerate(stems, 1):
        vote = np.asarray(Image.open(CONSENSUS / dst / "mask_vote.png"),
                          dtype=np.float32) / 255.0
        spine_v = np.asarray(Image.open(CONSENSUS / dst / "spine_vote.png"),
                             dtype=np.float32) / 255.0
        radius = np.asarray(Image.open(CONSENSUS / dst / "radius.png"), dtype=np.float32)

        add_mask = Image.new("L", (FULL, FULL), 0)
        add_spine = Image.new("L", (FULL, FULL), 0)
        added = 0
        here = present[dst]

        for src in stems:
            gap = abs((times[dst] - times[src]).total_seconds()) / 86400.0
            if src == dst or gap > MAX_GAP_DAYS:
                continue
            for item in summary[src]:
                if item["area"] < 600:
                    continue
                pr = project(item, times[src], times[dst])
                if pr is None:
                    continue
                if len(here):
                    dlat = np.abs(here[:, 0] - pr["mlat"])
                    dlon = np.abs((here[:, 1] - pr["mlon"] + 180.0) % 360.0 - 180.0)
                    if np.any((dlat <= MATCH_LAT) & (dlon <= MATCH_LON)):
                        continue          # already annotated here
                sweep(pr["px"], pr["py"], pr["r"], add_mask)
                dr = ImageDraw.Draw(add_spine)
                dr.line(list(zip(pr["px"], pr["py"])), fill=1, width=3)
                added += 1

        if added:
            am = np.asarray(add_mask, bool)
            asp = spine_heatmap(np.asarray(add_spine, bool), SIGMA)
            vote = np.maximum(vote, am * PROP_VOTE)
            spine_v = np.maximum(spine_v, asp * PROP_VOTE)
            radius = np.where((radius <= 0) & am, distance_transform_edt(am), radius)
            n_added += added
            n_photo += 1

        dd = OUT / dst
        dd.mkdir(exist_ok=True)
        Image.fromarray(np.clip(vote * 255, 0, 255).astype(np.uint8)).save(
            dd / "mask_vote.png", optimize=True)
        Image.fromarray(np.clip(spine_v * 255, 0, 255).astype(np.uint8)).save(
            dd / "spine_vote.png", optimize=True)
        Image.fromarray(np.clip(radius, 0, 255).astype(np.uint8)).save(
            dd / "radius.png", optimize=True)

        if k % 100 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(stems)}  {n_added} filaments restored so far, "
                  f"{el:.0f}s elapsed, {el/k*(len(stems)-k):.0f}s left", flush=True)

    print(f"\n{n_added} filaments restored across {n_photo} of {len(stems)} photographs")
    print(f"mean {n_added/max(n_photo,1):.1f} per affected photograph, "
          f"written with vote {PROP_VOTE}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
