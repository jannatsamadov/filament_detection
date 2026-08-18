"""Animate a run of consecutive observations, so the rotation can be watched.

Two versions of the same run are written:

  raw       the flattened disks in time order, which is what the model sees
  derotated each frame remapped to heliographic coordinates, where a filament
            that survives should hold still

The second is a direct visual test of the ephemeris. Anything that stays put
across frames is correctly de-rotated; anything that slides is not, and the
label propagation built on this conversion placed 54% of its restored filaments
on material no darker than its surroundings.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRE = ROOT / "preprocessed" / "train"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles, heliographic_to_pixel   # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, OFFSET, SCALE    # noqa: E402


def longest_run(max_gap_days=2):
    stems = sorted(p.stem for p in PRE.glob("*.png"))
    t = sorted((datetime.strptime(s[:14], "%Y%m%d%H%M%S"), s) for s in stems)
    best, cur = [], [t[0]]
    for i in range(1, len(t)):
        if (t[i][0] - t[i - 1][0]).days <= max_gap_days:
            cur.append(t[i])
        else:
            if len(cur) > len(best):
                best = cur
            cur = [t[i]]
    return max(best, cur, key=len)


def load(stem, size=700):
    a = np.asarray(Image.open(PRE / f"{stem}.png"), dtype=np.float32) / SCALE + OFFSET
    lo, hi = 0.86, 1.10          # fixed window so brightness is comparable frame to frame
    v = np.clip((a - lo) / (hi - lo), 0, 1)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    v[((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) > (DISK_R * 0.995) ** 2] = 0.0
    return Image.fromarray((v * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)


def derotated(stem, t, size=700, nlat=360, nlon=720):
    """Remap the disk onto a heliographic grid; a persistent filament holds still."""
    a = np.asarray(Image.open(PRE / f"{stem}.png"), dtype=np.float32) / SCALE + OFFSET
    P, B0, L0 = solar_angles(t)
    lat = np.linspace(-75, 75, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    px, py, vis = heliographic_to_pixel(LAT, LON, DISK_CX, DISK_CY, DISK_R, P, B0, L0)
    xi = np.clip(px.astype(int), 0, a.shape[1] - 1)
    yi = np.clip(py.astype(int), 0, a.shape[0] - 1)
    out = a[yi, xi]
    rho = np.hypot(px - DISK_CX, py - DISK_CY) / DISK_R
    out[~vis | (rho > 0.95)] = np.nan
    lo, hi = 0.86, 1.10
    v = np.clip((out - lo) / (hi - lo), 0, 1)
    v = np.where(np.isnan(v), 0.0, v)
    img = Image.fromarray((v * 255).astype(np.uint8))
    return img.resize((size, int(size * nlat / nlon)), Image.BILINEAR)


def label(img, text, sub=""):
    im = img.convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
    d.text((8, 7), text, fill=(255, 255, 255))
    if sub:
        d.text((im.width - 8 - 7 * len(sub), 7), sub, fill=(160, 200, 255))
    return im


def main():
    run = longest_run()
    print(f"{len(run)} frames, {run[0][0].date()} to {run[-1][0].date()}")

    raw = [label(load(s), t.strftime("%Y-%m-%d %H:%M"), s[14:16]) for t, s in run]
    raw[0].save(FIGS / "12_rotation_raw.gif", save_all=True, append_images=raw[1:],
                duration=900, loop=0)
    print(f"wrote {FIGS / '12_rotation_raw.gif'}")

    der = [label(derotated(s, t), t.strftime("%Y-%m-%d %H:%M") + "  heliographic",
                 s[14:16]) for t, s in run]
    der[0].save(FIGS / "13_rotation_derotated.gif", save_all=True,
                append_images=der[1:], duration=900, loop=0)
    print(f"wrote {FIGS / '13_rotation_derotated.gif'}")
    print("\nIn the second, a surviving filament should stay in place. If it "
          "slides, the ephemeris is wrong.")


if __name__ == "__main__":
    main()
