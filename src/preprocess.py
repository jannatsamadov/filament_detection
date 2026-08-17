"""Precompute flattened images once so training does not pay for it every epoch.

Writes uint8 PNGs where the flattened value f (quiet sun = 1.0) is stored as
    byte = clip((f - 0.5) * 255, 0, 255)
so byte 128 is quiet sun, filaments land near 90-115, plage above 140, and
off-disk is 0. Also records each image's disk fit in disk_fits.json.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026"
OUT = ROOT / "preprocessed"
sys.path.insert(0, str(HERE))
from geometry import fit_disk, radial_coordinates   # noqa: E402
from photometry import flatten                      # noqa: E402

OFFSET, SCALE = 0.5, 255.0


def encode(flat):
    return np.clip((flat - OFFSET) * SCALE, 0, 255).astype(np.uint8)


def decode(byte_img):
    return byte_img.astype(np.float32) / SCALE + OFFSET


def run(split, n_rays=180):
    src = BASE / split / f"{split}_images"
    dst = OUT / split
    dst.mkdir(parents=True, exist_ok=True)
    fits = {}
    files = sorted(src.iterdir())
    t0 = time.time()
    for k, f in enumerate(files, 1):
        out_path = dst / (f.stem + ".png")
        if out_path.exists():
            continue
        img = np.array(Image.open(f).convert("L")).astype(np.float32)
        fit = fit_disk(img, n_rays=n_rays)
        rr, _ = radial_coordinates(img.shape, fit)
        flat = flatten(img, rr, r_max=0.995)
        Image.fromarray(encode(flat)).save(out_path, optimize=True)
        fits[f.stem] = dict(cx=fit.cx, cy=fit.cy, r=fit.r,
                            residual=fit.residual, n_points=fit.n_points)
        if k % 50 == 0:
            el = time.time() - t0
            print(f"  {split}: {k}/{len(files)}  {el:.0f}s elapsed, "
                  f"{el/k*(len(files)-k):.0f}s left", flush=True)
    fp = OUT / f"disk_fits_{split}.json"
    if fits:
        old = json.load(open(fp)) if fp.exists() else {}
        old.update(fits)
        json.dump(old, open(fp, "w"), indent=1)
    print(f"{split}: {len(files)} images ready in {dst}")
    if fits:
        res = np.array([v["residual"] for v in fits.values()])
        rad = np.array([v["r"] for v in fits.values()])
        print(f"   disk fit residual: median {np.median(res):.2f} px, max {res.max():.2f} px")
        print(f"   radius: {rad.min():.1f} - {rad.max():.1f} px (median {np.median(rad):.1f})")
        bad = [k for k, v in fits.items() if v["residual"] > 6]
        if bad:
            print(f"   WARNING: {len(bad)} images with residual > 6 px: {bad[:8]}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "test"]):
        run(s)
