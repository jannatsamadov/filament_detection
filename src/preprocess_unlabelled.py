"""Put the unlabelled GONG archive through the same preparation as the competition data.

Pretraining only transfers if the encoder sees the same kind of input during both
stages. The 566 annotated photographs are flattened against their own quiet-sun
radial profile and masked outside the limb, so these 3810 must be too -- otherwise
the features learned here would be features of limb darkening and scattered light,
which the fine-tuning data no longer contains.

Same code path as src/preprocess.py: locate the limb by the steepest radial
brightness drop, divide out the frame's own profile, zero everything past 0.995 R.
"""
from __future__ import annotations

import json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "gong_unlabelled"
OUT = ROOT / "preprocessed" / "unlabelled"
sys.path.insert(0, str(HERE))
from geometry import fit_disk, radial_coordinates   # noqa: E402
from photometry import flatten                      # noqa: E402

OFFSET, SCALE = 0.5, 255.0


def one(path: Path):
    dst = OUT / (path.stem + ".png")
    if dst.exists():
        return path.stem, None, True
    try:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        if img.shape != (2048, 2048):
            return path.stem, None, False
        fit = fit_disk(img, n_rays=180)
        rr, _ = radial_coordinates(img.shape, fit)
        flat = flatten(img, rr, r_max=0.995)
        Image.fromarray(np.clip((flat - OFFSET) * SCALE, 0, 255).astype(np.uint8)).save(
            dst, optimize=True)
        return path.stem, (fit.r, fit.residual), False
    except Exception:
        return path.stem, None, False


def main(workers=6):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("*.jpg"))
    print(f"{len(files)} unlabelled images -> {OUT}")
    t0, fits, bad, cached = time.time(), {}, [], 0
    with Pool(workers) as pool:
        for k, (stem, fit, was_cached) in enumerate(pool.imap_unordered(one, files, 8), 1):
            if was_cached:
                cached += 1
            elif fit is None:
                bad.append(stem)
            else:
                fits[stem] = {"r": fit[0], "residual": fit[1]}
            if k % 300 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(files)}  {el/60:.0f} min elapsed, "
                      f"{el/k*(len(files)-k)/60:.0f} min left", flush=True)

    if fits:
        p = OUT.parent / "disk_fits_unlabelled.json"
        old = json.load(open(p)) if p.exists() else {}
        old.update(fits)
        json.dump(old, open(p, "w"), indent=1)
        r = np.array([v["r"] for v in fits.values()])
        res = np.array([v["residual"] for v in fits.values()])
        print(f"\nradius: {r.min():.1f} - {r.max():.1f} px (median {np.median(r):.1f})")
        print(f"fit residual: median {np.median(res):.2f} px, "
              f"90th {np.percentile(res, 90):.2f}, max {res.max():.2f}")
        off = int((np.abs(r - 904) > 10).sum())
        print(f"radius off by more than 10 px: {off} "
              f"({100*off/len(r):.2f}%)  <- these should be dropped before training")
    print(f"{cached} already done, {len(bad)} failed")
    n = len(list(OUT.glob('*.png')))
    size = sum(f.stat().st_size for f in OUT.glob("*.png")) / 1e9
    print(f"{n} images ready ({size:.2f} GB) in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
