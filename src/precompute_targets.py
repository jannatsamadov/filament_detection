"""Precompute the three per-pixel targets once, per annotation set.

Building them on the fly cost 6.94 s per training step against 0.43 s of actual
GPU work: rasterising the polygons and running two 2048x2048 distance transforms
dominates, and with 1154 annotation sets a small LRU cache almost never hits.
Precomputing moves that work out of the training loop entirely.

Stored as uint8 PNGs, one directory per annotation set id:
  mask.png    0 or 255
  spine.png   Gaussian ridge, 0..255
  radius.png  distance to the mask boundary in pixels, clipped at 255
"""
from __future__ import annotations

import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from spine_data import rasterize, spine_heatmap, load_annotations   # noqa: E402

OUT = ROOT / "targets"
SIGMA = 3.0


def build_one(args):
    iid, anns = args
    d = OUT / iid
    if (d / "radius.png").exists():
        return iid, True
    d.mkdir(parents=True, exist_ok=True)
    mask, spine, radius = rasterize(anns)
    heat = spine_heatmap(spine, SIGMA)
    Image.fromarray((mask * 255).astype(np.uint8)).save(d / "mask.png", optimize=True)
    Image.fromarray(np.clip(heat * 255, 0, 255).astype(np.uint8)).save(d / "spine.png",
                                                                      optimize=True)
    Image.fromarray(np.clip(radius, 0, 255).astype(np.uint8)).save(d / "radius.png",
                                                                   optimize=True)
    return iid, False


def main(workers=6):
    by_id, fname = load_annotations()
    OUT.mkdir(exist_ok=True)
    jobs = [(iid, anns) for iid, anns in sorted(by_id.items())]
    print(f"{len(jobs)} annotation sets -> {OUT}")
    t0 = time.time()
    done = skipped = 0
    with Pool(workers) as pool:
        for k, (iid, was_cached) in enumerate(pool.imap_unordered(build_one, jobs, 8), 1):
            done += 1
            skipped += int(was_cached)
            if k % 100 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(jobs)}  {el:.0f}s elapsed, "
                      f"{el/k*(len(jobs)-k):.0f}s left", flush=True)
    size = sum(f.stat().st_size for f in OUT.rglob("*.png")) / 1e6
    print(f"done: {done} sets ({skipped} already present) in "
          f"{(time.time()-t0)/60:.1f} min, {size:.0f} MB on disk")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
