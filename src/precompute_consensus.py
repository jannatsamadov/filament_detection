"""Per-photograph targets that record how many annotators agreed.

Until now every annotator's version was a separate training sample, so a
filament that one expert of three marked was shown to the model once as
foreground and twice as background. The model resolved that contradiction by
siding with the majority: it recovers 71% of filaments two or more experts
agree on, but only 29% of those a single expert marked.

Here each photograph becomes one sample and the target becomes the *fraction of
annotators who marked the pixel* -- 1.0, 0.67, 0.33 -- so disagreement is
represented as graded confidence instead of contradictory labels. A threshold on
the output then means something physical: how much expert agreement to require.

Written per photograph (707), not per annotation set (1154):
  mask_vote.png    fraction of annotators whose mask covers the pixel
  spine_vote.png   mean of the annotators' spine heatmaps
  radius.png       mean half-width where any annotator drew a filament
  n_annotators     recorded in meta.json
"""
from __future__ import annotations

import collections, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "consensus"
sys.path.insert(0, str(HERE))
from spine_data import rasterize, spine_heatmap, load_annotations, FULL  # noqa: E402

SIGMA = 3.0


def build(args):
    stem, ann_sets = args
    d = OUT / stem
    if (d / "radius.png").exists():
        return True
    d.mkdir(parents=True, exist_ok=True)

    votes = np.zeros((FULL, FULL), np.float32)
    spine_sum = np.zeros((FULL, FULL), np.float32)
    rad_sum = np.zeros((FULL, FULL), np.float32)
    rad_cnt = np.zeros((FULL, FULL), np.float32)

    for anns in ann_sets:
        mask, spine, radius = rasterize(anns)
        votes += mask
        spine_sum += spine_heatmap(spine, SIGMA)
        rad_sum += radius * mask
        rad_cnt += mask

    n = len(ann_sets)
    votes /= n
    spine_sum /= n
    radius = np.divide(rad_sum, np.maximum(rad_cnt, 1.0))

    Image.fromarray(np.clip(votes * 255, 0, 255).astype(np.uint8)).save(
        d / "mask_vote.png", optimize=True)
    Image.fromarray(np.clip(spine_sum * 255, 0, 255).astype(np.uint8)).save(
        d / "spine_vote.png", optimize=True)
    Image.fromarray(np.clip(radius, 0, 255).astype(np.uint8)).save(
        d / "radius.png", optimize=True)
    json.dump({"n_annotators": n}, open(d / "meta.json", "w"))
    return False


def main(workers=6):
    by_id, fname = load_annotations()
    by_file = collections.defaultdict(list)
    for iid, f in fname.items():
        by_file[Path(f).stem].append(by_id[iid])
    jobs = sorted(by_file.items())
    counts = collections.Counter(len(v) for _, v in jobs)
    OUT.mkdir(exist_ok=True)
    print(f"{len(jobs)} photographs; annotators per photograph: {dict(sorted(counts.items()))}")

    t0 = time.time()
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(build, jobs, 4), 1):
            if k % 100 == 0:
                el = time.time() - t0
                print(f"  {k}/{len(jobs)}  {el:.0f}s elapsed, "
                      f"{el/k*(len(jobs)-k):.0f}s left", flush=True)
    size = sum(f.stat().st_size for f in OUT.rglob("*.png")) / 1e6
    print(f"done in {(time.time()-t0)/60:.1f} min, {size:.0f} MB")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
