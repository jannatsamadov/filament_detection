"""Run a checkpoint over the test set and write the competition CSV.

Submission format (from the competition page):
  filament_id,segmentation_rle
  20150125172714Mh_1,<RLE counts>
Only RLE *counts* are stored; the size is fixed at 2048x2048 and must not be
written. No quotation marks.

Predictions are forced not to overlap: Panoptic Quality assumes each pixel
belongs to at most one segment, so where two predicted masks collide the pixel
goes to the higher-scoring instance. Leaving overlaps in would inflate every
union and quietly depress IoU.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRE = ROOT / "preprocessed" / "test"
sys.path.insert(0, str(HERE))
from model import build_model                      # noqa: E402
from evaluate import predict_masks                 # noqa: E402
from thermal import Governor                       # noqa: E402

FULL = 2048


def resolve_overlaps(masks, scores):
    """Highest score wins each contested pixel; drop anything left empty."""
    if not masks:
        return [], []
    order = np.argsort(-np.asarray(scores))
    claimed = np.zeros((FULL, FULL), bool)
    out_m, out_s = [], []
    for i in order:
        m = masks[i] & ~claimed
        if m.sum() == 0:
            continue
        claimed |= m
        out_m.append(m); out_s.append(scores[i])
    return out_m, out_s


def encode_rle(mask):
    r = maskutil.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--mask-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=150,
                    help="drop predictions smaller than this many pixels at 2048")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(size=args.size).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']})")

    files = sorted(PRE.glob("*.png"))
    print(f"{len(files)} test images")
    gov = Governor(target_util=70, check_every=10, verbose=True)

    rows, per_image = [], []
    for k, f in enumerate(files, 1):
        arr = np.asarray(Image.open(f), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr)[None]
        img = torch.nn.functional.interpolate(
            img[None], size=(args.size, args.size), mode="bilinear",
            align_corners=False)[0]
        with gov.step():
            masks, scores = predict_masks(model, img, device, args.score_thresh,
                                          args.mask_thresh, min_area=args.min_area)
        masks, scores = resolve_overlaps(masks, scores)
        masks = [m for m in masks if m.sum() >= args.min_area]
        stem = f.stem
        for j, m in enumerate(masks, start=1):
            rows.append((f"{stem}_{j}", encode_rle(m)))
        per_image.append(len(masks))
        if k % 30 == 0:
            print(f"  {k}/{len(files)}  mean {np.mean(per_image):.1f} filaments/image",
                  flush=True)

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        fh.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            fh.write(f"{fid},{rle}\n")

    pi = np.array(per_image)
    print(f"\nwrote {out}  ({len(rows)} rows, {len(files)} images)")
    print(f"filaments per image: mean {pi.mean():.2f} median {np.median(pi):.0f} "
          f"min {pi.min()} max {pi.max()}   (ground truth in train: mean 7.1, median 7)")
    print(f"images with zero predictions: {(pi == 0).sum()}")
    print(f"governor: {gov.summary()}")


if __name__ == "__main__":
    main()
