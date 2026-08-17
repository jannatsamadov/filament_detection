"""Run the spine model over the test set and write the competition CSV.

Format, from the competition page: two columns, filament_id and segmentation_rle,
RLE *counts* only (the size is fixed at 2048x2048 and must not be written), no
quotation marks. Verify the result with src/verify_submission.py before using an
upload -- it decodes every mask the way the organisers' scorer will.

Instances are already non-overlapping by construction here (each is a swept curve
built from its own spine component), but overlaps are resolved anyway: Panoptic
Quality assumes one segment per pixel, and a contested pixel would otherwise
inflate both unions.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as maskutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))          # shared: gpu_lock.py, gpu_guard.py
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL, OFFSET, SCALE   # noqa: E402
from spine_model import build_spine_model                              # noqa: E402
from spine_infer import assemble                                       # noqa: E402
from spine_train import predict_frame                                  # noqa: E402
from gpu_guard import Governor                                          # noqa: E402

TEST = ROOT / "preprocessed" / "test"


def resolve_overlaps(masks):
    if not masks:
        return []
    claimed = np.zeros((FULL, FULL), bool)
    out = []
    for m in sorted(masks, key=lambda a: -a.sum()):
        m = m & ~claimed
        if m.sum() == 0:
            continue
        claimed |= m
        out.append(m)
    return out


def encode_rle(mask):
    r = maskutil.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_unet_r34_best.pt"))
    ap.add_argument("--out", default=str(ROOT / "submission.csv"))
    ap.add_argument("--spine-thresh", type=float, default=None,
                    help="default: the best value recorded by spine_report.py")
    ap.add_argument("--sem-thresh", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=120)
    ap.add_argument("--min-spine-px", type=int, default=None,
                    help="default: the tuned value from spine_calibrate.py")
    ap.add_argument("--result-json", default=str(ROOT / "result_spine_unet_r34.json"))
    args = ap.parse_args()

    spine_thresh = args.spine_thresh
    if spine_thresh is None:
        p = Path(args.result_json)
        if p.exists():
            j = json.load(open(p))
            spine_thresh = j.get("best_spine_thresh", 0.4)
            print(f"spine threshold {spine_thresh} taken from {p.name}")
        else:
            spine_thresh = 0.4
            print("spine threshold 0.4 (no tuned value found)")

    min_spine_px = args.min_spine_px
    if min_spine_px is None:
        p = Path(args.result_json.replace("result_", "calibration_"))
        min_spine_px = json.load(open(p)).get("best_min_spine_px", 12) if p.exists() else 12
    print(f"min_spine_px {min_spine_px}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}, val PQ {ck.get('PQ', float('nan')):.4f})")

    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    files = sorted(TEST.glob("*.png"))
    print(f"{len(files)} test images")
    gov = Governor(target_util=55, check_every=2)

    rows, per_image = [], []
    for k, f in enumerate(files, 1):
        img = np.asarray(Image.open(f), dtype=np.float32) / SCALE + OFFSET
        with gov.step():
            sem, spn, rad = predict_frame(model, img, device)
        sem *= ondisk; spn *= ondisk
        masks = assemble(spn, rad, sem, spine_thresh=spine_thresh,
                         sem_thresh=args.sem_thresh, min_area=args.min_area,
                         min_spine_px=min_spine_px)
        masks = resolve_overlaps(masks)
        masks = [m for m in masks if m.sum() >= args.min_area]
        for j, m in enumerate(masks, start=1):
            rows.append((f"{f.stem}_{j}", encode_rle(m)))
        per_image.append(len(masks))
        if k % 25 == 0:
            print(f"  {k}/{len(files)}  mean {np.mean(per_image):.1f} filaments/image",
                  flush=True)

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        fh.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            fh.write(f"{fid},{rle}\n")

    pi = np.array(per_image)
    print(f"\nwrote {out}  ({len(rows)} rows over {len(files)} images)")
    print(f"filaments per image: mean {pi.mean():.2f} median {np.median(pi):.0f} "
          f"min {pi.min()} max {pi.max()}   (training ground truth: mean 7.1, median 7)")
    print(f"images with no prediction: {(pi == 0).sum()}")
    print(f"governor: {gov.summary()}")


if __name__ == "__main__":
    main()
