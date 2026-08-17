"""Verify a submission CSV the way the platform will, before spending an upload.

Checks, in order:
  1. header and column names exactly as specified
  2. no quotation marks anywhere in the RLE field
  3. every filament_id unique, and its image part is a real test image
  4. every RLE decodes with pycocotools at size 2048x2048
  5. decoded masks are non-empty and lie inside the solar disk
  6. masks belonging to one image do not overlap
  7. round-trip: decode -> re-encode reproduces the same string
  8. per-image counts are sane against the training distribution

Run from a directory outside the project too -- it takes only the CSV path and
the test image folder, exactly like the organisers' scorer would.
"""
from __future__ import annotations

import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskutil

FULL = 2048
EXPECTED_HEADER = ["filament_id", "segmentation_rle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--test-dir", default=None,
                    help="folder of test JPEGs, to confirm every id is a real image")
    ap.add_argument("--disk-radius", type=float, default=904.0)
    ap.add_argument("--strict-disk", action="store_true",
                    help="fail (not warn) if any mask pixel lies off the solar disk")
    args = ap.parse_args()

    errors, warnings = [], []
    path = Path(args.csv_path)
    if not path.exists():
        print(f"FAIL: {path} does not exist"); sys.exit(1)

    raw = path.read_text()
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        print("FAIL: empty file"); sys.exit(1)

    # 1. header
    if rows[0] != EXPECTED_HEADER:
        errors.append(f"header is {rows[0]}, expected {EXPECTED_HEADER}")
    body = rows[1:]
    print(f"rows: {len(body)}")

    # 2. quotes
    if '"' in raw or "'" in raw:
        errors.append("file contains quotation marks; the spec forbids them")

    # 3. ids
    ids = [r[0] for r in body if len(r) >= 1]
    if len(set(ids)) != len(ids):
        dup = [i for i in set(ids) if ids.count(i) > 1][:5]
        errors.append(f"duplicate filament_id, e.g. {dup}")

    known = None
    if args.test_dir:
        known = {p.stem for p in Path(args.test_dir).iterdir()
                 if p.suffix.lower() in (".jpeg", ".jpg", ".png")}
        print(f"test images on disk: {len(known)}")

    by_image = defaultdict(list)
    bad_rows = 0
    for r in body:
        if len(r) != 2:
            bad_rows += 1
            continue
        fid, rle = r
        if "_" not in fid:
            errors.append(f"filament_id has no _<n> suffix: {fid}"); continue
        stem = fid.rsplit("_", 1)[0]
        by_image[stem].append((fid, rle))
    if bad_rows:
        errors.append(f"{bad_rows} rows do not have exactly 2 fields")

    if known is not None:
        unknown = set(by_image) - known
        missing = known - set(by_image)
        if unknown:
            errors.append(f"{len(unknown)} ids refer to images not in the test set, "
                          f"e.g. {sorted(unknown)[:3]}")
        if missing:
            warnings.append(f"{len(missing)} test images have no prediction at all, "
                            f"e.g. {sorted(missing)[:3]}")

    # 4-7. decode every mask
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    rr = np.sqrt((xx - 1024.0) ** 2 + (yy - 1024.0) ** 2) / args.disk_radius
    offdisk = rr > 1.0

    n_decoded = empty = roundtrip_fail = 0
    offdisk_masks = 0
    areas = []
    overlap_images = []
    for stem, entries in by_image.items():
        canvas = np.zeros((FULL, FULL), np.int32)
        for fid, rle in entries:
            obj = {"size": [FULL, FULL], "counts": rle.encode("ascii")}
            try:
                m = maskutil.decode(obj).astype(bool)
            except Exception as e:
                errors.append(f"{fid}: RLE will not decode ({e})")
                continue
            n_decoded += 1
            a = int(m.sum())
            areas.append(a)
            if a == 0:
                empty += 1
                continue
            if (m & offdisk).any():
                offdisk_masks += 1
            again = maskutil.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")
            if again != rle:
                roundtrip_fail += 1
            canvas += m.astype(np.int32)
        if (canvas > 1).any():
            overlap_images.append((stem, int((canvas > 1).sum())))

    print(f"decoded masks: {n_decoded}")
    if empty:
        errors.append(f"{empty} masks decode to zero pixels")
    if roundtrip_fail:
        errors.append(f"{roundtrip_fail} masks do not survive a decode/encode round trip")
    if overlap_images:
        errors.append(f"{len(overlap_images)} images have overlapping predicted masks, "
                      f"worst {max(overlap_images, key=lambda t: t[1])}")
    if offdisk_masks:
        msg = f"{offdisk_masks} masks contain pixels outside the solar disk (r>1.0)"
        (errors if args.strict_disk else warnings).append(msg)

    if areas:
        a = np.array(areas)
        print(f"mask area px: min {a.min()} median {np.median(a):.0f} "
              f"max {a.max()}   (train ground truth: median 1228)")
        counts = np.array([len(v) for v in by_image.values()])
        print(f"filaments per image: mean {counts.mean():.2f} median "
              f"{np.median(counts):.0f} max {counts.max()}   (train: mean 7.1, median 7)")
        if counts.mean() > 20:
            warnings.append(f"mean {counts.mean():.1f} filaments/image is far above the "
                            f"training mean of 7.1 -- likely flooding false positives")
        if a.min() < 50:
            warnings.append(f"smallest mask is {a.min()} px; the host warned that tiny "
                            f"detections are usually noise or sunspots")

    print()
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print(f"\nNOT SAFE TO SUBMIT: {len(errors)} error(s)")
        sys.exit(1)
    print(f"\nOK to submit ({len(warnings)} warning(s))")


if __name__ == "__main__":
    main()
