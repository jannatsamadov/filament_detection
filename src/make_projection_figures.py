"""Draw the annotations, and draw where rotation says they should be next time.

Two folders come out of this.

figures/annotated/ is every annotated photograph with its filaments outlined,
coloured by how many of the annotators drew each one. Filenames start with the
timestamp, so photographs from neighbouring days sit next to each other and can
be flipped through.

figures/projection/ is the check that matters: for each pair of photographs close
in time, the later one is shown with its own annotations in green and the earlier
one's annotations -- carried forward by solar rotation -- in red. If the rotation
is right the red lands on the green. If it does not, the picture says so faster
than any table.

Images are written at half size; a 2048 frame is unreadable on screen anyway and
the folder would be gigabytes.
"""
from __future__ import annotations

import collections, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026" / "train"
PREP = ROOT / "preprocessed" / "train"
OUT_A = ROOT / "figures" / "annotated"
OUT_P = ROOT / "figures" / "projection"
sys.path.insert(0, str(HERE))
from solar_ephemeris import solar_angles                          # noqa: E402
from propagate_labels import obs_time, summarise, project         # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL, OFFSET, SCALE  # noqa: E402

HALF = FULL // 2
MAX_GAP = 3.5
VOTE_COLOUR = {1: (80, 140, 255), 2: (255, 200, 40), 3: (60, 230, 90)}


def base_image(stem):
    p = PREP / f"{stem}.png"
    if not p.exists():
        return None
    a = np.asarray(Image.open(p), dtype=np.float32)
    # stretch the middle of the range so filaments are visible on screen
    a = np.clip((a - 60) * (255.0 / 120.0), 0, 255).astype(np.uint8)
    im = Image.fromarray(a).convert("RGB").resize((HALF, HALF), Image.BILINEAR)
    return im


def outline(dr, anns, colour, width=2, scale=0.5):
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) < 6:
                continue
            pts = [(x * scale, y * scale) for x, y in zip(poly[0::2], poly[1::2])]
            dr.line(pts + [pts[0]], fill=colour, width=width)


def vote_of(a, others):
    """How many annotators drew this filament, by overlap of the drawn outlines."""
    return a.get("_vote", 1)


def tag_votes(per_annotator):
    """Mark each annotation with how many annotators drew something on top of it."""
    masks = []
    for anns in per_annotator:
        im = Image.new("L", (FULL, FULL), 0)
        dr = ImageDraw.Draw(im)
        for a in anns:
            for poly in a["segmentation"]:
                if len(poly) >= 6:
                    dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        masks.append(np.asarray(im, np.uint8))
    total = sum(masks) if masks else None
    for anns in per_annotator:
        for a in anns:
            im = Image.new("L", (FULL, FULL), 0)
            dr = ImageDraw.Draw(im)
            for poly in a["segmentation"]:
                if len(poly) >= 6:
                    dr.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
            m = np.asarray(im, bool)
            a["_vote"] = int(np.median(total[m])) if m.sum() else 1
    return total


def main(limit_pairs=60):
    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_P.mkdir(parents=True, exist_ok=True)
    d = json.load(open(BASE / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    stem = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    ann_of = collections.defaultdict(list)
    for a in d["annotations"]:
        ann_of[a["image_id"]].append(a)
    photo_ids = collections.defaultdict(list)
    for iid in ann_of:
        photo_ids[stem[iid]].append(iid)
    photos = sorted(photo_ids)
    times = {s: obs_time(s) for s in photos}

    print(f"{len(photos)} annotated photographs -> {OUT_A}")
    per_photo = {}
    for k, s in enumerate(photos, 1):
        groups = [ann_of[i] for i in sorted(photo_ids[s])]
        tag_votes(groups)
        flat = [a for g in groups for a in g]
        per_photo[s] = flat
        im = base_image(s)
        if im is None:
            continue
        dr = ImageDraw.Draw(im)
        for a in flat:
            outline(dr, [a], VOTE_COLOUR.get(a["_vote"], (255, 80, 80)))
        n = len(photo_ids[s])
        dr.text((12, 12), f"{s}   {n} annotator(s)   {len(flat)} outlines",
                fill=(255, 255, 255))
        dr.text((12, 26), "green 3/3   yellow 2/3   blue 1/3", fill=(200, 200, 200))
        im.save(OUT_A / f"{s[:8]}_{s[8:14]}_{s[14:]}_{n}ann.png")
        if k % 100 == 0:
            print(f"  {k}/{len(photos)}", flush=True)

    pairs = [(a, b) for a, b in combinations(photos, 2)
             if 0.02 < abs((times[b] - times[a]).total_seconds()) / 86400 <= MAX_GAP]
    # show the best-supported pairs first: both photographs with three annotators
    pairs.sort(key=lambda p: (-(len(photo_ids[p[0]]) + len(photo_ids[p[1]])),
                              abs((times[p[1]] - times[p[0]]).total_seconds())))
    pairs = pairs[:limit_pairs]
    print(f"\n{len(pairs)} pairs -> {OUT_P}")

    for k, (sa, sb) in enumerate(pairs, 1):
        if times[sb] < times[sa]:
            sa, sb = sb, sa
        gap = (times[sb] - times[sa]).total_seconds() / 86400.0
        im = base_image(sb)
        if im is None:
            continue
        dr = ImageDraw.Draw(im)
        for a in per_photo[sb]:
            outline(dr, [a], VOTE_COLOUR.get(a["_vote"], (60, 230, 90)))

        ang = solar_angles(times[sa])
        n_proj = 0
        for a in per_photo[sa]:
            s = summarise(a, ang)
            if s is None:
                continue
            p = project(s, times[sa], times[sb])
            if p is None:
                continue
            pts = [(x * 0.5, y * 0.5) for x, y in zip(p["px"], p["py"])]
            if len(pts) >= 2:
                dr.line(pts, fill=(255, 60, 60), width=3)
                n_proj += 1
        dr.text((12, 12), f"{sb}  <-  {sa}", fill=(255, 255, 255))
        dr.text((12, 26), f"gap {gap:.2f} d   red = {n_proj} filaments carried "
                          f"forward by rotation", fill=(255, 140, 140))
        dr.text((12, 40), "green/yellow/blue = what the annotators drew in this frame",
                fill=(160, 220, 160))
        im.save(OUT_P / f"gap{gap:04.1f}d_{sa}_to_{sb}.png")
        if k % 20 == 0:
            print(f"  {k}/{len(pairs)}", flush=True)

    print(f"\nannotated: {len(list(OUT_A.glob('*.png')))} files")
    print(f"projection: {len(list(OUT_P.glob('*.png')))} files")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
