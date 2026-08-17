"""Patch dataset with spine-first targets, at the native 2048 resolution.

Three targets per pixel:

  semantic  filament or not
  spine     a Gaussian ridge along the annotated centreline
  radius    distance to the mask boundary, i.e. the local half-width

At inference the three combine into instances without anchors or NMS: threshold
the spine map, take connected components (measured: only 0.9% of spines come
within 10 px of another filament's spine, against 2.8% for the masks), fit a
smoothing spline to each, and sweep a disk of the predicted radius along it.
Sweeping the *annotated* spine and half-width reproduces the annotated mask at
median IoU 0.758 -- above the 0.595 that two human annotators reach with each
other -- so the representation is not what limits the score.

Resolution is deliberately native. Filaments are a median 8.2 px wide at 2048;
training the earlier detector at 1024 squeezed them to 4 px and erased the barbs.

Patches are sampled on filaments most of the time because they cover only about
0.4% of the disk, so uniform sampling would return almost nothing to learn from.
"""
from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026"
PRE = ROOT / "preprocessed"
ANN = BASE / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

TARGETS = ROOT / "targets"
FULL = 2048
DISK_CX, DISK_CY, DISK_R = 1024.0, 1024.0, 904.0
OFFSET, SCALE = 0.5, 255.0     # matches src/preprocess.py encoding

_DISK = None


def _disk_mask():
    """On-disk pixels. Cached module-wide: the geometry is identical in every
    frame (centre 1024,1024 to within half a pixel, radius 904 +- 0.8)."""
    global _DISK
    if _DISK is None:
        yy, xx = np.mgrid[0:FULL, 0:FULL]
        _DISK = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    return _DISK


def load_annotations():
    d = json.load(open(ANN))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}
    return by_id, fname


def make_split(fname, val_frac=0.2, seed=0):
    """Grouped by photograph: every annotator's version stays on one side."""
    files = sorted({v for v in fname.values()})
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_frac))
    val_files = set(files[:n_val])
    train_ids = [k for k, v in fname.items() if v not in val_files]
    val_ids = [k for k, v in fname.items() if v in val_files]
    return sorted(train_ids), sorted(val_ids), val_files


def rasterize(anns, size=FULL, spine_width=3):
    """Full-frame masks, spine polylines and per-pixel radius for one annotation set."""
    mask_img = Image.new("L", (size, size), 0)
    spine_img = Image.new("L", (size, size), 0)
    md, sd = ImageDraw.Draw(mask_img), ImageDraw.Draw(spine_img)
    for a in anns:
        p = a["segmentation"][0]
        md.polygon(list(zip(p[0::2], p[1::2])), fill=1)
        s = a["spine"]
        if len(s) >= 4:
            sd.line(list(zip(s[0::2], s[1::2])), fill=1, width=spine_width)
    mask = np.asarray(mask_img, bool)
    spine = np.asarray(spine_img, bool)
    radius = distance_transform_edt(mask).astype(np.float32)
    return mask, spine, radius


def spine_heatmap(spine, sigma=3.0):
    """exp(-d^2 / 2 sigma^2) with d the distance to the nearest spine pixel."""
    if not spine.any():
        return np.zeros(spine.shape, np.float32)
    d = distance_transform_edt(~spine)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


class SpinePatchDataset(Dataset):
    """Random patches with (image, semantic, spine, radius, valid) targets.

    `valid` marks on-disk pixels; everything off the solar limb is excluded from
    every loss, since off-disk material has no flat field and no annotations.
    """

    def __init__(self, image_ids, by_id, fname, patch=512, augment=True,
                 on_filament_prob=0.75, sigma=3.0, split="train",
                 patches_per_image=1, cache_size=24):
        self.ids = list(image_ids)
        self.by_id = by_id
        self.fname = fname
        self.patch = patch
        self.augment = augment
        self.on_filament_prob = on_filament_prob
        self.sigma = sigma
        self.dir = PRE / split
        self.patches_per_image = patches_per_image
        self._cache = collections.OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.ids) * self.patches_per_image

    # -- full-frame arrays ---------------------------------------------------
    def _frame(self, iid):
        """Read the precomputed targets. Building them here instead cost 6.94 s
        per training step against 0.43 s of GPU work, because two 2048x2048
        distance transforms per annotation set dominate everything else."""
        hit = self._cache.get(iid)
        if hit is not None:
            self._cache.move_to_end(iid)
            return hit

        img = np.asarray(Image.open(self.dir / (Path(self.fname[iid]).stem + ".png")),
                         dtype=np.float32) / SCALE + OFFSET
        tdir = TARGETS / iid
        if (tdir / "radius.png").exists():
            mask = np.asarray(Image.open(tdir / "mask.png")) > 127
            heat = np.asarray(Image.open(tdir / "spine.png"), dtype=np.float32) / 255.0
            radius = np.asarray(Image.open(tdir / "radius.png"), dtype=np.float32)
        else:                                    # fall back if not precomputed
            mask, spine, radius = rasterize(self.by_id[iid])
            heat = spine_heatmap(spine, self.sigma)

        item = (img.astype(np.float32), mask, heat, radius, _disk_mask())
        self._cache[iid] = item
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return item

    def _pick_origin(self, mask, valid):
        p = self.patch
        if random.random() < self.on_filament_prob and mask.any():
            ys, xs = np.nonzero(mask)
            k = random.randrange(len(xs))
            cx, cy = int(xs[k]), int(ys[k])
        else:
            ys, xs = np.nonzero(valid[::16, ::16])
            k = random.randrange(len(xs))
            cx, cy = int(xs[k]) * 16, int(ys[k]) * 16
        jitter = p // 4
        cx += random.randint(-jitter, jitter)
        cy += random.randint(-jitter, jitter)
        x0 = int(np.clip(cx - p // 2, 0, FULL - p))
        y0 = int(np.clip(cy - p // 2, 0, FULL - p))
        return x0, y0

    def __getitem__(self, index):
        iid = self.ids[index % len(self.ids)]
        img, mask, heat, radius, valid = self._frame(iid)
        p = self.patch
        x0, y0 = self._pick_origin(mask, valid)
        sl = (slice(y0, y0 + p), slice(x0, x0 + p))
        im = img[sl].copy()
        ma = mask[sl].astype(np.float32)
        he = heat[sl].copy()
        ra = radius[sl].copy()
        va = valid[sl].astype(np.float32)

        if self.augment:
            k = random.randrange(4)             # rotation by multiples of 90
            if k:
                im, ma, he, ra, va = (np.rot90(a, k).copy() for a in (im, ma, he, ra, va))
            if random.random() < 0.5:           # mirror
                im, ma, he, ra, va = (np.ascontiguousarray(a[:, ::-1])
                                      for a in (im, ma, he, ra, va))
            if random.random() < 0.3:           # mild photometric jitter
                im = im * random.uniform(0.97, 1.03) + random.uniform(-0.02, 0.02)

        return {
            "image": torch.from_numpy(im)[None],
            "mask": torch.from_numpy(ma)[None],
            "spine": torch.from_numpy(he)[None],
            "radius": torch.from_numpy(ra)[None],
            "valid": torch.from_numpy(va)[None],
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
CONSENSUS = ROOT / "consensus"
PHYSICS = ROOT / "physics"
PHYS_CHANNELS = ("coherence", "sin2t", "cos2t", "sinlat", "plagedist")


class ConsensusPatchDataset(Dataset):
    """One sample per photograph, targets graded by how many annotators agreed.

    Two changes from SpinePatchDataset, both answering measurements:

    * **Targets are vote fractions, not binary.** Treating each annotator's set
      as its own sample showed a disputed filament to the model once as
      foreground and twice as background; it sided with the majority and now
      recovers 71% of filaments two experts agree on but only 29% of those a
      single expert marked. A target of 0.33 says "one of three saw this"
      instead of contradicting itself, and a threshold on the output then means
      how much expert agreement to require.

    * **Physical input channels.** Fibril coherence and orientation (relative to
      the local heliographic east-west), latitude, and distance to plage. The
      image-only model learns "dark and elongated", and 84% of its false
      positives are dark structures no annotator marked.

    Augmentation is restricted to transforms that preserve solar geometry --
    the two mirrors and a half turn. Once latitude is an input channel the
    problem is no longer rotationally symmetric: a quarter-turned Sun would have
    latitude running across the image, which is not a Sun at all. The channels
    that change sign under a mirror are corrected analytically.
    """

    def __init__(self, files, patch=512, augment=True, on_filament_prob=0.75,
                 split="train", patches_per_image=4, cache_size=6,
                 use_physics=True):
        self.files = list(files)
        self.patch = patch
        self.augment = augment
        self.on_filament_prob = on_filament_prob
        self.dir = PRE / split
        self.split = split
        self.patches_per_image = patches_per_image
        self.use_physics = use_physics
        self._cache = collections.OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.files) * self.patches_per_image

    @property
    def in_channels(self):
        return 1 + (len(PHYS_CHANNELS) if self.use_physics else 0)

    def _frame(self, stem):
        hit = self._cache.get(stem)
        if hit is not None:
            self._cache.move_to_end(stem)
            return hit

        img = np.asarray(Image.open(self.dir / f"{stem}.png"),
                         dtype=np.float32) / SCALE + OFFSET
        planes = [img]
        if self.use_physics:
            for k in PHYS_CHANNELS:
                planes.append(np.asarray(Image.open(PHYSICS / self.split / stem / f"{k}.png"),
                                         dtype=np.float32) / 255.0)
        x = np.stack(planes)

        d = CONSENSUS / stem
        mask = np.asarray(Image.open(d / "mask_vote.png"), dtype=np.float32) / 255.0
        spine = np.asarray(Image.open(d / "spine_vote.png"), dtype=np.float32) / 255.0
        radius = np.asarray(Image.open(d / "radius.png"), dtype=np.float32)

        item = (x, mask, spine, radius, _disk_mask())
        self._cache[stem] = item
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return item

    def _pick_origin(self, mask, valid):
        p = self.patch
        if random.random() < self.on_filament_prob and (mask > 0.2).any():
            ys, xs = np.nonzero(mask > 0.2)
            k = random.randrange(len(xs))
            cx, cy = int(xs[k]), int(ys[k])
        else:
            ys, xs = np.nonzero(valid[::16, ::16])
            k = random.randrange(len(xs))
            cx, cy = int(xs[k]) * 16, int(ys[k]) * 16
        j = p // 4
        cx += random.randint(-j, j); cy += random.randint(-j, j)
        return (int(np.clip(cx - p // 2, 0, FULL - p)),
                int(np.clip(cy - p // 2, 0, FULL - p)))

    def __getitem__(self, index):
        stem = self.files[index % len(self.files)]
        x, mask, spine, radius, valid = self._frame(stem)
        p = self.patch
        x0, y0 = self._pick_origin(mask, valid)
        sl = (slice(y0, y0 + p), slice(x0, x0 + p))

        xi = x[:, sl[0], sl[1]].copy()
        ma = mask[sl].copy()
        sp = spine[sl].copy()
        ra = radius[sl].copy()
        va = valid[sl].astype(np.float32)

        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h:
                xi = xi[:, :, ::-1]; ma = ma[:, ::-1]; sp = sp[:, ::-1]
                ra = ra[:, ::-1]; va = va[:, ::-1]
            if flip_v:
                xi = xi[:, ::-1]; ma = ma[::-1]; sp = sp[::-1]
                ra = ra[::-1]; va = va[::-1]
            xi = np.ascontiguousarray(xi); ma = np.ascontiguousarray(ma)
            sp = np.ascontiguousarray(sp); ra = np.ascontiguousarray(ra)
            va = np.ascontiguousarray(va)

            if self.use_physics:
                i_sin2t = 1 + PHYS_CHANNELS.index("sin2t")
                i_sinlat = 1 + PHYS_CHANNELS.index("sinlat")
                # both channels are stored as 0.5*(value+1); a sign flip is 1-v.
                # A mirror reverses the sense of the ridge-to-east-west angle, so
                # sin2t changes sign under either mirror but not under both.
                if flip_h != flip_v:
                    xi[i_sin2t] = 1.0 - xi[i_sin2t]
                # latitude changes sign only under the north-south mirror
                if flip_v:
                    xi[i_sinlat] = 1.0 - xi[i_sinlat]

            if random.random() < 0.3:
                xi[0] = xi[0] * random.uniform(0.97, 1.03) + random.uniform(-0.02, 0.02)

        return {
            "image": torch.from_numpy(xi),
            "mask": torch.from_numpy(ma)[None],
            "spine": torch.from_numpy(sp)[None],
            "radius": torch.from_numpy(ra)[None],
            "valid": torch.from_numpy(va)[None],
        }


def photograph_split(val_frac=0.2, seed=0):
    """Train/val split over photographs, matching make_split's val set exactly."""
    _, fname = load_annotations()
    files = sorted({Path(v).stem for v in fname.values()})
    rng = random.Random(seed)
    order = sorted({v for v in fname.values()})
    rng.shuffle(order)
    n_val = int(round(len(order) * val_frac))
    val_files = {Path(f).stem for f in order[:n_val]}
    return sorted(set(files) - val_files), sorted(val_files)
