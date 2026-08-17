"""Dataset and train/val split for MAGFiLO.

Two things here are competition-specific and easy to get wrong:

1. `image_id` in the COCO file is `<annotator_batch>-<filename>`, so the same
   photograph appears up to three times with three different, disagreeing
   annotations. The split is therefore grouped by *filename*: every annotator's
   version of an image lands on the same side of the split. Splitting by
   image_id would put the same photograph in both train and validation and
   inflate the validation score.

2. Rotation about the disk centre is a physically valid augmentation here: the
   Sun has no preferred orientation in the image plane, and every image is
   already registered to centre (1024, 1024) with radius ~904 px, so a rotation
   maps solar surface onto solar surface. With only 707 photographs this is the
   cheapest real gain available.
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
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "MAGFiLO_1.0_Kaggle_2026"
PRE = ROOT / "preprocessed"
ANN = BASE / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

FULL = 2048
DISK_CX, DISK_CY = 1024.0, 1024.0


def load_annotations():
    d = json.load(open(ANN))
    by_id = collections.defaultdict(list)
    for a in d["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in d["images"]}
    return by_id, fname


def make_split(fname, val_frac=0.2, seed=0):
    """Group by photograph, not by annotation set."""
    files = sorted({v for v in fname.values()})
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_frac))
    val_files = set(files[:n_val])
    train_ids = [k for k, v in fname.items() if v not in val_files]
    val_ids = [k for k, v in fname.items() if v in val_files]
    return sorted(train_ids), sorted(val_ids), val_files


def _rotate_point(x, y, ang, cx=DISK_CX, cy=DISK_CY):
    ca, sa = math.cos(ang), math.sin(ang)
    dx, dy = x - cx, y - cy
    return cx + dx * ca - dy * sa, cy + dx * sa + dy * ca


class FilamentDataset(Dataset):
    """Yields (image tensor [1,S,S] in [0,1], target dict) for torchvision detectors."""

    def __init__(self, image_ids, by_id, fname, size=1024, augment=False,
                 min_area_px=40, split="train"):
        self.ids = list(image_ids)
        self.by_id = by_id
        self.fname = fname
        self.size = size
        self.augment = augment
        self.min_area_px = min_area_px
        self.dir = PRE / split
        self.scale = size / FULL

    def __len__(self):
        return len(self.ids)

    def _polys(self, iid):
        return [a["segmentation"][0] for a in self.by_id[iid]]

    def __getitem__(self, i):
        iid = self.ids[i]
        img = Image.open(self.dir / (Path(self.fname[iid]).stem + ".png"))
        polys = self._polys(iid)

        ang = 0.0
        flip_h = flip_v = False
        if self.augment:
            ang = random.uniform(0, 2 * math.pi)
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            img = img.rotate(math.degrees(ang), resample=Image.BILINEAR,
                             center=(DISK_CX, DISK_CY), fillcolor=0)
            if flip_h:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_v:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None]           # [1, S, S]

        masks, boxes = [], []
        for poly in polys:
            pts = list(zip(poly[0::2], poly[1::2]))
            if self.augment:
                if ang:
                    pts = [_rotate_point(x, y, ang) for x, y in pts]
                if flip_h:
                    pts = [(FULL - 1 - x, y) for x, y in pts]
                if flip_v:
                    pts = [(x, FULL - 1 - y) for x, y in pts]
            pts = [(x * self.scale, y * self.scale) for x, y in pts]
            if len(pts) < 3:
                continue
            m = Image.new("L", (self.size, self.size), 0)
            ImageDraw.Draw(m).polygon(pts, fill=1)
            m = np.asarray(m, dtype=np.uint8)
            if m.sum() * (FULL / self.size) ** 2 < self.min_area_px:
                continue
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            masks.append(m)
            boxes.append([x0, y0, x1, y1])

        if masks:
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            masks_t = torch.zeros((0, self.size, self.size), dtype=torch.uint8)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": torch.ones((len(masks),), dtype=torch.int64),
            "masks": masks_t,
            "image_id": torch.tensor([i]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            if len(masks) else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))
