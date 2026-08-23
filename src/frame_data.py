"""Full photographs, with targets that can change between epochs.

Two departures from the patch dataset.

**The whole frame, not a crop.** Training on 512 px windows taught the spine
head to end filaments at the window border, and at inference there is no border
-- the model sees all 2048 px and the only real edge is the limb. A full frame
at batch one peaks at 4749 MiB, which the card has.

**Targets assembled from a label store, not read as fixed images.** Each
photograph carries an instance map and a per-filament count of what speaks for
and against it. The mask target is the filament's current score painted over its
pixels, so when the counts move between epochs the targets move with them. The
scores live in one json the trainer rewrites each epoch; workers are respawned
per epoch and read it back.

Channels are split by how they vary. Seven vary per pixel and travel as bytes:
the flattened image, the raw image, and the five physical channels. Eleven are
one number for the whole photograph -- which telescope, where the solar axis
points, the time of year -- and travel as scalars, expanded to planes on the GPU
where it costs nothing.

Both images are given. Pretraining ran on flattened frames, so dropping them
would waste it; the raw frame is there because flattening visibly damages parts
of the disk. The model can use either.
"""
from __future__ import annotations

import collections, json, math, random, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from spine_data import (FULL, OFFSET, SCALE, PHYS_CHANNELS, PRE, PHYSICS,  # noqa: E402
                        _disk_mask)
from solar_ephemeris import solar_angles                                  # noqa: E402

RAW = ROOT / "MAGFiLO_1.0_Kaggle_2026"
LABELS = ROOT / "labels_v5"
STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
PLANES = ("flat", "raw") + PHYS_CHANNELS
N_PLANES = len(PLANES)
N_SCALARS = len(STATIONS) + 5          # station, sinP, cosP, B0, doy sin/cos
IN_CHANNELS = N_PLANES + N_SCALARS
SIGMA = 3.0
N_CHIRALITY = 4                        # Left, Right, Unidentifiable, Ambiguous

I_SIN2T = PLANES.index("sin2t")
I_SINLAT = PLANES.index("sinlat")
I_SINP = len(STATIONS)
I_B0 = len(STATIONS) + 2


def scalars_for(stem):
    """Everything about a photograph that is one number, not a picture."""
    v = np.zeros(N_SCALARS, np.float32)
    st = stem[14:16]
    if st in STATIONS:
        v[STATIONS.index(st)] = 1.0
    t = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    P, B0, _ = solar_angles(t)
    v[I_SINP] = math.sin(math.radians(P))
    v[I_SINP + 1] = math.cos(math.radians(P))
    v[I_B0] = B0 / 10.0                      # +-7.2 degrees, kept near unit scale
    doy = t.timetuple().tm_yday / 365.25
    v[I_B0 + 1] = math.sin(2 * math.pi * doy)
    v[I_B0 + 2] = math.cos(2 * math.pi * doy)
    return v


def load_weights(path=None):
    p = Path(path) if path else LABELS / "weights.json"
    if p.exists():
        return json.load(open(p))
    # first epoch: take the opening counts from the label store
    w = {}
    for d in sorted(LABELS.iterdir()):
        if not d.is_dir():
            continue
        meta = json.load(open(d / "meta.json"))
        w[d.name] = {str(f["id"]): [f["a"], f["b"], bool(f["frozen"])]
                     for f in meta["filaments"]}
    return w


class FullFrameDataset(Dataset):
    def __init__(self, stems, split="train", augment=True, weights=None,
                 weights_path=None, cache_size=2, threshold=0.75, soft=True):
        self.stems = list(stems)
        self.split = split
        self.augment = augment
        self.threshold = threshold
        self.soft = soft
        self.weights = weights if weights is not None else load_weights(weights_path)
        self._cache = collections.OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.stems)

    @property
    def in_channels(self):
        return IN_CHANNELS

    def _planes(self, stem):
        hit = self._cache.get(stem)
        if hit is not None:
            self._cache.move_to_end(stem)
            return hit
        flat = np.asarray(Image.open(PRE / self.split / f"{stem}.png"), np.uint8)
        raw = Image.open(RAW / self.split / f"{self.split}_images" / f"{stem}.jpeg")
        raw = np.asarray(raw.convert("L"), np.uint8)
        planes = [flat, raw]
        for k in PHYS_CHANNELS:
            planes.append(np.asarray(
                Image.open(PHYSICS / self.split / stem / f"{k}.png"), np.uint8))
        x = np.stack(planes)
        self._cache[stem] = x
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return x

    def _targets(self, stem):
        d = LABELS / stem
        inst = np.asarray(Image.open(d / "inst.png"), np.uint16)
        spid = np.asarray(Image.open(d / "spine_id.png"), np.uint16)
        radius = np.asarray(Image.open(d / "radius.png"), np.float32)
        meta = json.load(open(d / "meta.json"))
        w = self.weights.get(stem, {})

        n = int(inst.max())
        score = np.zeros(n + 1, np.float32)
        chir = np.zeros(n + 1, np.uint8)
        for f in meta["filaments"]:
            i = f["id"]
            if i > n:
                continue
            a, b = w.get(str(i), [f["a"], f["b"], f["frozen"]])[:2]
            s = a / max(a + b, 1e-6)
            # A target of 0.4 tells the model this is probably background; the
            # threshold is what decides, so the target is the decision unless
            # soft targets are asked for.
            score[i] = s if self.soft else float(s >= self.threshold)
            chir[i] = max(0, min(N_CHIRALITY - 1, f["chirality"] - 1))

        mask = score[inst]
        spine = gaussian_filter(score[spid], SIGMA)
        if spine.max() > 0:
            spine = spine / max(float(gaussian_filter(
                np.ones((1, 1), np.float32), SIGMA).max()), 1e-6)
        np.clip(spine, 0.0, 1.0, out=spine)
        chirality = chir[inst].astype(np.int64)
        present = (inst > 0)
        return mask, spine, radius, chirality, present, inst.astype(np.int16)

    def __getitem__(self, i):
        stem = self.stems[i]
        x = self._planes(stem)
        mask, spine, radius, chirality, present, inst = self._targets(stem)
        sc = scalars_for(stem).copy()
        valid = _disk_mask()

        if self.augment:
            fh, fv = random.random() < 0.5, random.random() < 0.5
            if fh:
                x = x[:, :, ::-1]; mask = mask[:, ::-1]; spine = spine[:, ::-1]
                radius = radius[:, ::-1]; chirality = chirality[:, ::-1]
                present = present[:, ::-1]; valid = valid[:, ::-1]
                inst = inst[:, ::-1]
            if fv:
                x = x[:, ::-1]; mask = mask[::-1]; spine = spine[::-1]
                radius = radius[::-1]; chirality = chirality[::-1]
                present = present[::-1]; valid = valid[::-1]
                inst = inst[::-1]
            x = np.ascontiguousarray(x)
            mask = np.ascontiguousarray(mask); spine = np.ascontiguousarray(spine)
            radius = np.ascontiguousarray(radius)
            chirality = np.ascontiguousarray(chirality)
            present = np.ascontiguousarray(present)
            valid = np.ascontiguousarray(valid); inst = np.ascontiguousarray(inst)
            # channels stored as 0.5*(v+1): a sign flip is 255-v in bytes
            if fh != fv:
                x[I_SIN2T] = 255 - x[I_SIN2T]
                sc[I_SINP] = -sc[I_SINP]        # a mirror negates the axis angle
            if fv:
                x[I_SINLAT] = 255 - x[I_SINLAT]
                sc[I_B0] = -sc[I_B0]            # north and south swap
            # chirality is handedness: a mirror turns left into right
            if fh != fv:
                left, right = chirality == 0, chirality == 1
                chirality[left], chirality[right] = 1, 0

        return {
            "planes": torch.from_numpy(x),                       # uint8 (7,H,W)
            "scalars": torch.from_numpy(sc),                     # float32 (11,)
            "mask": torch.from_numpy(mask.astype(np.float32))[None],
            "spine": torch.from_numpy(spine.astype(np.float32))[None],
            "radius": torch.from_numpy(radius.astype(np.float32))[None],
            "chirality": torch.from_numpy(chirality),            # int64 (H,W)
            "present": torch.from_numpy(present.astype(np.float32))[None],
            "inst": torch.from_numpy(inst.astype(np.int16))[None],
            "valid": torch.from_numpy(valid.astype(np.float32))[None],
            "stem": stem,
        }


def to_input(planes, scalars, device):
    """Bytes and scalars in, the model's float input out.

    The flattened plane is decoded the way preprocess.py wrote it; everything
    else is a straight 0-1 scaling. Scalars become constant planes here rather
    than in the dataset, where they would have cost 11 full-size arrays per
    sample of host memory and PCIe traffic for no information.
    """
    p = planes.to(device, non_blocking=True).float()
    x = torch.empty_like(p)
    x[:, 0] = p[:, 0] / SCALE + OFFSET
    x[:, 1] = p[:, 1] / 255.0
    x[:, 2:] = p[:, 2:] / 255.0
    s = scalars.to(device, non_blocking=True)
    b, _, h, w = x.shape
    planes_s = s[:, :, None, None].expand(b, s.shape[1], h, w)
    return torch.cat([x, planes_s], 1)
