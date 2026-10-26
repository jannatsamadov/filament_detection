"""Full-frame U-Net with heavy augmentation, trained on Kaggle's GPUs.

Two trainings run side by side, one per T4, identical in everything but the
augmentation:

  flip   horizontal and vertical mirrors only -- the control
  full   any-angle rotation, mirrors, contrast down and up, gamma, a little
         brightness, blur or sharpening, sensor noise, JPEG re-compression

The difference in validation PQ between the two is what the augmentation is
worth; neither number means much on its own.

Ranges come from the frame-to-frame spread measured on the competition images
(src/check_aug_ranges.py): contrast varies 1.7-1.85x between frames and fine-
structure sharpness 2.2-2.3x, but brightness only 1.05x. So contrast and blur
are pushed past the observed range, and brightness is barely moved -- a large
shift would produce frames unlike any the test set holds.

Carried over from measurement rather than taste:
  * native 2048 resolution: filaments are a median 8.2 px wide, 4 px at 1024
  * whole frames, no crops: a crop edge cuts filaments and teaches false ends
  * the disk sits at (1024, 1024), r = 904, in every frame, so rotating about
    the frame centre keeps it in place and never pulls in empty corners
  * the split is over photographs (seed 0, 20%), the same as every earlier
    run, and PQ is pooled over annotation sets the way the leaderboard does it

Off Kaggle the script never touches the local GPU (its fan is broken); it runs
a tiny CPU smoke test instead. `--preview` draws the augmentations.
"""
from __future__ import annotations

import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
if not ON_KAGGLE:
    # "-1", not "": on Windows an empty value deletes the variable and every GPU
    # stays visible -- which is how the first local smoke test ran on the card
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # before torch is imported
    # pretrained weights live with the project on D:, not in the user profile on C:
    os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / ".torchhub"))

import argparse, collections, json, math, pickle, random, subprocess, sys, time

# torch before cv2: the other order crashed at random (access violations in torch,
# cv2 and numpy alike) once both were loaded in one Windows process
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mu
from torchvision.models import resnet34, resnet50

cv2.setNumThreads(1)
if not ON_KAGGLE and torch.cuda.is_available():
    sys.exit("refusing to run: the local GPU is visible (broken fan) -- this script trains only on Kaggle")

# Edit these, not the command line: a Kaggle script kernel runs without arguments.
SMOKE = False
# "train": the two trainings.  "dump": load best_*.pt from an attached run and write
# every validation and test probability map, plain and with D4 TTA, so that
# post-processing can be explored offline without spending GPU quota on it.
MODE = "train"
if not ON_KAGGLE:                            # local checks may set them from the environment
    SMOKE = SMOKE or os.environ.get("FIL_SMOKE") == "1"
    MODE = os.environ.get("FIL_MODE", MODE)

FULL = 2048
CX = CY = 1024.0
R_DISK = 904.0
HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working") if ON_KAGGLE else HERE / "out"
CACHE = Path("/tmp/fil") if ON_KAGGLE else HERE / "out" / "cache"     # ~5 GB, not kept
# The arms of a run, one per GPU. Each is a full training; only what is listed
# differs from ARM_DEFAULTS. Run 1 (2026-10-26) was flip vs full: what the
# augmentation is worth.
ARM_DEFAULTS = dict(aug="full", encoder="resnet34", target="soft", init="imagenet", all_data=False)
RUNS = {
    # run 1: what the augmentation is worth
    "r1": {"flip": dict(aug="flip"), "full": dict(aug="full")},
    # run 2, on top of heavy augmentation: a target that rewards single-annotator
    # filaments, and a deeper encoder (slower, so fewer epochs)
    "r2": {"union": dict(target="union"), "r50": dict(encoder="resnet50", epochs=45)},
    # run 3: the solar-pretrained encoder (v4's, 3772 GONG frames) against ImageNet,
    # validated; and the final R50 on all 707 photographs. With no held-out frames
    # left, that arm trains a fixed number of epochs and uses the post-processing
    # tuned for R50 in run 2 (D4: 0.75 / 0 / 350).
    "r3": {"solar": dict(init="solar"),
           "r50all": dict(encoder="resnet50", epochs=45, all_data=True,
                          post=dict(thresh=0.75, merge=0, min_area=350))},
}
RUN = "r1"                                   # kernels/build.py sets this per kernel
ARMS = RUNS[RUN]

# post-processing grid, scored on every validation pass. The first run's grid
# (0.3-0.7, areas to 500) put its best point on the edge at 0.7 / 500 by epoch 9.
THRESH = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MERGE = (0, 3, 7)            # dilation radius that decides which pieces are one filament
MIN_AREA = (100, 250, 500, 1000, 2000)


# ---------------------------------------------------------------- data

def find_data():
    root = Path("/kaggle/input") if ON_KAGGLE else HERE.parents[1] / "MAGFiLO_1.0_Kaggle_2026"
    ann = next(root.rglob("*_train.json"), None)
    if ann is None:
        raise FileNotFoundError(f"no training annotations under {root}")
    tr = next(p for p in root.rglob("train_images") if p.is_dir())
    te = next(p for p in root.rglob("test_images") if p.is_dir())
    return ann, tr, te


def photograph_split(fname, val_frac=0.2, seed=0):
    """Grouped by photograph, identical to src/spine_data.make_split."""
    files = sorted(set(fname.values()))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(round(len(files) * val_frac))
    return sorted(files[n_val:]), sorted(files[:n_val])


def ann_mask(a):
    """Rasterised the way the scorer does it: polygon -> COCO RLE -> mask."""
    return mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], FULL, FULL))).astype(bool)


def prepare(smoke, with_train=True):
    """Decode every frame once into memory-mapped arrays the workers share."""
    ann_path, tr_dir, te_dir = find_data()
    coco = json.load(open(ann_path))
    by_id = collections.defaultdict(list)
    for a in coco["annotations"]:
        by_id[a["image_id"]].append(a)
    fname = {i["id"]: i["file_name"] for i in coco["images"]}
    sets_of = collections.defaultdict(list)
    for sid, f in fname.items():
        sets_of[f].append(sid)

    train_files, val_files = photograph_split(fname)
    test_files = sorted(p.name for p in te_dir.glob("*.jpeg"))
    if smoke:
        # off Kaggle, enough validation frames that some carry several annotators
        n_tr, n_va, n_te = (16, 8, 6) if ON_KAGGLE else (3, 10, 2)
        train_files, val_files, test_files = train_files[:n_tr], val_files[:n_va], test_files[:n_te]
    val_set = set(val_files)
    val_ids = sorted(k for k, v in fname.items() if v in val_set)

    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # training target: per pixel, the fraction of annotators who drew a filament there
    if with_train:
        img = np.lib.format.open_memmap(CACHE / "train_img.npy", "w+", np.uint8, (len(train_files), FULL, FULL))
        msk = np.lib.format.open_memmap(CACHE / "train_msk.npy", "w+", np.uint8, (len(train_files), FULL, FULL))
        for i, f in enumerate(train_files):
            img[i] = cv2.imread(str(tr_dir / f), cv2.IMREAD_GRAYSCALE)
            acc = np.zeros((FULL, FULL), np.float32)
            for sid in sets_of[f]:
                u = np.zeros((FULL, FULL), bool)
                for a in by_id[sid]:
                    u |= ann_mask(a)
                acc += u
            msk[i] = np.round(255.0 * acc / len(sets_of[f])).astype(np.uint8)
        img.flush(); msk.flush(); del img, msk

    # validation frames keep their targets too, for an arm that trains on all of them
    vimg = np.lib.format.open_memmap(CACHE / "val_img.npy", "w+", np.uint8, (len(val_files), FULL, FULL))
    vmsk = np.lib.format.open_memmap(CACHE / "val_msk.npy", "w+", np.uint8, (len(val_files), FULL, FULL))
    for i, f in enumerate(val_files):
        vimg[i] = cv2.imread(str(tr_dir / f), cv2.IMREAD_GRAYSCALE)
        acc = np.zeros((FULL, FULL), np.float32)
        for sid in sets_of[f]:
            u = np.zeros((FULL, FULL), bool)
            for a in by_id[sid]:
                u |= ann_mask(a)
            acc += u
        vmsk[i] = np.round(255.0 * acc / len(sets_of[f])).astype(np.uint8)
    vimg.flush(); vmsk.flush(); del vimg, vmsk

    # ground truth per annotation set, each filament kept as a bbox crop
    index = {f: i for i, f in enumerate(val_files)}
    gt = []
    for sid in val_ids:
        crops = []
        for a in by_id[sid]:
            m = ann_mask(a)
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue                      # an empty polygon is not a filament to find
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crops.append((int(y0), int(x0), m[y0:y1, x0:x1].copy()))
        gt.append({"set": sid, "frame": index[fname[sid]], "crops": crops})
    pickle.dump(gt, open(CACHE / "val_gt.pkl", "wb"))

    meta = dict(train_files=train_files, val_files=val_files, val_ids=val_ids,
                test_files=test_files, test_dir=str(te_dir))
    json.dump(meta, open(CACHE / "meta.json", "w"))
    print(f"prepared {len(train_files)} train / {len(val_files)} val photographs "
          f"({len(val_ids)} val annotation sets), {len(test_files)} test, "
          f"in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- augmentation

_W = None


def disk_weight():
    """1 on the disk, 0 off it, feathered over 24 px so the limb stays natural."""
    global _W
    if _W is None:
        yy, xx = np.mgrid[:FULL, :FULL]
        r = np.hypot(xx - CX, yy - CY).astype(np.float32)
        _W = np.clip((R_DISK + 12.0 - r) / 24.0, 0.0, 1.0)
    return _W


def log_uniform(lo, hi):
    return math.exp(random.uniform(math.log(lo), math.log(hi)))


def geometric(im, ma):
    """Any-angle rotation about the disk centre, then the two mirrors."""
    M = cv2.getRotationMatrix2D((CX, CY), random.uniform(0.0, 360.0), 1.0)
    bg = float(np.median(im[:48, :48]))
    im = cv2.warpAffine(im, M, (FULL, FULL), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    ma = cv2.warpAffine(ma, M, (FULL, FULL), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return augment_flip(im, ma)


def photometric(im):
    x = im.astype(np.float32)
    med = float(np.median(x[724:1324, 724:1324]))
    y = x
    if random.random() < 0.8:                        # contrast, both down and up
        y = med + log_uniform(0.6, 1.65) * (y - med)
    if random.random() < 0.5:                        # gamma
        y = 255.0 * np.power(np.clip(y, 0.0, 255.0) / 255.0, log_uniform(0.75, 1.33))
    if random.random() < 0.5:                        # brightness: frames barely differ in it
        y = y + random.uniform(-5.0, 5.0)
    x = x + disk_weight() * (y - x)

    u = random.random()
    if u < 0.35:                                     # worse seeing
        x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.5, 1.8))
    elif u < 0.65:                                   # better seeing: unsharp mask
        s, a = random.uniform(1.0, 3.0), random.uniform(0.3, 1.2)
        x = x + a * (x - cv2.GaussianBlur(x, (0, 0), s))
    if random.random() < 0.3:                        # quiet-Sun scatter is 4.4 levels
        rng = np.random.default_rng(random.getrandbits(32))
        x = x + rng.normal(0.0, random.uniform(1.0, 4.0), x.shape).astype(np.float32)

    im = np.clip(x + 0.5, 0, 255).astype(np.uint8)
    if random.random() < 0.4:                        # the test frames are JPEG, quality 92
        _, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, random.randint(70, 95)])
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return im


def augment_full(im, ma):
    im, ma = geometric(im, ma)
    return np.ascontiguousarray(photometric(im)), ma


def augment_flip(im, ma):
    if random.random() < 0.5:
        im, ma = im[:, ::-1], ma[:, ::-1]
    if random.random() < 0.5:
        im, ma = im[::-1], ma[::-1]
    return np.ascontiguousarray(im), np.ascontiguousarray(ma)


def to_input(im_u8):
    return torch.from_numpy(im_u8).float().div_(255.0).sub_(0.5).div_(0.25)


class Frames(torch.utils.data.Dataset):
    """target: "soft" = the fraction of annotators who drew each pixel;
    "union" = drawn by anyone, which a single-annotator test set rewards."""

    def __init__(self, n, aug, target="soft", n_val=0):
        """n training frames, then n_val validation frames appended (all-data arm)."""
        self.n, self.n_val, self.aug, self.target = n, n_val, aug, target
        self.img = self.msk = None

    def __len__(self):
        return self.n + self.n_val

    def __getitem__(self, i):
        if self.img is None:                         # opened per worker process
            self.img = np.load(CACHE / "train_img.npy", mmap_mode="r")
            self.msk = np.load(CACHE / "train_msk.npy", mmap_mode="r")
            if self.n_val:
                self.vimg = np.load(CACHE / "val_img.npy", mmap_mode="r")
                self.vmsk = np.load(CACHE / "val_msk.npy", mmap_mode="r")
        if i < self.n:
            im, ma = np.array(self.img[i]), np.array(self.msk[i])
        else:
            im, ma = np.array(self.vimg[i - self.n]), np.array(self.vmsk[i - self.n])
        if self.target == "union":
            ma = np.where(ma > 0, 255, 0).astype(np.uint8)
        im, ma = (augment_full if self.aug == "full" else augment_flip)(im, ma)
        return to_input(im)[None], torch.from_numpy(ma).float().div_(255.0)[None]


# ---------------------------------------------------------------- model

def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


ENCODERS = {  # constructor, channels of layer1..layer4
    "resnet34": (resnet34, (64, 128, 256, 512)),
    "resnet50": (resnet50, (256, 512, 1024, 2048)),
}


class UNet(nn.Module):
    """ResNet encoder, five decoder stages, one output: filament probability.

    The grey input inherits the ImageNet first layer summed over RGB -- the same
    response to a grey image as repeating it three times, at a third of the work.
    """

    def __init__(self, encoder="resnet34", pretrained=True, width=(256, 128, 64, 32, 16)):
        super().__init__()
        fn, ch = ENCODERS[encoder]
        net = fn(weights="DEFAULT" if pretrained else None)
        conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.copy_(net.conv1.weight.sum(dim=1, keepdim=True))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = net.layer1, net.layer2, net.layer3, net.layer4
        c = width
        self.dec4 = _block(ch[3] + ch[2], c[0])
        self.dec3 = _block(c[0] + ch[1], c[1])
        self.dec2 = _block(c[1] + ch[0], c[2])
        self.dec1 = _block(c[2] + 64, c[3])
        self.dec0 = _block(c[3], c[4])
        self.head = nn.Conv2d(c[4], 1, 1)
        nn.init.constant_(self.head.bias, -4.0)       # filaments are ~0.4% of the disk

    @staticmethod
    def _up(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(self.pool(s))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d = self.dec4(torch.cat([self._up(e4, e3), e3], 1))
        d = self.dec3(torch.cat([self._up(d, e2), e2], 1))
        d = self.dec2(torch.cat([self._up(d, e1), e1], 1))
        d = self.dec1(torch.cat([self._up(d, s), s], 1))
        return self.head(self.dec0(self._up(d, x)))


def load_solar(model, tag=""):
    """Start from the U-Net v4 pretrained on 3772 unlabelled GONG frames.

    That checkpoint uses the same module names (stem, enc1-4, dec4-0), so every
    tensor whose name and shape match is copied and only the output head starts
    fresh. Solar weights exist for ResNet-34 only.
    """
    roots = [Path("/kaggle/input")] if ON_KAGGLE else [HERE.parents[1] / "checkpoints"]
    path = next((p for r in roots for p in r.rglob("pretrain_r34_encoder.pt")), None)
    if path is None:
        sys.exit("solar init asked for, but pretrain_r34_encoder.pt is not attached")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    print(f"[{tag}] solar init from {path.name}: {len(ok)} of {len(own)} tensors loaded, "
          f"{len(sd) - len(ok)} of the checkpoint's skipped", flush=True)


def loss_fn(logit, y, valid, pos_weight=10.0):
    """BCE + soft Dice, on-disk pixels only. Targets are soft (annotator fraction)."""
    pw = torch.tensor(pos_weight, device=logit.device)
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw, reduction="none")
    bce = (bce * valid).sum() / (valid.sum() * logit.shape[0])
    p, t = torch.sigmoid(logit) * valid, y * valid
    dice = 1.0 - (2.0 * (p * t).sum((1, 2, 3)) + 1.0) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1.0)
    return bce + dice.mean(), bce.detach(), dice.mean().detach()


# ---------------------------------------------------------------- scoring

def components(prob_u8, t, k):
    """Threshold, then group pieces lying within 2k px of each other into one filament."""
    b = (prob_u8 >= int(round(255 * t))).astype(np.uint8)
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        lab = cv2.connectedComponents(cv2.dilate(b, ker), connectivity=8)[1]
        lab[b == 0] = 0
    else:
        lab = cv2.connectedComponents(b, connectivity=8)[1]
    return lab, np.bincount(lab.ravel())


def pq_counts(crops, lab, area, keep):
    """TP, FP, FN and summed IoU for one annotation set against one label image.

    Equivalent to src/metrics.pq_from_masks (checked by --check-metric): at IoU
    > 0.5 a match is unique, so greedy matching by descending IoU is exact.
    """
    pairs = []
    for g, (y0, x0, c) in enumerate(crops):
        lv = lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c]
        lv = lv[keep[lv]]
        if lv.size == 0:
            continue
        ids, inter = np.unique(lv, return_counts=True)
        ga = int(c.sum())
        for j, it in zip(ids.tolist(), inter.tolist()):
            iou = it / (ga + int(area[j]) - it)
            if iou > 0.5:
                pairs.append((iou, g, j))
    pairs.sort(reverse=True)
    used_g, used_p, s = set(), set(), 0.0
    for iou, g, j in pairs:
        if g not in used_g and j not in used_p:
            used_g.add(g); used_p.add(j); s += iou
    tp = len(used_g)
    return tp, int(keep.sum()) - tp, len(crops) - tp, s


def summarise(c):
    tp, fp, fn, s = c
    den = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=s / den if den else 0.0, SQ=s / tp if tp else 0.0,
                RQ=tp / den if den else 0.0, TP=tp, FP=fp, FN=fn)


def sweep(probs, gt, n_first=60):
    """Score the whole post-processing grid; return it, best first."""
    by_frame = collections.defaultdict(list)
    for i, g in enumerate(gt):
        by_frame[g["frame"]].append((i, g["crops"]))
    rows = []
    for t in THRESH:
        for k in MERGE:
            tot = {a: np.zeros(4) for a in MIN_AREA}
            first = {a: np.zeros(4) for a in MIN_AREA}
            for fi, sets in by_frame.items():
                lab, area = components(probs[fi], t, k)
                for a in MIN_AREA:
                    keep = area >= a
                    keep[0] = False
                    for i, crops in sets:
                        c = np.array(pq_counts(crops, lab, area, keep), float)
                        tot[a] += c
                        if i < n_first:
                            first[a] += c
            for a in MIN_AREA:
                r = summarise(tot[a])
                r.update(thresh=t, merge=k, min_area=a, PQ60=summarise(first[a])["PQ"])
                rows.append(r)
    return sorted(rows, key=lambda r: -r["PQ"])


def fmt(device):
    """channels_last feeds the T4's tensor cores; on CPU it crashes backward at 2048."""
    return torch.channels_last if device.type == "cuda" else torch.contiguous_format


@torch.no_grad()
def predict(model, im_u8, device, ondisk, tta="plain"):
    """Filament probability as uint8.

    tta: "plain", "flip4" (the four mirrors) or "d4" (mirrors with and without a
    transpose -- all eight symmetries of the square, each mapped back). The disk
    is centred, so every one of them keeps it in place.
    """
    x = to_input(im_u8)[None, None].to(device)
    flips = ([], [3], [2], [2, 3]) if tta in ("flip4", "d4") else ([],)
    views = [(t, d) for t in ((False, True) if tta == "d4" else (False,)) for d in flips]
    acc = 0.0
    for tr, dims in views:
        xi = x.transpose(2, 3) if tr else x
        xi = torch.flip(xi, dims) if dims else xi
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xi.contiguous(memory_format=fmt(device))).float())
        p = torch.flip(p, dims) if dims else p
        acc = acc + (p.transpose(2, 3) if tr else p)
    p = acc[0, 0] / len(views)
    return (p * ondisk * 255.0).round().byte().cpu().numpy()


# ---------------------------------------------------------------- one training

def train_worker(args):
    tag = args.arm
    cfg = {**ARM_DEFAULTS, **ARMS[tag]}
    epochs = args.epochs if SMOKE else int(cfg.get("epochs", args.epochs))   # smoke caps every arm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.load(open(CACHE / "meta.json"))
    gt = pickle.load(open(CACHE / "val_gt.pkl", "rb"))
    vimg = np.load(CACHE / "val_img.npy", mmap_mode="r")
    n_train = len(meta["train_files"])
    random.seed(1000 + sum(map(ord, tag))); torch.manual_seed(0)
    print(f"[{tag}] {cfg}, {epochs} epochs", flush=True)

    yy, xx = np.mgrid[:FULL, :FULL]
    ondisk_np = (((xx - CX) ** 2 + (yy - CY) ** 2) <= (R_DISK * 0.995) ** 2).astype(np.float32)
    ondisk = torch.from_numpy(ondisk_np).to(device)
    valid = ondisk[None, None]

    model = UNet(cfg["encoder"], pretrained=cfg["init"] == "imagenet")
    if cfg["init"] == "solar":
        load_solar(model, tag)
    model = model.to(device).to(memory_format=fmt(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_val_train = len(meta["val_files"]) if cfg["all_data"] else 0
    loader = torch.utils.data.DataLoader(
        Frames(n_train, cfg["aug"], cfg["target"], n_val_train),
        batch_size=int(cfg.get("batch", args.batch)),
        shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0)
    steps = epochs * len(loader)
    warm = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    t_start, budget = time.time(), args.hours * 3600.0
    hist, vals, best = [], [], None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = bce_s = dice_s = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=fmt(device))
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(x)
            loss, b, d = loss_fn(logit.float(), y, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); bce_s += b.item(); dice_s += d.item()
        n = len(loader)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        h = dict(epoch=epoch, loss=tot / n, bce=bce_s / n, dice=dice_s / n,
                 lr=sched.get_last_lr()[0], sec=time.time() - t0, sec_per_step=(time.time() - t0) / n,
                 gpu_mem_gib=mem)
        hist.append(h)
        print(f"[{tag}] epoch {epoch}/{epochs} loss {h['loss']:.4f} (bce {h['bce']:.4f} "
              f"dice {h['dice']:.4f}) {h['sec']:.0f}s {h['sec_per_step']:.2f}s/step {mem:.1f} GiB", flush=True)

        if cfg["all_data"]:
            # nothing is held out, so nothing to select on: keep the latest epoch,
            # with the post-processing tuned on the same architecture in run 2
            post = dict(cfg["post"], PQ=None)
            best = dict(post, epoch=epoch)
            torch.save({"model": model.state_dict(), "epoch": epoch, "post": post, "tag": tag,
                        "aug": cfg["aug"], "encoder": cfg["encoder"], "cfg": cfg},
                       WORK / f"best_{tag}.pt")
        elif epoch % args.val_every == 0 or epoch == epochs:
            t1 = time.time()
            model.eval()
            probs = [predict(model, np.array(vimg[i]), device, ondisk) for i in range(len(vimg))]
            grid = sweep(probs, gt)
            top = dict(grid[0], epoch=epoch, sec=time.time() - t1)
            vals.append(top)
            print(f"[{tag}] val epoch {epoch}: PQ {top['PQ']:.4f} SQ {top['SQ']:.3f} RQ {top['RQ']:.3f} "
                  f"TP {top['TP']} FP {top['FP']} FN {top['FN']}  PQ(60 sets) {top['PQ60']:.4f}  "
                  f"at thresh {top['thresh']} merge {top['merge']} min_area {top['min_area']}  "
                  f"({top['sec']:.0f}s)", flush=True)
            if best is None or top["PQ"] > best["PQ"]:
                best = dict(top, grid=grid[:40])
                torch.save({"model": model.state_dict(), "epoch": epoch, "post": top, "tag": tag,
                            "aug": cfg["aug"], "encoder": cfg["encoder"], "cfg": cfg},
                           WORK / f"best_{tag}.pt")
                for i in range(min(6, len(probs))):        # for figures made after download
                    cv2.imwrite(str(WORK / f"valprob_{tag}_{Path(meta['val_files'][i]).stem}.png"), probs[i])

        json.dump(dict(aug=tag, args=vars(args), history=hist, val=vals, best=best),
                  open(WORK / f"result_{tag}.json", "w"), indent=1)
        elapsed = time.time() - t_start
        per_epoch = elapsed / epoch
        if elapsed + 1.5 * per_epoch + 900 > budget:
            print(f"[{tag}] stopping after epoch {epoch}: time budget {args.hours} h", flush=True)
            break

    # test predictions with the best checkpoint and its tuned post-processing
    ck = torch.load(WORK / f"best_{tag}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    post = ck["post"]
    rows, per_image = [], []
    for f in meta["test_files"]:
        im = cv2.imread(str(Path(meta["test_dir"]) / f), cv2.IMREAD_GRAYSCALE)
        lab, area = components(predict(model, im, device, ondisk), post["thresh"], post["merge"])
        ids = [j for j in range(1, len(area)) if area[j] >= post["min_area"]]
        for n_, j in enumerate(ids, 1):
            r = mu.encode(np.asfortranarray((lab == j).astype(np.uint8)))
            rows.append(f"{Path(f).stem}_{n_},{r['counts'].decode('ascii')}")
        per_image.append(len(ids))
    with open(WORK / f"submission_{tag}.csv", "w", newline="") as fh:
        fh.write("filament_id,segmentation_rle\n" + "".join(r + "\n" for r in rows))
    res = json.load(open(WORK / f"result_{tag}.json"))
    res["test"] = dict(rows=len(rows), images=len(per_image), mean_per_image=float(np.mean(per_image)),
                       empty_images=int(sum(1 for p in per_image if p == 0)), from_epoch=ck["epoch"])
    json.dump(res, open(WORK / f"result_{tag}.json", "w"), indent=1)
    print(f"[{tag}] wrote submission_{tag}.csv: {len(rows)} rows, {np.mean(per_image):.1f} per image, "
          f"from epoch {ck['epoch']}", flush=True)
    # for the offline post-processing kernel; an all-data model has no honest val maps
    write_maps(model, tag, ck, device, ondisk, val=not cfg["all_data"])


# ---------------------------------------------------------------- dump mode

def dump_worker(args):
    """Every val and test probability map of one checkpoint, plain and D4 TTA, as PNG."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    tag = ck.get("tag", ck["aug"])                 # run 1 checkpoints carry only "aug"
    model = UNet(ck.get("encoder", "resnet34"), pretrained=False).to(device).to(memory_format=fmt(device))
    model.load_state_dict(ck["model"]); model.eval()
    yy, xx = np.mgrid[:FULL, :FULL]
    ondisk = torch.from_numpy((((xx - CX) ** 2 + (yy - CY) ** 2) <= (R_DISK * 0.995) ** 2)
                              .astype(np.float32)).to(device)
    write_maps(model, tag, ck, device, ondisk)


def write_maps(model, tag, ck, device, ondisk, val=True):
    import shutil
    meta = json.load(open(CACHE / "meta.json"))
    _, tr_dir, _ = find_data()
    model.eval()
    root = WORK / f"dump_{tag}"
    t0 = time.time()
    splits = [("val", tr_dir, meta["val_files"])] if val else []
    for split, folder, files in splits + [("test", Path(meta["test_dir"]), meta["test_files"])]:
        for view in ("plain", "d4"):
            (root / f"{split}_{view}").mkdir(parents=True, exist_ok=True)
        for f in files:
            im = cv2.imread(str(folder / f), cv2.IMREAD_GRAYSCALE)
            for view in ("plain", "d4"):
                p = predict(model, im, device, ondisk, tta=view)
                cv2.imwrite(str(root / f"{split}_{view}" / f"{Path(f).stem}.png"), p)
        print(f"[{tag}] {split}: {len(files)} frames, {time.time() - t0:.0f}s", flush=True)
    # one archive per model: the CLI fetches outputs file by file
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    json.dump(dict(aug=tag, epoch=ck["epoch"], post=ck["post"], val_maps=val,
                   val_files=meta["val_files"], test_files=meta["test_files"]),
              open(WORK / f"dump_{tag}.json", "w"), indent=1)
    print(f"[{tag}] wrote dump_{tag}.tar (checkpoint epoch {ck['epoch']})", flush=True)


def dump_launch():
    ckpts = sorted(Path("/kaggle/input").rglob("best_*.pt")) if ON_KAGGLE else sorted(WORK.glob("best_*.pt"))
    print("checkpoints:", [str(c) for c in ckpts], flush=True)
    if not ckpts:
        sys.exit("no best_*.pt found -- attach the training run as a kernel source")
    prepare(SMOKE, with_train=False)
    procs = []
    for i, c in enumerate(ckpts):
        env = dict(os.environ)
        if ON_KAGGLE:
            env["CUDA_VISIBLE_DEVICES"] = str(i % max(1, torch.cuda.device_count()))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--ckpt", str(c)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    print("exit codes:", codes, flush=True)


# ---------------------------------------------------------------- checks run off Kaggle

def check_metric():
    """pq_counts must agree with src/metrics.pq_from_masks on real masks."""
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from metrics import pq_from_masks, aggregate
    gt = pickle.load(open(CACHE / "val_gt.pkl", "rb"))
    frames = collections.defaultdict(list)
    for g in gt:
        frames[g["frame"]].append(g)
    ours, ref = np.zeros(4), []
    for sets in frames.values():
        if len(sets) < 2:
            continue
        # annotator A's drawing, as a label image, plays the prediction for the others
        lab = np.zeros((FULL, FULL), np.int32)
        for j, (y0, x0, c) in enumerate(sets[0]["crops"], 1):
            lab[y0:y0 + c.shape[0], x0:x0 + c.shape[1]][c] = j
        area = np.bincount(lab.ravel())
        keep = area > 0; keep[0] = False
        preds = [lab == j for j in range(1, len(area)) if keep[j]]
        for g in sets[1:]:
            ours += np.array(pq_counts(g["crops"], lab, area, keep), float)
            full = []
            for y0, x0, c in g["crops"]:
                m = np.zeros((FULL, FULL), bool)
                m[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
                full.append(m)
            ref.append(pq_from_masks(full, preds))
    a, b = summarise(ours), aggregate(ref)
    print(f"pq_counts      PQ {a['PQ']:.6f} TP {a['TP']} FP {a['FP']} FN {a['FN']}")
    print(f"pq_from_masks  PQ {b['PQ']:.6f} TP {b['TP']} FP {b['FP']} FN {b['FN']}")
    assert abs(a["PQ"] - b["PQ"]) < 1e-9 and (a["TP"], a["FP"], a["FN"]) == (b["TP"], b["FP"], b["FN"])
    print("metric check passed")


def preview():
    """One frame, seven random augmentations: whole disk above, the same patch below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, tr_dir, _ = find_data()
    f = json.load(open(CACHE / "meta.json"))["val_files"][0]
    im0 = cv2.imread(str(tr_dir / f), cv2.IMREAD_GRAYSCALE)
    ma0 = np.zeros_like(im0)
    # centre the patch on the largest filament drawn on this frame
    crops = [c for g in pickle.load(open(CACHE / "val_gt.pkl", "rb")) if g["frame"] == 0 for c in g["crops"]]
    y0, x0, c = max(crops, key=lambda t: t[2].sum())
    py = int(np.clip(y0 + c.shape[0] // 2 - 125, 0, FULL - 250))
    px = int(np.clip(x0 + c.shape[1] // 2 - 125, 0, FULL - 250))
    random.seed(7)
    fig, ax = plt.subplots(2, 8, figsize=(24, 6.6))
    for k in range(8):
        if k == 0:
            g, p = im0, im0
        else:
            p = photometric(im0)              # the patch shows it unrotated, at a fixed place
            g, _ = geometric(p, ma0)
        ax[0, k].imshow(g[::4, ::4], cmap="gray", vmin=0, vmax=255)
        ax[1, k].imshow(p[py:py + 250, px:px + 250], cmap="gray", vmin=0, vmax=255)
        ax[0, k].set_title("original" if k == 0 else f"augmented {k}", fontsize=12)
        for a in ax[:, k]:
            a.set_xticks([]); a.set_yticks([])
    ax[1, 0].set_ylabel("250 px around a filament\n(unrotated)", fontsize=11)
    fig.suptitle(f"augmentation preview -- {f}", fontsize=14)
    fig.tight_layout()
    out = HERE.parents[1] / "figures" / "aug_preview.png"
    fig.savefig(out, dpi=80)
    print(f"wrote {out}")


# ---------------------------------------------------------------- launcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    # 60 epochs: the smoke run measured 0.78 s per step of two frames on a T4, so an
    # epoch of 566 frames is ~3.7 min, ~4.7 min with validation every third -- ~4.7 h
    ap.add_argument("--epochs", type=int, default=(2 if ON_KAGGLE else 1) if SMOKE else 60)
    ap.add_argument("--batch", type=int, default=2 if ON_KAGGLE else 1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-every", type=int, default=1 if SMOKE else 3)
    ap.add_argument("--hours", type=float, default=0.5 if SMOKE else 10.8)
    ap.add_argument("--workers", type=int, default=2 if ON_KAGGLE else 0)
    ap.add_argument("--check-metric", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--ckpt", help="dump mode: the checkpoint this worker writes maps for")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return dump_worker(args) if MODE == "dump" else train_worker(args)
    if MODE == "dump":
        return dump_launch()
    if MODE == "post":
        return kaggle_post()      # defined only in the kernel kernels/build.py merges with post_local.py
    if args.check_metric or args.preview:
        if not (CACHE / "meta.json").exists():
            prepare(smoke=True)
        return check_metric() if args.check_metric else preview()

    # device_count reads NVML; anything that names a device would create a CUDA
    # context in this process and hold GPU 0 memory for the whole run
    print(f"torch {torch.__version__}  gpus {torch.cuda.device_count()}  "
          f"cpus {os.cpu_count()}  smoke {SMOKE}", flush=True)
    if ON_KAGGLE:
        subprocess.call(["nvidia-smi", "-L"])
    prepare(SMOKE)

    base = [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--epochs", str(args.epochs), "--batch", str(args.batch), "--lr", str(args.lr),
            "--val-every", str(args.val_every), "--hours", str(args.hours),
            "--workers", str(args.workers)]
    n_gpu = torch.cuda.device_count()
    t0 = time.time()
    if n_gpu >= 2:
        procs = []
        for i, aug in enumerate(ARMS):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            log = open(WORK / f"log_{aug}.txt", "w")
            procs.append((aug, subprocess.Popen(base + ["--arm", aug], env=env, stdout=log,
                                                stderr=subprocess.STDOUT), log))
        last = time.time()
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(15)
            if time.time() - last > (60 if SMOKE else 600):
                last = time.time()
                for aug, _, _ in procs:
                    lines = (WORK / f"log_{aug}.txt").read_text().strip().splitlines()
                    print(f"  {(time.time() - t0) / 3600:.2f} h  {lines[-1] if lines else ''}", flush=True)
        codes = {aug: p.returncode for aug, p, _ in procs}
    else:
        # one device: run them in turn, each with half the time
        codes = {}
        for aug in ARMS:
            cmd = base + ["--arm", aug]
            cmd[cmd.index("--hours") + 1] = str(args.hours / 2)
            with open(WORK / f"log_{aug}.txt", "w") as log:
                codes[aug] = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)

    summary = {}
    for aug in ARMS:
        print(f"\n===== {aug} (exit {codes[aug]}) =====")
        print((WORK / f"log_{aug}.txt").read_text()[-3000:])
        p = WORK / f"result_{aug}.json"
        if p.exists():
            r = json.load(open(p))
            summary[aug] = dict(best={k: v for k, v in (r.get("best") or {}).items() if k != "grid"},
                                epochs=len(r["history"]), test=r.get("test"))
    json.dump(summary, open(WORK / "summary.json", "w"), indent=1)
    print("\nsummary:", json.dumps(summary, indent=1))
    print(f"total {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
