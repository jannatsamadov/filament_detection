"""PixNet: the user's per-pixel neural network, trained from scratch (2026-11-12).

Each pixel is classified from the table of what is known about it -- where it is on the Sun
(sine of latitude, longitude from the central meridian, mu), its brightness raw and relative to the
disk -- together with the same for every pixel around it out to ~130 px at half resolution (~260 px
at 2048), and the photograph's own columns: date (solar-cycle phase), season, hour, station, B0,
the disk's median and spread. A table holding every neighbour as a column would have ~50 000
columns per pixel; a convolution is the same small per-pixel network slid over the frame with
shared weights, so the model is built as one:

  stem   the frame at half resolution (1024) with the per-pixel columns as channels
  body   12 conv layers 3x3 with ReLU and BatchNorm, in residual pairs dilated 1, 2, 4, 8, 16, 1
  FiLM   the photograph's columns scale and shift the channels of every layer
  head   a per-pixel MLP (1x1 convs), back to 2048, with a small full-resolution branch for the edges
  ~6 M parameters, no pretrained weights.

Two arms, one per T4, validated on fold 0 exactly as the U-Nets and the Qwen segmenter were:
  scratch  random weights, then the 566 annotated photographs: the model as the user asked
  pre      first one pass over 2000 unlabelled GONG frames with the fold-0 U-Net ensemble's map as
           the target (for the Qwen segmenter that was worth +0.013), then the same
Each writes dump_pixnet_<arm>.tar (val_plain, test_plain) in the U-Nets' dump format and
pixnet_<arm>.json with the full post-processing sweep on fold 0.
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")            # before run.py hides the card
import json, math, random, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- torch before cv2
from run import torch, cv2, np, nn, F  # noqa: E402
import qwen_seg as Q  # noqa: E402  -- crops, loss, scoring, pseudo-labels, governor

SMOKE = os.environ.get("FIL_PIXNET_SMOKE") == "1"
ARMS = tuple(os.environ.get("FIL_PIXNET_ARMS", "scratch,pre").split(","))   # filament-pixnet-rot: table,tablerot
WIDTH = 224                      # 12 layers of 224 channels: ~5.4 M weights in the body
DILATIONS = (1, 2, 4, 8, 16, 1)  # each a residual pair of 3x3 layers: a 129 px radius at half resolution
STATIONS = ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
N_META = 15
CROP = 512 if SMOKE else 768     # full-resolution px; the body runs at half that
# on the laptop's RTX 2060 (6 GB, duty-cycled to 72%): half the batch, fewer epochs (Kaggle's GPU quota ran out 11-12)
LOCAL = not run.ON_KAGGLE
BATCH = 2 if LOCAL else 4
EPOCHS = 1 if SMOKE else (24 if LOCAL else 30)
VAL_AT = (1,) if SMOKE else ((8, 12, 16, 20, 24) if LOCAL else (10, 15, 20, 25, 30))
N_PSEUDO = 6 if SMOKE else 2000
BUDGET_H = 0.5 if SMOKE else (9.0 if LOCAL else 5.5)  # per arm: past this, stop, keep the best epoch, write everything
if LOCAL:                        # its own full fold-0 cache: out/cache holds a smoke one
    run.CACHE_ROOT = HERE / "out" / "cache_full"
    run.CACHE = run.CACHE_ROOT
LR = 1e-3
WORK = run.WORK
# "cont" (laptop, 11-12): Kaggle's scratch model was still rising at its last epoch (quick val 0.354, 0.358,
# 0.370 at 20, 25, 30), so it goes on from there with a second, lower cosine
CONT_INIT = WORK / "pixnet_scratch_kaggle.pt"   # filament-pixnet's pixnet_scratch.pt (epoch 30, fp16)
CONT_EPOCHS = 1 if SMOKE else 12
CONT_VAL_AT = (1,) if SMOKE else (4, 8, 12)
CONT_LR = 3e-4
# a full 2048 px forward pass does not fit the laptop's 6 GB (OOM at 5.4 GB, 11-12): 2x2 tiles of 1024 px,
# each run with a 256 px margin of context (the receptive field is ~260 px), only the core kept
TILE, TILE_M = (1024, 256) if LOCAL else (0, 0)
Q.N_PSEUDO, Q.CROP, Q.MARGIN = N_PSEUDO, CROP, 64
Q.TMP = Path("/tmp/pixnet") if run.ON_KAGGLE else HERE / "out" / "pixnet"


# ---------------------------------------------------------------- the photograph's own columns

_DISK = None


def disk_mask():
    global _DISK
    if _DISK is None:
        yy, xx = np.mgrid[:run.FULL:4, :run.FULL:4]
        _DISK = torch.from_numpy(((xx - run.CX) ** 2 + (yy - run.CY) ** 2) <= (run.R_DISK * 0.98) ** 2)
    return _DISK


def meta_vec(stem, x, b0):
    """(N_META,): date, season, hour, station, B0, and the disk's median and spread in x's units."""
    t = datetime.strptime(stem[:14], "%Y%m%d%H%M%S")
    doy = (t.timetuple().tm_yday - 1) / 365.25
    hr = (t.hour + t.minute / 60.0) / 24.0
    st = [0.0] * (len(STATIONS) + 1)
    st[STATIONS.index(stem[14:16]) if stem[14:16] in STATIONS else -1] = 1.0
    v = x.detach().float().cpu()[::4, ::4][disk_mask()]
    return torch.tensor([(t.year + doy - 2016.5) / 6.0, math.sin(2 * math.pi * doy), math.cos(2 * math.pi * doy),
                         math.sin(2 * math.pi * hr), math.cos(2 * math.pi * hr), *st, float(b0) / 7.25,
                         float(v.median()), float(v.std())], dtype=torch.float32)


MED, SPREAD = N_META - 2, N_META - 1


# ---------------------------------------------------------------- the user's pixel table as channels (11-14)

TAB_N = 35                       # pixel_gbm.PIX_FEAT_A: position, time, station, darkness, ridge, photograph
_TAB = {}


def tab_norm():
    """Mean and spread of the 35 table columns over the disks of 16 photographs (computed once, kept)."""
    if "mu" not in _TAB:
        import pixel_gbm as G
        p = WORK / "pixnet_table_norm.json"
        if p.exists():
            d = json.load(open(p))
        else:
            tr = run.find_data()[1]
            files = sorted(x.name for x in tr.glob("*.jpeg"))[::40][:16]
            X = np.concatenate([G.pix_features(cv2.imread(str(tr / f), cv2.IMREAD_GRAYSCALE), Path(f).stem)[::7]
                                for f in files])
            d = dict(columns=G.PIX_FEAT_A, mu=X.mean(0).tolist(), sd=X.std(0).tolist())
            json.dump(d, open(p, "w"), indent=1)
        _TAB["mu"] = np.array(d["mu"], np.float32)
        _TAB["sd"] = np.maximum(np.array(d["sd"], np.float32), 1e-6)
    return _TAB["mu"], _TAB["sd"]


def table(im_u8, stem):
    """(35, 1024, 1024) float16: the pixel table's columns of this photograph, each to mean 0 and spread 1,
    0 off the disk. A neural network needs comparable scales (year ~2014 next to contrast ~1); a tree does not."""
    import pixel_gbm as G
    mu, sd = tab_norm()
    T = np.zeros((TAB_N, G.PIX_HALF, G.PIX_HALF), np.float16)
    T[:, G.PIX_G[3]] = ((G.pix_features(im_u8, stem) - mu) / sd).T
    return torch.from_numpy(T)


def _one_thread(_):
    cv2.setNumThreads(1)


# ---------------------------------------------------------------- the Sun's rotation as channels (user, 11-14)
# The same slots as the RF table (pixel_nb): before/after x 0-24, 24-48, 48-84 h, each the nearest other photograph
# of the same pool (training among training, validation among validation, test among test), carried to this
# photograph's moment by differential rotation (kfold.carry). Per slot: its contrast there, "a photograph is
# there", and the gap; plus how far each pixel moves per hour (vx, vy). The convolution sees the carried
# neighbourhood by itself, so no radius is chosen.

ROT_N = 6 * 3 + 2
ROT_DIR = Path("/tmp/pixnet_rot") if run.ON_KAGGLE else HERE / "out" / "pixnet_rot"
_ROT = {}


def _half_grid():
    i = 2.0 * np.arange(1024) + 0.5                  # the 1024 grid's pixel centres in 2048 px (as pixel_gbm.pix_grid)
    yy, xx = np.meshgrid(i, i, indexing="ij")
    on = np.hypot(xx - run.CX, yy - run.CY) < run.R_DISK
    return np.stack([yy.ravel(), xx.ravel()], axis=1), on


def rot_job(job):
    """One photograph: its six slots' other photographs carried here, as uint8 contrast x 100 (0 = nothing)."""
    import kfold as KF
    import pixel_nb as NB
    stem, slots = job
    pts, on = _half_grid()
    b0 = run.b0_deg(stem)
    out = np.zeros((6, 1024, 1024), np.uint8)
    for k, s in enumerate(slots):
        if s is None:
            continue
        path, dt = s
        c = NB.contrast(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        yb, xb, vis = KF.carry(pts, b0, run.b0_deg(Path(path).stem), dt)
        v = cv2.remap(c, xb.reshape(1024, 1024).astype(np.float32), yb.reshape(1024, 1024).astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        v = np.where(vis.reshape(1024, 1024) & on, v, 0.0)
        out[k] = np.clip(np.round(v * 100.0), 0, 255).astype(np.uint8)
    np.save(ROT_DIR / f"{stem}.npy", out)
    return stem


def rot_prepare(meta, procs):
    """Carries every photograph's slots once (CPU) before training; the slots are chosen within the full pools."""
    import multiprocessing as mp
    import pixel_nb as NB
    ann, tr_dir, te_dir = run.find_data()
    fname = {i["id"]: i["file_name"] for i in json.load(open(ann))["images"]}
    train_all, val_all = run.photograph_split(fname, 0)
    test_all = sorted(p.name for p in Path(te_dir).glob("*.jpeg"))
    ROT_DIR.mkdir(parents=True, exist_ok=True)
    dts, jobs = {}, []
    use = set(meta["train_files"]) | set(meta["val_files"]) | set(meta["test_files"])
    for files, d in ((train_all, tr_dir), (val_all, tr_dir), (test_all, Path(te_dir))):
        slots = NB.rot_slots(files)
        for f in files:
            if f in use:
                stem = Path(f).stem
                dts[stem] = [None if s is None else s[1] * 24.0 for s in slots[f]]
                jobs.append((stem, [None if s is None else (str(d / s[0]), s[1]) for s in slots[f]]))
    json.dump(dts, open(ROT_DIR / "dt.json", "w"))
    t0 = time.time()
    with mp.Pool(procs) as pool:
        for _ in pool.imap_unordered(rot_job, jobs):
            pass
    fill = np.mean([[x is not None for x in v] for v in dts.values()], axis=0).round(3).tolist()
    print(f"rotation slots carried for {len(jobs)} photographs in {(time.time() - t0) / 60:.1f} min; "
          f"share filled per slot {fill}", flush=True)


def rot_channels(stem, b0):
    """(20, 1024, 1024) float16: per slot contrast, "there", gap/84 h; then vx, vy (px/h at 2048) / 10."""
    import kfold as KF
    if "dt" not in _ROT:
        _ROT["dt"] = json.load(open(ROT_DIR / "dt.json"))
        _ROT["grid"] = _half_grid()
    a = np.load(ROT_DIR / f"{stem}.npy")
    ch = []
    for k, dt in enumerate(_ROT["dt"][stem]):
        there = (a[k] > 0).astype(np.float32)
        ch += [a[k].astype(np.float32) / 100.0, there, there * ((dt or 0.0) / 84.0)]
    pts, on = _ROT["grid"]
    y1, x1, _ = KF.carry(pts, b0, b0, 1.0 / 24)
    ch += [np.where(on, (x1 - pts[:, 1]).reshape(1024, 1024) / 10.0, 0.0),
           np.where(on, (y1 - pts[:, 0]).reshape(1024, 1024) / 10.0, 0.0)]
    return torch.from_numpy(np.stack(ch).astype(np.float16))


class MetaFrames(run.Frames):
    """Fold 0's training photographs, augmented as the U-Nets' `full` arm, with the photograph's columns."""

    def __init__(self, files):
        super().__init__(len(files), "full", b0=[run.b0_deg(f) for f in files])
        self.stems = [Path(f).stem for f in files]

    def __getitem__(self, i):
        x, y, g = super().__getitem__(i)
        return x, y, g, meta_vec(self.stems[i], x[0], g[4])


class TableFrames(MetaFrames):
    """The user's table (11-14): no rotation or mirror; contrast, sharpness or brightness alone or all three
    (pixel_nb.augment, the RF table's copies); the 35 columns of the augmented photograph as extra channels;
    with rot, also the rotation channels (the other photographs are not augmented)."""

    def __init__(self, files, rot=False):
        super().__init__(files)
        self.rot = rot

    def __getitem__(self, i):
        import pixel_nb as NB
        if self.img is None:
            self.img = np.load(run.CACHE / "train_img.npy", mmap_mode="r")
            self.msk = np.load(run.CACHE / "train_msk.npy", mmap_mode="r")
        im, ma = np.array(self.img[i]), np.array(self.msk[i])
        rng = np.random.default_rng(random.getrandbits(32))
        im = NB.augment(im, NB.VERSIONS[int(rng.integers(len(NB.VERSIONS)))], rng)
        g = torch.tensor([1.0, 0.0, 0.0, 1.0, self.b0[i]], dtype=torch.float32)
        x = run.to_input(im)[None]
        tab = table(im, self.stems[i])
        if self.rot:
            tab = torch.cat([tab, rot_channels(self.stems[i], self.b0[i])])
        return x, torch.from_numpy(ma).float().div_(255.0)[None], g, meta_vec(self.stems[i], x[0], self.b0[i]), tab


class MetaPseudo(torch.utils.data.Dataset):
    """Unlabelled frames with the ensemble's map as the target, augmented the same way."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, pl = self.items[i]
        im = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        ma = cv2.imread(str(pl), cv2.IMREAD_GRAYSCALE)
        run._AUG_S = np.eye(2)
        im, ma = run.augment_full(im, ma)
        b0 = run.b0_deg(Path(img).name)
        g = torch.tensor([*run._AUG_S.ravel(), b0], dtype=torch.float32)
        x = run.to_input(im)[None]
        return x, torch.from_numpy(ma).float().div_(255.0)[None], g, meta_vec(Path(img).stem, x[0], b0)


# ---------------------------------------------------------------- the network

def _cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Pair(nn.Module):
    """Two dilated 3x3 layers with a residual; FiLM scales and shifts both after BatchNorm."""

    def __init__(self, w, d):
        super().__init__()
        self.c1 = nn.Conv2d(w, w, 3, padding=d, dilation=d, bias=False)
        self.c2 = nn.Conv2d(w, w, 3, padding=d, dilation=d, bias=False)
        self.b1, self.b2 = nn.BatchNorm2d(w), nn.BatchNorm2d(w)

    def forward(self, x, film):
        g1, s1, g2, s2 = film[:, :, None, None].to(x.dtype).chunk(4, 1)
        h = F.relu(self.b1(self.c1(x)) * (1 + g1) + s1)
        return F.relu(x + self.b2(self.c2(h)) * (1 + g2) + s2)


class PixNet(nn.Module):
    IN_CH = 5                    # x, x relative to the disk, sin latitude, longitude, mu

    def __init__(self, width=WIDTH, dil=DILATIONS, extra=0):
        super().__init__()
        self.extra = extra                            # the table's channels (arm `table`), joined at half resolution
        self.stem = _cbr(self.IN_CH + extra, width)
        self.blocks = nn.ModuleList([Pair(width, d) for d in dil])
        self.film = nn.Sequential(nn.Linear(N_META, 128), nn.ReLU(inplace=True), nn.Linear(128, 4 * width * len(dil)))
        nn.init.zeros_(self.film[2].weight)           # starts as a plain network
        nn.init.zeros_(self.film[2].bias)
        self.mlp = nn.Sequential(nn.Conv2d(width, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                 nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.edge = nn.Sequential(_cbr(1, 16), _cbr(16, 16))
        self.out = nn.Sequential(_cbr(128 + 16 + 1, 32), nn.Conv2d(32, 1, 1))
        nn.init.constant_(self.out[1].bias, -4.0)
        self.ckpt = False

    def forward(self, x, geo, meta, tab=None):
        rel = (x - meta[:, MED, None, None, None]) / meta[:, SPREAD, None, None, None].clamp_min(0.05)
        z = F.avg_pool2d(torch.cat([x, rel, geo], 1), 2)
        if self.extra:
            z = torch.cat([z, tab.to(z.dtype)], 1)
        h = self.stem(z)
        for blk, f in zip(self.blocks, self.film(meta).chunk(len(self.blocks), 1)):
            if self.ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, f, use_reentrant=False)
            else:
                h = blk(h, f)
        h = F.interpolate(self.mlp(h), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([h, self.edge(x), x], 1))


# ---------------------------------------------------------------- training and scoring

def crops(x, y, geo, valid, tab=None):
    """One random CROP window per frame; the loss ignores its outer Q.MARGIN. tab (CPU, half resolution) is
    cut at the same window and only then moved to the card."""
    B, _, H, W = x.shape
    inner = torch.zeros(1, CROP, CROP, device=x.device)
    inner[:, Q.MARGIN:CROP - Q.MARGIN, Q.MARGIN:CROP - Q.MARGIN] = 1.0
    out = [[], [], [], [], []]
    for b in range(B):
        y0, x0 = random.randint(0, H - CROP) // 2 * 2, random.randint(0, W - CROP) // 2 * 2
        win = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        out[0].append(x[b, :, win[0], win[1]])
        out[1].append(y[b, :, win[0], win[1]])
        out[2].append(geo[b, :, win[0], win[1]])
        out[3].append(valid[:, win[0], win[1]] * inner)
        if tab is not None:
            out[4].append(tab[b, :, y0 // 2:(y0 + CROP) // 2, x0 // 2:(x0 + CROP) // 2])
    return [torch.stack(o) for o in out[:4]] + [torch.stack(out[4]).to(x.device) if tab is not None else None]


@torch.no_grad()
def predict(model, im_u8, stem, device, ondisk):
    b0 = run.b0_deg(stem)
    tab = None
    if getattr(model, "extra", 0):
        tab = table(im_u8, stem)
        if model.extra > TAB_N:
            tab = torch.cat([tab, rot_channels(stem, b0)])
        tab = tab[None].to(device)
    x = run.to_input(im_u8)
    meta = meta_vec(stem, x, b0)[None].to(device)
    geo = run.geo_channels(torch.tensor([[1.0, 0.0, 0.0, 1.0, b0]]), device)
    x = x[None, None].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        if not TILE:
            p = torch.sigmoid(model(x, geo, meta, tab).float())[0, 0]
        else:                                    # the laptop: a 2048 px forward pass needs > 5.4 GB
            H = x.shape[-1]
            p = torch.empty((H, H), device=device)
            for y0 in range(0, H, TILE):
                for x0 in range(0, H, TILE):
                    a0, a1 = max(y0 - TILE_M, 0), min(y0 + TILE + TILE_M, H)
                    c0, c1 = max(x0 - TILE_M, 0), min(x0 + TILE + TILE_M, H)
                    tt = tab[..., a0 // 2:a1 // 2, c0 // 2:c1 // 2] if tab is not None else None
                    o = torch.sigmoid(model(x[..., a0:a1, c0:c1], geo[..., a0:a1, c0:c1], meta, tt).float())[0, 0]
                    p[y0:y0 + TILE, x0:x0 + TILE] = o[y0 - a0:y0 - a0 + TILE, x0 - c0:x0 - c0 + TILE]
    return (p * ondisk * 255.0).round().byte().cpu().numpy()


def val_maps(model, device, ondisk, files, img_dir, gov=None):
    """Full-frame maps. On the laptop (11-12) the training's cached blocks plus a 2048 px forward pass
    overflowed the 6 GB card; Windows then moved memory to system RAM ("sysmem fallback"): the GPU read
    100 %, the validation took 30 min instead of ~1, and the next epoch never finished. So the cache is
    emptied around it, and each frame runs inside the governor like a training step."""
    import contextlib
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = []
    for f in files:
        with (gov.step() if gov is not None else contextlib.nullcontext()):
            out.append(predict(model, cv2.imread(str(Path(img_dir) / f), cv2.IMREAD_GRAYSCALE), Path(f).stem,
                               device, ondisk))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    return out


def epoch_loop(model, opt, sched, scaler, loader, device, gov, tag, stage, ep):
    wdisk = torch.from_numpy(run.disk_weight()).to(device)[None]
    t0, tot, n = time.time(), 0.0, 0
    for batch in loader:
        x, y, g, meta = batch[:4]
        tab = batch[4] if len(batch) > 4 else None          # arm `table`: stays on the CPU until cropped
        x, y, meta = x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta.to(device)
        geo = run.geo_channels(g, device)
        xc, yc, gc_, vc, tc = crops(x, y, geo, wdisk, tab)
        with gov.step():
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(xc, gc_, meta, tc)
            loss = Q.loss_fn(logit.float(), yc, vc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
        tot += loss.item(); n += 1
    print(f"[{tag}] {stage} epoch {ep}: loss {tot / max(n, 1):.4f}, {n} steps, {(time.time() - t0) / 60:.1f} min",
          flush=True)


def arm_main(arm):
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    run.set_fold(0)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = __import__("pickle").load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_files = meta["val_files"][:4] if SMOKE else meta["val_files"]
    if SMOKE:
        gt = [g for g in gt if g["frame"] < 4]
    tr_dir = run.find_data()[1]
    ondisk = Q.disk(device)
    gov = Q.governor()
    if LOCAL and device.type == "cuda":
        # past this share PyTorch raises out-of-memory instead of letting Windows spill to system RAM
        torch.cuda.set_per_process_memory_fraction(0.9)

    if arm in ("table", "tablerot"):
        tab_norm()                    # once, before the workers start
    extra = {"table": TAB_N, "tablerot": TAB_N + ROT_N}.get(arm, 0)
    model = PixNet(extra=extra).to(device)   # NCHW: channels_last was 5x slower (laptop probe, 11-12)
    model.ckpt = device.type == "cuda"
    print(f"[{arm}] PixNet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, "
          f"{2 * len(DILATIONS)} body layers, width {WIDTH}", flush=True)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    t0 = time.time()
    elapsed = [0.0]                  # hours spent before a restart, from the resume file

    def spent():
        return elapsed[0] + time.time() - t0

    def stage(ds, epochs, name, val_at=(), lr=LR, best0=None):
        # the table costs ~1.5 s of CPU per photograph, but on the laptop 3 workers ran out of committed memory
        # ("bad allocation", CUDA OOM with 4 GB free; 11-14: 7.5 GB of the 37 GB commit limit was free)
        table_arm = isinstance(ds, TableFrames)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                             num_workers=3 if run.ON_KAGGLE else 1, drop_last=True,
                                             persistent_workers=True, worker_init_fn=_one_thread if table_arm else None)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total = max(1, epochs * len(loader))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        best, start = best0 or (-1.0, None, 0), 1
        # the laptop faults at random under load: every epoch is saved, and a restart carries on from it
        ck_path = WORK / f"resume_{arm}_{name}.pt"
        if ck_path.exists():
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            best, start = ck["best"], ck["epoch"] + 1
            elapsed[0] = ck["elapsed"]
            print(f"[{arm}] {name}: resumed after epoch {ck['epoch']} (best {best[0]:.4f} at {best[2]})", flush=True)
        for ep in range(start, epochs + 1):
            t_ep = time.time()
            epoch_loop(model, opt, sched, scaler, loader, device, gov, arm, name, ep)
            late = spent() + (time.time() - t_ep) > BUDGET_H * 3600     # the next epoch would not fit
            if ep in val_at or (val_at and late):
                r = Q.score(val_maps(model, device, ondisk, val_files, tr_dir, gov), gt, quick=True)[0]
                print(f"[{arm}] {name} epoch {ep}: val PQ {r['PQ']:.4f} at t {r['thresh']} merge {r['merge']} "
                      f"area {r['min_area']}", flush=True)
                if r["PQ"] > best[0]:
                    best = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), best=best, epoch=ep, elapsed=spent()), ck_path)
            if late:
                print(f"[{arm}] {name}: time budget reached after epoch {ep}", flush=True)
                break
        return best

    if arm == "pre":
        items = Q.pseudo_labels(device)
        stage(MetaPseudo(items), 1, "pretrain")
        print(f"[{arm}] pretrained on {len(items)} frames in {(time.time() - t0) / 60:.0f} min", flush=True)
    files = meta["train_files"][:6] if SMOKE else meta["train_files"]
    if arm == "cont":
        model.load_state_dict(torch.load(CONT_INIT, map_location=device))
        r = Q.score(val_maps(model, device, ondisk, val_files, tr_dir, gov), gt, quick=True)[0]
        print(f"[{arm}] start (Kaggle scratch, epoch 30): quick val PQ {r['PQ']:.4f}", flush=True)
        best0 = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
        best_pq, best_sd, best_ep = stage(MetaFrames(files), CONT_EPOCHS, "train", CONT_VAL_AT, CONT_LR, best0)
    else:
        ds = TableFrames(files, rot=arm == "tablerot") if arm in ("table", "tablerot") else MetaFrames(files)
        best_pq, best_sd, best_ep = stage(ds, EPOCHS, "train", VAL_AT)
    model.load_state_dict(best_sd)
    print(f"[{arm}] best epoch {best_ep} (quick val PQ {best_pq:.4f}), {spent() / 60:.0f} min", flush=True)

    tag = f"pixnet_{arm}"
    root = WORK / f"dump_{tag}"
    te_dir = Path(meta["test_dir"])
    test_files = meta["test_files"][:2] if SMOKE else meta["test_files"]
    probs = val_maps(model, device, ondisk, val_files, tr_dir, gov)
    (root / "val_plain").mkdir(parents=True, exist_ok=True)
    (root / "test_plain").mkdir(parents=True, exist_ok=True)
    for f, p in zip(val_files, probs):
        cv2.imwrite(str(root / "val_plain" / f"{Path(f).stem}.png"), p)
    for f, p in zip(test_files, val_maps(model, device, ondisk, test_files, te_dir, gov)):
        cv2.imwrite(str(root / "test_plain" / f"{Path(f).stem}.png"), p)
    model.eval()
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    grid = Q.score(probs, gt, quick=False)
    res = dict(arm=arm, params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2), best_epoch=best_ep,
               quick_val_pq=best_pq, best=grid[0], top5=grid[:5], minutes=round(spent() / 60, 1), smoke=SMOKE,
               where="laptop RTX 2060" if LOCAL else "Kaggle T4", batch=BATCH,
               epochs=f"30 on Kaggle + {CONT_EPOCHS} at lr {CONT_LR}" if arm == "cont" else EPOCHS)
    json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    torch.save({k: v.half() for k, v in best_sd.items()}, WORK / f"{tag}.pt")
    print(f"[{arm}] fold 0 val PQ {grid[0]['PQ']:.4f} (SQ {grid[0]['SQ']:.3f}, RQ {grid[0]['RQ']:.3f}) at "
          f"t {grid[0]['thresh']} merge {grid[0]['merge']} area {grid[0]['min_area']}", flush=True)


def kaggle_main():
    t0 = time.time()
    run.set_fold(0)
    run.prepare(SMOKE, with_train=True)       # once, before the two arms share the cache
    print(f"fold 0 cache ready in {time.time() - t0:.0f}s", flush=True)
    if "tablerot" in ARMS:
        rot_prepare(json.load(open(run.CACHE / "meta.json")), os.cpu_count() or 4)
    if any(a in ("table", "tablerot") for a in ARMS):
        tab_norm()                            # written once, read by both arms
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(WORK / f"log_pixnet_{arm}.txt", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (WORK / f"log_pixnet_{arm}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][-160:] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n"
              + (WORK / f"log_pixnet_{arm}.txt").read_text(errors="replace")[-3000:], flush=True)


def local_main(arm="scratch"):
    """The laptop: one arm on the one card; the fold-0 cache is built once (CPU, a few minutes)."""
    run.set_fold(0)
    meta = run.CACHE / "meta.json"           # written last: a crash halfway leaves none, and the cache is rebuilt
    if not meta.exists() or len(json.load(open(meta))["train_files"]) < (1 if SMOKE else 500):
        run.prepare(SMOKE, with_train=True)
    arm_main(arm)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ARMS + ("cont", "table", "tablerot"):
        local_main(sys.argv[1]) if LOCAL else arm_main(sys.argv[1])
    else:
        kaggle_main()
