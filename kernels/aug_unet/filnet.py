"""FilNet: a filament network built from nothing, Claude's own design (2026-11-16, the user's request).

What goes in, and why (how one finds a filament by eye, and what the from-scratch literature says):

  input    A filament is "darker than the quiet Sun around it", not "dark": so the frame is first turned into
           contrast = brightness / the quiet Sun's brightness at the same mu (limb darkening removed, measured
           on the photograph itself), and into "darker than its surroundings" at two sizes (~30 px: the body
           against its channel; ~130 px: against the region). Fixed arithmetic, no weights. With the raw
           brightness and the position on the sphere (sin latitude, longitude, mu) that is 7 channels.
  network  A plain residual U-Net, six sizes (1, 1/2 ... 1/32), widths 24-384, ~18 M weights, no pretrained
           weights, no attention. Full resolution is kept (IoU > 0.5 on a 10 px wide filament is decided at
           its edge); 1/32 sees ~1000 px, enough to tell a filament from a sunspot group or a plage.
  training nnU-Net's recipe (Isensee et al. 2021), which is what makes from-scratch U-Nets match pretrained
           ones on a few hundred images: small windows (512 px) rather than whole frames, half of them centred
           on a filament, batch 12, so a window holds ~5x more filament pixels than the same area of a whole
           frame; loss at three sizes (deep supervision); an averaged copy of the weights (EMA) is the model.
  loss     BCE (pos_weight 10) + Tversky, soft target (share of experts), as every other model here.

Two arms, one per T4, fold 0, scored exactly as the U-Nets / PixNet / SolarUNet were:
  phys  the 7 channels above
  raw   control: brightness + position only (4 channels), everything else identical
Each writes dump_filnet_<arm>.tar (val_plain, val_d4, test_plain, test_d4), filnet_<arm>.pt, filnet_<arm>.json.
    FIL_FILNET_SMOKE=1 python kernels/aug_unet/filnet.py phys      # the laptop, a few frames
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")            # before run.py hides the card
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import copy, json, math, pickle, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- torch before cv2
from run import torch, cv2, np, nn, F  # noqa: E402
import qwen_seg as Q  # noqa: E402  -- scoring, governor, disk

SMOKE = os.environ.get("FIL_FILNET_SMOKE") == "1"
ARMS = tuple(os.environ.get("FIL_FILNET_ARMS", "phys,raw").split(","))
LOCAL = not run.ON_KAGGLE
if LOCAL:                        # the full fold-0 cache (out/cache holds a smoke one); a smoke run reads a few frames of it
    run.CACHE_ROOT = HERE / "out" / "cache_full"
    run.CACHE = run.CACHE_ROOT
WIDTHS = (24, 48, 96, 192, 320, 384)
DEPTH = (1, 2, 2, 2, 2, 2)
CROP = 384 if (SMOKE and LOCAL) else 512
FRAMES = 2 if LOCAL else 4       # photographs per step
PER = 2 if LOCAL else 3          # windows per photograph: batch 12 on a T4
MARGIN = 16                      # the loss ignores a window's outer rim (zero padding there, not Sun)
EPOCHS = int(os.environ.get("FIL_FILNET_SMOKE_EP", "1")) if SMOKE else 400     # an upper bound: after epoch 1 the schedule is fitted to the time budget
BUDGET_H = 0.3 if SMOKE else float(os.environ.get("FIL_FILNET_BUDGET", "4.6"))
LR, WD = 1e-3, 1e-2
ALPHA, BETA = 0.6, 0.4
DEEP = (1.0, 0.5, 0.25)          # loss weights at 1, 1/2, 1/4
WORK = run.WORK


# ---------------------------------------------------------------- the fixed front end (no weights)

def _ring_mean(v, w, idx, nb):
    s = torch.zeros(v.shape[0], nb, device=v.device).scatter_add_(1, idx, (v * w).flatten(1))
    n = torch.zeros(v.shape[0], nb, device=v.device).scatter_add_(1, idx, w.flatten(1))
    return s / n.clamp(min=1.0), n


def _local(c, on, pool, k):
    """c minus its mean over ~pool*k px, the mean taken over disk pixels only."""
    a, b = F.avg_pool2d(c * on, pool), F.avg_pool2d(on, pool)
    a = F.avg_pool2d(a, k, 1, k // 2, count_include_pad=False)
    b = F.avg_pool2d(b, k, 1, k // 2, count_include_pad=False)
    m = F.interpolate(a / b.clamp(min=0.05), size=c.shape[-2:], mode="bilinear", align_corners=False)
    return (c - m) * on


@torch.no_grad()
def front(x, geo, arm):
    """x (B,1,H,W) in run.to_input units, geo (B,3,H,W) -> the network's input channels, whole frames."""
    if arm == "raw":
        return torch.cat([x, geo], 1)
    x, geo = x.float(), geo.float()
    v = x * 0.25 + 0.5                                   # brightness 0..1
    mu = geo[:, 2:3]
    on = (mu > 0.0).float()
    nb = 48
    idx = (mu.clamp(0.0, 0.9999) * nb).long().flatten(1)
    p1, _ = _ring_mean(v, on, idx, nb)                   # mean brightness in 48 rings of mu
    q1 = p1.gather(1, idx).view_as(v)
    quiet = on * ((v / q1.clamp(min=0.02) - 1.0).abs() < 0.12).float()   # again without filaments, spots, plage
    p2, n2 = _ring_mean(v, quiet, idx, nb)
    p2 = torch.where(n2 > 20, p2, p1)
    c = (v / p2.gather(1, idx).view_as(v).clamp(min=0.02) - 1.0).clamp(-1.0, 1.0) * on
    return torch.cat([x, 5.0 * c, 8.0 * _local(c, on, 4, 7), 6.0 * _local(c, on, 8, 15), geo], 1)


IN_CH = {"phys": 7, "raw": 4}


# ---------------------------------------------------------------- the network

class Res(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        nn.init.zeros_(self.b2.weight)                   # each block starts as the identity

    def forward(self, x):
        h = F.silu(self.b1(self.c1(x)), inplace=True)
        h = self.b2(self.c2(h))
        return F.silu(h + (x if self.skip is None else self.skip(x)), inplace=True)


class FilNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        w = WIDTHS
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w[0], 3, padding=1, bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(True))
        self.enc = nn.ModuleList([nn.Sequential(*[Res(w[i], w[i]) for _ in range(DEPTH[i])]) for i in range(len(w))])
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(w[i], w[i + 1], 3, stride=2, padding=1, bias=False),
                                                 nn.BatchNorm2d(w[i + 1]), nn.SiLU(True)) for i in range(len(w) - 1)])
        self.dec = nn.ModuleList([Res(w[i + 1] + w[i], w[i]) for i in range(len(w) - 1)])
        self.heads = nn.ModuleList([nn.Conv2d(w[i], 1, 1) for i in range(len(DEEP))])
        for h in self.heads:
            nn.init.constant_(h.bias, -3.0)              # filaments are ~0.5% of the disk

    def forward(self, x, deep=False, feats=False):
        """feats: also the feature maps at every size, {"tok": 1/32, "d128": 1/16, ... "d2048": 1/1}, for the
        filament head (filnet_head.py)."""
        skips = []
        h = self.stem(x)
        for i, e in enumerate(self.enc):
            h = e(h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        fm = {"tok": h} if feats else None
        outs = [None] * len(DEEP)
        for i in range(len(self.dec) - 1, -1, -1):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.dec[i](torch.cat([h, skips[i]], 1))
            if feats:
                fm[f"d{run.FULL >> i}"] = h
            if i < len(DEEP) and (deep or i == 0):
                outs[i] = self.heads[i](h)
        out = outs if deep else outs[0]
        return (out, fm) if feats else out


# ---------------------------------------------------------------- loss

def loss_one(logit, y, w):
    """BCE + Tversky over the whole batch of windows (many hold no filament). Targets soft."""
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(10.0, device=logit.device),
                                             reduction="none")
    bce = (bce * w).sum() / w.sum().clamp(min=1.0)
    p = torch.sigmoid(logit)
    tp, fp, fn = (p * y * w).sum(), (p * (1 - y) * w).sum(), ((1 - p) * y * w).sum()
    return bce + 1.0 - (tp + 1.0) / (tp + ALPHA * fp + BETA * fn + 1.0)


def loss_fn(outs, y, w):
    tot = 0.0
    for i, (o, a) in enumerate(zip(outs, DEEP)):
        yi, wi = (y, w) if i == 0 else (F.avg_pool2d(y, 2 ** i), F.avg_pool2d(w, 2 ** i))
        tot = tot + a * loss_one(o.float(), yi, wi)
    return tot / sum(DEEP)


# ---------------------------------------------------------------- windows

def windows(inp, y, w):
    """PER windows of CROP px from each frame: every other one centred near a filament pixel, the rest anywhere
    on the disk. inp (B,C,H,W), y (B,1,H,W), w (1,1,H,W) disk weight."""
    B, _, H, W = inp.shape
    rim = torch.zeros(1, CROP, CROP, device=inp.device)
    rim[:, MARGIN:CROP - MARGIN, MARGIN:CROP - MARGIN] = 1.0
    xs, ys, ws = [], [], []
    for b in range(B):
        pos = torch.nonzero(y[b, 0, ::8, ::8] > 0.3)
        for k in range(PER):
            if k % 2 == 0 and len(pos):
                cy, cx = (pos[random.randrange(len(pos))] * 8).tolist()
                cy += random.randint(-CROP // 3, CROP // 3); cx += random.randint(-CROP // 3, CROP // 3)
            else:
                r, a = run.R_DISK * math.sqrt(random.random()), random.uniform(0, 2 * math.pi)
                cy, cx = int(run.CY + r * math.sin(a)), int(run.CX + r * math.cos(a))
            y0 = min(max(cy - CROP // 2, 0), H - CROP) // 32 * 32
            x0 = min(max(cx - CROP // 2, 0), W - CROP) // 32 * 32
            xs.append(inp[b, :, y0:y0 + CROP, x0:x0 + CROP])
            ys.append(y[b, :, y0:y0 + CROP, x0:x0 + CROP])
            ws.append(w[0, :, y0:y0 + CROP, x0:x0 + CROP] * rim)
    return torch.stack(xs).contiguous(memory_format=torch.channels_last), torch.stack(ys), torch.stack(ws)


# ---------------------------------------------------------------- prediction

def views():
    """(transpose?, flipped dims, the view's output-to-source map a, b, c, d in image axes) -- as sunet_d4.py."""
    for tr in (False, True):
        for dims in ([], [3], [2], [2, 3]):
            su = -1.0 if 3 in dims else 1.0
            sv = -1.0 if 2 in dims else 1.0
            yield tr, dims, ((0.0, sv, su, 0.0) if tr else (su, 0.0, 0.0, sv))


@torch.no_grad()
def predict(model, arm, im_u8, stem, device, ondisk, d4=False, gov=None):
    """uint8 map; with d4 the pair (plain, mean of the 8 views)."""
    import contextlib
    b0 = run.b0_deg(stem)
    x = run.to_input(im_u8)[None, None].to(device)
    acc, plain = 0.0, None
    for tr, dims, g in views():
        with (gov.step() if gov is not None else contextlib.nullcontext()):
            xi = x.transpose(2, 3) if tr else x
            xi = torch.flip(xi, dims) if dims else xi
            geo = run.geo_channels(torch.tensor([[*g, b0]]), device)
            inp = front(xi.contiguous(), geo, arm).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p = torch.sigmoid(model(inp).float())
            p = torch.flip(p, dims) if dims else p
            p = p.transpose(2, 3) if tr else p
            if device.type == "cuda":
                torch.cuda.synchronize()
        if plain is None:
            plain = p
            if not d4:
                break
        acc = acc + p

    def u8(q):
        return (q[0, 0] * ondisk * 255.0).round().byte().cpu().numpy()
    return (u8(plain), u8(acc / 8)) if d4 else u8(plain)


def val_maps(model, arm, device, ondisk, files, img_dir, gov=None, d4=False):
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = [predict(model, arm, cv2.imread(str(Path(img_dir) / f), cv2.IMREAD_GRAYSCALE), Path(f).stem, device,
                   ondisk, d4, gov) for f in files]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


class FoldFrames(run.Frames):
    """run.Frames that carries its fold's cache folder: on Windows a loader worker is a fresh process, in which
    run.CACHE is back at its default, so the frames would silently come from another fold."""

    def __init__(self, files, cache):
        super().__init__(len(files), "full", b0=[run.b0_deg(f) for f in files])
        self.cache = str(cache)

    def __getitem__(self, i):
        if self.img is None:
            self.img = np.load(Path(self.cache) / "train_img.npy", mmap_mode="r")
            self.msk = np.load(Path(self.cache) / "train_msk.npy", mmap_mode="r")
            if LOCAL:
                cv2.setNumThreads(2)                    # the laptop faults under sustained CPU load
        return super().__getitem__(i)


# ---------------------------------------------------------------- one arm

def arm_main(arm, fold=0):
    random.seed(fold); torch.manual_seed(fold); np.random.seed(fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = not LOCAL
    run.set_fold(fold)
    arm_in, arm = arm, (arm if fold == 0 else f"{arm}_f{fold}")      # arm: the name in logs and file names
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_files = meta["val_files"][:4] if SMOKE else meta["val_files"]
    if SMOKE:
        gt = [g for g in gt if g["frame"] < 4]
    tr_dir = run.find_data()[1]
    ondisk = Q.disk(device)
    gov = Q.governor()
    if LOCAL and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)

    model = FilNet(IN_CH[arm_in]).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{arm}] FilNet: {n_par / 1e6:.2f} M parameters, {IN_CH[arm_in]} input channels, windows {CROP} px, "
          f"batch {FRAMES * PER}", flush=True)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    wdisk = torch.from_numpy(run.disk_weight()).to(device)[None, None]
    t0 = time.time()

    def spent():
        return time.time() - t0

    files = meta["train_files"][:int(os.environ.get("FIL_FILNET_SMOKE_N", "6"))] if SMOKE else meta["train_files"]
    ds = FoldFrames(files, run.CACHE)
    nw = 2 if run.ON_KAGGLE else (0 if SMOKE else 1)
    loader = torch.utils.data.DataLoader(ds, batch_size=FRAMES, shuffle=True, drop_last=True, num_workers=nw,
                                         pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    plan = [max(1, EPOCHS * len(loader))]
    warm = 2 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / plan[0])))))
    best, n_ep, val_at, step = (-1.0, None, 0), EPOCHS, set(), 0
    for ep in range(1, EPOCHS + 1):
        te, tot, n = time.time(), 0.0, 0
        model.train()
        for x, y, g in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with gov.step():
                inp, yc, wc = windows(front(x, run.geo_channels(g, device), arm_in), y, wdisk)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    outs = model(inp, deep=True)
                loss = loss_fn(outs, yc, wc)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                step += 1
                d = min(0.999, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            tot += loss.item(); n += 1
        dt = time.time() - te
        if ep == 1 or ep % 5 == 0 or ep in val_at:
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
            print(f"[{arm}] epoch {ep}: loss {tot / max(n, 1):.4f}, {n} steps, {dt / 60:.2f} min, {mem:.1f} GiB, "
                  f"{spent() / 3600:.2f} h", flush=True)
        if ep == 1:
            left = BUDGET_H * 3600 - spent() - (60 if SMOKE else 1500)   # validation passes, D4 maps, writing
            n_ep = max(1, min(EPOCHS, 1 + int(left / max(dt, 1.0))))
            plan[0] = n_ep * len(loader)
            val_at = {n_ep} | {max(1, int(round(n_ep * q))) for q in ((0.4, 0.6, 0.75, 0.9) if BUDGET_H > 4.3 else (0.7, 0.85))}
            print(f"[{arm}] {dt / 60:.2f} min per epoch -> {n_ep} epochs ({plan[0]} steps, "
                  f"{plan[0] * FRAMES * PER} windows), validation at {sorted(val_at)}", flush=True)
        if ep in val_at:
            r = Q.score(val_maps(ema, arm_in, device, ondisk, val_files, tr_dir, gov), gt, quick=True)[0]
            print(f"[{arm}] epoch {ep}: val PQ {r['PQ']:.4f} (SQ {r['SQ']:.3f}, RQ {r['RQ']:.3f}) at t {r['thresh']} "
                  f"merge {r['merge']} area {r['min_area']}", flush=True)
            if r["PQ"] > best[0]:
                best = (r["PQ"], {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}, ep)
                torch.save({k: v.half() if v.is_floating_point() else v for k, v in best[1].items()},
                           WORK / f"filnet_{arm}.pt")
        if ep >= n_ep:
            break
    best_pq, best_sd, best_ep = best
    ema.load_state_dict(best_sd)
    print(f"[{arm}] best epoch {best_ep} (quick val PQ {best_pq:.4f}), {spent() / 60:.0f} min", flush=True)

    tag = f"filnet_{arm}"
    root = WORK / f"dump_{tag}"
    te_dir = Path(meta["test_dir"])
    test_files = meta["test_files"][:2] if SMOKE else meta["test_files"]
    val = val_maps(ema, arm_in, device, ondisk, val_files, tr_dir, gov, d4=True)
    for sub in ("val_plain", "val_d4", "test_plain", "test_d4"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for f, (p, q) in zip(val_files, val):
        cv2.imwrite(str(root / "val_plain" / f"{Path(f).stem}.png"), p)
        cv2.imwrite(str(root / "val_d4" / f"{Path(f).stem}.png"), q)
    for f, (p, q) in zip(test_files, val_maps(ema, arm_in, device, ondisk, test_files, te_dir, gov, d4=True)):
        cv2.imwrite(str(root / "test_plain" / f"{Path(f).stem}.png"), p)
        cv2.imwrite(str(root / "test_d4" / f"{Path(f).stem}.png"), q)
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    print(f"[{arm}] maps written, {spent() / 60:.0f} min", flush=True)

    res = dict(arm=arm, params_M=round(n_par / 1e6, 2), best_epoch=best_ep, epochs=n_ep, quick_val_pq=best_pq,
               minutes=round(spent() / 60, 1), smoke=SMOKE, where="laptop RTX 2060" if LOCAL else "Kaggle T4",
               batch=FRAMES * PER, crop=CROP, steps=step)
    for name, maps in (("plain", [p for p, _ in val]), ("d4", [q for _, q in val])):
        grid = Q.score(maps, gt, quick=False)
        res[name] = dict(best=grid[0], top5=grid[:5])
        print(f"[{arm}] fold 0 val PQ, {name}: {grid[0]['PQ']:.4f} (SQ {grid[0]['SQ']:.3f}, RQ {grid[0]['RQ']:.3f}) at "
              f"t {grid[0]['thresh']} merge {grid[0]['merge']} area {grid[0]['min_area']}", flush=True)
        json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)

    im = cv2.imread(str(te_dir / test_files[0]), cv2.IMREAD_GRAYSCALE)
    ts = []
    for i in range(7):
        t1 = time.perf_counter()
        predict(ema, arm_in, im, Path(test_files[0]).stem, device, ondisk)
        ts.append(time.perf_counter() - t1)
    res["sec_per_photo"] = round(float(np.median(ts[2:])), 3)
    json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    print(f"[{arm}] {res['sec_per_photo']} s per photograph (one pass)", flush=True)


def fold_main(arm, fold):
    """One further fold, in a process of its own: its cache, the training, then the cache is dropped."""
    run.set_fold(fold)
    run.prepare(SMOKE, with_train=True)
    arm_main(arm, fold)
    if run.ON_KAGGLE and fold:
        shutil.rmtree(run.CACHE, ignore_errors=True)


def kaggle_folds(spec):
    """FIL_FILNET_FOLDS = "1;2" or "1,3;2,4": per T4 (separated by ;) the folds it trains one after another."""
    import threading
    t0 = time.time()
    arm = ARMS[0]

    def chain(gpu, folds):
        for k in folds:
            with open(WORK / f"log_filnet_{arm}_f{k}.txt", "w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), arm, str(k)],
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)

    chains = [[int(k) for k in part.split(",")] for part in spec.split(";")]
    threads = [threading.Thread(target=chain, args=(i, c)) for i, c in enumerate(chains)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(600)
        for c in chains:
            for k in c:
                f = WORK / f"log_filnet_{arm}_f{k}.txt"
                lines = [x for x in f.read_text(errors="replace").splitlines() if x.strip()] if f.exists() else []
                if lines:
                    print(f"  {(time.time() - t0) / 3600:.2f} h  fold {k}: {lines[-1][-170:]}", flush=True)
    for c in chains:
        for k in c:
            print(f"\n===== fold {k} =====\n" + (WORK / f"log_filnet_{arm}_f{k}.txt").read_text(errors="replace")[-5000:],
                  flush=True)


def kaggle_main():
    if os.environ.get("FIL_FILNET_FOLDS"):
        return kaggle_folds(os.environ["FIL_FILNET_FOLDS"])
    t0 = time.time()
    run.set_fold(0)
    run.prepare(SMOKE, with_train=True)
    print(f"fold 0 cache ready in {time.time() - t0:.0f}s", flush=True)
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(WORK / f"log_filnet_{arm}.txt", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (WORK / f"log_filnet_{arm}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][-170:] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n"
              + (WORK / f"log_filnet_{arm}.txt").read_text(errors="replace")[-5000:], flush=True)


def local_main(arm):
    run.set_fold(0)
    run.CACHE = run.CACHE_ROOT
    meta = run.CACHE / "meta.json"
    if not meta.exists() or len(json.load(open(meta))["train_files"]) < (1 if SMOKE else 500):
        run.prepare(SMOKE, with_train=True)
    arm_main(arm)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in IN_CH:
        fold_main(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] in IN_CH:
        local_main(sys.argv[1]) if LOCAL else arm_main(sys.argv[1])
    else:
        kaggle_main()
