"""One compact network for the whole solar disk, built to the user's design (2026-11-15).

The user's spec: first a per-pixel analysis as in PixNet, then the frame halved and analysed again, attention
at the smallest size, then back up to full size; turning pixels into filaments as the model's last wing.

  input   2048 x 2048, the whole disk in one pass (no crops, no tiles): brightness, brightness relative to the
          disk's median and spread, sin(latitude), longitude from the central meridian, mu
  FiLM    the photograph's own 15 numbers (date, season, hour, station, B0, disk median and spread) scale and
          shift every block, as in PixNet
  left    2048 (16 ch, the per-pixel stage) -> 1024 (32) -> 512 (64) -> 256 (128) -> 128 (256)
  middle  64 x 64: 4 attention layers over all 4096 positions of the disk, positions given as the solar
          coordinates and image coordinates
  right   back up through 128, 256, 512, 1024 to 2048 with the left wing's maps joined at each size
  heads   (1) filament probability per pixel;
          (2) the filament head: for each candidate outline the network's own features, averaged inside it and
              in a ring around it, go to a learned selector that predicts the IoU an expert would give it.
              Over the whole pool of outlines (8 thresholds x 4 join radii) that picks each filament's own
              outline, which is where the fixed rules lose the most (izah.md par. 17: oracle 0.519 -> 0.578).

Loss: BCE (pos_weight 10) + Tversky (alpha 0.6 on false positives, beta 0.4 on misses), on the disk.
Two arms, one per T4, fold 0, validated exactly as the U-Nets, PixNet and the Qwen segmenter were:
  scratch  random weights, the 566 annotated photographs: the user's spec as stated
  pre      Claude's variant: first one pass over 2000 unlabelled GONG frames with the fold-0 U-Net ensemble's
           map as the target (worth +0.013 for the Qwen segmenter), then the same training
Each writes dump_sunet_<arm>.tar (val_plain, test_plain), sunet_<arm>.pt, sunet_<arm>.json and
stage2_<arm>.pkl (per validation photograph: candidates' 21 numbers, network features, IoU targets).
    FIL_SUNET_SMOKE=1 python kernels/aug_unet/solar_unet.py scratch      # the laptop, a few frames
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")            # before run.py hides the card
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # full frames fragment the cache
import json, math, pickle, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- torch before cv2
from run import torch, cv2, np, nn, F  # noqa: E402
import pixnet as PX  # noqa: E402  -- photograph columns, datasets, fold-0 cache paths (sets Q's settings too)
import qwen_seg as Q  # noqa: E402  -- pseudo-labels, scoring, governor
import candidate_pool as C  # noqa: E402  -- the outline pool, its 21 numbers, IoU targets, conflicts, selection

SMOKE = os.environ.get("FIL_SUNET_SMOKE") == "1"
ARMS = tuple(os.environ.get("FIL_SUNET_ARMS", "scratch,pre").split(","))
LOCAL = not run.ON_KAGGLE
WIDTHS = (16, 32, 64, 128, 256)  # 2048, 1024, 512, 256, 128
D_ATT, N_ATT, HEADS = 256, 4, 8  # the 64 x 64 middle
BATCH = 1 if LOCAL else 2
EPOCHS = 1 if SMOKE else 60      # an upper bound: after epoch 1 the schedule is fitted to the time budget
BUDGET_H = 0.4 if SMOKE else 5.0
N_PSEUDO = 6 if SMOKE else 2000
LR = 1e-3
ALPHA, BETA = 0.6, 0.4
WORK = run.WORK
Q.N_PSEUDO = PX.N_PSEUDO = N_PSEUDO
# every block and the head are recomputed in backward: with only the 2048-512 px blocks so, one 2048 px frame
# still needed > 5.4 GB on the laptop (smoke test, 11-15); this keeps batch 2 well inside a T4's 15 GB
CKPT_LEVELS = 5


# ---------------------------------------------------------------- the network

class Block(nn.Module):
    """Two 3x3 convolutions with a residual; FiLM scales and shifts both after BatchNorm."""

    def __init__(self, cin, cout, dil=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil, bias=False)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=dil, dilation=dil, bias=False)
        self.b1, self.b2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.n_film = 4 * cout

    def forward(self, x, film):
        a = self.b1(self.c1(x))
        # FiLM in the convolution's dtype (float16 under autocast), not the input's: the first block's input is
        # float32, and casting to it made every 2048 px activation float32 (twice the memory; smoke test, 11-15)
        g1, s1, g2, s2 = film[:, :, None, None].to(a.dtype).chunk(4, 1)
        h = F.relu(a * (1 + g1) + s1)
        return F.relu(self.skip(x).to(a.dtype) + self.b2(self.c2(h)) * (1 + g2) + s2)


class Attention(nn.Module):
    def __init__(self, d=D_ATT, heads=HEADS):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        for lin in (self.proj, self.mlp[2]):          # each layer starts as "pass through": a plain U-Net first
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, t):
        B, N, D = t.shape
        q, k, v = self.qkv(self.n1(t)).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        t = t + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return t + self.mlp(self.n2(t))


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


N_FOURIER = 8


class SolarUNet(nn.Module):
    IN_CH = 5

    def __init__(self, w=WIDTHS):
        super().__init__()
        self.enc = nn.ModuleList([nn.ModuleList([Block(self.IN_CH, w[0])])]
                                 + [nn.ModuleList([Block(w[i], w[i], 1), Block(w[i], w[i], 2 if i >= 3 else 1)])
                                    for i in range(1, len(w))])
        self.down = nn.ModuleList([_down(w[i], w[i + 1]) for i in range(len(w) - 1)] + [_down(w[-1], D_ATT)])
        self.pos = nn.Linear(3 + 4 * N_FOURIER, D_ATT)
        self.att = nn.ModuleList([Attention() for _ in range(N_ATT)])
        ups = [D_ATT] + list(w[::-1])                  # 256 (middle), 256, 128, 64, 32, 16
        self.dec = nn.ModuleList([Block(ups[i] + w[-1 - i], w[-1 - i]) for i in range(len(w))])
        self.head = nn.Sequential(nn.Conv2d(w[0] + 1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        nn.init.constant_(self.head[2].bias, -4.0)    # filaments are ~0.5% of the frame
        blocks = [b for lv in self.enc for b in lv] + list(self.dec)
        self.n_film = [b.n_film for b in blocks]
        self.film = nn.Sequential(nn.Linear(PX.N_META, 128), nn.ReLU(inplace=True), nn.Linear(128, sum(self.n_film)))
        nn.init.zeros_(self.film[2].weight)           # starts as a plain network; the photograph's numbers
        nn.init.zeros_(self.film[2].bias)             # enter only as far as they help
        self.ckpt = False

    def _run(self, blk, x, f, level):
        if self.ckpt and self.training and level < CKPT_LEVELS:
            return torch.utils.checkpoint.checkpoint(blk, x, f, use_reentrant=False)
        return blk(x, f)

    def _pos(self, geo, size):
        g = F.adaptive_avg_pool2d(geo.float(), size)                     # solar coordinates of each position
        c = torch.linspace(-1, 1, size, device=geo.device) * (run.FULL / 2 / run.R_DISK)
        v, u = torch.meshgrid(c, c, indexing="ij")
        fr = (2.0 ** torch.arange(N_FOURIER, device=geo.device)) * math.pi / 2
        four = torch.cat([torch.sin(u[None] * fr[:, None, None]), torch.cos(u[None] * fr[:, None, None]),
                          torch.sin(v[None] * fr[:, None, None]), torch.cos(v[None] * fr[:, None, None])])
        four = four[None].expand(g.shape[0], -1, -1, -1)
        return self.pos(torch.cat([g, four], 1).flatten(2).transpose(1, 2))

    @property
    def n_enc(self):
        return sum(len(lv) for lv in self.enc)

    def films(self, meta):
        f = list(self.film(meta).split(self.n_film, 1))
        return f[:self.n_enc], f[self.n_enc:]

    def encode(self, x, geo, meta, films):
        """The left wing and the 64 x 64 attention: the skips and the tokens (B, 4096, D)."""
        rel = (x - meta[:, PX.MED, None, None, None]) / meta[:, PX.SPREAD, None, None, None].clamp_min(0.05)
        films = iter(films)
        h, skips = torch.cat([x, rel, geo.to(x.dtype)], 1), []
        for i, lv in enumerate(self.enc):
            if i:
                h = self.down[i - 1](h)
            for blk in lv:
                h = self._run(blk, h, next(films), i)
            skips.append(h)
        h = self.down[-1](h)                                            # 64 x 64
        t = h.flatten(2).transpose(1, 2) + self._pos(geo, h.shape[-1]).to(h.dtype)
        for a in self.att:     # 8 heads x 4096 x 4096 weights per layer when the fused kernel is not available
            t = (torch.utils.checkpoint.checkpoint(a, t, use_reentrant=False) if self.ckpt and self.training
                 else a(t))
        return skips, t

    def forward(self, x, geo, meta, feats=False):
        fe, fd = self.films(meta)
        skips, t = self.encode(x, geo, meta, fe)
        return self.decode(skips, t, x, fd, feats)

    def decode(self, skips, t, x, films, feats=False):
        B, N, D = t.shape
        S = int(round(N ** 0.5))
        films = iter(films)
        h = t.transpose(1, 2).reshape(B, D, S, S)
        out = {"tok": h}
        for j, blk in enumerate(self.dec):
            s = skips[-1 - j]
            # autocast runs bilinear upsampling in float32: cast back, or every join is float32 (768 MB at 2048)
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False).to(s.dtype)
            h = self._run(blk, torch.cat([h, s], 1), next(films), len(self.dec) - 1 - j)
            out[f"d{s.shape[-1]}"] = h
        hx = torch.cat([h, x.to(h.dtype)], 1)              # x is float32: the join would otherwise promote
        if self.ckpt and self.training:
            logit = torch.utils.checkpoint.checkpoint(self.head, hx, use_reentrant=False)
        else:
            logit = self.head(hx)
        return (logit, out) if feats else logit


def loss_fn(logit, y, w):
    """BCE + Tversky on the disk (w: 1 on it, 0 off it, feathered at the limb). Targets are soft (expert share)."""
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(10.0, device=logit.device),
                                             reduction="none")
    bce = (bce * w).sum() / (w.sum() * logit.shape[0])
    p = torch.sigmoid(logit)
    tp = (p * y * w).sum((1, 2, 3))
    fp = (p * (1 - y) * w).sum((1, 2, 3))
    fn = ((1 - p) * y * w).sum((1, 2, 3))
    tv = 1.0 - (tp + 1.0) / (tp + ALPHA * fp + BETA * fn + 1.0)
    return bce + tv.mean()


# ---------------------------------------------------------------- prediction

def _inputs(im_u8, stem, device):
    b0 = run.b0_deg(stem)
    x = run.to_input(im_u8)
    meta = PX.meta_vec(stem, x, b0)[None].to(device)
    geo = run.geo_channels(torch.tensor([[1.0, 0.0, 0.0, 1.0, b0]]), device)
    return x[None, None].to(device), geo, meta


@torch.no_grad()
def predict(model, im_u8, stem, device, ondisk, feats=False):
    x, geo, meta = _inputs(im_u8, stem, device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(x, geo, meta, feats=feats)
    logit, fm = out if feats else (out, None)
    p = (torch.sigmoid(logit.float())[0, 0] * ondisk * 255.0).round().byte().cpu().numpy()
    return (p, fm) if feats else p


def val_maps(model, device, ondisk, files, img_dir, gov=None):
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


# ---------------------------------------------------------------- the filament head's data

LEVELS = (("d2048", 1), ("d1024", 2), ("d512", 4), ("d256", 8), ("d128", 16), ("tok", 32))
RING_LEVEL = ("d256", 8)


def pooled(fm, cands):
    """Per candidate: the network's features averaged inside the outline at every size, and at 256 px in a
    15 px ring around it. (n, 16+32+64+128+256+256+128) float16."""
    out = []
    flat = {k: fm[k][0].flatten(1) for k, _ in LEVELS}
    for c in cands:
        ys, xs = np.nonzero(c["m"])
        ys, xs = ys + c["y0"], xs + c["x0"]
        v = []
        for k, s in LEVELS:
            n = run.FULL // s
            idx = torch.from_numpy(np.unique((ys // s) * n + xs // s)).to(flat[k].device)
            v.append(flat[k][:, idx].float().mean(1))
        pad = 10
        u = np.pad(c["m"].astype(np.uint8), pad)
        ring = (cv2.dilate(u, C.RING) > 0) & ~u.astype(bool)
        ry, rx = np.nonzero(ring)
        ry, rx = np.clip(ry + c["y0"] - pad, 0, run.FULL - 1), np.clip(rx + c["x0"] - pad, 0, run.FULL - 1)
        k, s = RING_LEVEL
        idx = torch.from_numpy(np.unique((ry // s) * (run.FULL // s) + rx // s)).to(flat[k].device)
        v.append(flat[k][:, idx].float().mean(1))
        out.append(torch.cat(v).cpu().numpy())
    return np.array(out, np.float16) if out else np.zeros((0, sum(m for m in (16, 32, 64, 128, 256, 256, 128))), np.float16)


def stage2_frames(model, device, ondisk, files, img_dir, gt, gov=None):
    """For every validation photograph: the outline pool from this model's own map, the 21 numbers the
    submitted rule uses, the network features, each outline's IoU per expert set, and which outlines overlap."""
    import contextlib
    import collections
    by_frame = collections.defaultdict(list)
    for g in gt:
        by_frame[g["frame"]].append(g["crops"])
    model.eval()
    frames = []
    t0 = time.time()
    for i, f in enumerate(files):
        im = cv2.imread(str(Path(img_dir) / f), cv2.IMREAD_GRAYSCALE)
        stem = Path(f).stem
        with (gov.step() if gov is not None else contextlib.nullcontext()):
            p, fm = predict(model, im, stem, device, ondisk, feats=True)
        if SMOKE:          # an untrained network proposes nothing; the experts' share map exercises this path
            p = np.array(np.load(run.CACHE / "val_msk.npy", mmap_mode="r")[i])
        cands, X = C.candidates(p, im, run.b0_deg(stem))
        with (gov.step() if gov is not None else contextlib.nullcontext()):
            Z = pooled(fm, cands)
        del fm
        sets = by_frame[i]
        Y, best = C.targets(cands, sets)
        frames.append(dict(stem=stem, X=X, Z=Z, Y=Y.astype(np.float32), n_gt=[len(s) for s in sets],
                           conflict=C.conflicts(cands), one=np.array([C.ONE_SETTING in c["from"] for c in cands], bool),
                           area=np.array([c["area"] for c in cands], np.int32)))
        if (i + 1) % 20 == 0:
            print(f"  stage 2: {i + 1}/{len(files)} photographs, {time.time() - t0:.0f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    return frames


# ---------------------------------------------------------------- the filament head: learned selection, scored

def tally(frames, scores, cut, restrict):
    tot = np.zeros(4)
    for f, s in zip(frames, scores):
        allowed = f["one"] if restrict else np.ones(len(s), bool)
        chosen = [n for n in C.select(s, f["conflict"], allowed) if s[n] >= cut]
        y = f["Y"][chosen] if chosen else np.zeros((0, len(f["n_gt"])))
        for q, n_gt in enumerate(f["n_gt"]):
            tp = int((y[:, q] > 0).sum())
            tot += (tp, len(chosen) - tp, n_gt - tp, float(y[:, q].sum()))
    return tot


def summary(tot):
    tp, fp, fn, iou = tot
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9)
    sq = iou / max(tp, 1e-9)
    return dict(PQ=round(sq * rq, 4), SQ=round(sq, 4), RQ=round(rq, 4), TP=int(tp), FP=int(fp), FN=int(fn))


VARIANTS = {   # name -> (only the one setting t 0.6 / join 3?, which numbers)
    "fixed rules (t 0.6, join 3, area 100), keep all": (True, None),
    "one outline per filament, 21 numbers (the submitted rule's kind)": (True, "X"),
    "one outline, 21 numbers + network features": (True, "XZ"),
    "whole pool (each filament its own outline), 21 numbers": (False, "X"),
    "whole pool, 21 numbers + network features": (False, "XZ"),
    "whole pool, network features only": (False, "Z"),
}


def _design(f, kind):
    return {"X": f["X"], "Z": f["Z"].astype(np.float32), "XZ": np.hstack([f["X"], f["Z"].astype(np.float32)])}[kind]


def _fit(frames, kind, restrict, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xs, ys = [], []
    for f in frames:
        rows = f["one"] if restrict else np.ones(len(f["one"]), bool)
        if rows.any():
            Xs.append(_design(f, kind)[rows]); ys.append(f["Y"][rows].mean(1))
    m = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
                                      random_state=seed)
    m.fit(np.vstack(Xs), np.concatenate(ys))
    return m


def _scores(m, frames, kind):
    return [m.predict(_design(f, kind)) if len(f["one"]) else np.zeros(0) for f in frames]


def stage2_eval(frames, seed=0):
    """Nested halves of the validation photographs: the selector is fitted on one half, its cut chosen on that
    half's own out-of-fold scores (two inner halves), then applied to the other half; and the other way round.
    Every PQ here is on photographs the selector never saw."""
    cuts = np.round(np.arange(0.02, 0.6, 0.01), 2)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    halves = [sorted(idx[::2]), sorted(idx[1::2])]
    res = {}
    for name, (restrict, kind) in VARIANTS.items():
        tot = np.zeros(4)
        for h in range(2):
            test = [frames[i] for i in halves[h]]
            fit = [frames[i] for i in halves[1 - h]]
            if kind is None:
                tot += tally(test, [np.ones(len(f["one"])) for f in test], 0.5, True)
                continue
            inner = [fit[0::2], fit[1::2]]
            oof = [None] * 2
            for j in range(2):
                oof[j] = _scores(_fit(inner[1 - j], kind, restrict, seed), inner[j], kind)
            fr_in, sc_in = inner[0] + inner[1], oof[0] + oof[1]
            pq = [summary(tally(fr_in, sc_in, c, restrict))["PQ"] for c in cuts]
            cut = float(cuts[int(np.argmax(pq))])
            tot += tally(test, _scores(_fit(fit, kind, restrict, seed), test, kind), cut, restrict)
        res[name] = summary(tot)
        print(f"  {name:66s} PQ {res[name]['PQ']:.4f}  SQ {res[name]['SQ']:.3f}  RQ {res[name]['RQ']:.3f}  "
              f"TP {res[name]['TP']} FP {res[name]['FP']} FN {res[name]['FN']}", flush=True)
    return res


# ---------------------------------------------------------------- training

def arm_main(arm):
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = not LOCAL          # its trial workspaces do not fit the laptop's 6 GB
    run.set_fold(0)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_files = meta["val_files"][:4] if SMOKE else meta["val_files"]
    if SMOKE:
        gt = [g for g in gt if g["frame"] < 4]
    tr_dir = run.find_data()[1]
    ondisk = Q.disk(device)
    gov = Q.governor()
    if LOCAL and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)   # out-of-memory instead of a silent spill to system RAM

    model = SolarUNet().to(device)
    model.ckpt = device.type == "cuda"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{arm}] one network: {n_par / 1e6:.2f} M parameters (attention {sum(p.numel() for p in model.att.parameters()) / 1e6:.2f} M)",
          flush=True)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    t0 = time.time()
    wdisk = torch.from_numpy(run.disk_weight()).to(device)[None, None]

    def spent():
        return time.time() - t0

    def epoch(loader, opt, sched, name, ep):
        te, tot, n = time.time(), 0.0, 0
        for x, y, g, mt in loader:
            x, y, mt = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mt.to(device)
            geo = run.geo_channels(g, device)
            with gov.step():
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logit = model(x, geo, mt)
                loss = loss_fn(logit.float(), y, wdisk)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
            tot += loss.item(); n += 1
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        print(f"[{arm}] {name} epoch {ep}: loss {tot / max(n, 1):.4f}, {n} steps, {(time.time() - te) / 60:.1f} min, "
              f"{mem:.1f} GiB, {spent() / 3600:.2f} h", flush=True)
        return time.time() - te

    def stage(ds, epochs, name, validate):
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                                             num_workers=2 if run.ON_KAGGLE else 0, pin_memory=device.type == "cuda",
                                             persistent_workers=run.ON_KAGGLE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        plan = [max(1, epochs * len(loader))]          # steps; refitted to the budget after the first epoch
        warm = len(loader)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / plan[0])))))
        best, n_ep, val_at = (-1.0, None, 0), epochs, set()
        for ep in range(1, epochs + 1):
            dt = epoch(loader, opt, sched, name, ep)
            if ep == 1 and validate:
                left = BUDGET_H * 3600 - spent() - 900       # 15 min for the validation passes and writing
                n_ep = max(1, min(epochs, 1 + int(left / max(dt, 1.0))))
                plan[0] = n_ep * len(loader)
                val_at = {n_ep} | {max(1, int(round(n_ep * q))) for q in (0.5, 0.65, 0.8, 0.9)}
                print(f"[{arm}] {name}: {dt / 60:.1f} min per epoch -> {n_ep} epochs, validation at {sorted(val_at)}",
                      flush=True)
            if validate and ep in val_at:
                r = Q.score(val_maps(model, device, ondisk, val_files, tr_dir, gov), gt, quick=True)[0]
                print(f"[{arm}] {name} epoch {ep}: val PQ {r['PQ']:.4f} at t {r['thresh']} merge {r['merge']} "
                      f"area {r['min_area']}", flush=True)
                if r["PQ"] > best[0]:
                    best = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            if ep >= n_ep:
                break
        return best

    if arm == "pre":
        items = Q.pseudo_labels(device)
        stage(PX.MetaPseudo(items), 1, "pretrain", validate=False)
        print(f"[{arm}] pretrained on {len(items)} frames, {spent() / 60:.0f} min", flush=True)
    files = meta["train_files"][:6] if SMOKE else meta["train_files"]
    best_pq, best_sd, best_ep = stage(PX.MetaFrames(files), EPOCHS, "train", validate=True)
    model.load_state_dict(best_sd)
    print(f"[{arm}] best epoch {best_ep} (quick val PQ {best_pq:.4f}), {spent() / 60:.0f} min", flush=True)
    tag = f"sunet_{arm}"
    torch.save({k: v.half() for k, v in best_sd.items()}, WORK / f"{tag}.pt")

    # maps in the U-Nets' dump format, written before anything that could fail
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
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    grid = Q.score(probs, gt, quick=False)

    # seconds per photograph on this card (one forward pass; median of 5 after 2 warm-ups)
    im = cv2.imread(str(te_dir / test_files[0]), cv2.IMREAD_GRAYSCALE)
    model.eval()
    ts = []
    for i in range(7):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        predict(model, im, Path(test_files[0]).stem, device, ondisk)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t1)
    sec = float(np.median(ts[2:]))
    res = dict(arm=arm, params_M=round(n_par / 1e6, 2), best_epoch=best_ep, quick_val_pq=best_pq, best=grid[0],
               top5=grid[:5], minutes=round(spent() / 60, 1), smoke=SMOKE, sec_per_photo=round(sec, 3),
               where="laptop RTX 2060" if LOCAL else "Kaggle T4", batch=BATCH)
    json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    print(f"[{arm}] fold 0 val PQ {grid[0]['PQ']:.4f} (SQ {grid[0]['SQ']:.3f}, RQ {grid[0]['RQ']:.3f}) at "
          f"t {grid[0]['thresh']} merge {grid[0]['merge']} area {grid[0]['min_area']}; {sec:.3f} s per photograph",
          flush=True)

    # the filament head: its data, then its score on held-out halves
    try:
        frames = stage2_frames(model, device, ondisk, val_files, tr_dir, gt, gov)
        pickle.dump(frames, open(WORK / f"stage2_{tag}.pkl", "wb"))
        print(f"[{arm}] stage 2: {sum(len(f['one']) for f in frames)} outlines on {len(frames)} photographs", flush=True)
        res["stage2"] = stage2_eval(frames)
        json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    except Exception as e:                                  # the maps and weights are already written
        import traceback
        traceback.print_exc()
        print(f"[{arm}] stage 2 failed: {e!r}", flush=True)


def kaggle_main():
    t0 = time.time()
    run.set_fold(0)
    run.prepare(SMOKE, with_train=True)       # once, before the two arms share the cache
    print(f"fold 0 cache ready in {time.time() - t0:.0f}s", flush=True)
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(WORK / f"log_sunet_{arm}.txt", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (WORK / f"log_sunet_{arm}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][-160:] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n"
              + (WORK / f"log_sunet_{arm}.txt").read_text(errors="replace")[-4000:], flush=True)


def local_main(arm="scratch"):
    run.set_fold(0)
    meta = run.CACHE / "meta.json"
    if not meta.exists() or len(json.load(open(meta))["train_files"]) < (1 if SMOKE else 500):
        run.prepare(SMOKE, with_train=True)
    arm_main(arm)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("scratch", "pre"):
        local_main(sys.argv[1]) if LOCAL else arm_main(sys.argv[1])
    else:
        kaggle_main()
