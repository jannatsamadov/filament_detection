"""Time attention on the compact network: the user's idea (2026-11-15).

A filament changes little over a few days while the Sun turns under it, so another photograph taken within
84 h carries information about this one. Here the network looks at it through attention:

  each neighbour photograph goes through the same left wing and 64 x 64 attention (no gradient: its tokens are
  read, not trained); the photograph's own tokens then attend to the neighbours' tokens in 2 cross-attention
  layers of 8 heads. Where to look is set by positions on the Sun: each token carries its latitude and longitude,
  and a neighbour's longitude is moved by a rotation law the network learns,
        omega(phi) = A + B sin^2(phi) + C sin^4(phi)  [deg/day],    A = B = C = 0 at the start,
  times the time between the two photographs. Nothing about the Sun's rotation is given; if the learned A, B, C
  come out near the measured ones (Snodgrass: 14.71, -2.39, -1.78 for photospheric Doppler features), the
  attention found it. A learned "nothing here" token lets a photograph without a neighbour, or a region the
  neighbour does not show, be left alone; every new layer starts as "pass through", so training starts exactly
  at the compact network it is built on.

Rule (1) holds: at inference a test photograph's neighbours are other test photographs only (validation:
validation photographs only). In training they come from the training photographs and the unlabelled GONG
frames, each kept with probability 0.6, because most test photographs have none within 84 h (32 %).

Two arms on the two T4s, both fine-tuned from the better filament-solar-unet arm with the same schedule:
  time   with the neighbours
  ctrl   the same network and training, with no neighbour ever given: the control
Each writes dump_stime_<arm>.tar, stime_<arm>.pt and stime_<arm>.json (PQ on all validation photographs and on
those that have a neighbour, and the learned rotation law).
    FIL_STIME_SMOKE=1 python kernels/aug_unet/solar_time.py time      # the laptop, a few frames
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, math, pickle, random, shutil, subprocess, sys, tarfile, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import solar_unet as S  # noqa: E402
import run  # noqa: E402
from run import torch, cv2, np, nn, F  # noqa: E402
import pixnet as PX  # noqa: E402
import qwen_seg as Q  # noqa: E402

SMOKE = os.environ.get("FIL_STIME_SMOKE") == "1"
ARMS = ("time", "ctrl")
LOCAL = not run.ON_KAGGLE
MAX_DT_H = 84.0
N_NB = 2                         # the nearest earlier and the nearest later photograph
KEEP_P = 0.6
N_XATT = 2
N_FOURIER = 8
EPOCHS = 1 if SMOKE else 30
BUDGET_H = 0.4 if SMOKE else 2.7
LR, ROT_LR = 3e-4, 0.02          # the three rotation numbers move ~0.02 deg/day per step, so they can reach ~14
BATCH = 1 if LOCAL else 2
WORK = run.WORK
SNODGRASS = (14.71, -2.39, -1.78)


def hours(stem):
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").timestamp() / 3600.0


# ---------------------------------------------------------------- neighbours

class Pool:
    """Photographs of one pool by time; near() gives the nearest earlier and later one within 84 h,
    with dt = (this photograph's time - the neighbour's) in hours."""

    def __init__(self, items):
        self.items = sorted(items, key=lambda it: hours(it[0]))
        self.t = np.array([hours(s) for s, _ in self.items])

    def near(self, stem):
        d = self.t - hours(stem)
        out = []
        for side in (d < 0, d > 0):
            idx = np.nonzero(side & (np.abs(d) <= MAX_DT_H))[0]
            idx = [i for i in idx if self.items[i][0] != stem]
            if idx:
                i = min(idx, key=lambda i: abs(d[i]))
                out.append((self.items[i], -d[i]))
        return out


def _gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im.shape != (run.FULL, run.FULL):
        im = cv2.resize(im, (run.FULL, run.FULL), interpolation=cv2.INTER_AREA)
    return im


class _Cached:
    """A photograph of the fold cache, read lazily (the arrays are memory-mapped per worker)."""
    _arr = {}

    def __init__(self, name, i):
        self.name, self.i = name, i

    def __call__(self):
        if self.name not in self._arr:
            self._arr[self.name] = np.load(run.CACHE / f"{self.name}.npy", mmap_mode="r")
        return np.array(self._arr[self.name][self.i])


class _File:
    def __init__(self, path):
        self.path = path

    def __call__(self):
        return _gray(self.path)


def gong_frames():
    if run.ON_KAGGLE:
        frames = sorted(p for p in Path("/kaggle/input").rglob("*.jpg") if "gong-unlabelled" in str(p))
        tar = next((p for p in Path("/kaggle/input").rglob("frames.tar")), None)
        if not frames and tar is not None:
            fr = Path("/tmp/stime_frames")
            fr.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as t:
                t.extractall(fr, filter="data")
            frames = sorted(fr.glob("*.jpg"))
    else:
        frames = sorted((Q.ROOT / "gong_unlabelled").glob("*.jpg"))
    return frames


def nb_tensors(near, augment):
    """Up to N_NB neighbours as batch-free tensors: images, geometry maps' source, photograph numbers, dt, present."""
    xs = torch.zeros(N_NB, 1, run.FULL, run.FULL)
    gs = torch.zeros(N_NB, 5)
    gs[:, 0] = gs[:, 3] = 1.0
    ms = torch.zeros(N_NB, PX.N_META)
    ms[:, PX.SPREAD] = 1.0
    dts = torch.zeros(N_NB)
    ok = torch.zeros(N_NB, dtype=torch.bool)
    for j, ((s, load), dt) in enumerate(near[:N_NB]):
        if augment and random.random() > KEEP_P:
            continue
        im = load()
        run._AUG_S = np.eye(2)
        if augment:
            im, _ = run.augment_flip(im, im)
        b0 = run.b0_deg(s)
        gs[j] = torch.tensor([*run._AUG_S.ravel(), b0], dtype=torch.float32)
        x = run.to_input(np.ascontiguousarray(im))
        xs[j, 0] = x
        ms[j] = PX.meta_vec(s, x, b0)
        dts[j] = float(dt)
        ok[j] = True
    return xs, gs, ms, dts, ok


class TimeFrames(PX.MetaFrames):
    def __init__(self, files, pool):
        super().__init__(files)
        self.pool = pool

    def __getitem__(self, i):
        x, y, g, mt = super().__getitem__(i)
        near = self.pool.near(self.stems[i]) if self.pool is not None else []
        return (x, y, g, mt) + nb_tensors(near, augment=True)


# ---------------------------------------------------------------- the network

def fourier(v, n=N_FOURIER):
    f = (2.0 ** torch.arange(n, device=v.device, dtype=torch.float32)) * math.pi / 2
    a = v[..., None] * f
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class CrossAttention(nn.Module):
    def __init__(self, d=S.D_ATT, heads=S.HEADS):
        super().__init__()
        self.h = heads
        self.nq, self.nk, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        for lin in (self.proj, self.mlp[2]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, t, pq, m, pk, mask):
        B, N, D = t.shape
        M, hd = m.shape[1], D // self.h
        q = self.q(self.nq(t) + pq).reshape(B, N, self.h, hd).transpose(1, 2)
        mk = self.nk(m)
        k = self.k(mk + pk).reshape(B, M, self.h, hd).transpose(1, 2)
        v = self.v(mk).reshape(B, M, self.h, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=mask[:, None, None, :])
        t = t + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return t + self.mlp(self.n2(t))


class TimeSolarUNet(S.SolarUNet):
    def __init__(self):
        super().__init__()
        d = S.D_ATT
        self.rot = nn.Parameter(torch.zeros(3))       # A, B, C of omega(phi), deg/day
        self.xp = nn.Linear(4 * N_FOURIER, d)          # position on the Sun -> token space (shared by both sides)
        self.xdt = nn.Linear(2 * 4 + 1, d)             # how long ago / ahead
        self.null = nn.Parameter(torch.randn(1, d) * 0.02)   # the "nothing here" token, always attendable
        self.xatt = nn.ModuleList([CrossAttention() for _ in range(N_XATT)])

    def omega(self, sinlat):
        s2 = sinlat * sinlat
        return self.rot[0] + self.rot[1] * s2 + self.rot[2] * s2 * s2

    def sunpos(self, g64, dt_days=None):
        """(B, 3, S, S) pooled geometry -> (B, S*S, D). With dt, the longitude is carried forward by omega * dt."""
        sl, lon = g64[:, 0], g64[:, 1] * 90.0
        if dt_days is not None:
            lon = lon + self.omega(sl) * dt_days[:, None, None]
        f = torch.cat([fourier(sl), fourier(lon / 90.0)], -1)          # (B, S, S, 32)
        return self.xp(f.flatten(1, 2))

    def forward(self, x, geo, meta, nb=None, feats=False):
        fe, fd = self.films(meta)
        skips, t = self.encode(x, geo, meta, fe)
        B, N, D = t.shape
        Sz = int(round(N ** 0.5))
        g64 = F.adaptive_avg_pool2d(geo.float(), Sz)
        pq = self.sunpos(g64).to(t.dtype)
        mem, pk, mask = [self.null[0:1].expand(B, 1, D).to(t.dtype)], [torch.zeros(B, 1, D, device=t.device, dtype=t.dtype)], \
            [torch.ones(B, 1, dtype=torch.bool, device=t.device)]
        if nb is not None:
            xs, gs, ms, dts, ok = nb
            for j in range(xs.shape[1]):
                if not bool(ok[:, j].any()):
                    continue
                gj = run.geo_channels(gs[:, j], x.device)
                with torch.no_grad():
                    fej, _ = self.films(ms[:, j])
                    _, tj = self.encode(xs[:, j], gj, ms[:, j], fej)
                gj64 = F.adaptive_avg_pool2d(gj, Sz)
                dt_days = dts[:, j].float() / 24.0
                dtf = torch.cat([fourier(dts[:, j].float() / MAX_DT_H, 4), (dts[:, j].float().abs() / MAX_DT_H)[:, None]], -1)
                mem.append(tj.detach())
                pk.append((self.sunpos(gj64, dt_days) + self.xdt(dtf)[:, None, :]).to(t.dtype))
                mask.append((gj64[:, 2].flatten(1) > 0.05) & ok[:, j:j + 1])
        m, k_pos, msk = torch.cat(mem, 1), torch.cat(pk, 1), torch.cat(mask, 1)
        for xa in self.xatt:
            t = xa(t, pq, m, k_pos, msk)
        return self.decode(skips, t, x, fd, feats)


# ---------------------------------------------------------------- prediction and scoring

@torch.no_grad()
def predict(model, im_u8, stem, device, ondisk, pool):
    x, geo, meta = S._inputs(im_u8, stem, device)
    nb = None
    if pool is not None:
        xs, gs, ms, dts, ok = nb_tensors(pool.near(stem), augment=False)
        if ok.any():
            nb = tuple(v[None].to(device) for v in (xs, gs, ms, dts, ok))
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logit = model(x, geo, meta, nb)
    return (torch.sigmoid(logit.float())[0, 0] * ondisk * 255.0).round().byte().cpu().numpy()


def maps(model, device, ondisk, files, img_dir, pool, gov=None):
    import contextlib
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out = []
    for f in files:
        with (gov.step() if gov is not None else contextlib.nullcontext()):
            out.append(predict(model, _gray(Path(img_dir) / f), Path(f).stem, device, ondisk, pool))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    return out


def score_subset(probs, gt, idx, quick):
    new = {f: n for n, f in enumerate(idx)}
    sub = [dict(g, frame=new[g["frame"]]) for g in gt if g["frame"] in new]
    return Q.score([probs[f] for f in idx], sub, quick=quick)[0] if idx else None


# ---------------------------------------------------------------- training

def base_weights():
    """The better filament-solar-unet arm by its full fold-0 grid PQ."""
    roots = [Path("/kaggle/input")] if run.ON_KAGGLE else [HERE.parent / "solar_unet" / "out_kaggle", WORK]
    best = None
    for r in roots:
        for js in r.rglob("sunet_*.json"):
            res = json.load(open(js))
            pt = js.with_suffix(".pt")
            if pt.exists() and (best is None or res["best"]["PQ"] > best[0]):
                best = (res["best"]["PQ"], pt, res["arm"])
    return best


def arm_main(arm):
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = not LOCAL
    run.set_fold(0)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = pickle.load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_files = meta["val_files"][:4] if SMOKE else meta["val_files"]
    if SMOKE:
        gt = [g for g in gt if g["frame"] < 4]
    tr_dir, te_dir = run.find_data()[1], Path(meta["test_dir"])
    ondisk = Q.disk(device)
    gov = Q.governor()
    if LOCAL and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.9)

    model = TimeSolarUNet().to(device)
    model.ckpt = device.type == "cuda"
    base = base_weights()
    if base is not None:
        sd = {k: v.float() for k, v in torch.load(base[1], map_location="cpu").items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.split(".")[0] in ("rot", "xp", "xdt", "null", "xatt") for k in missing), \
            (missing, unexpected)
        print(f"[{arm}] from filament-solar-unet arm {base[2]} (fold-0 PQ {base[0]:.4f}): {base[1]}", flush=True)
    elif not SMOKE:
        sys.exit("no filament-solar-unet weights attached")
    print(f"[{arm}] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)

    files = meta["train_files"][:6] if SMOKE else meta["train_files"]
    pool_tr = None
    if arm == "time":
        items = [(Path(f).stem, _Cached("train_img", i)) for i, f in enumerate(meta["train_files"])]
        items += [(p.stem, _File(p)) for p in gong_frames()]
        pool_tr = Pool(items)
        n_has = sum(bool(pool_tr.near(Path(f).stem)) for f in files)
        print(f"[{arm}] training pool: {len(items)} photographs; {n_has}/{len(files)} training photographs have a "
              f"neighbour within {MAX_DT_H:.0f} h", flush=True)
    pool_val = Pool([(Path(f).stem, _File(tr_dir / f)) for f in val_files]) if arm == "time" else None
    pool_te = Pool([(Path(f).stem, _File(te_dir / f)) for f in meta["test_files"]]) if arm == "time" else None
    # the validation photographs that have a neighbour among the validation photographs: both arms report them
    vpool = Pool([(Path(f).stem, None) for f in val_files])
    has_nb = [n for n, f in enumerate(val_files) if vpool.near(Path(f).stem)]
    print(f"[{arm}] {len(has_nb)}/{len(val_files)} validation photographs have a neighbour within {MAX_DT_H:.0f} h",
          flush=True)

    loader = torch.utils.data.DataLoader(TimeFrames(files, pool_tr), batch_size=BATCH, shuffle=True, drop_last=True,
                                         num_workers=2 if run.ON_KAGGLE else 0, pin_memory=device.type == "cuda",
                                         persistent_workers=run.ON_KAGGLE)
    rot = [model.rot]
    rest = [p for n, p in model.named_parameters() if n != "rot"]
    # the rotation numbers' gradient passes through layers that start at zero: tiny at first, and Adam's default
    # eps (1e-8) would swallow it (smoke test: A, B, C stayed 0.00); a smaller eps lets them move from the start
    opt = torch.optim.AdamW([dict(params=rest, lr=LR, weight_decay=1e-4),
                             dict(params=rot, lr=ROT_LR, weight_decay=0.0, eps=1e-16)])
    plan = [max(1, EPOCHS * len(loader))]
    warm = len(loader) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / plan[0])))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    wdisk = torch.from_numpy(run.disk_weight()).to(device)[None, None]
    t0 = time.time()
    hist, best, n_ep, val_at = [], (-1.0, None, 0), EPOCHS, set()
    for ep in range(1, EPOCHS + 1):
        te, tot, n = time.time(), 0.0, 0
        for batch in loader:
            x, y, g, mt = (v.to(device, non_blocking=True) for v in batch[:4])
            nb = tuple(v.to(device, non_blocking=True) for v in batch[4:]) if arm == "time" else None
            geo = run.geo_channels(g, device)
            with gov.step():
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logit = model(x, geo, mt, nb)
                loss = S.loss_fn(logit.float(), y, wdisk)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(rest, 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
            tot += loss.item(); n += 1
        dt_ep = time.time() - te
        A, B, Cc = (float(v) for v in model.rot.detach().cpu())
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        print(f"[{arm}] epoch {ep}: loss {tot / max(n, 1):.4f}, {dt_ep / 60:.1f} min, {mem:.1f} GiB; rotation law "
              f"A {A:.2f} B {B:.2f} C {Cc:.2f} deg/day", flush=True)
        if ep == 1:
            left = BUDGET_H * 3600 - (time.time() - t0) - 900
            n_ep = max(1, min(EPOCHS, 1 + int(left / max(dt_ep, 1.0))))
            plan[0] = n_ep * len(loader)
            val_at = {n_ep} | {max(1, int(round(n_ep * q))) for q in (0.4, 0.6, 0.8)}
            print(f"[{arm}] {n_ep} epochs, validation at {sorted(val_at)}", flush=True)
        if ep in val_at:
            probs = maps(model, device, ondisk, val_files, tr_dir, pool_val, gov)
            r = Q.score(probs, gt, quick=True)[0]
            rs = score_subset(probs, gt, has_nb, quick=True)
            hist.append(dict(epoch=ep, PQ=r["PQ"], PQ_with_neighbour=rs["PQ"] if rs else None, rot=[A, B, Cc]))
            print(f"[{arm}] epoch {ep}: val PQ {r['PQ']:.4f}; on the {len(has_nb)} with a neighbour "
                  f"{rs['PQ'] if rs else float('nan'):.4f}", flush=True)
            if r["PQ"] > best[0]:
                best = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep >= n_ep:
            break

    model.load_state_dict(best[1])
    if SMOKE and pool_tr is not None:          # the prediction path with neighbours (no smoke validation photo has one)
        s0 = next(Path(f).stem for f in meta["train_files"] if pool_tr.near(Path(f).stem))
        i0 = [Path(f).stem for f in meta["train_files"]].index(s0)
        p0 = predict(model, _Cached("train_img", i0)(), s0, device, ondisk, pool_tr)
        print(f"[{arm}] smoke: prediction with {len(pool_tr.near(s0))} neighbour(s): {p0.shape}, max {p0.max()}", flush=True)
    tag = f"stime_{arm}"
    torch.save({k: v.half() for k, v in best[1].items()}, WORK / f"{tag}.pt")
    root = WORK / f"dump_{tag}"
    (root / "val_plain").mkdir(parents=True, exist_ok=True)
    (root / "test_plain").mkdir(parents=True, exist_ok=True)
    probs = maps(model, device, ondisk, val_files, tr_dir, pool_val, gov)
    for f, p in zip(val_files, probs):
        cv2.imwrite(str(root / "val_plain" / f"{Path(f).stem}.png"), p)
    test_files = meta["test_files"][:2] if SMOKE else meta["test_files"]
    for f, p in zip(test_files, maps(model, device, ondisk, test_files, te_dir, pool_te, gov)):
        cv2.imwrite(str(root / "test_plain" / f"{Path(f).stem}.png"), p)
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    grid = Q.score(probs, gt, quick=False)
    sub = score_subset(probs, gt, has_nb, quick=False)
    A, B, Cc = (float(v) for v in model.rot.detach().cpu())
    res = dict(arm=arm, base=str(base[1]) if base else None, best_epoch=best[2], quick_val_pq=best[0], best=grid[0],
               with_neighbour=dict(photographs=len(has_nb), best=sub), history=hist,
               rotation_law=dict(A=A, B=B, C=Cc, snodgrass=SNODGRASS), minutes=round((time.time() - t0) / 60, 1),
               smoke=SMOKE, where="laptop RTX 2060" if LOCAL else "Kaggle T4")
    json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    print(f"[{arm}] fold 0 val PQ {grid[0]['PQ']:.4f}; with a neighbour ({len(has_nb)}) "
          f"{sub['PQ'] if sub else float('nan'):.4f}; rotation law A {A:.2f} B {B:.2f} C {Cc:.2f}", flush=True)


def kaggle_main():
    t0 = time.time()
    run.set_fold(0)
    run.prepare(SMOKE, with_train=True)
    print(f"fold 0 cache ready in {time.time() - t0:.0f}s", flush=True)
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(WORK / f"log_stime_{arm}.txt", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (WORK / f"log_stime_{arm}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][-160:] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n"
              + (WORK / f"log_stime_{arm}.txt").read_text(errors="replace")[-4000:], flush=True)


def local_main(arm):
    run.set_fold(0)
    m = run.CACHE / "meta.json"
    if not m.exists() or len(json.load(open(m))["train_files"]) < (1 if SMOKE else 500):
        run.prepare(SMOKE, with_train=True)
    arm_main(arm)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ARMS:
        local_main(sys.argv[1]) if LOCAL else arm_main(sys.argv[1])
    else:
        kaggle_main()
