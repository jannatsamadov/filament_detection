"""A segmentation model built on Qwen3-VL's own eye, pretrained on unlabelled GONG frames.

Qwen's language half cannot draw a mask (it answers in text, boxes at best), and
PQ scores masks, so the model here is Qwen3-VL-2B's vision encoder -- 24
transformer blocks on 16 px patches, the weights Qwen learned from its image
corpus -- with a mask head: the features of blocks 6, 12, 18 and 24 at 1/16
scale, fused, and brought back to full resolution with a small convolutional
branch on the raw frame for the edges. Nothing in it is a U-Net encoder, so its
mistakes are its own: a third family next to the U-Nets and YOLO.

Two arms, one per T4, both validated on fold 0 exactly as the U-Nets were:

  pre   1. pretrain: one pass over 3810 unlabelled GONG frames (none from a test
           day); the target is the fold-0 five-model U-Net ensemble's map (plain),
           so nothing of fold 0's validation labels leaks in
        2. fine-tune on fold 0's 566 annotated photographs, soft targets
  ft    step 2 only -- the control: what the pretraining is worth

Training uses 1024 px crops at native resolution (4096 tokens each; the loss
ignores a 48 px margin, so crop edges teach no false ends); inference runs on
the whole 2048 frame at once (16384 tokens). Each arm writes
dump_qwenseg_<arm>.tar in the U-Nets' dump format (val_plain, test_plain), so
every existing tool (sweep, keep-rule, ensembles) reads it as one more model,
and qwenseg_<arm>.json with the full post-processing sweep on fold 0.
"""
from __future__ import annotations

import os
os.environ.setdefault("FIL_LOCAL_GPU", "1")            # before run.py hides the card
import json, math, random, shutil, subprocess, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if not Path("/kaggle/working").exists():
    os.environ.setdefault("HF_HOME", str(ROOT / ".hfhome"))     # the model cache stays on D:
sys.path.insert(0, str(HERE))
import run  # noqa: E402  -- torch before cv2
from run import torch, cv2, np, nn, F  # noqa: E402

SMOKE = os.environ.get("FIL_QSEG_SMOKE") == "1"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ARMS = ("pre", "ft")
# 11-15, kernel filament-qwen-seg2: both arms were still rising at epoch 12 (pre: 0.391, 0.414,
# 0.417 at 6, 9, 12), so each continues from its saved weights (filament-qwen-seg's
# qwenseg_<arm>.pt) for CONT_EPOCHS more at half the learning rate; the loaded model is
# scored first, so an arm that does not improve keeps its old weights. Arms pre2 and ft2.
CONT = os.environ.get("FIL_QSEG_CONT") == "1"
CONT_EPOCHS = 1 if SMOKE else 9
CONT_VAL_AT = (1,) if SMOKE else (3, 6, 9)
CONT_LR = 0.5
if CONT:
    ARMS = ("pre2", "ft2")
TAPS = (5, 11, 17, 23)
FREEZE = 6                       # patch embedding and blocks 0-5 stay as Qwen learned them
CROP = 512 if SMOKE else 1024
MARGIN = 48
BATCH = 2
EPOCHS = 1 if SMOKE else 12
VAL_AT = (1,) if SMOKE else (6, 9, 12)
N_PSEUDO = 6 if SMOKE else 2000  # unlabelled frames used for pretraining (of 3810), evenly spread in time
BUDGET_H = 3.0 if CONT else 5.0  # per arm: past this, stop training, keep the best epoch, write everything
LR_VIT, LR_HEAD = 2e-5, 5e-4
ENS0 = ("flip", "full", "r50", "solar", "union")    # fold-0 models: never saw fold 0's validation labels
WORK = run.WORK
TMP = Path("/tmp/qseg") if run.ON_KAGGLE else HERE / "out" / "qseg"
P, M, T = 16, 2, 2               # Qwen3-VL: patch, spatial merge, temporal patch


# ---------------------------------------------------------------- Qwen's vision encoder

def load_vision():
    """Only the vision tower of Qwen3-VL-2B, with SDPA attention (eager would need GBs per image)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = AutoConfig.from_pretrained(MODEL).vision_config
    vis = Qwen3VLVisionModel._from_config(cfg, attn_implementation="sdpa")
    sd = {}
    with safe_open(hf_hub_download(MODEL, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = f.get_tensor(k).float()
    missing, unexpected = vis.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return vis.float()


def patchify(x):
    """(B, 1, H, W) Qwen-normalised grey -> Qwen's flattened patches, in its processor's order."""
    B, _, H, W = x.shape
    gh, gw = H // P, W // P
    x = x.expand(B, 3, H, W)[:, None].expand(B, T, 3, H, W)
    x = x.reshape(B, T, 3, gh // M, M, P, gw // M, M, P).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    thw = torch.tensor([[1, gh, gw]] * B, device=x.device)
    return x.reshape(B * gh * gw, 3 * T * P * P), thw


def unpatch(h, B, gh, gw):
    """Tokens in merge-window order -> (B, D, gh, gw)."""
    D = h.shape[-1]
    h = h.reshape(B, gh // M, gw // M, M, M, D).permute(0, 5, 1, 3, 2, 4)
    return h.reshape(B, D, gh, gw)


def _cbr(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class QwenSeg(nn.Module):
    def __init__(self, vis):
        super().__init__()
        self.vis = vis
        D = vis.config.hidden_size
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(D, 128, 1, bias=False), nn.BatchNorm2d(128),
                                                 nn.ReLU(inplace=True)) for _ in TAPS])
        self.fuse = run._block(len(TAPS) * 128, 256)
        self.c1, self.c2, self.c3 = _cbr(1, 32, 2), _cbr(32, 64, 2), _cbr(64, 128, 2)
        self.u8 = run._block(256 + 128, 128)
        self.u4 = run._block(128 + 64, 64)
        self.u2 = run._block(64 + 32, 32)
        self.u1 = run._block(32 + 1, 16)
        self.head = nn.Conv2d(16, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)
        self._taps = {}
        for i in TAPS:
            vis.blocks[i].register_forward_hook(lambda m, a, o, i=i: self._taps.__setitem__(i, o))

    @staticmethod
    def _up(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        """x: run.to_input's normalisation, (v - 0.5) / 0.25; Qwen wants (v - 0.5) / 0.5."""
        B, _, H, W = x.shape
        pix, thw = patchify(x * 0.5)
        self._taps.clear()
        self.vis(pix, grid_thw=thw)
        gh, gw = H // P, W // P
        f = self.fuse(torch.cat([pr(unpatch(self._taps[i], B, gh, gw)) for pr, i in zip(self.proj, TAPS)], 1))
        s1 = self.c1(x)
        s2 = self.c2(s1)
        s3 = self.c3(s2)
        d = self.u8(torch.cat([self._up(f, s3), s3], 1))
        d = self.u4(torch.cat([self._up(d, s2), s2], 1))
        d = self.u2(torch.cat([self._up(d, s1), s1], 1))
        d = self.u1(torch.cat([self._up(d, x), x], 1))
        return self.head(d)


# ---------------------------------------------------------------- data

class Pseudo(torch.utils.data.Dataset):
    """Unlabelled frames with the ensemble's map as the target, augmented as the U-Nets were."""

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
        return run.to_input(im)[None], torch.from_numpy(ma).float().div_(255.0)[None], torch.zeros(5)


def crops(x, y, valid):
    """One random CROP x CROP window per frame; the loss ignores its outer MARGIN."""
    B, _, H, W = x.shape
    xs, ys, vs = [], [], []
    inner = torch.zeros(1, CROP, CROP, device=x.device)
    inner[:, MARGIN:CROP - MARGIN, MARGIN:CROP - MARGIN] = 1.0
    for b in range(B):
        y0, x0 = random.randint(0, H - CROP), random.randint(0, W - CROP)
        xs.append(x[b, :, y0:y0 + CROP, x0:x0 + CROP])
        ys.append(y[b, :, y0:y0 + CROP, x0:x0 + CROP])
        vs.append(valid[:, y0:y0 + CROP, x0:x0 + CROP] * inner)
    return torch.stack(xs), torch.stack(ys), torch.stack(vs)


# ---------------------------------------------------------------- pseudo-labels

def pseudo_labels(device):
    """The fold-0 ensemble (plain) on every unlabelled frame, as uint8 PNG; cached."""
    pl = TMP / "pl"
    pl.mkdir(parents=True, exist_ok=True)
    if run.ON_KAGGLE:
        # Kaggle unpacks the uploaded frames.tar into the dataset as single JPEGs
        frames = sorted(p for p in Path("/kaggle/input").rglob("*.jpg") if "gong-unlabelled" in str(p))
        tar = next((p for p in Path("/kaggle/input").rglob("frames.tar")), None)
        if not frames and tar is not None:
            fr = TMP / "frames"
            fr.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar) as t:
                t.extractall(fr, filter="data")
            frames = sorted(fr.glob("*.jpg"))
    else:
        frames = sorted((ROOT / "gong_unlabelled").glob("*.jpg"))
    if not frames:
        raise RuntimeError("no unlabelled frames attached (dataset jannatsamadov/filament-gong-unlabelled)")
    print(f"{len(frames)} unlabelled frames", flush=True)
    if len(frames) > N_PSEUDO:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, N_PSEUDO)]
    todo = [f for f in frames if not (pl / f"{f.stem}.png").exists()]
    if todo:
        roots = [Path("/kaggle/input")] if run.ON_KAGGLE else [HERE / "out", ROOT / "kernels"]
        models = []
        for tag in ENS0:
            ck_path = next((p for r in roots for p in r.rglob(f"best_{tag}.pt")), None)
            if ck_path is None:
                print(f"pseudo-labels: best_{tag}.pt not attached, skipped", flush=True)
                continue
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            m = run.UNet(ck.get("encoder", "resnet34"), pretrained=False,
                         in_ch=run.model_in_ch(ck.get("cfg", {}))).to(device).to(memory_format=run.fmt(device))
            m.load_state_dict(ck["model"]); m.eval()
            models.append(m)
        if not models:
            raise RuntimeError("no fold-0 U-Net checkpoint found for the pseudo-labels")
        print(f"pseudo-labels from {len(models)} fold-0 models on {len(todo)} frames", flush=True)
        ondisk = disk(device)
        t0 = time.time()
        for n, f in enumerate(todo):
            im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            acc = sum(run.predict(m, im, device, ondisk, "plain").astype(np.float32) for m in models)
            cv2.imwrite(str(pl / f"{f.stem}.png"), np.round(acc / len(models)).astype(np.uint8))
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(todo)} pseudo-labels, {time.time() - t0:.0f}s", flush=True)
        del models
        torch.cuda.empty_cache()
    return [(f, pl / f"{f.stem}.png") for f in frames]


def disk(device):
    yy, xx = np.mgrid[:run.FULL, :run.FULL]
    return torch.from_numpy((((xx - run.CX) ** 2 + (yy - run.CY) ** 2) <= (run.R_DISK * 0.995) ** 2)
                            .astype(np.float32)).to(device)


# ---------------------------------------------------------------- training and scoring

@torch.no_grad()
def predict(model, im_u8, device, ondisk):
    x = run.to_input(im_u8)[None, None].to(device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.sigmoid(model(x).float())[0, 0]
    return (p * ondisk * 255.0).round().byte().cpu().numpy()


def val_maps(model, device, ondisk, meta, files=None):
    tr_dir = run.find_data()[1]
    files = files or meta["val_files"]
    model.eval()
    out = [predict(model, cv2.imread(str(tr_dir / f), cv2.IMREAD_GRAYSCALE), device, ondisk) for f in files]
    model.train()
    return out


def score(probs, gt, quick):
    if quick:
        run.THRESH, run.MERGE, run.MIN_AREA = (0.5, 0.6, 0.7), (3,), (100, 250)
    else:
        run.THRESH, run.MERGE, run.MIN_AREA = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9), (0, 3, 7), (100, 250, 500, 1000, 2000)
    return run.sweep(probs, gt)


def loss_fn(logit, y, valid, pos_weight=10.0):
    """run.loss_fn's BCE + soft Dice, with a per-sample valid mask (the crop's disk and inner part)."""
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pos_weight, device=logit.device),
                                             reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    p, t = torch.sigmoid(logit) * valid, y * valid
    dice = 1.0 - (2.0 * (p * t).sum((1, 2, 3)) + 1.0) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1.0)
    return bce + dice.mean()


def epoch_loop(model, opt, sched, scaler, loader, device, gov, tag, stage, ep):
    wdisk = torch.from_numpy(run.disk_weight()).to(device)[None]
    t0, tot, n = time.time(), 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        xc, yc, vc = crops(x, y, wdisk)
        with gov.step():
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logit = model(xc)
            loss = loss_fn(logit.float(), yc, vc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
        tot += loss.item(); n += 1
    print(f"[{tag}] {stage} epoch {ep}: loss {tot / max(n, 1):.4f}, {n} steps, {(time.time() - t0) / 60:.1f} min",
          flush=True)


def governor():
    if run.ON_KAGGLE:
        class _No:
            def step(self):
                import contextlib
                return contextlib.nullcontext()
        return _No()
    sys.path.insert(0, str(ROOT.parent))
    from gpu_guard import Governor
    return Governor(target_util=72)


def arm_main(arm):
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run.set_fold(0)
    meta = json.load(open(run.CACHE / "meta.json"))
    gt = __import__("pickle").load(open(run.CACHE / "val_gt.pkl", "rb"))
    val_files = meta["val_files"][:4] if SMOKE else meta["val_files"]
    if SMOKE:
        keep = set(range(4))
        gt = [g for g in gt if g["frame"] in keep]
    ondisk = disk(device)
    gov = governor()

    model = QwenSeg(load_vision()).to(device)
    model.vis.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.vis.patch_embed.parameters():
        p.requires_grad_(False)
    model.vis.pos_embed.weight.requires_grad_(False)
    for b in model.vis.blocks[:FREEZE]:
        for p in b.parameters():
            p.requires_grad_(False)
    vit = [p for p in model.vis.parameters() if p.requires_grad]
    head = [p for n_, p in model.named_parameters() if not n_.startswith("vis.")]
    print(f"[{arm}] trainable: ViT {sum(p.numel() for p in vit) / 1e6:.0f}M, head {sum(p.numel() for p in head) / 1e6:.1f}M",
          flush=True)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    def stage(ds, epochs, name, val_at=(), lr=1.0, best=(-1.0, None, 0)):
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=3 if run.ON_KAGGLE else 0,
                                             drop_last=True, persistent_workers=run.ON_KAGGLE)
        opt = torch.optim.AdamW([{"params": vit, "lr": LR_VIT * lr}, {"params": head, "lr": LR_HEAD * lr}],
                                weight_decay=0.01)
        total = max(1, epochs * len(loader))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
        for ep in range(1, epochs + 1):
            t_ep = time.time()
            epoch_loop(model, opt, sched, scaler, loader, device, gov, arm, name, ep)
            late = (time.time() - t0) + (time.time() - t_ep) > BUDGET_H * 3600     # the next epoch would not fit
            if ep in val_at or (val_at and late and best[1] is None):
                r = score(val_maps(model, device, ondisk, meta, val_files), gt, quick=True)[0]
                print(f"[{arm}] {name} epoch {ep}: val PQ {r['PQ']:.4f} at t {r['thresh']} merge {r['merge']} "
                      f"area {r['min_area']}", flush=True)
                if r["PQ"] > best[0]:
                    best = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
            if late:
                print(f"[{arm}] {name}: time budget reached after epoch {ep}", flush=True)
                break
        return best

    t0 = time.time()
    n_train = 6 if SMOKE else len(meta["train_files"])
    if CONT:
        roots = [Path("/kaggle/input")] if run.ON_KAGGLE else [HERE / "out", ROOT / "kernels"]
        ck = next((p for r in roots for p in r.rglob(f"qwenseg_{arm[:-1]}.pt")), None)
        if ck is None:
            raise RuntimeError(f"qwenseg_{arm[:-1]}.pt not attached (kernel filament-qwen-seg)")
        sd = torch.load(ck, map_location="cpu")
        model.load_state_dict({k: v.float() for k, v in sd.items()})
        del sd
        r = score(val_maps(model, device, ondisk, meta, val_files), gt, quick=True)[0]
        print(f"[{arm}] loaded {ck}: val PQ {r['PQ']:.4f} at t {r['thresh']} merge {r['merge']} area {r['min_area']}",
              flush=True)
        start = (r["PQ"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
        best_pq, best_sd, best_ep = stage(run.Frames(n_train, "full"), CONT_EPOCHS, "continue", CONT_VAL_AT,
                                          lr=CONT_LR, best=start)
    else:
        if arm == "pre":
            items = pseudo_labels(device)
            stage(Pseudo(items), 1, "pretrain")
            print(f"[{arm}] pretrained on {len(items)} frames in {(time.time() - t0) / 60:.0f} min", flush=True)
        best_pq, best_sd, best_ep = stage(run.Frames(n_train, "full"), EPOCHS, "finetune", VAL_AT)
    model.load_state_dict(best_sd)
    print(f"[{arm}] best epoch {best_ep} (quick val PQ {best_pq:.4f}), {(time.time() - t0) / 60:.0f} min", flush=True)

    # maps in the U-Nets' dump format, then the full sweep
    tag = f"qwenseg_{arm}"
    root = WORK / f"dump_{tag}"
    te_dir = Path(meta["test_dir"])
    test_files = meta["test_files"][:2] if SMOKE else meta["test_files"]
    probs = val_maps(model, device, ondisk, meta, val_files)
    (root / "val_plain").mkdir(parents=True, exist_ok=True)
    (root / "test_plain").mkdir(parents=True, exist_ok=True)
    for f, p in zip(val_files, probs):
        cv2.imwrite(str(root / "val_plain" / f"{Path(f).stem}.png"), p)
    model.eval()
    for f in test_files:
        cv2.imwrite(str(root / "test_plain" / f"{Path(f).stem}.png"),
                    predict(model, cv2.imread(str(te_dir / f), cv2.IMREAD_GRAYSCALE), device, ondisk))
    shutil.make_archive(str(root), "tar", root)
    shutil.rmtree(root)
    grid = score(probs, gt, quick=False)
    res = dict(arm=arm, best_epoch=best_ep, quick_val_pq=best_pq, best=grid[0], top5=grid[:5],
               minutes=round((time.time() - t0) / 60, 1), smoke=SMOKE)
    json.dump(res, open(WORK / f"{tag}.json", "w"), indent=1, default=float)
    json.dump(dict(aug=tag, epoch=best_ep, post=dict(thresh=grid[0]["thresh"], merge=grid[0]["merge"],
                                                    min_area=grid[0]["min_area"]),
                   val_maps=True, val_files=val_files, test_files=test_files),
              open(WORK / f"dump_{tag}.json", "w"), indent=1)
    torch.save({k: v.half() for k, v in best_sd.items()}, WORK / f"{tag}.pt")
    print(f"[{arm}] fold 0 val PQ {grid[0]['PQ']:.4f} (SQ {grid[0]['SQ']:.3f}, RQ {grid[0]['RQ']:.3f}) at "
          f"t {grid[0]['thresh']} merge {grid[0]['merge']} area {grid[0]['min_area']}", flush=True)


def kaggle_main():
    t0 = time.time()
    run.set_fold(0)
    run.prepare(SMOKE, with_train=True)       # once, before the two arms share the cache
    print(f"fold 0 cache ready in {time.time() - t0:.0f}s", flush=True)
    procs = []
    for i, arm in enumerate(ARMS):
        log = open(WORK / f"log_qseg_{arm}.txt", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        procs.append((arm, subprocess.Popen([sys.executable, str(Path(__file__).resolve()), arm], env=env,
                                            stdout=log, stderr=subprocess.STDOUT)))
    last = time.time()
    while any(p.poll() is None for _, p in procs):
        time.sleep(20)
        if time.time() - last > 600:
            last = time.time()
            for arm, _ in procs:
                lines = [x for x in (WORK / f"log_qseg_{arm}.txt").read_text(errors="replace").splitlines() if x.strip()]
                print(f"  {(time.time() - t0) / 3600:.2f} h  {arm}: {lines[-1][-160:] if lines else ''}", flush=True)
    for arm, p in procs:
        print(f"\n===== {arm} (exit {p.returncode}) =====\n"
              + (WORK / f"log_qseg_{arm}.txt").read_text(errors="replace")[-3000:], flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ARMS:
        arm_main(sys.argv[1])
    else:
        kaggle_main()
