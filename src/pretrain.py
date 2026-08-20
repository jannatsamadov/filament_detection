"""Teach the encoder what the Sun looks like, before asking it to find filaments.

The 566 annotated photographs are the binding limit here, and the encoder starts
from ImageNet weights -- a network trained on cats and cars. This pretrains it on
3772 unlabelled GONG H-alpha frames instead, spanning 2010 to 2024 and balanced
across the six stations, so it arrives at the labelled data already knowing how a
solar disk is put together.

The task is reconstruction from a masked view: blank out square blocks of the
patch and ask the network to fill them in. Nothing is labelled -- to fill a hole
convincingly the network has to have learned that filaments are long dark threads
that continue past the gap, that plage is bright and clumpy, that the
chromospheric network has a scale. Those are exactly the features the filament
task needs.

Input is the flattened, limb-masked image, identical to what fine-tuning uses;
pretraining on raw frames would teach features of limb darkening that the
fine-tuning data no longer contains.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))          # shared: gpu_lock.py, gpu_guard.py
from spine_model import SpineUNet                                   # noqa: E402
from spine_data import DISK_CX, DISK_CY, DISK_R, FULL, OFFSET, SCALE  # noqa: E402
from gpu_guard import Governor, gpu_status, other_gpu_processes      # noqa: E402
from gpu_lock import GpuLock                                         # noqa: E402

UNLAB = ROOT / "preprocessed" / "unlabelled"
CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


class MaskedPatches(Dataset):
    """Random on-disk patches, with square blocks blanked out."""

    def __init__(self, files, patch=512, patches_per_image=2, block=48,
                 mask_frac=0.5, augment=True, cache_size=8):
        self.files = list(files)
        self.patch, self.per = patch, patches_per_image
        self.block, self.mask_frac = block, mask_frac
        self.augment = augment
        self._cache, self._cache_size = {}, cache_size
        yy, xx = np.mgrid[0:FULL, 0:FULL]
        self.disk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.99) ** 2

    def __len__(self):
        return len(self.files) * self.per

    def _img(self, path):
        hit = self._cache.get(path)
        if hit is not None:
            return hit
        a = np.asarray(Image.open(path), dtype=np.float32) / SCALE + OFFSET
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[path] = a
        return a

    def __getitem__(self, i):
        path = self.files[i % len(self.files)]
        a = self._img(path)
        p = self.patch
        # sample a patch that is mostly on the disk
        for _ in range(12):
            y0 = random.randint(0, FULL - p)
            x0 = random.randint(0, FULL - p)
            sub_disk = self.disk[y0:y0 + p, x0:x0 + p]
            if sub_disk.mean() > 0.85:
                break
        img = a[y0:y0 + p, x0:x0 + p].copy()
        valid = sub_disk.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img, valid = img[:, ::-1].copy(), valid[:, ::-1].copy()
            if random.random() < 0.5:
                img, valid = img[::-1].copy(), valid[::-1].copy()
            k = random.randrange(4)
            if k:
                img, valid = np.rot90(img, k).copy(), np.rot90(valid, k).copy()

        # block mask
        nb = p // self.block
        keep = np.random.rand(nb, nb) > self.mask_frac
        mask = np.kron(keep, np.ones((self.block, self.block), np.float32))
        mask = mask[:p, :p]
        if mask.shape != img.shape:
            m = np.ones_like(img)
            m[:mask.shape[0], :mask.shape[1]] = mask
            mask = m

        return {
            "target": torch.from_numpy(img)[None],
            "input": torch.from_numpy(img * mask)[None],
            "mask": torch.from_numpy(mask)[None],       # 1 = shown, 0 = hidden
            "valid": torch.from_numpy(valid)[None],
        }


class ReconUNet(nn.Module):
    """SpineUNet's body with a single reconstruction output."""

    def __init__(self, encoder="resnet34"):
        super().__init__()
        self.net = SpineUNet(encoder=encoder, pretrained=True, in_channels=1)
        c = self.net.head_sem.in_channels
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        n = self.net
        s = n.stem(x)
        e1 = n.enc1(n.pool(s))
        e2 = n.enc2(e1)
        e3 = n.enc3(e2)
        e4 = n.enc4(e3)
        d = n.dec4(torch.cat([n._up(e4, e3), e3], 1))
        d = n.dec3(torch.cat([n._up(d, e2), e2], 1))
        d = n.dec2(torch.cat([n._up(d, e1), e1], 1))
        d = n.dec1(torch.cat([n._up(d, s), s], 1))
        d = n.dec0(n._up(d, x))
        return self.head(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--patches-per-image", type=int, default=2)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--target-util", type=int, default=55)
    ap.add_argument("--tag", default="pretrain_r34")
    ap.add_argument("--resume", default="")
    ap.add_argument("--start-below", type=int, default=68)
    ap.add_argument("--wait-for-gpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! no CUDA"); sys.exit(1)
    others = other_gpu_processes()
    if others:
        print(f"!! another training process is on the GPU: {others}"); sys.exit(1)
    _, t0c, _ = gpu_status()
    while t0c is not None and t0c > args.start_below:
        print(f"   GPU at {t0c}C, waiting for {args.start_below}C", flush=True)
        time.sleep(20); _, t0c, _ = gpu_status()
    torch.backends.cudnn.benchmark = True

    files = sorted(UNLAB.glob("*.png"))
    if not files:
        print(f"!! no images in {UNLAB}"); sys.exit(1)
    ds = MaskedPatches(files, patch=args.patch, patches_per_image=args.patches_per_image,
                       block=args.block, mask_frac=args.mask_frac)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    model = ReconUNet(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(dl), 1), pct_start=0.1)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start_epoch = ck["epoch"] + 1
        try:
            sched.load_state_dict(ck["sched"])
        except Exception:
            pass
        print(f"resumed at epoch {start_epoch}")

    lock = GpuLock("filament_detection", args.tag, wait=args.wait_for_gpu).acquire()
    gov = Governor(target_util=args.target_util)
    print(f"{len(files)} unlabelled images, {len(dl)} steps/epoch, {args.epochs} epochs")
    print(f"masking {100*args.mask_frac:.0f}% in {args.block}px blocks")

    log = ROOT / f"trainlog_{args.tag}.csv"
    if not log.exists():
        log.write_text("epoch,step,loss_hidden,loss_all,lr,temp,secs\n")

    try:
        for epoch in range(start_epoch, args.epochs):
            lock.progress(f"epoch {epoch}/{args.epochs}")
            t_ep, run, run_all = time.time(), [], []
            for i, b in enumerate(dl):
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                with gov.step():
                    with torch.amp.autocast("cuda"):
                        out = model(b["input"])
                        hidden = (1.0 - b["mask"]) * b["valid"]
                        # score only what was hidden, and only on the solar disk
                        loss = (F.l1_loss(out, b["target"], reduction="none") * hidden).sum()
                        loss = loss / hidden.sum().clamp(min=1.0)
                        with torch.no_grad():
                            all_l = ((F.l1_loss(out, b["target"], reduction="none")
                                      * b["valid"]).sum() / b["valid"].sum().clamp(min=1.0))
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update(); sched.step()
                    torch.cuda.synchronize()
                run.append(float(loss.detach())); run_all.append(float(all_l))
                if (i + 1) % 200 == 0:
                    _, t, _ = gpu_status()
                    el = time.time() - t_ep
                    print(f"  e{epoch} [{i+1}/{len(dl)}] hidden {np.mean(run[-200:]):.5f} "
                          f"all {np.mean(run_all[-200:]):.5f} | {t}C | {el:.0f}s", flush=True)
                    with open(log, "a") as fh:
                        fh.write(f"{epoch},{i+1},{np.mean(run[-200:]):.6f},"
                                 f"{np.mean(run_all[-200:]):.6f},"
                                 f"{sched.get_last_lr()[0]:.6f},{t},{el:.0f}\n")

            print(f"epoch {epoch} done in {(time.time()-t_ep)/60:.1f} min, "
                  f"hidden {np.mean(run):.5f} | {gov.summary()}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       CKPT / f"{args.tag}_last.pt")
            # the encoder alone, in the form build_spine_model expects
            torch.save({k: v for k, v in model.net.state_dict().items()
                        if k.startswith(("stem", "pool", "enc", "dec"))},
                       CKPT / f"{args.tag}_encoder.pt")
    finally:
        lock.release()

    print(f"\nfinished. encoder written to {CKPT / f'{args.tag}_encoder.pt'}")


if __name__ == "__main__":
    main()
