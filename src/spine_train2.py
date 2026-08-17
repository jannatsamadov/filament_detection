"""Train the spine U-Net with consensus targets and physical input channels.

What changed from spine_train.py, and why:

  consensus targets   Each photograph is one sample and the target is the
                      fraction of annotators who marked the pixel. Previously
                      each annotator's set was a separate sample, which showed a
                      disputed filament once as foreground and twice as
                      background; the model sided with the majority and recovers
                      71% of filaments two experts agree on against 29% of those
                      only one marked.

  physical channels   Fibril coherence and orientation relative to the local
                      heliographic east-west, latitude, distance to plage. The
                      image-only model reached PQ 0.28 officially, and 84% of
                      its false positives were dark structures no annotator
                      marked -- it had learned "dark and elongated", which is
                      not what a filament is.

  longer schedule     The first run stopped at 20 epochs with the loss still
                      falling.

Validation still scores each annotator's set separately against that annotator's
own labels, so the numbers stay comparable with the earlier runs.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))          # shared: gpu_lock.py, gpu_guard.py
from spine_data import (ConsensusPatchDataset, photograph_split, collate,
                        load_annotations, make_split, PHYS_CHANNELS, PHYSICS,
                        FULL, PRE, DISK_CX, DISK_CY, DISK_R, OFFSET, SCALE)  # noqa: E402
from spine_model import build_spine_model, SpineLoss     # noqa: E402
from spine_infer import assemble                         # noqa: E402
from gpu_guard import Governor, gpu_status, other_gpu_processes  # noqa: E402
from gpu_lock import GpuLock                                     # noqa: E402
from metrics import pq_from_masks, aggregate             # noqa: E402
from evaluate import gt_masks_full                       # noqa: E402

CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


def load_stack(stem, split="train", use_physics=True):
    """Flattened image plus the physical channels, as one array."""
    img = np.asarray(Image.open(PRE / split / f"{stem}.png"),
                     dtype=np.float32) / SCALE + OFFSET
    planes = [img]
    if use_physics:
        for k in PHYS_CHANNELS:
            planes.append(np.asarray(Image.open(PHYSICS / split / stem / f"{k}.png"),
                                     dtype=np.float32) / 255.0)
    return np.stack(planes)


@torch.inference_mode()
def predict_frame(model, stack, device, patch=512, stride=384, tta=False):
    """Tile a full frame; optionally average over the four geometry-preserving
    mirrors, which is the only symmetry group left once latitude is an input."""
    c, h, w = stack.shape
    acc = np.zeros((3, h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    ramp = (np.hanning(patch)[:, None] * np.hanning(patch)[None, :]).astype(np.float32) + 1e-3

    variants = [(False, False)]
    if tta:
        variants = [(False, False), (True, False), (False, True), (True, True)]

    i_sin2t = 1 + PHYS_CHANNELS.index("sin2t") if c > 1 else None
    i_sinlat = 1 + PHYS_CHANNELS.index("sinlat") if c > 1 else None

    for fh, fv in variants:
        s = stack
        if fh:
            s = s[:, :, ::-1]
        if fv:
            s = s[:, ::-1]
        s = np.ascontiguousarray(s)
        if c > 1:
            if fh != fv:
                s[i_sin2t] = 1.0 - s[i_sin2t]
            if fv:
                s[i_sinlat] = 1.0 - s[i_sinlat]

        out = np.zeros((3, h, w), np.float32)
        cnt = np.zeros((h, w), np.float32)
        ys = list(range(0, h - patch + 1, stride)) or [0]
        xs = list(range(0, w - patch + 1, stride)) or [0]
        if ys[-1] != h - patch:
            ys.append(h - patch)
        if xs[-1] != w - patch:
            xs.append(w - patch)
        for y0 in ys:
            tiles = [s[:, y0:y0 + patch, x0:x0 + patch] for x0 in xs]
            batch = torch.from_numpy(np.stack(tiles)).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                o = model(batch)
            sem = torch.sigmoid(o["sem"].float())[:, 0].cpu().numpy()
            spn = torch.sigmoid(o["spine"].float())[:, 0].cpu().numpy()
            rad = o["radius"].float()[:, 0].cpu().numpy()
            for x0, a, b, r in zip(xs, sem, spn, rad):
                out[0, y0:y0 + patch, x0:x0 + patch] += a * ramp
                out[1, y0:y0 + patch, x0:x0 + patch] += b * ramp
                out[2, y0:y0 + patch, x0:x0 + patch] += r * ramp
                cnt[y0:y0 + patch, x0:x0 + patch] += ramp
        out /= np.maximum(cnt, 1e-6)
        if fv:
            out = out[:, ::-1]
        if fh:
            out = out[:, :, ::-1]
        acc += out
        wsum += 1.0
    acc /= wsum.max()
    return acc[0], acc[1], acc[2]


def validate(model, val_ids, by_id, fname, device, limit=25, governor=None,
             spine_thresh=0.5, sem_thresh=0.5, min_spine_px=100, use_physics=True,
             tta=False):
    model.eval()
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    cache, results = {}, []
    for iid in val_ids[:limit]:
        stem = Path(fname[iid]).stem
        if stem not in cache:
            stack = load_stack(stem, "train", use_physics)
            sem, spn, rad = predict_frame(model, stack, device, tta=tta)
            cache[stem] = (sem * ondisk, spn * ondisk, rad)
            if governor is not None:
                governor._regulate()
        sem, spn, rad = cache[stem]
        preds = assemble(spn, rad, sem, spine_thresh=spine_thresh,
                         sem_thresh=sem_thresh, min_spine_px=min_spine_px)
        results.append(pq_from_masks(gt_masks_full(by_id[iid]), preds))
    model.train()
    return aggregate(results), aggregate(results, mode="image")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--patches-per-image", type=int, default=6)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--val-limit", type=int, default=25)
    ap.add_argument("--target-util", type=int, default=55)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--tag", default="spine_v2")
    ap.add_argument("--resume", default="")
    ap.add_argument("--start-below", type=int, default=68)
    ap.add_argument("--wait-for-gpu", action="store_true",
                    help="queue behind whoever holds the shared GPU lock")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! no CUDA"); sys.exit(1)
    others = other_gpu_processes()
    if others:
        print(f"!! another process is on the GPU: {others}"); sys.exit(1)
    _, t0, _ = gpu_status()
    while t0 is not None and t0 > args.start_below:
        print(f"   GPU at {t0}C, waiting for {args.start_below}C", flush=True)
        time.sleep(20); _, t0, _ = gpu_status()
    torch.backends.cudnn.benchmark = True

    use_physics = not args.no_physics
    train_files, _ = photograph_split()
    by_id, fname = load_annotations()
    _, val_ids, _ = make_split(fname)

    ds = ConsensusPatchDataset(train_files, patch=args.patch, augment=True,
                               patches_per_image=args.patches_per_image,
                               use_physics=use_physics)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, collate_fn=collate,
                                     pin_memory=True, drop_last=True,
                                     persistent_workers=args.workers > 0)

    model = build_spine_model(args.encoder, in_channels=ds.in_channels).to(device)
    crit = SpineLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(dl), 1), pct_start=0.15)

    start_epoch, best = 0, -1.0
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
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.encoder} U-Net, {n_par/1e6:.2f} M params, {ds.in_channels} input "
          f"channels ({'image + physics' if use_physics else 'image only'})")
    print(f"{len(train_files)} training photographs, {len(dl)} steps/epoch, "
          f"{args.epochs} epochs")

    log = ROOT / f"trainlog_{args.tag}.csv"
    if not log.exists():
        log.write_text("epoch,step,loss,sem_dice,spine_dice,radius_l1,lr,temp,secs\n")
    vlog = ROOT / f"valog_{args.tag}.json"

    try:
      for epoch in range(start_epoch, args.epochs):
          lock.progress(f"epoch {epoch}/{args.epochs}")
          t_ep, run, parts = time.time(), [], []
          for i, batch in enumerate(dl):
              batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
              with gov.step():
                  with torch.amp.autocast("cuda"):
                      out = model(batch["image"])
                      loss, comp = crit(out, batch)
                  opt.zero_grad(set_to_none=True)
                  scaler.scale(loss).backward()
                  scaler.unscale_(opt)
                  torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                  scaler.step(opt); scaler.update(); sched.step()
                  torch.cuda.synchronize()
              run.append(float(loss.detach()))
              parts.append({k: float(v) for k, v in comp.items()})
              if (i + 1) % 150 == 0:
                  _, t, _ = gpu_status()
                  el = time.time() - t_ep
                  pm = {k: np.mean([p[k] for p in parts[-150:]]) for k in parts[0]}
                  print(f"  e{epoch} [{i+1}/{len(dl)}] loss {np.mean(run[-150:]):.4f} "
                        f"sem_dice {pm['sem_dice']:.3f} spine_dice {pm['spine_dice']:.3f} "
                        f"| {t}C | {el:.0f}s", flush=True)
                  with open(log, "a") as fh:
                      fh.write(f"{epoch},{i+1},{np.mean(run[-150:]):.5f},"
                               f"{pm['sem_dice']:.5f},{pm['spine_dice']:.5f},"
                               f"{pm['radius_l1']:.5f},{sched.get_last_lr()[0]:.6f},"
                               f"{t},{el:.0f}\n")

          pm = {k: np.mean([p[k] for p in parts]) for k in parts[0]}
          print(f"epoch {epoch} done in {(time.time()-t_ep)/60:.1f} min, "
                f"loss {np.mean(run):.4f} (sem_dice {pm['sem_dice']:.3f}, "
                f"spine_dice {pm['spine_dice']:.3f}) | {gov.summary()}", flush=True)
          torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                      "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                      "epoch": epoch, "args": vars(args)},
                     CKPT / f"{args.tag}_last.pt")

          if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
              tv = time.time()
              dsm, imm = validate(model, val_ids, by_id, fname, device,
                                  limit=args.val_limit, governor=gov,
                                  use_physics=use_physics)
              print(f"  VAL ({args.val_limit} sets, {time.time()-tv:.0f}s): "
                    f"PQ={dsm['PQ']:.4f} SQ={dsm['SQ']:.4f} RQ={dsm['RQ']:.4f} "
                    f"TP={dsm['TP']} FP={dsm['FP']} FN={dsm['FN']}  "
                    f"[image-only model reached 0.333 val / 0.28 official]", flush=True)
              with open(vlog, "a") as fh:
                  fh.write(json.dumps({"epoch": epoch, "dataset": dsm, "image": imm}) + "\n")
              if dsm["PQ"] > best:
                  best = dsm["PQ"]
                  torch.save({"model": model.state_dict(), "epoch": epoch, "PQ": best,
                              "args": vars(args)}, CKPT / f"{args.tag}_best.pt")
                  print(f"  new best PQ {best:.4f}", flush=True)
    finally:
        lock.release()

    print(f"\nfinished. best PQ {best:.4f}. {gov.summary()}")


if __name__ == "__main__":
    main()
