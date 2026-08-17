"""Train the spine U-Net, with full-frame Panoptic Quality validation.

Validation runs the real inference path -- tile the 2048 frame, stitch the three
maps, assemble instances, score with PQ -- so the number reported during training
is the number that matters, not a proxy.

Feeding the assembly its own ground truth scores PQ 0.734, which is the ceiling
of this pipeline and already twice what two human annotators reach with each
other (0.348). Anything below that is the model, not the representation.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
from spine_data import (SpinePatchDataset, load_annotations, make_split, collate,
                        FULL, PRE, DISK_CX, DISK_CY, DISK_R, OFFSET, SCALE)  # noqa: E402
from spine_model import build_spine_model, SpineLoss                        # noqa: E402
from spine_infer import assemble                                            # noqa: E402
from thermal import Governor, gpu_status, other_gpu_processes               # noqa: E402
from metrics import pq_from_masks, aggregate                                # noqa: E402
from evaluate import gt_masks_full                                          # noqa: E402

from PIL import Image

CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


@torch.inference_mode()
def predict_frame(model, image, device, patch=512, stride=384, amp=True):
    """Tile a full 2048 frame and stitch the three maps with a cosine window."""
    h, w = image.shape
    sem = np.zeros((h, w), np.float32)
    spn = np.zeros((h, w), np.float32)
    rad = np.zeros((h, w), np.float32)
    acc = np.zeros((h, w), np.float32)

    ramp = np.hanning(patch)[:, None] * np.hanning(patch)[None, :]
    ramp = ramp.astype(np.float32) + 1e-3

    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)

    for y0 in ys:
        tiles, coords = [], []
        for x0 in xs:
            tiles.append(image[y0:y0 + patch, x0:x0 + patch])
            coords.append((y0, x0))
        batch = torch.from_numpy(np.stack(tiles))[:, None].to(device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(batch)
        s = torch.sigmoid(out["sem"].float())[:, 0].cpu().numpy()
        p = torch.sigmoid(out["spine"].float())[:, 0].cpu().numpy()
        r = out["radius"].float()[:, 0].cpu().numpy()
        for (y0, x0), si, pi, ri in zip(coords, s, p, r):
            sem[y0:y0 + patch, x0:x0 + patch] += si * ramp
            spn[y0:y0 + patch, x0:x0 + patch] += pi * ramp
            rad[y0:y0 + patch, x0:x0 + patch] += ri * ramp
            acc[y0:y0 + patch, x0:x0 + patch] += ramp
    acc = np.maximum(acc, 1e-6)
    return sem / acc, spn / acc, rad / acc


def load_frame(fname, split="train"):
    p = PRE / split / (Path(fname).stem + ".png")
    return np.asarray(Image.open(p), dtype=np.float32) / SCALE + OFFSET


def validate(model, val_ids, by_id, fname, device, limit=25, governor=None,
             spine_thresh=0.4, sem_thresh=0.5, min_spine_px=12, min_area=60,
             verbose=False, return_results=False):
    model.eval()
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    results = []
    for i, iid in enumerate(val_ids[:limit]):
        img = load_frame(fname[iid])
        sem, spn, rad = predict_frame(model, img, device)
        sem *= ondisk; spn *= ondisk
        preds = assemble(spn, rad, sem, spine_thresh=spine_thresh,
                         sem_thresh=sem_thresh, min_spine_px=min_spine_px,
                         min_area=min_area)
        results.append(pq_from_masks(gt_masks_full(by_id[iid]), preds))
        if governor is not None:
            governor._regulate()
        if verbose and (i + 1) % 10 == 0:
            print(f"    val {i+1}/{limit} running PQ={aggregate(results)['PQ']:.4f}", flush=True)
    model.train()
    if return_results:
        return aggregate(results), aggregate(results, mode="image"), results
    return aggregate(results), aggregate(results, mode="image")


@torch.inference_mode()
def cached_maps(model, val_ids, fname, device, limit):
    """Predict the three maps once so a threshold sweep does not re-run the net.

    A sweep over thresholds only changes the assembly, never the maps, and the
    network pass is the expensive half -- caching turns an 18-point sweep from
    an hour of GPU time into one pass plus cheap CPU work.
    """
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    model.eval()
    out = {}
    for iid in val_ids[:limit]:
        sem, spn, rad = predict_frame(model, load_frame(fname[iid]), device)
        out[iid] = (sem * ondisk, spn * ondisk, rad)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--patches-per-image", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--val-limit", type=int, default=25)
    ap.add_argument("--target-util", type=int, default=55)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="spine_unet_r34")
    ap.add_argument("--resume", default="")
    ap.add_argument("--start-below", type=int, default=68)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! no CUDA -- refusing to train on CPU"); sys.exit(1)
    others = other_gpu_processes()
    if others:
        print(f"!! another process is already computing on the GPU: {others}"); sys.exit(1)
    _, t0, _ = gpu_status()
    while t0 is not None and t0 > args.start_below:
        print(f"   GPU at {t0}C, waiting for {args.start_below}C", flush=True)
        time.sleep(20)
        _, t0, _ = gpu_status()
    torch.backends.cudnn.benchmark = True

    by_id, fname = load_annotations()
    train_ids, val_ids, val_files = make_split(fname)
    print(f"split: {len(train_ids)} train annotation-sets, {len(val_ids)} val "
          f"({len(val_files)} photographs)")

    ds = SpinePatchDataset(train_ids, by_id, fname, patch=args.patch, augment=True,
                           patches_per_image=args.patches_per_image)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, collate_fn=collate,
                                     pin_memory=True, drop_last=True,
                                     persistent_workers=args.workers > 0)

    model = build_spine_model(args.encoder).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    crit = SpineLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(args.epochs * len(dl), 1), pct_start=0.2)

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

    gov = Governor(target_util=args.target_util)
    print(f"{args.encoder} U-Net, {n_par/1e6:.2f} M parameters, patch {args.patch}, "
          f"batch {args.batch}, {len(dl)} steps/epoch")
    print(f"thermal: target {args.target_util}% duty, pause 78C, resume 70C")

    log = ROOT / f"trainlog_{args.tag}.csv"
    if not log.exists():
        log.write_text("epoch,step,loss,sem_bce,sem_dice,spine_bce,spine_dice,"
                       "radius_l1,lr,temp,secs\n")
    vlog = ROOT / f"valog_{args.tag}.json"

    best = -1.0
    for epoch in range(start_epoch, args.epochs):
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
                scaler.step(opt); scaler.update()
                sched.step()
                torch.cuda.synchronize()
            run.append(float(loss.detach()))
            parts.append({k: float(v) for k, v in comp.items()})
            if (i + 1) % 100 == 0:
                _, t, _ = gpu_status()
                el = time.time() - t_ep
                pm = {k: np.mean([p[k] for p in parts[-100:]]) for k in parts[0]}
                print(f"  e{epoch} [{i+1}/{len(dl)}] loss {np.mean(run[-100:]):.4f} "
                      f"sem_dice {pm['sem_dice']:.3f} spine_dice {pm['spine_dice']:.3f} "
                      f"rad_l1 {pm['radius_l1']:.2f} | {t}C | {el:.0f}s", flush=True)
                with open(log, "a") as fh:
                    fh.write(f"{epoch},{i+1},{np.mean(run[-100:]):.5f},"
                             f"{pm['sem_bce']:.5f},{pm['sem_dice']:.5f},"
                             f"{pm['spine_bce']:.5f},{pm['spine_dice']:.5f},"
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
                                limit=args.val_limit, governor=gov)
            print(f"  VAL ({args.val_limit} frames, {time.time()-tv:.0f}s): "
                  f"PQ={dsm['PQ']:.4f} SQ={dsm['SQ']:.4f} RQ={dsm['RQ']:.4f} "
                  f"TP={dsm['TP']} FP={dsm['FP']} FN={dsm['FN']} | "
                  f"per-image PQ={imm['PQ']:.4f}  [ceiling 0.734, human 0.348]",
                  flush=True)
            with open(vlog, "a") as fh:
                fh.write(json.dumps({"epoch": epoch, "dataset": dsm, "image": imm}) + "\n")
            if dsm["PQ"] > best:
                best = dsm["PQ"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "PQ": best, "args": vars(args)},
                           CKPT / f"{args.tag}_best.pt")
                print(f"  new best PQ {best:.4f} -> {args.tag}_best.pt", flush=True)

    print(f"\nfinished. best PQ {best:.4f}. {gov.summary()}")


if __name__ == "__main__":
    main()
