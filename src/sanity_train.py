"""Short train-then-evaluate run, to catch a collapsing configuration in minutes.

The first full attempt spent 40 minutes before revealing that the model had
collapsed to predicting nothing. This trains for a few hundred steps at the real
settings and immediately checks that the model still emits non-empty masks that
land on filaments, which is the cheapest possible guard against repeating that.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from data import FilamentDataset, load_annotations, make_split, collate  # noqa: E402
from model import build_model, forward_images                            # noqa: E402
from thermal import Governor, gpu_status, other_gpu_processes            # noqa: E402
from evaluate import predict_masks, gt_masks_full                        # noqa: E402
from metrics import pq_from_masks, aggregate                             # noqa: E402


def check(model, ds, by_id, device, n_img=5, tag=""):
    model.eval()
    best_ious, n_pred, areas = [], 0, []
    res = []
    for i in range(n_img):
        img, _ = ds[i]
        preds, _ = predict_masks(model, img, device, score_thresh=0.3)
        gts = gt_masks_full(by_id[ds.ids[i]])
        n_pred += len(preds)
        areas += [int(p.sum()) for p in preds]
        for g in gts:
            b = 0.0
            for p in preds:
                u = (g | p).sum()
                if u:
                    b = max(b, (g & p).sum() / u)
            best_ious.append(b)
        res.append(pq_from_masks(gts, preds))
    a = aggregate(res)
    print(f"  [{tag}] non-empty preds {n_pred:4d} | pred area median "
          f"{int(np.median(areas)) if areas else 0:5d} | best IoU per GT: "
          f"mean {np.mean(best_ious):.3f} max {max(best_ious):.3f} | "
          f"PQ {a['PQ']:.4f} TP {a['TP']} FP {a['FP']} FN {a['FN']}", flush=True)
    model.train()
    return np.mean(best_ious), max(best_ious), n_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--check-every", type=int, default=200)
    ap.add_argument("--target-util", type=int, default=55)
    ap.add_argument("--no-small-anchors", action="store_true",
                    help="keep torchvision's default anchor sizes")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--constant-lr", action="store_true",
                    help="skip OneCycle; hold lr fixed so a collapse cannot be "
                         "blamed on the schedule")
    args = ap.parse_args()

    device = torch.device("cuda")
    others = other_gpu_processes()
    if others:
        print(f"!! another GPU process is running: {others}"); sys.exit(1)
    torch.backends.cudnn.benchmark = True

    by_id, fname = load_annotations()
    train_ids, val_ids, _ = make_split(fname)
    ds_tr = FilamentDataset(train_ids, by_id, fname, size=args.size, augment=True)
    ds_va = FilamentDataset(val_ids, by_id, fname, size=args.size, augment=False)
    dl = torch.utils.data.DataLoader(ds_tr, batch_size=1, shuffle=True, num_workers=2,
                                     collate_fn=collate, pin_memory=True)

    model = build_model(size=args.size,
                        small_anchors=not args.no_small_anchors).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.constant_lr:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        sched.total_steps = 10 ** 9
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=max(args.steps // args.accum, 1),
            pct_start=0.15)

    gov = Governor(target_util=args.target_util)
    print(f"lr {args.lr}, {args.steps} steps, size {args.size}, "
          f"anchors={'default' if args.no_small_anchors else 'small'}, "
          f"amp={use_amp}, schedule={'constant' if args.constant_lr else 'onecycle'}")
    print("  before any training:")
    check(model, ds_va, by_id, device, tag="step 0")

    t0, run, i = time.time(), [], 0
    opt.zero_grad(set_to_none=True)
    while i < args.steps:
        for images, targets in dl:
            if i >= args.steps:
                break
            with gov.step():
                imgs = forward_images(model, [im.to(device) for im in images])
                tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
                if tgts[0]["boxes"].numel() == 0:
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    losses = model(imgs, tgts)
                    loss = sum(losses.values())
                scaler.scale(loss / args.accum).backward()
                if (i + 1) % args.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 10.0)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if sched.last_epoch < sched.total_steps - 1:
                        sched.step()
                torch.cuda.synchronize()
            run.append(float(loss.detach()))
            i += 1
            if i % 100 == 0:
                lc = {k: round(float(v.detach()), 4) for k, v in losses.items()}
                print(f"    parts {lc}", flush=True)
                print(f"  step {i}: loss {np.mean(run[-100:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.5f} | {gov.summary()}", flush=True)
            if i % args.check_every == 0:
                check(model, ds_va, by_id, device, tag=f"step {i}")

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    m_iou, mx_iou, n_pred = check(model, ds_va, by_id, device, tag="final")
    print()
    if n_pred == 0:
        print("VERDICT: COLLAPSED -- the model predicts nothing. Lower the lr further.")
    elif mx_iou < 0.05:
        print("VERDICT: BROKEN -- predictions never touch a filament. Not a training-length issue.")
    elif mx_iou < 0.3:
        print("VERDICT: alive but weak -- predictions land near filaments. More epochs needed.")
    else:
        print("VERDICT: healthy -- real overlap already. Launch the full run.")


if __name__ == "__main__":
    main()
