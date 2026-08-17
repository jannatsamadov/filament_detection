"""Train Mask R-CNN on MAGFiLO under this machine's thermal limits.

The GPU fan on this laptop does not work, so every step is duty-cycled by
src/thermal.py: utilisation is held at or below 72%, and the run pauses above
78 C and resumes near 70 C. At the default 50% target that costs roughly 2x wall
time. Checkpoints are written every epoch so a run can be stopped and resumed.

The script refuses to start if another process is already computing on the GPU.
Two concurrent trainings once drove the card to 81 C: duty-cycling only bounds
the load of the process doing the cycling, so a second uncontrolled run defeats
the whole policy. Pass --allow-shared-gpu to override deliberately.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from data import FilamentDataset, load_annotations, make_split, collate  # noqa: E402
from model import build_model, forward_images                            # noqa: E402
from thermal import Governor, gpu_status, other_gpu_processes            # noqa: E402
from evaluate import evaluate                                            # noqa: E402

CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--val-limit", type=int, default=60)
    ap.add_argument("--smoke", type=int, default=0, help="run N steps and stop")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--tag", type=str, default="maskrcnn_r50")
    ap.add_argument("--target-util", type=int, default=50)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="start even if another process is already on the GPU")
    ap.add_argument("--start-below", type=int, default=68,
                    help="refuse to start until the GPU is at or below this temperature")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! no CUDA -- refusing to train on CPU"); sys.exit(1)
    torch.backends.cudnn.benchmark = True

    if not args.allow_shared_gpu:
        others = other_gpu_processes()
        if others:
            print(f"!! another process is already computing on the GPU: {others}\n"
                  f"   Duty-cycling cannot bound a load it does not own. Stop it first, "
                  f"or pass --allow-shared-gpu.")
            sys.exit(1)
    u0, t0, _ = gpu_status()
    if t0 is not None and t0 > args.start_below:
        print(f"!! GPU is at {t0}C; waiting for it to fall to {args.start_below}C "
              f"before starting", flush=True)
        while True:
            time.sleep(15)
            _, t0, _ = gpu_status()
            if t0 is None or t0 <= args.start_below:
                break
        print(f"   starting at {t0}C", flush=True)

    by_id, fname = load_annotations()
    train_ids, val_ids, val_files = make_split(fname)
    print(f"split: {len(train_ids)} train annotation-sets, {len(val_ids)} val "
          f"({len(val_files)} val photographs, grouped by filename)")

    ds_tr = FilamentDataset(train_ids, by_id, fname, size=args.size, augment=True)
    ds_va = FilamentDataset(val_ids, by_id, fname, size=args.size, augment=False)
    dl = torch.utils.data.DataLoader(
        ds_tr, batch_size=1, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)

    model = build_model(size=args.size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    total_steps = max(args.epochs * len(dl) // args.accum, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.15)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); start_epoch = ck["epoch"] + 1
        try:
            sched.load_state_dict(ck["sched"])
        except Exception:
            pass
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    gov = Governor(target_util=args.target_util)
    u, t, m = gpu_status()
    print(f"GPU before start: util={u}% temp={t}C mem={m}MiB")
    print(f"thermal policy: target {args.target_util}% util, hard cap 72%, "
          f"pause at 78C, resume at 70C  (initial sleep factor "
          f"{gov.sleep_factor:.2f}x compute time)")

    log_path = ROOT / f"trainlog_{args.tag}.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,loss,loss_classifier,loss_box,loss_mask,"
                            "loss_objectness,loss_rpn_box,lr,util,temp,mem,secs\n")

    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        run = []
        opt.zero_grad(set_to_none=True)
        for i, (images, targets) in enumerate(dl):
            with gov.step():
                imgs = forward_images(model, [im.to(device, non_blocking=True) for im in images])
                tgts = [{k: v.to(device) for k, v in t_.items()} for t_ in targets]
                if tgts[0]["boxes"].numel() == 0:
                    continue
                with torch.amp.autocast("cuda"):
                    losses = model(imgs, tgts)
                    loss = sum(losses.values())
                scaler.scale(loss / args.accum).backward()
                if (i + 1) % args.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 10.0)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if sched.last_epoch < total_steps - 1:
                        sched.step()
                    step += 1
                # CUDA work is queued asynchronously; without this the governor
                # would time only the enqueue and duty-cycle against a fiction.
                torch.cuda.synchronize()
            run.append(float(loss.detach()))

            if (i + 1) % 50 == 0:
                u, t, m = gpu_status()
                el = time.time() - t_epoch
                print(f"  e{epoch} [{i+1}/{len(dl)}] loss {np.mean(run[-50:]):.4f} "
                      f"lr {sched.get_last_lr()[0]:.5f} | util {u}% {t}C {m}MiB | "
                      f"{el:.0f}s, {el/(i+1)*(len(dl)-i-1):.0f}s left", flush=True)
                def _v(key):
                    x = losses.get(key)
                    return float(x.detach()) if hasattr(x, "detach") else 0.0
                with open(log_path, "a") as fh:
                    fh.write(f"{epoch},{i+1},{np.mean(run[-50:]):.5f},"
                             f"{_v('loss_classifier'):.5f},"
                             f"{_v('loss_box_reg'):.5f},"
                             f"{_v('loss_mask'):.5f},"
                             f"{_v('loss_objectness'):.5f},"
                             f"{_v('loss_rpn_box_reg'):.5f},"
                             f"{sched.get_last_lr()[0]:.6f},{u},{t},{m},{el:.0f}\n")

            if args.smoke and i + 1 >= args.smoke:
                print(f"\nSMOKE TEST DONE: {args.smoke} steps in "
                      f"{time.time()-t_epoch:.1f}s "
                      f"({(time.time()-t_epoch)/args.smoke:.2f}s per step wall)")
                print(f"  peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
                      f"allocated, {torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved")
                print(f"  governor: {gov.summary()}")
                print(f"  mean loss: {np.mean(run):.4f}")
                est = (time.time() - t_epoch) / args.smoke * len(dl)
                print(f"  => one epoch would take ~{est/60:.1f} min "
                      f"({len(dl)} steps); {args.epochs} epochs ~{est*args.epochs/3600:.2f} h")
                return

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "scaler": scaler.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "args": vars(args)}
        torch.save(ck, CKPT / f"{args.tag}_last.pt")
        print(f"epoch {epoch} done in {(time.time()-t_epoch)/60:.1f} min, "
              f"loss {np.mean(run):.4f} | {gov.summary()}", flush=True)

        if (epoch + 1) % args.val_every == 0:
            t_val = time.time()
            _, ds_m, im_m = evaluate(model, ds_va, by_id, device, score_thresh=0.5,
                                     limit=args.val_limit, governor=gov)
            print(f"  VAL (thr 0.50, {args.val_limit} sets, {time.time()-t_val:.0f}s): "
                  f"dataset PQ={ds_m['PQ']:.4f} SQ={ds_m['SQ']:.4f} RQ={ds_m['RQ']:.4f} "
                  f"TP={ds_m['TP']} FP={ds_m['FP']} FN={ds_m['FN']} | "
                  f"per-image PQ={im_m['PQ']:.4f}", flush=True)
            with open(ROOT / f"valog_{args.tag}.json", "a") as fh:
                fh.write(json.dumps({"epoch": epoch, "dataset": ds_m, "image": im_m}) + "\n")

    print(f"\nfinished. {gov.summary()}")


if __name__ == "__main__":
    main()
