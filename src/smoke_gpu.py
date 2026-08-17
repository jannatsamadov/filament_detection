"""Check the metric on known cases, then measure what the model costs on this GPU.

Run before any real training: it answers "does it fit in 6 GB" and "how hot does
it get", which on this machine decides whether the job may run at all.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import pq_from_masks                 # noqa: E402
from model import build_model, forward_images     # noqa: E402
from thermal import Governor, gpu_status          # noqa: E402


def check_metric():
    print("== metric checks ==")
    S = 64

    def box(x0, y0, x1, y1):
        m = np.zeros((S, S), bool); m[y0:y1, x0:x1] = True; return m

    a = box(0, 0, 20, 20)
    r = pq_from_masks([a], [a])
    assert abs(r.pq - 1.0) < 1e-9, r
    print(f"  identical mask            -> {r}   (expect PQ=1)")

    r = pq_from_masks([a], [])
    print(f"  missed entirely           -> {r}   (expect PQ=0, FN=1)")
    assert r.fn == 1 and r.pq == 0.0

    r = pq_from_masks([], [a])
    print(f"  invented one              -> {r}   (expect PQ=0, FP=1)")
    assert r.fp == 1

    # two equal GT filaments merged into one prediction: IoU is exactly 0.5 with
    # each, the threshold is strictly greater than 0.5, so neither matches.
    g1, g2 = box(0, 0, 10, 10), box(20, 0, 30, 10)
    merged = g1 | g2
    r = pq_from_masks([g1, g2], [merged])
    print(f"  two GT merged into one    -> {r}   (expect TP=0, FP=1, FN=2)")
    assert (r.tp, r.fp, r.fn) == (0, 1, 2), r

    # one GT split into two predictions, each covering half
    p1, p2 = box(0, 0, 10, 5), box(0, 5, 10, 10)
    r = pq_from_masks([g1], [p1, p2])
    print(f"  one GT split into two     -> {r}   (expect TP=0, FP=2, FN=1)")
    assert (r.tp, r.fp, r.fn) == (0, 2, 1), r

    # partial overlap above threshold
    p = box(0, 0, 10, 9)
    r = pq_from_masks([g1], [p])
    print(f"  90% overlap               -> {r}   (expect TP=1, IoU=0.9)")
    assert r.tp == 1 and abs(r.ious[0] - 0.9) < 1e-9
    print("  all metric checks passed\n")


def probe(size=1024, steps=12, n_obj=8):
    print(f"== GPU probe: Mask R-CNN R50-FPN v2 at {size}x{size}, batch 1, AMP ==")
    device = torch.device("cuda")
    u, t, m = gpu_status()
    print(f"  before: util={u}% temp={t}C mem={m}MiB")

    model = build_model(size=size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"  parameters: {sum(p.numel() for p in params)/1e6:.1f} M")
    opt = torch.optim.SGD(params, lr=0.001, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    gov = Governor(target_util=70, check_every=3)

    torch.manual_seed(0)
    times = []
    model.train()
    for i in range(steps):
        img = torch.rand(1, size, size, device=device)
        boxes, masks = [], []
        for _ in range(n_obj):
            x0 = np.random.randint(0, size - 80); y0 = np.random.randint(0, size - 80)
            w = np.random.randint(20, 70); h = np.random.randint(20, 70)
            boxes.append([x0, y0, x0 + w, y0 + h])
            mm = torch.zeros(size, size, dtype=torch.uint8, device=device)
            mm[y0:y0 + h, x0:x0 + w] = 1
            masks.append(mm)
        tgt = {"boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
               "labels": torch.ones(n_obj, dtype=torch.int64, device=device),
               "masks": torch.stack(masks)}
        t0 = time.perf_counter()
        with gov.step():
            imgs = forward_images(model, [img])
            with torch.amp.autocast("cuda"):
                losses = model(imgs, [tgt])
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if i == 2:
            torch.cuda.reset_peak_memory_stats()

    u, t, m = gpu_status()
    compute = float(np.median(times[3:]))
    print(f"  compute per step   : {compute:.3f} s")
    print(f"  wall per step      : {compute*(1+gov.sleep_factor):.3f} s "
          f"(governor sleeps {gov.sleep_factor:.2f}x)")
    print(f"  peak GPU memory    : {torch.cuda.max_memory_allocated()/1e9:.2f} GB allocated, "
          f"{torch.cuda.max_memory_reserved()/1e9:.2f} GB reserved  (card has 6.0 GB)")
    print(f"  after: util={u}% temp={t}C mem={m}MiB")
    print(f"  governor: {gov.summary()}")
    return compute * (1 + gov.sleep_factor)


if __name__ == "__main__":
    check_metric()
    wall = probe()
    for n_steps, name in ((924, "one epoch (~924 annotation sets)"),):
        print(f"\n  estimate: {name} ~ {wall*n_steps/60:.1f} min "
              f"=> 6 epochs ~ {wall*n_steps*6/3600:.2f} h")
