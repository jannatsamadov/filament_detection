"""v5: full frames, evidence-weighted labels, and labels that learn.

Three stretches, and the shape matters.

  epochs 1-6     warm-up. Human labels only. A model one epoch old answers at
                 random and its opinion would corrupt the labels.
  epochs 7-30    self-training. Each filament keeps a count of what speaks for
                 it and against it; every epoch the model adds one vote.
  epochs 31-36   settle. The labels are frozen and the learning rate decays to
                 zero, so the last weights fit a target that stops moving.

The vote is taken from the forward pass the training step makes anyway, so it
costs nothing. What it must not do is ask the model whether it agrees with the
label it was just trained on -- a model fitted to a target of 0.4 answers 0.4 and
votes against, every epoch, and the label spirals to zero. So the question is
put against the neighbourhood instead: is the response inside this filament
higher than in the background around it? That is answerable independently of
what the filament was labelled.

Human agreement is a floor. A filament every annotator drew cannot be moved at
all, and no filament can be pushed below where the people who drew it put it.
"""
from __future__ import annotations

import argparse, collections, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))            # shared gpu_guard, gpu_lock
from frame_data import (FullFrameDataset, to_input, load_weights, LABELS,   # noqa: E402
                        IN_CHANNELS, N_CHIRALITY, scalars_for)
from spine_data import photograph_split, load_annotations, make_split, FULL  # noqa: E402
from spine_model import build_spine_model, SpineLoss                    # noqa: E402
from query_head import (QueryLoss, targets_from_inst,                   # noqa: E402
                        instances_from_queries)
from spine_infer import assemble                                        # noqa: E402
from metrics import pq_from_masks, aggregate                            # noqa: E402
from evaluate import gt_masks_full                                      # noqa: E402
from gpu_guard import Governor, gpu_status, other_gpu_processes         # noqa: E402
from gpu_lock import GpuLock                                            # noqa: E402

CKPT = ROOT / "checkpoints"
CKPT.mkdir(exist_ok=True)
VOTE_MARGIN = 0.10         # how much higher inside must be to count as a yes
VOTE_FLOOR = 0.20          # below this the model is not seeing anything at all
MAX_MODEL_VOTES = 30


def instance_votes(sem_prob, inst, valid, scale=4, ring=33):
    """Per filament: is the response inside higher than just outside?

    Both numbers come from the map the training step already produced.
    Comparing inside against the local background asks whether the model sees
    structure here -- a different question from whether it reproduces the label
    it was trained on, which a fitted model always would.

    Everything is done at quarter resolution and in one pass. A ring computed
    separately for each filament meant a 31x31 dilation per instance per step,
    which is billions of operations; a single box average of the background,
    read back under each filament, gives the same comparison for one pass.
    """
    p = torch.nn.functional.avg_pool2d(sem_prob * valid, scale)
    ii = inst[:, :, ::scale, ::scale].long()
    v = (valid[:, :, ::scale, ::scale] > 0.5)
    bg = ((ii == 0) & v).float()
    pad = ring // 2
    num = torch.nn.functional.avg_pool2d(p * bg, ring, 1, pad, count_include_pad=False)
    den = torch.nn.functional.avg_pool2d(bg, ring, 1, pad, count_include_pad=False)
    local_bg = num / den.clamp(min=1e-4)

    flat = ii.reshape(-1)
    n = int(flat.max()) + 1
    if n <= 1:
        return {}
    cnt = torch.bincount(flat, minlength=n).float()
    ins = torch.bincount(flat, weights=p.reshape(-1).float(), minlength=n)
    aro = torch.bincount(flat, weights=local_bg.reshape(-1).float(), minlength=n)
    cnt = cnt.clamp(min=1.0)
    ins, aro, cnt = (t.cpu().numpy() for t in (ins / cnt, aro / cnt, cnt))

    out = {}
    for i in range(1, n):
        if cnt[i] < 4:                      # too small at this scale to judge
            continue
        yes = bool(ins[i] > aro[i] + VOTE_MARGIN and ins[i] > VOTE_FLOOR)
        out[i] = (float(ins[i]), float(aro[i]), yes)
    return out


def apply_votes(weights, stem, votes, opening):
    """Add one vote per filament, respecting the floors."""
    w = weights.setdefault(stem, {})
    moved = 0
    for i, (inside, around, yes) in votes.items():
        key = str(i)
        rec = w.get(key)
        if rec is None:
            continue
        a, b, frozen = rec[0], rec[1], rec[2]
        if frozen:
            continue
        n_votes = rec[3] if len(rec) > 3 else 0
        if n_votes >= MAX_MODEL_VOTES:
            continue
        if yes:
            a += 1.0
        else:
            b += 1.0
        # the people who drew it set a floor the model may not push under
        floor_a, floor_b = opening.get(stem, {}).get(key, (a, b))
        if a / max(a + b, 1e-6) < floor_a / max(floor_a + floor_b, 1e-6):
            a = floor_a / max(floor_a + floor_b, 1e-6) * (a + b)
            b = (a + b) - a if (a + b) else b
        w[key] = [round(float(a), 3), round(float(b), 3), bool(frozen), n_votes + 1]
        moved += 1
    return moved


def predict_frame(model, planes, scalars, device, tta=False, want_query=False):
    """Full-frame inference, optionally averaged over the geometry-safe mirrors."""
    from frame_data import I_SIN2T, I_SINLAT, I_SINP, I_B0, N_PLANES
    outs, q = [], None
    combos = [(False, False)] if not tta else [(False, False), (True, False),
                                               (False, True), (True, True)]
    for fh, fv in combos:
        p, s = planes.clone(), scalars.clone()
        if fh:
            p = torch.flip(p, dims=[3])
        if fv:
            p = torch.flip(p, dims=[2])
        if fh != fv:
            p[:, I_SIN2T] = 255 - p[:, I_SIN2T]
            s[:, I_SINP] = -s[:, I_SINP]
        if fv:
            p[:, I_SINLAT] = 255 - p[:, I_SINLAT]
            s[:, I_B0] = -s[:, I_B0]
        x = to_input(p, s, device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            o = model(x)
        sem = torch.sigmoid(o["sem"].float())
        spn = torch.sigmoid(o["spine"].float())
        rad = o["radius"].float()
        if fv:
            sem, spn, rad = (torch.flip(t, dims=[2]) for t in (sem, spn, rad))
        if fh:
            sem, spn, rad = (torch.flip(t, dims=[3]) for t in (sem, spn, rad))
        outs.append((sem, spn, rad))
        if want_query and not fh and not fv:
            q = o.get("query")
    sem = torch.stack([o[0] for o in outs]).mean(0)[0, 0].cpu().numpy()
    spn = torch.stack([o[1] for o in outs]).mean(0)[0, 0].cpu().numpy()
    rad = torch.stack([o[2] for o in outs]).mean(0)[0, 0].cpu().numpy()
    if want_query:
        return sem, spn, rad, q
    return sem, spn, rad


def validate(model, val_stems, by_id, fname, device, gov, ondisk,
             thresholds=(0.5, 0.7, 0.85, 0.9), min_spine_px=600, limit=60):
    """PQ over whole photographs, at the best of a few spine thresholds."""
    from frame_data import FullFrameDataset as _F
    ds = _F(val_stems[:limit], split="train", augment=False)
    stem_ids = collections.defaultdict(list)
    for iid, f in fname.items():
        stem_ids[Path(f).stem].append(iid)

    model.eval()
    per_thresh = {t: [] for t in thresholds}
    query_res, query_conf = [], []
    for k in range(len(ds)):
        item = ds[k]
        planes = item["planes"][None]
        scalars = item["scalars"][None]
        with gov.step():
            sem, spn, rad, q = predict_frame(model, planes, scalars, device,
                                             want_query=True)
        sem *= ondisk; spn *= ondisk
        gts = [gt_masks_full(by_id[iid]) for iid in stem_ids[item["stem"]]]
        for t in thresholds:
            preds = assemble(spn, rad, sem, spine_thresh=t, sem_thresh=0.5,
                             min_spine_px=min_spine_px, min_area=60)
            for g in gts:
                per_thresh[t].append(pq_from_masks(g, preds))
        if q is not None:
            masks, conf = instances_from_queries(
                q["score"][0].float(), q["masks"][0].float(), sem, sem.shape)
            for g in gts:
                query_res.append(pq_from_masks(g, masks))
            query_conf.append(len(masks))
    model.train()
    best_t, best = None, None
    for t in thresholds:
        a = aggregate(per_thresh[t])
        if best is None or a["PQ"] > best["PQ"]:
            best_t, best = t, a
    qa = aggregate(query_res) if query_res else None
    return best_t, best, qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--target-util", type=int, default=55)
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--val-limit", type=int, default=60)
    ap.add_argument("--w-chirality", type=float, default=0.2)
    ap.add_argument("--n-queries", type=int, default=50)
    ap.add_argument("--w-query", type=float, default=1.0)
    ap.add_argument("--init-from", default=str(CKPT / "spine_v4_best.pt"))
    ap.add_argument("--tag", default="spine_v5")
    ap.add_argument("--resume", default="")
    ap.add_argument("--start-below", type=int, default=68)
    ap.add_argument("--wait-for-gpu", action="store_true")
    ap.add_argument("--no-self-training", action="store_true")
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

    train_stems, val_stems = photograph_split()
    by_id, fname = load_annotations()
    print(f"{len(train_stems)} training photographs, {len(val_stems)} validation")

    weights_path = ROOT / f"weights_{args.tag}.json"
    weights = load_weights(weights_path if weights_path.exists() else None)
    opening = {s: {k: (v[0], v[1]) for k, v in d.items()}
               for s, d in load_weights(None).items()}
    if not weights_path.exists():
        json.dump(weights, open(weights_path, "w"))
    n_fil = sum(len(v) for v in weights.values())
    n_frozen = sum(1 for v in weights.values() for r in v.values() if r[2])
    above = sum(1 for v in weights.values() for r in v.values()
                if r[0] / max(r[0] + r[1], 1e-6) >= 0.75)
    print(f"{n_fil} filaments in the label store: {n_frozen} frozen, "
          f"{above} already above 0.75")

    model = build_spine_model(args.encoder, pretrained=False,
                              in_channels=IN_CHANNELS,
                              n_chirality=N_CHIRALITY,
                              n_queries=args.n_queries).to(device)
    if args.init_from and Path(args.init_from).exists() and not args.resume:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        own = model.state_dict()
        loaded = skipped = 0
        for k, v in sd.items():
            if k not in own:
                skipped += 1; continue
            if own[k].shape == v.shape:
                own[k] = v; loaded += 1
            elif k == "stem.0.weight" and v.shape[1] <= own[k].shape[1]:
                # v4 had 6 channels; the new ones start at zero, so the model
                # begins exactly where v4 left off and has to earn the rest
                w = own[k].clone(); w[:, :v.shape[1]] = v; own[k] = w; loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own)
        print(f"init from {Path(args.init_from).name}: {loaded} tensors, {skipped} skipped")

    crit = SpineLoss(w_chirality=args.w_chirality).to(device)
    qcrit = QueryLoss().to(device) if args.n_queries else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    steps = args.epochs * len(train_stems)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(steps, 1), pct_start=0.1)
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

    yy, xx = np.mgrid[0:FULL, 0:FULL]
    from spine_data import DISK_CX, DISK_CY, DISK_R
    ondisk = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2

    lock = GpuLock("filament_detection", args.tag, wait=args.wait_for_gpu).acquire()
    gov = Governor(target_util=args.target_util)
    log = ROOT / f"trainlog_{args.tag}.csv"
    if not log.exists():
        log.write_text("epoch,step,loss,sem_dice,spine_dice,chir,lr,temp,secs\n")
    vlog = ROOT / f"vallog_{args.tag}.json"
    history = json.load(open(vlog)) if vlog.exists() else []
    best = max([h["PQ"] for h in history], default=0.0)

    self_start = args.warmup
    self_end = args.epochs - args.settle
    print(f"phases: warm-up 1-{self_start}, self-training {self_start+1}-{self_end}, "
          f"settle {self_end+1}-{args.epochs}")

    try:
        for epoch in range(start_epoch, args.epochs):
            phase = ("warmup" if epoch < self_start else
                     "self" if epoch < self_end else "settle")
            lock.progress(f"epoch {epoch+1}/{args.epochs} ({phase})")
            ds = FullFrameDataset(train_stems, split="train", augment=True,
                                  weights=weights)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=1, shuffle=True, num_workers=args.workers,
                pin_memory=True, persistent_workers=False)

            t_ep, run = time.time(), collections.defaultdict(list)
            voting = phase == "self" and not args.no_self_training
            moved = 0
            for i, b in enumerate(dl):
                stem = b.pop("stem")[0]
                planes, scalars = b.pop("planes"), b.pop("scalars")
                inst = b.pop("inst").to(device, non_blocking=True)
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                with gov.step():
                    x = to_input(planes, scalars, device)
                    with torch.amp.autocast("cuda"):
                        out = model(x)
                        loss, parts = crit(out, b)
                        if qcrit is not None and "query" in out:
                            sc = {int(k): v[0] / max(v[0] + v[1], 1e-6)
                                  for k, v in weights.get(stem, {}).items()}
                            tg, tw = targets_from_inst(
                                inst[0], sc, out["query"]["masks"].shape[-2:])
                            ql, qparts = qcrit(out["query"], [(tg, tw)])
                            loss = loss + args.w_query * ql
                            parts.update(qparts)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt); scaler.update(); sched.step()
                    if voting:
                        with torch.no_grad():
                            sem_p = torch.sigmoid(out["sem"].float())
                            v = instance_votes(sem_p, inst, b["valid"])
                        moved += apply_votes(weights, stem, v, opening)
                    torch.cuda.synchronize()
                run["loss"].append(float(loss.detach()))
                for k, v in parts.items():
                    run[k].append(float(v))
                if (i + 1) % 100 == 0:
                    _, t, _ = gpu_status()
                    el = time.time() - t_ep
                    print(f"  e{epoch+1} [{i+1}/{len(dl)}] loss "
                          f"{np.mean(run['loss'][-100:]):.4f} "
                          f"(sem {np.mean(run['sem_dice'][-100:]):.3f}, "
                          f"spine {np.mean(run['spine_dice'][-100:]):.3f}, "
                          f"q {np.mean(run.get('q_score', [0])[-100:]):.3f}/"
                          f"{np.mean(run.get('q_dice', [0])[-100:]):.3f}) | "
                          f"{t}C | {el/60:.0f} min", flush=True)
                    with open(log, "a") as fh:
                        fh.write(f"{epoch+1},{i+1},{np.mean(run['loss'][-100:]):.5f},"
                                 f"{np.mean(run['sem_dice'][-100:]):.5f},"
                                 f"{np.mean(run['spine_dice'][-100:]):.5f},"
                                 f"{np.mean(run.get('chirality_ce', [0])[-100:]):.5f},"
                                 f"{sched.get_last_lr()[0]:.7f},{t},{el:.0f}\n")

            if voting:
                json.dump(weights, open(weights_path, "w"))
                above = sum(1 for v in weights.values() for r in v.values()
                            if r[0] / max(r[0] + r[1], 1e-6) >= 0.75)
                print(f"  votes applied to {moved} filaments; {above} now above 0.75",
                      flush=True)

            print(f"epoch {epoch+1} ({phase}) done in {(time.time()-t_ep)/60:.1f} min, "
                  f"loss {np.mean(run['loss']):.4f} | {gov.summary()}", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       CKPT / f"{args.tag}_last.pt")

            if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
                t0 = time.time()
                thr, a, qa = validate(model, val_stems, by_id, fname, device, gov,
                                      ondisk, limit=args.val_limit)
                a = dict(a); a.update(epoch=epoch + 1, thresh=thr, phase=phase)
                if qa:
                    a["query"] = dict(qa)
                    print(f"  query head: PQ={qa['PQ']:.4f} TP={qa['TP']} "
                          f"FP={qa['FP']} FN={qa['FN']}", flush=True)
                history.append(a); json.dump(history, open(vlog, "w"), indent=1)
                print(f"VAL e{epoch+1} ({time.time()-t0:.0f}s, thresh {thr}): "
                      f"PQ={a['PQ']:.4f} SQ={a['SQ']:.4f} RQ={a['RQ']:.4f} "
                      f"TP={a['TP']} FP={a['FP']} FN={a['FN']}"
                      f"   [v4 reached 0.3495 on 60 sets]", flush=True)
                if a["PQ"] > best:
                    best = a["PQ"]
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "PQ": best, "thresh": thr, "args": vars(args)},
                               CKPT / f"{args.tag}_best.pt")
                    print(f"  new best PQ {best:.4f}", flush=True)
    finally:
        lock.release()
    print(f"\nfinished. best PQ {best:.4f}")


if __name__ == "__main__":
    main()
