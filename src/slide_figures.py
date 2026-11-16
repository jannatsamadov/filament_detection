"""Figures built for the conference slides, not for diagnosis.

spine_report.py already draws everything a developer wants to see at once, which
is exactly why it is unusable from the back of a lecture hall. These are the same
results with one idea per figure and lines thick enough to survive a projector:

  slide_pred_vs_gt.png    three full disks, expert outline against ours
  slide_zoom.png          crops at 4x, so an 8-pixel filament is visible at all
  slide_disagreement.png  two experts on one frame, in two colours

The third needs no model and no GPU. It carries the argument the numbers make --
that two experts agree with each other at PQ 0.348 -- in a form the audience can
check with their own eyes in two seconds.
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torchhub"))
FIGS = ROOT / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent))
from spine_data import (load_annotations, make_split, PHYS_CHANNELS, PHYSICS,
                        FULL, PRE, DISK_CX, DISK_CY, DISK_R, OFFSET, SCALE)  # noqa
from spine_model import build_spine_model                    # noqa: E402
from spine_infer import assemble                             # noqa: E402
from spine_train2 import load_stack, predict_frame           # noqa: E402
from metrics import pq_from_masks                            # noqa: E402
from evaluate import gt_masks_full                           # noqa: E402

TRUTH, OURS = "#00d24a", "#ff7a00"          # green expert, orange model
AGREE, ALONE = "#00d24a", "#ff2f2f"         # both experts drew it, or only one


def pair_iou(a, b):
    """IoU of every mask in a against every mask in b.

    Done on flat indices rather than full-frame logical_and: a filament covers a
    few thousand pixels of four million, so intersecting index arrays is three
    orders of magnitude less work than intersecting the frames.
    """
    ai = [np.flatnonzero(np.asarray(m, bool).ravel()) for m in a]
    bi = [np.flatnonzero(np.asarray(m, bool).ravel()) for m in b]
    out = np.zeros((len(a), len(b)))
    for i, x in enumerate(ai):
        for j, y in enumerate(bi):
            inter = np.intersect1d(x, y, assume_unique=True).size
            if inter:
                out[i, j] = inter / (x.size + y.size - inter)
    return out


def frame_image(stem, split="train"):
    """The flattened frame, back in display range."""
    a = np.asarray(Image.open(PRE / split / f"{stem}.png"), np.float32) / SCALE + OFFSET
    return a


def rim_of(masks, width=3):
    """Boundary pixels of the union of a set of masks.

    ax.contour on a 2048-square mask allocates a contour generator per filament
    and exhausts the heap at twenty of them. Eroding the union costs one pass.
    The width matters more than it looks: a panel is about 570 pixels on a
    projector, so a one-pixel rim on a 2048 frame is a two-thirds chance of
    landing between sample points and vanishing.
    """
    if not len(masks):
        return None
    u = np.zeros((FULL, FULL), bool)
    for m in masks:
        u |= np.asarray(m, bool)
    rim = u ^ binary_erosion(u, iterations=1)
    if width > 1:
        rim = binary_dilation(rim, iterations=width - 1)
    return rim


def compose(img, layers, vmin=0.55, vmax=1.15):
    """Grey frame plus coloured outlines, as one RGB image.

    Drawn into the array rather than layered as RGBA: matplotlib renders a
    full-size RGBA overlay as a solid fill once the axes is zoomed to the disk,
    and compositing here sidesteps the resampling entirely.
    """
    g = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    for masks, colour, width in layers:
        rim = rim_of(masks, width)
        if rim is None:
            continue
        rgb[rim] = np.round(np.array(matplotlib.colors.to_rgb(colour)) * 255)
    return rgb


def disk_view(ax, rgb, pad=40):
    lo, hi = int(DISK_CY - DISK_R - pad), int(DISK_CY + DISK_R + pad)
    ax.imshow(rgb, interpolation="antialiased")
    ax.set_xlim(DISK_CX - DISK_R - pad, DISK_CX + DISK_R + pad)
    ax.set_ylim(hi, lo)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def sets_for(file_name, by_id, fname):
    """Every annotator's set for one photograph, as lists of masks."""
    ids = [i for i, v in fname.items() if v == file_name]
    return [gt_masks_full(by_id[i]) for i in sorted(ids)]


# ----------------------------------------------------------------- prediction

def predict(model, stem, device, tta=True):
    stack = load_stack(stem, "train", use_physics=True)
    sem, spn, rad = predict_frame(model, stack, device, tta=tta)
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    on = ((xx - DISK_CX) ** 2 + (yy - DISK_CY) ** 2) <= (DISK_R * 0.995) ** 2
    sem *= on; spn *= on
    return sem, spn, rad


def fig_pred_vs_gt(picks, out):
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, picks):
        disk_view(ax, compose(p["img"], [(p["gt"], TRUTH, 4), (p["pred"], OURS, 4)]))
        ax.set_title(f"{p['stem'][:8]}   PQ = {p['pq']:.2f}   "
                     f"{len(p['pred'])} found / {len(p['gt'])} drawn",
                     fontsize=13, pad=8)
    handles = [Line2D([], [], color=TRUTH, lw=3, label="expert annotation"),
               Line2D([], [], color=OURS, lw=3, label="our model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def crop_of(mask, half=150):
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    cy = int(np.clip(cy, half, FULL - half)); cx = int(np.clip(cx, half, FULL - half))
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def fig_zoom(picks, out, half=150):
    """One matched filament, one the model missed, one it invented."""
    cases = []
    for p in picks:
        ious = pair_iou(p["gt"], p["pred"])
        hit_g = ious.max(1) > 0.5 if ious.size else np.zeros(len(p["gt"]), bool)
        hit_p = ious.max(0) > 0.5 if ious.size else np.zeros(len(p["pred"]), bool)
        for i in np.nonzero(hit_g)[0][:1]:
            j = int(ious[i].argmax())
            cases.append(("found", p, [p["gt"][i]], [p["pred"][j]],
                          f"IoU {ious[i, j]:.2f}"))
        for i in np.nonzero(~hit_g)[0][:1]:
            cases.append(("missed", p, [p["gt"][i]], [], "drawn, not found"))
        for j in np.nonzero(~hit_p)[0][:1]:
            cases.append(("invented", p, [], [p["pred"][j]], "found, not drawn"))
    cases = cases[:6]
    if not cases:
        print("no zoom cases"); return
    ncol = min(3, len(cases))
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.6 * nrow))
    for ax, (kind, p, g, q, note) in zip(np.atleast_1d(axes).ravel(), cases):
        sy, sx = crop_of((g or q)[0], half)
        ax.imshow(p["img"][sy, sx], cmap="gray", vmin=0.55, vmax=1.15,
                  interpolation="bilinear")
        for m in g:
            ax.contour(m[sy, sx].astype(float), levels=[0.5], colors=[TRUTH], linewidths=2.0)
        for m in q:
            ax.contour(m[sy, sx].astype(float), levels=[0.5], colors=[OURS], linewidths=2.0)
        ax.set_title(f"{kind} — {note}", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[len(cases):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_disagreement(by_id, fname, out, want=3):
    """Frames where two experts drew the same Sun differently."""
    multi = {}
    for i, v in fname.items():
        multi.setdefault(v, []).append(i)
    cands = [(v, ids) for v, ids in multi.items() if len(ids) >= 2]
    cands.sort(key=lambda kv: -len(by_id[kv[1][0]]))
    picks = []
    for v, ids in cands:
        a, b = gt_masks_full(by_id[sorted(ids)[0]]), gt_masks_full(by_id[sorted(ids)[1]])
        if not a or not b:
            continue
        r = pq_from_masks(a, b)
        picks.append((v, a, b, r.pq))
        if len(picks) >= want:
            break
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.9))
    n_both = n_solo = 0
    for ax, (v, a, b, pq) in zip(np.atleast_1d(axes), picks):
        iou = pair_iou(a, b)
        a_hit = iou.max(1) > 0.5 if iou.size else np.zeros(len(a), bool)
        b_hit = iou.max(0) > 0.5 if iou.size else np.zeros(len(b), bool)
        both = [a[i] for i in np.nonzero(a_hit)[0]]
        solo = ([a[i] for i in np.nonzero(~a_hit)[0]]
                + [b[j] for j in np.nonzero(~b_hit)[0]])
        n_both += len(both); n_solo += len(solo)
        disk_view(ax, compose(frame_image(Path(v).stem),
                              [(solo, ALONE, 4), (both, AGREE, 4)]))
        ax.set_title(f"{len(both)} drawn by both   ·   {len(solo)} by one only\n"
                     f"agreement PQ = {pq:.2f}", fontsize=13, pad=8)
    handles = [Line2D([], [], color=AGREE, lw=3, label="both experts drew it"),
               Line2D([], [], color=ALONE, lw=3, label="only one expert drew it")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}  ({n_both} drawn by both, {n_solo} by one only; "
          f"mean expert-vs-expert PQ {np.mean([p[3] for p in picks]):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "spine_v2_best.pt"))
    ap.add_argument("--spine-thresh", type=float, default=0.95)
    ap.add_argument("--min-spine-px", type=int, default=600)
    ap.add_argument("--scan", type=int, default=10, help="frames to score")
    ap.add_argument("--show", type=int, default=3, help="frames to draw")
    ap.add_argument("--no-gpu", action="store_true", help="only the expert figure")
    args = ap.parse_args()

    by_id, fname = load_annotations()
    if args.no_gpu:
        fig_disagreement(by_id, fname, FIGS / "slide_disagreement.png")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    in_ch = 1 + len(PHYS_CHANNELS)
    model = build_spine_model(ck["args"].get("encoder", "resnet34"),
                              pretrained=False, in_channels=in_ch).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {Path(args.ckpt).name} (epoch {ck['epoch']}, {in_ch} channels)")

    _, val_ids, val_files = make_split(fname)
    stems = sorted({Path(f).stem for f in val_files})[:args.scan]

    picks = []
    for k, stem in enumerate(stems):
        sem, spn, rad = predict(model, stem, device)
        pred = assemble(spn, rad, sem, spine_thresh=args.spine_thresh,
                        min_spine_px=args.min_spine_px)
        gts = sets_for(stem + ".jpeg", by_id, fname)
        if not gts:
            continue
        gt = max(gts, key=len)
        pq = pq_from_masks(gt, pred).pq
        picks.append({"stem": stem, "img": frame_image(stem), "gt": gt,
                      "pred": pred, "pq": pq})
        print(f"  {stem}: {len(pred)} predicted, {len(gt)} drawn, PQ {pq:.3f}",
              flush=True)

    picks.sort(key=lambda p: -p["pq"])
    chosen = [picks[0], picks[len(picks) // 2], picks[-1]][:args.show]
    fig_pred_vs_gt(chosen, FIGS / "slide_pred_vs_gt.png")
    fig_zoom(chosen, FIGS / "slide_zoom.png")
    fig_disagreement(by_id, fname, FIGS / "slide_disagreement.png")


if __name__ == "__main__":
    main()
