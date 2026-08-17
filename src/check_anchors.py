"""Can the RPN's anchors even match these filaments?

Mask R-CNN's region proposal network matches anchors to ground truth by box IoU
and calls an anchor positive only above rpn_fg_iou_thresh (0.7 by default).
Filaments are long, curved and thin -- they fill only 24% of their own bounding
box -- so it is worth checking directly, without training anything, how well the
anchor grid can cover them. If most ground-truth boxes cannot reach 0.7 with any
anchor, the RPN gets almost no positives and the detector has no chance.
"""
from __future__ import annotations

import collections, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from data import load_annotations, make_split      # noqa: E402

FULL = 2048


def anchor_grid(size, sizes, ratios, strides=(4, 8, 16, 32, 64)):
    """Approximate torchvision's FPN anchors for a square image."""
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    ag = AnchorGenerator(sizes, ratios)
    feats = [torch.zeros(1, 1, size // s, size // s) for s in strides]
    image_list = type("IL", (), {"tensors": torch.zeros(1, 3, size, size),
                                 "image_sizes": [(size, size)]})()
    anchors = ag(image_list, feats)[0]
    return anchors.numpy()


def max_iou_per_box(boxes, anchors, chunk=20000):
    """boxes: [N,4]; anchors: [M,4]. Returns best IoU for each box."""
    ax0, ay0, ax1, ay1 = anchors.T
    a_area = (ax1 - ax0) * (ay1 - ay0)
    best = np.zeros(len(boxes))
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        b_area = (x1 - x0) * (y1 - y0)
        b = 0.0
        for s in range(0, len(anchors), chunk):
            sl = slice(s, s + chunk)
            iw = np.minimum(x1, ax1[sl]) - np.maximum(x0, ax0[sl])
            ih = np.minimum(y1, ay1[sl]) - np.maximum(y0, ay0[sl])
            np.clip(iw, 0, None, out=iw); np.clip(ih, 0, None, out=ih)
            inter = iw * ih
            iou = inter / (a_area[sl] + b_area - inter + 1e-9)
            m = iou.max()
            if m > b:
                b = m
        best[i] = b
    return best


def main(size=1024):
    by_id, fname = load_annotations()
    train_ids, _, _ = make_split(fname)
    scale = size / FULL

    boxes, fills = [], []
    for iid in train_ids[:400]:
        for a in by_id[iid]:
            x, y, w, h = a["bbox"]
            if w < 2 or h < 2:
                continue
            boxes.append([x * scale, y * scale, (x + w) * scale, (y + h) * scale])
            fills.append(a["area"] / (w * h))
    boxes = np.array(boxes, dtype=np.float64)
    fills = np.array(fills)
    print(f"{len(boxes)} ground-truth boxes at {size}px")
    bw = boxes[:, 2] - boxes[:, 0]; bh = boxes[:, 3] - boxes[:, 1]
    print(f"  box width : median {np.median(bw):.0f} px  p10 {np.percentile(bw,10):.0f}  "
          f"p90 {np.percentile(bw,90):.0f}")
    print(f"  box height: median {np.median(bh):.0f} px  p10 {np.percentile(bh,10):.0f}  "
          f"p90 {np.percentile(bh,90):.0f}")
    print(f"  mask fills {np.median(fills)*100:.0f}% of its box (median)\n")

    configs = {
        "torchvision default": (((32,), (64,), (128,), (256,), (512,)),
                                ((0.5, 1.0, 2.0),) * 5),
        "our small anchors": (((16,), (32,), (64,), (128,), (256,)),
                              ((0.5, 1.0, 2.0),) * 5),
        "small + more ratios": (((16,), (32,), (64,), (128,), (256,)),
                                ((0.25, 0.5, 1.0, 2.0, 4.0),) * 5),
        "multi-size per level": (((16, 24, 32), (32, 48, 64), (64, 96, 128),
                                  (128, 192, 256), (256, 384, 512)),
                                 ((0.5, 1.0, 2.0),) * 5),
    }

    print(f"{'configuration':<24} {'anchors':>9} {'max IoU: median':>16} "
          f"{'>=0.7':>7} {'>=0.5':>7} {'>=0.3':>7}")
    for name, (sizes, ratios) in configs.items():
        anchors = anchor_grid(size, sizes, ratios)
        best = max_iou_per_box(boxes, anchors)
        print(f"{name:<24} {len(anchors):>9,} {np.median(best):>16.3f} "
              f"{100*np.mean(best>=0.7):>6.1f}% {100*np.mean(best>=0.5):>6.1f}% "
              f"{100*np.mean(best>=0.3):>6.1f}%")

    print("\n>=0.7 is the share of filaments the RPN can call positive normally.")
    print("Below that it relies on allow_low_quality_matches, which contributes")
    print("a single anchor per object -- a very thin training signal.")


if __name__ == "__main__":
    main()
