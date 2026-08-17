"""Mask R-CNN configured for solar filaments.

Why a proposal-based detector rather than a semantic segmenter plus connected
components: filaments are effectively separate objects in these images (only 7%
have a neighbour within 20 px, and only 7.3% have a bounding box overlapping any
other), so each proposal can own one filament and fragmentation -- which Panoptic
Quality punishes hard -- cannot arise by construction. A 28x28 RoI mask head is
enough: reconstructing the ground-truth polygons through 28x28 gives median
IoU 0.965, so the head does not throw away barbs.

Anchors are shifted one octave down from the torchvision default because the
objects are small: median bounding-box long side is 96 px at 2048, i.e. 48 px at
the 1024 training resolution.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

__all__ = ["build_model", "GrayToRGB"]


class GrayToRGB(nn.Module):
    """Repeat the single H-alpha channel so the COCO-pretrained backbone applies."""

    def forward(self, images):
        return [im.repeat(3, 1, 1) if im.shape[0] == 1 else im for im in images]


def build_model(size=1024, num_classes=2, score_thresh=0.05,
                detections_per_img=200, small_anchors=True, pretrained=True,
                rpn_fg_iou_thresh=0.6):
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=size, max_size=size,
        box_score_thresh=score_thresh,
        box_detections_per_img=detections_per_img,
        rpn_fg_iou_thresh=rpn_fg_iou_thresh,
    )

    if small_anchors:
        # Three sizes per FPN level, not one. Filament boxes are median 38x35 px
        # at 1024 and the default octave spacing (32, 64, 128, ...) leaves the
        # sizes in between uncovered: measured over 2665 boxes, only 33% could
        # reach IoU 0.7 with any default anchor, and 29% with a single small size
        # per level. Three sizes per level lifts that to 70%. Below the threshold
        # the RPN falls back to one anchor per object, which is too thin a signal
        # to learn from -- that is what collapsed the first two training runs.
        sizes = ((16, 24, 32), (32, 48, 64), (64, 96, 128),
                 (128, 192, 256), (256, 384, 512))
        ratios = ((0.5, 1.0, 2.0),) * len(sizes)
        old_n = model.rpn.anchor_generator.num_anchors_per_location()[0]
        model.rpn.anchor_generator = AnchorGenerator(sizes, ratios)
        new_n = model.rpn.anchor_generator.num_anchors_per_location()[0]
        # Only rebuild the RPN head if the anchor count actually changed. It does
        # not here -- default and custom sizes both give 3 anchors per location --
        # and rebuilding anyway threw away the COCO-pretrained RPN classifier and
        # box regressor. Together with too high a learning rate that collapsed the
        # model to predicting nothing at all by the second epoch.
        if old_n != new_n:
            head = model.rpn.head
            in_ch = head.conv[0][0].in_channels
            head.cls_logits = nn.Conv2d(in_ch, new_n, kernel_size=1, stride=1)
            head.bbox_pred = nn.Conv2d(in_ch, new_n * 4, kernel_size=1, stride=1)
            for layer in (head.cls_logits, head.bbox_pred):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)

    model.gray_to_rgb = GrayToRGB()
    return model


def forward_images(model, images):
    """Expand grayscale to RGB before the detector's own transform."""
    return model.gray_to_rgb(list(images))
