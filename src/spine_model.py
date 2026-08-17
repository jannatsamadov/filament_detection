"""U-Net with a ResNet encoder, three heads: semantic, spine, radius.

Chosen after Mask R-CNN failed on this data. A proposal detector matches anchors
to ground truth by box IoU, and a filament is a median 8.2 x 125 px curved ribbon
filling 24% of its own box; measured over 2665 boxes, only a third could reach
IoU 0.7 with any default anchor, so the region proposal network never received
enough positives and every configuration collapsed to predicting nothing.

This model has no boxes, no anchors and no NMS. Every parameter serves the
per-pixel output, where Mask R-CNN spent a third of its 45.9 M on a box head
that was receiving almost no positive proposals.

The single input channel is handled by summing the pretrained first-layer
weights across RGB rather than repeating the channel: same response to a grey
image, a third of the first-layer work.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34

__all__ = ["SpineUNet", "build_spine_model", "SpineLoss"]


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SpineUNet(nn.Module):
    """Encoder-decoder producing (semantic logit, spine logit, radius)."""

    def __init__(self, encoder="resnet34", pretrained=True, in_channels=1,
                 decoder_width=(256, 128, 64, 32, 16)):
        super().__init__()
        self.in_channels = in_channels
        fn = {"resnet18": resnet18, "resnet34": resnet34}[encoder]
        net = fn(weights="DEFAULT" if pretrained else None)

        # Channel 0 is the H-alpha image and inherits the pretrained RGB filters
        # collapsed into one channel. Any extra physical channels start at zero,
        # so the model begins exactly where the image-only model did and has to
        # earn any use it makes of them.
        w = net.conv1.weight.data.sum(dim=1, keepdim=True)
        conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv1.weight.zero_()
            conv1.weight[:, :1] = w
        net.conv1 = conv1

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2,  64
        self.pool = net.maxpool                                    # /4
        self.enc1, self.enc2 = net.layer1, net.layer2              # /4 64, /8 128
        self.enc3, self.enc4 = net.layer3, net.layer4              # /16 256, /32 512

        c = decoder_width
        self.dec4 = _conv_block(512 + 256, c[0])
        self.dec3 = _conv_block(c[0] + 128, c[1])
        self.dec2 = _conv_block(c[1] + 64, c[2])
        self.dec1 = _conv_block(c[2] + 64, c[3])
        self.dec0 = _conv_block(c[3], c[4])

        self.head_sem = nn.Conv2d(c[4], 1, 1)
        self.head_spine = nn.Conv2d(c[4], 1, 1)
        self.head_radius = nn.Conv2d(c[4], 1, 1)
        # Start with a low prior on both foreground classes: filaments cover
        # about 0.4% of the disk, so an unbiased head spends its early updates
        # unlearning a 50/50 prior.
        for h in (self.head_sem, self.head_spine):
            nn.init.constant_(h.bias, -4.0)

    @staticmethod
    def _up(x, like):
        return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s = self.stem(x)          # /2
        e1 = self.enc1(self.pool(s))   # /4
        e2 = self.enc2(e1)             # /8
        e3 = self.enc3(e2)             # /16
        e4 = self.enc4(e3)             # /32

        d = self.dec4(torch.cat([self._up(e4, e3), e3], 1))
        d = self.dec3(torch.cat([self._up(d, e2), e2], 1))
        d = self.dec2(torch.cat([self._up(d, e1), e1], 1))
        d = self.dec1(torch.cat([self._up(d, s), s], 1))
        d = self.dec0(self._up(d, x))

        return {
            "sem": self.head_sem(d),
            "spine": self.head_spine(d),
            "radius": F.softplus(self.head_radius(d)),   # a width is never negative
        }


def build_spine_model(encoder="resnet34", pretrained=True, in_channels=1):
    return SpineUNet(encoder=encoder, pretrained=pretrained, in_channels=in_channels)


class SpineLoss(nn.Module):
    """Dice+BCE on the two maps, L1 on radius inside filaments only.

    Every term is restricted to on-disk pixels: off-disk has no flat field and no
    annotations, and it is nearly 40% of every frame.
    """

    def __init__(self, w_sem=1.0, w_spine=1.0, w_radius=0.1, pos_weight=20.0):
        super().__init__()
        self.w_sem, self.w_spine, self.w_radius = w_sem, w_spine, w_radius
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    @staticmethod
    def _dice(prob, target, valid, eps=1.0):
        p, t = prob * valid, target * valid
        num = 2.0 * (p * t).sum(dim=(1, 2, 3)) + eps
        den = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + eps
        return (1.0 - num / den).mean()

    def forward(self, out, batch):
        valid = batch["valid"]
        n = valid.sum().clamp(min=1.0)

        sem_bce = F.binary_cross_entropy_with_logits(
            out["sem"], batch["mask"], pos_weight=self.pos_weight, reduction="none")
        sem_bce = (sem_bce * valid).sum() / n
        sem_dice = self._dice(torch.sigmoid(out["sem"]), batch["mask"], valid)

        sp_bce = F.binary_cross_entropy_with_logits(
            out["spine"], batch["spine"], pos_weight=self.pos_weight, reduction="none")
        sp_bce = (sp_bce * valid).sum() / n
        sp_dice = self._dice(torch.sigmoid(out["spine"]), batch["spine"], valid)

        inside = batch["mask"] * valid
        rad = (F.l1_loss(out["radius"], batch["radius"], reduction="none") * inside).sum()
        rad = rad / inside.sum().clamp(min=1.0)

        total = (self.w_sem * (sem_bce + sem_dice)
                 + self.w_spine * (sp_bce + sp_dice)
                 + self.w_radius * rad)
        return total, {"sem_bce": sem_bce.detach(), "sem_dice": sem_dice.detach(),
                       "spine_bce": sp_bce.detach(), "spine_dice": sp_dice.detach(),
                       "radius_l1": rad.detach()}
