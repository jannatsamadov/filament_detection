"""An object-level head, so the model is trained the way it is scored.

The pipeline until now cut a probability map at a threshold and called each
connected patch a filament. That map is trained per pixel: the loss asks whether
each pixel is right, never whether the right *number* of objects came out. The
metric asks exactly that -- a filament that should not be there costs half a
false positive whatever its pixels look like. Measured at a loose threshold,
only 39.5% of the candidates are real, and ranking them afterwards recovers
nothing (+0.005 against a validation set that moves by 0.03).

So the decision moves into the network. A fixed set of queries each look at the
image and answer two questions: is there a filament here, and which pixels are
it. Training matches the queries one-to-one against the filaments that are
really there, by Hungarian assignment; whatever is left over is trained to say
"nothing". Producing a spurious object is then penalised directly, which is the
thing a per-pixel loss cannot express.

The masks the queries produce are coarse -- a quarter resolution, where a
filament eight pixels wide is two. That is deliberate. Coarse is enough to say
which object is which, and the semantic head already draws boundaries at an
average overlap of 0.64, which is the part that would suffer at low resolution.
Each query supplies identity; the semantic map supplies the edge.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class QueryHead(nn.Module):
    """N queries attending to the encoder, each returning a score and a mask.

    Kept small on purpose. A frame holds at most 17 filaments and averages 7, so
    50 queries is already generous, and the attention runs over the /16 feature
    map where 2048 pixels become 128 -- 16k keys, which 50 queries can afford.
    """

    def __init__(self, n_queries=50, d_model=128, n_layers=3, n_heads=4,
                 feat_ch=256, mask_ch=32, mask_dim=32):
        super().__init__()
        self.n_queries = n_queries
        self.query = nn.Embedding(n_queries, d_model)
        self.pos = nn.Parameter(torch.zeros(1, d_model, 1, 1))
        self.input_proj = nn.Conv2d(feat_ch, d_model, 1)
        self.mask_proj = nn.Conv2d(mask_ch, mask_dim, 1)

        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.score = nn.Linear(d_model, 1)
        self.to_mask = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True),
                                     nn.Linear(d_model, mask_dim))
        # start pessimistic: most queries should say "nothing"
        nn.init.constant_(self.score.bias, -4.0)

    @staticmethod
    def _sincos(d, h, w, device):
        """Fixed 2-D position encoding; queries need to know where they looked."""
        y = torch.arange(h, device=device).float() / max(h - 1, 1)
        x = torch.arange(w, device=device).float() / max(w - 1, 1)
        y, x = torch.meshgrid(y, x, indexing="ij")          # both (h,w)
        c = d // 4
        f = torch.exp(torch.arange(c, device=device).float() * (-math.log(1e4) / c))
        pe = torch.cat([torch.sin(y[..., None] * f), torch.cos(y[..., None] * f),
                        torch.sin(x[..., None] * f), torch.cos(x[..., None] * f)], -1)
        pe = pe.permute(2, 0, 1)[None]                      # (1, 4c, h, w)
        if pe.shape[1] < d:                                 # d not divisible by 4
            pe = F.pad(pe, (0, 0, 0, 0, 0, d - pe.shape[1]))
        return pe

    def forward(self, feat, mask_feat):
        b = feat.shape[0]
        f = self.input_proj(feat)
        f = f + self._sincos(f.shape[1], f.shape[2], f.shape[3], f.device)
        seq = f.flatten(2).transpose(1, 2)                       # (B, HW, D)
        q = self.query.weight[None].expand(b, -1, -1)            # (B, N, D)
        q = self.decoder(q, seq)
        score = self.score(q).squeeze(-1)                        # (B, N)
        emb = self.to_mask(q)                                    # (B, N, C)
        m = self.mask_proj(mask_feat)                            # (B, C, h, w)
        masks = torch.einsum("bnc,bchw->bnhw", emb, m)           # (B, N, h, w)
        return {"score": score, "masks": masks}


def _dice_cost(pred, tgt, eps=1.0):
    """Pairwise dice between every prediction and every target."""
    # under autocast the masks come back half precision; the matching runs in
    # float, where a matrix of N x G accumulations is not going to overflow
    p = pred.float().clamp(-30, 30).sigmoid().flatten(1)         # (N, P)
    t = tgt.float().flatten(1)                                   # (G, P)
    num = 2.0 * (p @ t.T) + eps
    den = p.sum(-1)[:, None] + t.sum(-1)[None, :] + eps
    return 1.0 - num / den


def _bce_cost(pred, tgt):
    # an untrained mask head can emit logits in the hundreds; clamping keeps the
    # pairwise sums finite, and a non-finite cost makes the assignment refuse
    p = pred.float().clamp(-30, 30).flatten(1)
    t = tgt.float().flatten(1)
    pos = F.binary_cross_entropy_with_logits(p, torch.ones_like(p), reduction="none")
    neg = F.binary_cross_entropy_with_logits(p, torch.zeros_like(p), reduction="none")
    return (pos @ t.T + neg @ (1.0 - t).T) / p.shape[1]


class QueryLoss(nn.Module):
    """Hungarian matching, then mask and score losses on the assignment.

    Ground-truth filaments carry a confidence between zero and one -- how much
    the annotators and the cross-day evidence support them. That confidence
    becomes the weight of the match, so an uncertain filament pulls less hard
    than one every annotator drew, and the model is not told a doubtful label is
    certain.
    """

    def __init__(self, w_score=2.0, w_mask=5.0, w_dice=5.0, no_object=0.1,
                 min_score=0.5):
        super().__init__()
        self.w_score, self.w_mask, self.w_dice = w_score, w_mask, w_dice
        self.no_object = no_object
        self.min_score = min_score

    @torch.no_grad()
    def match(self, pred_masks, tgt_masks, pred_score):
        cost = (self.w_mask * _bce_cost(pred_masks, tgt_masks)
                + self.w_dice * _dice_cost(pred_masks, tgt_masks)
                - self.w_score * pred_score.float().sigmoid()[:, None])
        c = torch.nan_to_num(cost.float(), nan=1e4, posinf=1e4, neginf=-1e4)
        i, j = linear_sum_assignment(c.cpu().numpy())
        return torch.as_tensor(i), torch.as_tensor(j)

    def forward(self, out, targets):
        """targets: list per batch item of (masks (G,h,w) float, weights (G,))."""
        total = pred_masks = None
        n_match = 0
        parts = {"q_score": 0.0, "q_mask": 0.0, "q_dice": 0.0}
        for b, (tgt, w) in enumerate(targets):
            pred_masks = out["masks"][b]
            score = out["score"][b]
            keep = w >= self.min_score
            tgt, w = tgt[keep], w[keep]
            label = torch.zeros_like(score)
            if len(tgt):
                i, j = self.match(pred_masks, tgt, score)
                i, j = i.to(score.device), j.to(score.device)
                label[i] = 1.0
                pm, tm, ww = pred_masks[i], tgt[j], w[j]
                pm, tm = pm.float(), tm.float()
                bce = F.binary_cross_entropy_with_logits(
                    pm, tm, reduction="none").flatten(1).mean(1)
                p = pm.sigmoid().flatten(1)
                t = tm.flatten(1)
                dice = 1.0 - (2 * (p * t).sum(1) + 1.0) / (p.sum(1) + t.sum(1) + 1.0)
                mask_l = (bce * ww).sum() / ww.sum().clamp(min=1e-6)
                dice_l = (dice * ww).sum() / ww.sum().clamp(min=1e-6)
                n_match += len(i)
            else:
                mask_l = pred_masks.sum() * 0.0
                dice_l = mask_l
            # queries left unmatched are trained to say nothing, but gently --
            # there are 50 of them and at most 17 filaments, so an unweighted
            # term would drown the positives
            weight = torch.where(label > 0.5, torch.ones_like(label),
                                 torch.full_like(label, self.no_object))
            sc = F.binary_cross_entropy_with_logits(score.float(), label.float(),
                                                    reduction="none")
            sc = (sc * weight).sum() / weight.sum()
            l = self.w_score * sc + self.w_mask * mask_l + self.w_dice * dice_l
            total = l if total is None else total + l
            parts["q_score"] += float(sc.detach())
            parts["q_mask"] += float(mask_l.detach())
            parts["q_dice"] += float(dice_l.detach())
        n = max(len(targets), 1)
        for k in parts:
            parts[k] /= n
        parts["q_matched"] = n_match / n
        return total / n, parts


def targets_from_inst(inst, scores, size):
    """Per-instance binary masks at the query head's resolution, plus weights.

    inst is the full-resolution instance map; scores maps instance id to how
    much the evidence supports it.
    """
    small = F.interpolate(inst.float()[None], size=size, mode="nearest")[0, 0]
    out, w = [], []
    for i, s in scores.items():
        m = (small == i)
        if m.sum() < 2:
            continue
        out.append(m.float())
        w.append(s)
    if not out:
        z = torch.zeros((0,) + tuple(size), device=inst.device)
        return z, torch.zeros(0, device=inst.device)
    return torch.stack(out), torch.tensor(w, device=inst.device, dtype=torch.float32)


@torch.no_grad()
def instances_from_queries(score, masks, sem, full_size, score_thresh=0.5,
                           mask_thresh=0.0, sem_thresh=0.5, min_area=60):
    """Turn the head's output into final masks, edges taken from the semantic map.

    The query says which object this is; the semantic head says where its
    boundary runs. Overlaps go to the more confident query, since Panoptic
    Quality assumes one segment per pixel and a contested pixel would inflate
    both unions.
    """
    keep = (score.sigmoid() > score_thresh).nonzero(as_tuple=True)[0]
    if not len(keep):
        return [], []
    order = keep[score[keep].argsort(descending=True)]
    up = F.interpolate(masks[order][None], size=full_size, mode="bilinear",
                       align_corners=False)[0] > mask_thresh
    sem_bin = torch.as_tensor(sem >= sem_thresh, device=up.device)
    claimed = torch.zeros(full_size, dtype=torch.bool, device=up.device)
    out, conf = [], []
    for k in range(len(order)):
        m = up[k] & sem_bin & ~claimed
        if int(m.sum()) < min_area:
            continue
        claimed |= m
        out.append(m.cpu().numpy())
        conf.append(float(score[order[k]].sigmoid()))
    return out, conf
