"""Photometric standardisation of GONG H-alpha full-disk images.

The images arrive already registered: disk centre (1024, 1024) +- 0.5 px, solar
radius 904 +- 0.8 px, central brightness 139 +- 3. What is *not* standardised is
how steeply each image dims towards the limb -- brightness(0.9 R)/brightness(0.1 R)
ranges from 0.72 to 0.85 across images, and varies that much within a single
station too. That spread is atmospheric (seeing, transparency, airmass), not solar.

Flattening removes it: we measure each image's own quiet-sun radial profile and
divide it out, so quiet chromosphere sits at 1.0 everywhere on the disk and a
filament's contrast means the same thing at disk centre as near the limb.

The profile is taken with a high percentile rather than the median: filaments are
dark, so a median profile is dragged down wherever filaments cluster, which would
partly flatten away the very signal we want.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["quiet_sun_profile", "flatten", "standardise"]


def quiet_sun_profile(img, rr, n_bins=200, r_max=1.0, pct=75, smooth=3.0):
    """Quiet-sun brightness as a function of r/R_sun.

    pct : percentile per radial bin. 75 tracks the quiet chromosphere while
          ignoring filaments and sunspots, which are dark minorities.
    """
    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.digitize(rr.ravel(), edges) - 1
    v = img.ravel()
    prof = np.full(n_bins, np.nan, dtype=np.float32)
    order = np.argsort(idx, kind="stable")
    idx_s, v_s = idx[order], v[order]
    bounds = np.searchsorted(idx_s, np.arange(n_bins + 1))
    for i in range(n_bins):
        a, b = bounds[i], bounds[i + 1]
        if b - a > 50:
            prof[i] = np.percentile(v_s[a:b], pct)
    # fill any empty bins, then smooth
    good = np.isfinite(prof)
    if good.sum() < 5:
        raise ValueError("radial profile could not be measured")
    centres = 0.5 * (edges[1:] + edges[:-1])
    prof = np.interp(centres, centres[good], prof[good])
    return centres, gaussian_filter1d(prof, smooth)


def flatten(img, rr, r_max=1.0, pct=75, floor=1e-3):
    """Divide out the image's own limb-darkening profile.

    Returns an array where quiet disk is ~1.0. Pixels beyond r_max are set to 0:
    off-disk material (prominences, the scattered-light halo) has no meaningful
    flat-field and must not leak into the model as if it were disk signal.
    """
    centres, prof = quiet_sun_profile(img, rr, r_max=r_max, pct=pct)
    denom = np.interp(np.clip(rr, 0, r_max), centres, prof)
    out = np.where(rr <= r_max, img / np.maximum(denom, floor), 0.0)
    return out.astype(np.float32)


def standardise(img, fit, r_max=0.995, pct=75):
    """Full pipeline for one image: flatten, then z-score the on-disk pixels.

    fit : a DiskFit from geometry.fit_disk

    Returns (standardised image, on-disk boolean mask). The z-score uses the
    on-disk pixels only, so the large black off-disk area cannot skew the scale.
    """
    from geometry import radial_coordinates
    rr, _ = radial_coordinates(img.shape, fit)
    flat = flatten(img, rr, r_max=r_max, pct=pct)
    disk = rr <= r_max
    vals = flat[disk]
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    out = np.where(disk, (flat - mu) / max(sd, 1e-6), 0.0)
    return out.astype(np.float32), disk
