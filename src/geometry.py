"""Solar disk geometry for GONG H-alpha images.

The naive approach (threshold + centroid) fails on these images because a faint
scattered-light halo surrounds the disk and gets counted as solar surface -- it
overestimated the radius by ~13% on the first image we tried.

The real limb is unmistakable in the radial derivative: H-alpha limb brightening
produces a sharp bright rim followed by the steepest brightness drop in the whole
image. We find that drop along many rays and fit a circle to the resulting points.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["DiskFit", "fit_disk", "radial_coordinates"]


class DiskFit:
    """Result of a disk fit: centre (cx, cy) and radius r, all in pixels."""

    __slots__ = ("cx", "cy", "r", "residual", "n_points")

    def __init__(self, cx: float, cy: float, r: float, residual: float, n_points: int):
        self.cx, self.cy, self.r = cx, cy, r
        self.residual = residual      # RMS of edge points about the fitted circle, px
        self.n_points = n_points      # edge points surviving outlier rejection

    def __repr__(self) -> str:
        return (f"DiskFit(cx={self.cx:.1f}, cy={self.cy:.1f}, r={self.r:.1f}, "
                f"residual={self.residual:.2f}px, n={self.n_points})")


def _sample_ray(img, cx, cy, angle, r_max, step=1.0):
    """Bilinear-sample brightness outward from (cx, cy) along one ray."""
    r = np.arange(0.0, r_max, step)
    x = cx + r * np.cos(angle)
    y = cy + r * np.sin(angle)
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return r, v


def _fit_circle(x, y):
    """Algebraic (Kasa) circle fit. Returns cx, cy, r."""
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cx, cy, r


def fit_disk(img, n_rays=720, n_iter=3, smooth=2.0, tol=0.05) -> DiskFit:
    """Locate the solar limb.

    img : 2-D float array (a grayscale H-alpha full-disk image)
    tol : keep only rays whose limb radius is within this fraction of the median

    Two things here are deliberate, both learned from failures on this dataset:

    * The search starts at the frame centre, not at a brightness centroid. These
      images are registered to the frame centre, and on the 5% of frames with a
      bright asymmetric halo the centroid is pulled sideways; the ray length was
      then capped by the near frame edge and rays pointing the other way were
      truncated before ever reaching the limb, producing nonsense edge points.
      That corrupted the fit on 33 of 707 training images, with radii as small
      as 571 px against a true 904 px.

    * Outliers are rejected against the *median* per-ray radius rather than a
      standard deviation. When many rays fail at once the standard deviation is
      inflated by the failures themselves and rejects nothing.

    The inner 30% of each ray is ignored so sunspots and filaments near disk
    centre cannot masquerade as the limb.
    """
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape

    cx, cy = w / 2.0, h / 2.0
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ex = ey = None

    for _ in range(n_iter):
        # Generous ray length, measured from the frame centre so that an
        # off-centre estimate cannot truncate rays before the limb.
        r_max = min(w, h) / 2.0 * 0.98
        radii, keep_ang = [], []
        for ang in angles:
            r, v = _sample_ray(img, cx, cy, ang, r_max)
            if len(r) < 50:
                continue
            v = gaussian_filter1d(v, smooth)
            d = np.gradient(v)
            lo = int(0.30 * len(d))            # never look inside 30% of the ray
            k = lo + int(np.argmin(d[lo:]))    # steepest drop == the limb
            radii.append(r[k]); keep_ang.append(ang)

        radii = np.asarray(radii); keep_ang = np.asarray(keep_ang)
        if len(radii) < 20:
            break
        med = float(np.median(radii))
        ok = np.abs(radii - med) <= tol * med
        if ok.sum() < 20:
            ok = np.ones(len(radii), bool)
        rr_, aa_ = radii[ok], keep_ang[ok]
        ex = cx + rr_ * np.cos(aa_)
        ey = cy + rr_ * np.sin(aa_)
        cx, cy, rad = _fit_circle(ex, ey)

    if ex is None:
        return DiskFit(w / 2.0, h / 2.0, 904.0, float("nan"), 0)

    dist = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
    residual = float(np.sqrt(np.mean((dist - rad) ** 2)))

    # Last-resort sanity net: if the circle fit still disagrees with the raw
    # per-ray measurement, trust the rays and the frame centre.
    med_r = float(np.median(dist))
    if not np.isfinite(rad) or abs(rad - med_r) > tol * med_r:
        cx, cy, rad = w / 2.0, h / 2.0, med_r
        dist = np.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
        residual = float(np.sqrt(np.mean((dist - rad) ** 2)))

    return DiskFit(cx, cy, rad, residual, len(ex))


def radial_coordinates(shape, fit: DiskFit):
    """r / R_sun and position angle for every pixel."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dx, dy = xx - fit.cx, yy - fit.cy
    return np.sqrt(dx * dx + dy * dy) / fit.r, np.arctan2(dy, dx)
