"""Solar ephemeris and heliographic coordinates, from the observation time alone.

Needed to compare two images of the same Sun taken days apart. A filament sits
at a fixed heliographic position while the disk turns underneath it, so matching
must happen in heliographic coordinates -- in the image plane a day moves an
equatorial feature 217 px, about 1.7 filament lengths, and matching is hopeless.

Three angles define the view, all functions of time only:

  B0   heliographic latitude of the disk centre, +-7.25 deg over a year
  L0   Carrington longitude of the central meridian, decreasing ~13.2 deg/day
  P    position angle of the solar rotation axis on the sky, +-26.3 deg

Implemented from Meeus, *Astronomical Algorithms*, ch. 29 ("Ephemeris for
Physical Observations of the Sun"), which is good to about 0.01 deg -- far
finer than the pixel scale here, where 1 px at disk centre is 0.063 deg of
heliographic longitude.

**The competition frames are already solar-north-up: pass P = 0 for them.**
Measured 2026-11-09 by carrying the annotated filaments of 287 photograph pairs
(7 h - 2 d apart) to the later frame's time and counting how many land within
15 px of a filament drawn there: P applied 3.0%, P = 0 18.1% (no motion at all:
0.6%). The same test found B0 entering with the wrong sign in the two pixel
conversions below; fixed then. Both errors together are why the August label
propagation (v3) put more than half of its filaments on no dark material.

Observer position on Earth is ignored deliberately. The diurnal solar parallax
is 8.79 arcsec against a solar radius of about 960 arcsec, so two GONG stations
on opposite sides of the Earth see the disk centre shift by at most 0.9% of the
radius, roughly 8 px here -- one filament width, and far below the day-to-day
rotation this module exists to remove. See parallax_pixels() for the number.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np

__all__ = ["solar_angles", "pixel_to_heliographic", "heliographic_to_pixel",
           "rotation_rate", "parallax_pixels", "julian_day"]

DEG = math.pi / 180.0
SOLAR_PARALLAX_ARCSEC = 8.794
SOLAR_RADIUS_ARCSEC = 959.63


def julian_day(t: datetime) -> float:
    y, m = t.year, t.month
    d = (t.day + t.hour / 24.0 + t.minute / 1440.0 + t.second / 86400.0)
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d + b - 1524.5)


def solar_angles(t: datetime):
    """(P, B0, L0) in degrees for an observation time (UTC).

    P  : position angle of the north end of the rotation axis, east of north
    B0 : heliographic latitude of the disk centre
    L0 : Carrington longitude of the central meridian
    """
    jd = julian_day(t)
    theta = (jd - 2398220.0) * 360.0 / 25.38          # Carrington rotation phase
    i = 7.25                                           # inclination of the solar equator
    k = 73.6667 + 1.3958333 * (jd - 2396758.0) / 36525.0

    tau = jd - 2451545.0
    # Sun's geometric longitude and the obliquity, low-precision series (Meeus 25)
    L = 280.46646 + 0.9856474 * tau
    g = (357.52911 + 0.9856003 * tau) % 360.0
    lam = L + 1.914602 * math.sin(g * DEG) + 0.019993 * math.sin(2 * g * DEG)
    eps = 23.439291 - 3.563e-7 * tau
    lam_prime = lam - 0.00569 - 0.00478 * math.sin((125.04 - 1934.136 * tau / 36525.0) * DEG)

    x = math.atan(-math.cos(lam_prime * DEG) * math.tan(eps * DEG))
    y = math.atan(-math.cos((lam - k) * DEG) * math.tan(i * DEG))
    P = (x + y) / DEG

    B0 = math.asin(math.sin((lam - k) * DEG) * math.sin(i * DEG)) / DEG

    eta = math.atan2(-math.sin((lam - k) * DEG) * math.cos(i * DEG),
                     -math.cos((lam - k) * DEG))
    L0 = (eta / DEG - theta) % 360.0
    return P, B0, L0


def pixel_to_heliographic(x, y, cx, cy, r, P, B0, L0):
    """Image pixels to (latitude, Carrington longitude) in degrees.

    Returns NaN where the pixel falls outside the disk. x grows right, y grows
    down, which is why the vertical offset is negated before the P rotation.
    """
    dx = (np.asarray(x, float) - cx) / r
    dy = -(np.asarray(y, float) - cy) / r

    p = P * DEG
    xr = dx * math.cos(p) + dy * math.sin(p)
    yr = -dx * math.sin(p) + dy * math.cos(p)

    rho2 = xr ** 2 + yr ** 2
    with np.errstate(invalid="ignore"):
        zr = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
    off = rho2 > 1.0

    b = B0 * DEG
    # rotate about the x axis by B0 to reach heliographic frame; the disk centre
    # (yr 0, zr 1) must land on latitude +B0. Until 2026-11-09 both signs here were
    # flipped, putting it at -B0 (see the module docstring)
    yh = yr * math.cos(b) + zr * math.sin(b)
    zh = -yr * math.sin(b) + zr * math.cos(b)

    lat = np.degrees(np.arcsin(np.clip(yh, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xr, zh)) + L0

    lat = np.where(off, np.nan, lat)
    lon = np.where(off, np.nan, (lon + 180.0) % 360.0 - 180.0 + L0 - L0)
    lon = np.where(off, np.nan, np.degrees(np.arctan2(xr, zh)) + L0)
    return lat, lon % 360.0


def heliographic_to_pixel(lat, lon, cx, cy, r, P, B0, L0):
    """Carrington (lat, lon) in degrees back to image pixels.

    The third return value is True where the point is on the visible hemisphere.
    """
    la = np.radians(np.asarray(lat, float))
    dlon = np.radians(np.asarray(lon, float) - L0)
    b = B0 * DEG

    xh = np.cos(la) * np.sin(dlon)
    yh = np.sin(la)
    zh = np.cos(la) * np.cos(dlon)

    yr = yh * math.cos(b) - zh * math.sin(b)
    zr = yh * math.sin(b) + zh * math.cos(b)
    xr = xh

    p = P * DEG
    dx = xr * math.cos(p) - yr * math.sin(p)
    dy = xr * math.sin(p) + yr * math.cos(p)

    return cx + dx * r, cy - dy * r, zr > 0.0


def rotation_rate(lat_deg, law="snodgrass", synodic=True):
    """Rotation rate in degrees per day at a heliographic latitude.

    'snodgrass'  magnetic-feature rate, 14.713 - 2.396 sin^2 - 1.787 sin^4
    'howard'     H-alpha plasma rate,   14.522 - 2.84 sin^2
    'rigid'      the Carrington rate, 14.184

    Filaments are anchored in the large-scale field and rotate more rigidly than
    the photospheric plasma, so the right law is an empirical question for this
    dataset rather than a settled one; measure_rotation.py answers it by aligning
    annotations across day pairs.

    Subtracting Earth's orbital motion (0.9856 deg/day) gives the synodic rate,
    which is what an observer on Earth actually sees.
    """
    s = np.sin(np.radians(np.asarray(lat_deg, float))) ** 2
    if law == "snodgrass":
        w = 14.713 - 2.396 * s - 1.787 * s * s
    elif law == "howard":
        w = 14.522 - 2.84 * s
    elif law == "rigid":
        w = np.full_like(s, 14.184)
    else:
        raise ValueError(law)
    return w - 0.9856 if synodic else w


def parallax_pixels(disk_radius_px=904.0):
    """How far two Earth-surface observers disagree about the disk centre.

    The full diurnal parallax is 8.794 arcsec; the solar radius is 959.63. The
    shift is therefore under one percent of the radius, which is why the
    observing station can be ignored when matching filaments across images.
    """
    return SOLAR_PARALLAX_ARCSEC / SOLAR_RADIUS_ARCSEC * disk_radius_px


if __name__ == "__main__":
    # Meeus, example 29.a: 1992 October 13.0 TD -> P = 26.27, B0 = 5.99, L0 = 238.63
    P, B0, L0 = solar_angles(datetime(1992, 10, 13))
    print(f"1992-10-13  P={P:7.2f}  B0={B0:6.2f}  L0={L0:7.2f}")
    print(f"   Meeus 29.a  P=  26.27  B0=  5.99  L0= 238.63")
    print()
    print(f"station parallax: {parallax_pixels():.1f} px of a {904:.0f} px radius "
          f"({100*parallax_pixels()/904:.2f}% -- about one filament width)")
    print()
    print(f"{'lat':>5} {'sidereal':>9} {'synodic':>9} {'px/day at disk centre':>22}")
    for lat in (0, 30, 60):
        w = rotation_rate(lat, synodic=True)
        px = 904.0 * math.cos(math.radians(lat)) * math.radians(float(w))
        print(f"{lat:5d} {float(rotation_rate(lat, synodic=False)):9.2f} "
              f"{float(w):9.2f} {px:22.0f}")
