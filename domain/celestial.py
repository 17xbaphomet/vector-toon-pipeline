"""Celestial sphere projection: view along walking heading.

Azimuth: 0=N, 90=E, 180=S, 270=W.
Screen: left/right of view_az, horizon → zenith.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_LAT = 51.0
DEFAULT_LON = 10.0
AZ_HALF_FOV = 90.0
ALT_MAX = 90.0


@dataclass(frozen=True, slots=True)
class Star:
    name: str
    ra_hours: float
    dec_deg: float
    mag: float


BRIGHT_STARS: tuple[Star, ...] = (
    Star("Sirius", 6.7525, -16.716, -1.46),
    Star("Arcturus", 14.2610, 19.182, -0.05),
    Star("Vega", 18.6156, 38.783, 0.03),
    Star("Capella", 5.2782, 45.998, 0.08),
    Star("Rigel", 5.2423, -8.202, 0.13),
    Star("Procyon", 7.6550, 5.225, 0.34),
    Star("Betelgeuse", 5.9195, 7.407, 0.42),
    Star("Altair", 19.8464, 8.868, 0.76),
    Star("Aldebaran", 4.5987, 16.509, 0.86),
    Star("Antares", 16.4901, -26.432, 0.96),
    Star("Spica", 13.4199, -11.161, 0.97),
    Star("Pollux", 7.7553, 28.026, 1.14),
    Star("Fomalhaut", 22.9608, -29.622, 1.16),
    Star("Deneb", 20.6905, 45.280, 1.25),
    Star("Regulus", 10.1395, 11.967, 1.35),
    Star("Adhara", 6.7229, -28.972, 1.50),
    Star("Castor", 7.5766, 31.888, 1.58),
    Star("Bellatrix", 5.4189, 6.349, 1.64),
    Star("Elnath", 5.4382, 28.607, 1.65),
    Star("Alnilam", 5.6036, -1.202, 1.69),
    Star("Alioth", 12.9005, 55.960, 1.77),
    Star("Alnitak", 5.6794, -1.943, 1.77),
    Star("Dubhe", 11.0621, 61.751, 1.79),
    Star("Mirfak", 3.4054, 49.861, 1.80),
    Star("Wezen", 7.1399, -26.393, 1.83),
    Star("Alkaid", 13.7923, 49.313, 1.86),
    Star("Alhena", 6.6287, 16.399, 1.90),
    Star("Peacock", 20.4275, -56.735, 1.91),
    Star("Mirzam", 6.3783, -17.956, 1.98),
    Star("Alphard", 9.4598, -8.659, 2.00),
    Star("Algieba", 10.3329, 19.842, 2.01),
    Star("Hamal", 2.1195, 23.462, 2.01),
    Star("Diphda", 0.7265, -17.987, 2.04),
    Star("Nunki", 18.9211, -26.297, 2.05),
    Star("Menkalinan", 5.9919, 44.948, 2.06),
    Star("Alpheratz", 0.1398, 29.091, 2.07),
    Star("Mirach", 1.1622, 35.620, 2.07),
    Star("Polaris", 2.5303, 89.264, 1.98),
    Star("Kochab", 14.8451, 74.155, 2.07),
    Star("Rasalhague", 17.5822, 12.560, 2.08),
    Star("Algol", 3.1361, 40.956, 2.09),
    Star("Denebola", 11.8177, 14.572, 2.14),
    Star("Cih", 0.6751, 56.537, 2.15),
    Star("Etamin", 17.9434, 51.489, 2.24),
    Star("Schedar", 0.6751, 56.537, 2.24),
    Star("Mintaka", 5.5334, -0.299, 2.25),
    Star("Caph", 0.1529, 59.150, 2.28),
    Star("Alphecca", 15.5781, 26.715, 2.22),
    Star("Mizar", 13.3987, 54.925, 2.27),
    Star("Sadr", 20.3705, 40.257, 2.23),
)


@dataclass(frozen=True, slots=True)
class SkyPoint:
    name: str
    alt_deg: float
    az_deg: float
    x: float
    y: float
    mag: float
    kind: str


def _julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc = dt.astimezone(timezone.utc)
    y, m, d = utc.year, utc.month, utc.day
    hh = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    return jd + hh / 24.0


def local_sidereal_time_deg(dt: datetime, lon_deg: float = DEFAULT_LON) -> float:
    jd = _julian_date(dt)
    T = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - T * T * T / 38710000.0
    )
    return (gmst + lon_deg) % 360.0


def equatorial_to_horizontal(
    ra_hours: float,
    dec_deg: float,
    dt: datetime,
    lat_deg: float = DEFAULT_LAT,
    lon_deg: float = DEFAULT_LON,
) -> tuple[float, float]:
    lst = local_sidereal_time_deg(dt, lon_deg)
    ra_deg = ra_hours * 15.0
    ha = math.radians((lst - ra_deg) % 360.0)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)
    sin_alt = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(ha)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    cos_az = (
        (math.sin(dec) - math.sin(math.radians(alt)) * math.sin(lat))
        / (math.cos(math.radians(alt)) * math.cos(lat) + 1e-12)
    )
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if math.sin(ha) > 0:
        az = 360.0 - az
    return alt, az


def heading_project(
    alt_deg: float,
    az_deg: float,
    width: int,
    height: int,
    *,
    view_az_deg: float = 180.0,
    horizon_y_frac: float = 0.55,
    az_half_fov: float = AZ_HALF_FOV,
) -> tuple[float, float] | None:
    """Project onto camera facing view_az_deg (walking heading)."""
    if alt_deg < -1.0:
        return None
    d_az = ((az_deg - view_az_deg + 540.0) % 360.0) - 180.0
    if abs(d_az) > az_half_fov:
        return None
    horizon_y = height * horizon_y_frac
    zenith_y = height * 0.02
    x = width * 0.5 + (d_az / az_half_fov) * (width * 0.5)
    t = max(0.0, min(1.0, alt_deg / 90.0))
    y = horizon_y + (zenith_y - horizon_y) * t
    return x, y


def south_facing_project(
    alt_deg: float,
    az_deg: float,
    width: int,
    height: int,
    *,
    horizon_y_frac: float = 0.55,
    az_half_fov: float = AZ_HALF_FOV,
) -> tuple[float, float] | None:
    return heading_project(
        alt_deg, az_deg, width, height,
        view_az_deg=180.0,
        horizon_y_frac=horizon_y_frac,
        az_half_fov=az_half_fov,
    )


def project_stars(
    dt: datetime,
    width: int,
    height: int,
    *,
    lat_deg: float = DEFAULT_LAT,
    lon_deg: float = DEFAULT_LON,
    view_az_deg: float = 180.0,
    max_mag: float = 2.5,
    min_alt: float = 0.0,
) -> list[SkyPoint]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
    out: list[SkyPoint] = []
    for s in BRIGHT_STARS:
        if s.mag > max_mag:
            continue
        alt, az = equatorial_to_horizontal(s.ra_hours, s.dec_deg, dt, lat_deg, lon_deg)
        if alt < min_alt:
            continue
        xy = heading_project(alt, az, width, height, view_az_deg=view_az_deg)
        if xy is None:
            continue
        out.append(SkyPoint(name=s.name, alt_deg=alt, az_deg=az, x=xy[0], y=xy[1], mag=s.mag, kind="star"))
    return out


def project_body(
    alt_deg: float,
    az_deg: float,
    width: int,
    height: int,
    name: str,
    kind: str,
    mag: float = 0.0,
    view_az_deg: float = 180.0,
) -> SkyPoint | None:
    xy = heading_project(alt_deg, az_deg, width, height, view_az_deg=view_az_deg)
    if xy is None:
        return None
    return SkyPoint(name=name, alt_deg=alt_deg, az_deg=az_deg, x=xy[0], y=xy[1], mag=mag, kind=kind)
