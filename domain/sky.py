"""Sun / moon positions from clock time and real lunar phase + scene grading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SYNODIC_MONTH = 29.530588853
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
DEFAULT_LAT_DEG = 51.0


@dataclass(frozen=True, slots=True)
class CelestialState:
    day_frac: float
    sun_alt_deg: float
    sun_az_deg: float
    moon_alt_deg: float
    moon_az_deg: float
    moon_phase: float
    moon_illumination: float
    is_day: bool
    local_time: datetime


@dataclass(frozen=True, slots=True)
class SceneGrade:
    """Color grade applied to the whole scene (landscape + objects + character)."""

    brightness: float   # multiply RGB, 1.0 = neutral
    saturation: float   # 1.0 = full, 0.0 = grey
    # Additive tint after saturation (r,g,b in -1..1-ish scale, applied as offset*255)
    tint_r: float
    tint_g: float
    tint_b: float


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


def moon_phase_at(dt: datetime) -> tuple[float, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - REF_NEW_MOON
    days = delta.total_seconds() / 86400.0
    phase = (days % SYNODIC_MONTH) / SYNODIC_MONTH
    illumination = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return phase, illumination


def _solar_altitude_azimuth(
    dt: datetime, lat_deg: float = DEFAULT_LAT_DEG
) -> tuple[float, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    doy = dt.timetuple().tm_yday
    decl = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    ha = 15.0 * (hour - 12.0)
    lat = math.radians(lat_deg)
    dec = math.radians(decl)
    ha_r = math.radians(ha)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha_r)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    cos_az = (math.sin(dec) - math.sin(lat) * math.sin(math.radians(alt))) / (
        math.cos(lat) * math.cos(math.radians(alt)) + 1e-9
    )
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if ha > 0:
        az = 360.0 - az
    return alt, az


def _moon_altitude_azimuth(
    dt: datetime, phase: float, lat_deg: float = DEFAULT_LAT_DEG
) -> tuple[float, float]:
    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg)
    elong = phase * 360.0
    moon_az = (sun_az + elong) % 360.0
    hour = dt.hour + dt.minute / 60.0
    moon_hour = (hour - phase * 12.0) % 24.0
    ha = 15.0 * (moon_hour - 12.0)
    doy = dt.timetuple().tm_yday
    decl = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    lat = math.radians(lat_deg)
    dec = math.radians(decl)
    ha_r = math.radians(ha)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha_r)
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    return alt, moon_az


def celestial_at(
    dt: datetime | None = None,
    *,
    tz: str = "Europe/Berlin",
    lat_deg: float = DEFAULT_LAT_DEG,
) -> CelestialState:
    if dt is None:
        dt = datetime.now(ZoneInfo(tz))
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        dt = dt.astimezone(ZoneInfo(tz))

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg)
    phase, illum = moon_phase_at(dt)
    moon_alt, moon_az = _moon_altitude_azimuth(dt, phase, lat_deg)

    return CelestialState(
        day_frac=hour / 24.0,
        sun_alt_deg=sun_alt,
        sun_az_deg=sun_az,
        moon_alt_deg=moon_alt,
        moon_az_deg=moon_az,
        moon_phase=phase,
        moon_illumination=illum,
        is_day=sun_alt > 0.0,
        local_time=dt,
    )


def sky_colors(state: CelestialState) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Top and bottom sky gradient RGB from sun altitude."""
    alt = state.sun_alt_deg
    if alt > 15:
        return (90, 170, 230), (180, 220, 245)
    if alt > 0:
        t = alt / 15.0
        return _lerp((255, 140, 80), (90, 170, 230), t), _lerp(
            (255, 200, 120), (180, 220, 245), t
        )
    if alt > -8:
        t = (alt + 8) / 8.0
        return _lerp((20, 24, 60), (255, 140, 80), t), _lerp(
            (40, 40, 80), (255, 200, 120), t
        )
    return (10, 14, 40), (25, 30, 55)


def scene_grade(state: CelestialState) -> SceneGrade:
    """
    Full-scene color grade from sun altitude.

    Day  → neutral brightness/sat
    Golden hour → warm tint, slight sat boost
    Twilight → cooler, dimmer, lower sat
    Night → dark, desaturated, blue tint
    """
    alt = state.sun_alt_deg
    if alt > 20:
        return SceneGrade(1.0, 1.0, 0.0, 0.0, 0.0)
    if alt > 5:
        # Soft golden
        t = (alt - 5) / 15.0  # 1 at 20°, 0 at 5°
        return SceneGrade(
            brightness=0.92 + 0.08 * t,
            saturation=1.05 - 0.05 * t,
            tint_r=0.06 * (1 - t),
            tint_g=0.02 * (1 - t),
            tint_b=-0.03 * (1 - t),
        )
    if alt > 0:
        # Strong golden hour
        t = alt / 5.0
        return SceneGrade(
            brightness=0.78 + 0.14 * t,
            saturation=1.1,
            tint_r=0.12 * (1 - 0.5 * t),
            tint_g=0.04 * (1 - 0.5 * t),
            tint_b=-0.06 * (1 - 0.5 * t),
        )
    if alt > -6:
        # Twilight
        t = (alt + 6) / 6.0  # 0 deep twilight, 1 sunset
        return SceneGrade(
            brightness=0.45 + 0.33 * t,
            saturation=0.55 + 0.45 * t,
            tint_r=0.04 * t,
            tint_g=0.0,
            tint_b=0.10 * (1 - t) + 0.02 * t,
        )
    if alt > -12:
        # Civil → nautical night
        t = (alt + 12) / 6.0
        return SceneGrade(
            brightness=0.28 + 0.17 * t,
            saturation=0.35 + 0.20 * t,
            tint_r=-0.02,
            tint_g=0.0,
            tint_b=0.12,
        )
    # Deep night
    return SceneGrade(
        brightness=0.22,
        saturation=0.30,
        tint_r=-0.03,
        tint_g=0.0,
        tint_b=0.14,
    )


def _lerp(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def alt_az_to_screen(
    alt_deg: float,
    az_deg: float,
    width: int,
    height: int,
    *,
    horizon_y: float | None = None,
    zenith_y: float = 20.0,
) -> tuple[float, float] | None:
    if alt_deg < -2:
        return None
    if horizon_y is None:
        horizon_y = height * 0.55
    az = az_deg % 360.0
    if 0 <= az <= 180:
        x_frac = 0.1 + 0.8 * (az / 180.0)
    else:
        x_frac = 0.9 - 0.8 * ((az - 180.0) / 180.0)
    t = max(0.0, min(1.0, alt_deg / 75.0))
    y = horizon_y + (zenith_y - horizon_y) * t
    return width * x_frac, y
