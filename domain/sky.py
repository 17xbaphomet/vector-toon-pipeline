"""Sun / moon positions from clock time and real lunar phase."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Synodic month (new moon → new moon)
SYNODIC_MONTH = 29.530588853
# Reference new moon: 2000-01-06 18:14 UTC (known new moon)
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)

# Approximate latitude for central Germany (for solar altitude)
DEFAULT_LAT_DEG = 51.0


@dataclass(frozen=True, slots=True)
class CelestialState:
    """Everything needed to draw the sky at one moment."""

    # 0..1 fraction of day (0=midnight, 0.5=noon)
    day_frac: float
    # Solar altitude degrees (−90..+90); negative = below horizon
    sun_alt_deg: float
    # Solar azimuth degrees (0=N, 90=E, 180=S, 270=W) — used for left/right
    sun_az_deg: float
    # Moon altitude / azimuth
    moon_alt_deg: float
    moon_az_deg: float
    # 0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter
    moon_phase: float
    # Illuminated fraction 0..1
    moon_illumination: float
    # True if sun is above horizon
    is_day: bool
    # Local datetime used
    local_time: datetime


def _julian_date(dt: datetime) -> float:
    """Julian Date from timezone-aware datetime."""
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
    """
    Real lunar phase from date.

    Returns (phase 0..1, illumination 0..1).
    phase: 0=new, 0.25=first quarter, 0.5=full, 0.75=last quarter
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - REF_NEW_MOON
    days = delta.total_seconds() / 86400.0
    phase = (days % SYNODIC_MONTH) / SYNODIC_MONTH
    # Illumination ≈ (1 − cos(2π · phase)) / 2
    illumination = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return phase, illumination


def _solar_altitude_azimuth(
    dt: datetime, lat_deg: float = DEFAULT_LAT_DEG
) -> tuple[float, float]:
    """
    Approximate solar altitude & azimuth for mid-latitudes.
    Good enough for cartoon sky positioning.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local = dt

    # Day of year
    doy = local.timetuple().tm_yday
    # Solar declination (approx)
    decl = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    # Equation of time neglected; use clock hour as local solar hour
    hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    # Hour angle: 0 at solar noon, −15° per hour before noon
    ha = 15.0 * (hour - 12.0)

    lat = math.radians(lat_deg)
    dec = math.radians(decl)
    ha_r = math.radians(ha)

    sin_alt = (
        math.sin(lat) * math.sin(dec)
        + math.cos(lat) * math.cos(dec) * math.cos(ha_r)
    )
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))

    cos_az = (
        (math.sin(dec) - math.sin(lat) * math.sin(math.radians(alt)))
        / (math.cos(lat) * math.cos(math.radians(alt)) + 1e-9)
    )
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if ha > 0:
        az = 360.0 - az  # afternoon → west

    return alt, az


def _moon_altitude_azimuth(
    dt: datetime, phase: float, lat_deg: float = DEFAULT_LAT_DEG
) -> tuple[float, float]:
    """
    Approximate moon position relative to sun.

    Full moon ≈ opposite the sun; new moon ≈ near the sun.
    Altitude modulated by phase and time of day for a natural arc.
    """
    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg)
    # Moon elongation ≈ phase * 360°
    elong = phase * 360.0
    moon_az = (sun_az + elong) % 360.0

    # Rough altitude: opposite to sun when full, similar when new
    # Shift hour for moon: full moon highest around midnight
    hour = dt.hour + dt.minute / 60.0
    # Moon's "local hour" offset by phase*12h
    moon_hour = (hour - phase * 12.0) % 24.0
    ha = 15.0 * (moon_hour - 12.0)
    # Declination roughly follows sun with lag — use similar formula
    doy = dt.timetuple().tm_yday
    decl = 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    # Moon declination wobbles ±5° around ecliptic — ignore for cartoon
    lat = math.radians(lat_deg)
    dec = math.radians(decl)
    ha_r = math.radians(ha)
    sin_alt = (
        math.sin(lat) * math.sin(dec)
        + math.cos(lat) * math.cos(dec) * math.cos(ha_r)
    )
    alt = math.degrees(math.asin(max(-1.0, min(1.0, sin_alt))))
    return alt, moon_az


def celestial_at(
    dt: datetime | None = None,
    *,
    tz: str = "Europe/Berlin",
    lat_deg: float = DEFAULT_LAT_DEG,
) -> CelestialState:
    """Compute sun/moon state for a local datetime (default: now in tz)."""
    if dt is None:
        dt = datetime.now(ZoneInfo(tz))
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        dt = dt.astimezone(ZoneInfo(tz))

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    day_frac = hour / 24.0

    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg)
    phase, illum = moon_phase_at(dt)
    moon_alt, moon_az = _moon_altitude_azimuth(dt, phase, lat_deg)

    return CelestialState(
        day_frac=day_frac,
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
    """
    Top and bottom sky gradient RGB from sun altitude.
    Night → dawn → day → dusk → night.
    """
    alt = state.sun_alt_deg
    if alt > 15:
        # Day
        return (90, 170, 230), (180, 220, 245)
    if alt > 0:
        # Golden hour
        t = alt / 15.0
        top = _lerp((255, 140, 80), (90, 170, 230), t)
        bot = _lerp((255, 200, 120), (180, 220, 245), t)
        return top, bot
    if alt > -8:
        # Twilight
        t = (alt + 8) / 8.0
        top = _lerp((20, 24, 60), (255, 140, 80), t)
        bot = _lerp((40, 40, 80), (255, 200, 120), t)
        return top, bot
    # Night
    return (10, 14, 40), (25, 30, 55)


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
    """
    Map altitude/azimuth to screen position for a side-scroller.

    Azimuth 90° (east) → left, 270° (west) → right (walking-right view).
    Returns None if below horizon.
    """
    if alt_deg < -2:
        return None
    if horizon_y is None:
        horizon_y = height * 0.55

    # X from azimuth: east(90)=0.15, south(180)=0.5, west(270)=0.85
    # Fold so path arcs across the sky
    az = az_deg % 360.0
    # Prefer the southern arc for northern hemisphere viewers
    if 0 <= az <= 180:
        x_frac = 0.1 + 0.8 * (az / 180.0)
    else:
        # Night side / north — push to edges
        x_frac = 0.9 - 0.8 * ((az - 180.0) / 180.0)

    # Y from altitude: 0°=horizon, 90°=zenith
    t = max(0.0, min(1.0, alt_deg / 75.0))
    y = horizon_y + (zenith_y - horizon_y) * t
    x = width * x_frac
    return x, y
