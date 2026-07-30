"""Tageszeitenberechnung: Sonne, Mondphase, Himmelsfarben, Szenen-Grade.

Basis: geografische Breite (Default 51° ≈ Mitte DE), lokale Uhrzeit,
echte Mondphase (synodischer Monat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo

SYNODIC_MONTH = 29.530588853
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
DEFAULT_LAT_DEG = 51.0
DEFAULT_LON_DEG = 10.0  # approx central Germany


class Tageszeit(str, Enum):
    """Named day period derived from solar altitude."""

    NACHT = "nacht"
    MORGENDÄMMERUNG = "morgendaemmerung"
    SONNENAUFGANG = "sonnenaufgang"
    VORMITTAG = "vormittag"
    MITTAG = "mittag"
    NACHMITTAG = "nachmittag"
    SONNENUNTERGANG = "sonnenuntergang"
    ABENDDÄMMERUNG = "abenddaemmerung"

    @property
    def label(self) -> str:
        return {
            Tageszeit.NACHT: "Nacht",
            Tageszeit.MORGENDÄMMERUNG: "Morgendämmerung",
            Tageszeit.SONNENAUFGANG: "Sonnenaufgang",
            Tageszeit.VORMITTAG: "Vormittag",
            Tageszeit.MITTAG: "Mittag",
            Tageszeit.NACHMITTAG: "Nachmittag",
            Tageszeit.SONNENUNTERGANG: "Sonnenuntergang",
            Tageszeit.ABENDDÄMMERUNG: "Abenddämmerung",
        }[self]


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
    tageszeit: Tageszeit
    # Approximate solar times for this day (local hour float)
    sunrise_hour: float
    sunset_hour: float
    solar_noon_hour: float


@dataclass(frozen=True, slots=True)
class SceneGrade:
    brightness: float
    saturation: float
    tint_r: float
    tint_g: float
    tint_b: float


# ── time helpers ─────────────────────────────────────────────────────


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


def _day_of_year(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def _solar_declination(doy: int) -> float:
    """Solar declination in degrees (approx)."""
    return 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))


def _equation_of_time(doy: int) -> float:
    """Equation of time in minutes (approx)."""
    B = math.radians(360.0 / 365.0 * (doy - 81))
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def _hour_angle(dt: datetime, lon_deg: float = DEFAULT_LON_DEG) -> float:
    """Local hour angle in degrees (0 at solar noon)."""
    doy = _day_of_year(dt)
    eot = _equation_of_time(doy)  # minutes
    # Local solar time
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    # Longitude correction from timezone meridian is approximate;
    # for Europe/Berlin CET/CEST the standard meridian is 15°E
    # Use lon relative to 15°
    lon_corr = (lon_deg - 15.0) / 15.0  # hours
    solar_time = hour + eot / 60.0 + lon_corr
    return 15.0 * (solar_time - 12.0)


def _solar_altitude_azimuth(
    dt: datetime,
    lat_deg: float = DEFAULT_LAT_DEG,
    lon_deg: float = DEFAULT_LON_DEG,
) -> tuple[float, float]:
    doy = _day_of_year(dt)
    decl = _solar_declination(doy)
    ha = _hour_angle(dt, lon_deg)
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


def sunrise_sunset_hours(
    dt: datetime,
    lat_deg: float = DEFAULT_LAT_DEG,
    lon_deg: float = DEFAULT_LON_DEG,
) -> tuple[float, float, float]:
    """
    Approximate sunrise, solar noon, sunset as local clock hours (float).

    Uses solar altitude = 0° (geometric horizon, no refraction).
    """
    doy = _day_of_year(dt)
    decl = math.radians(_solar_declination(doy))
    lat = math.radians(lat_deg)
    cos_ha = -math.tan(lat) * math.tan(decl)
    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))  # degrees from noon to rise/set

    eot = _equation_of_time(doy) / 60.0  # hours
    lon_corr = (lon_deg - 15.0) / 15.0
    noon = 12.0 - eot - lon_corr
    sunrise = noon - ha / 15.0
    sunset = noon + ha / 15.0
    return sunrise, noon, sunset


def tageszeit_from_altitude(
    sun_alt_deg: float,
    hour: float,
    sunrise: float,
    sunset: float,
    noon: float,
) -> Tageszeit:
    """Map solar altitude + clock context → named Tageszeit."""
    if sun_alt_deg < -12:
        return Tageszeit.NACHT
    if sun_alt_deg < -4:
        # Dawn vs dusk by clock relative to noon
        return Tageszeit.MORGENDÄMMERUNG if hour < noon else Tageszeit.ABENDDÄMMERUNG
    if sun_alt_deg < 2:
        return Tageszeit.SONNENAUFGANG if hour < noon else Tageszeit.SONNENUNTERGANG
    if sun_alt_deg >= 40 or abs(hour - noon) < 1.5:
        return Tageszeit.MITTAG
    if hour < noon:
        return Tageszeit.VORMITTAG
    return Tageszeit.NACHMITTAG


# ── moon ─────────────────────────────────────────────────────────────


def moon_phase_at(dt: datetime) -> tuple[float, float]:
    """Real lunar phase (0=new…0.5=full…1=new) and illumination 0..1."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (dt.astimezone(timezone.utc) - REF_NEW_MOON).total_seconds() / 86400.0
    phase = (days % SYNODIC_MONTH) / SYNODIC_MONTH
    illumination = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return phase, illumination


def _moon_altitude_azimuth(
    dt: datetime, phase: float, lat_deg: float = DEFAULT_LAT_DEG
) -> tuple[float, float]:
    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg)
    moon_az = (sun_az + phase * 360.0) % 360.0
    hour = dt.hour + dt.minute / 60.0
    moon_hour = (hour - phase * 12.0) % 24.0
    # Reuse solar geometry with shifted hour angle
    fake = dt.replace(
        hour=int(moon_hour) % 24,
        minute=int((moon_hour % 1) * 60),
        second=0,
        microsecond=0,
    )
    alt, _ = _solar_altitude_azimuth(fake, lat_deg)
    return alt, moon_az


# ── public API ───────────────────────────────────────────────────────


def celestial_at(
    dt: datetime | None = None,
    *,
    tz: str = "Europe/Berlin",
    lat_deg: float = DEFAULT_LAT_DEG,
    lon_deg: float = DEFAULT_LON_DEG,
) -> CelestialState:
    """Full celestial + Tageszeit state for a local datetime."""
    if dt is None:
        dt = datetime.now(ZoneInfo(tz))
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    else:
        dt = dt.astimezone(ZoneInfo(tz))

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    sun_alt, sun_az = _solar_altitude_azimuth(dt, lat_deg, lon_deg)
    sunrise, noon, sunset = sunrise_sunset_hours(dt, lat_deg, lon_deg)
    phase, illum = moon_phase_at(dt)
    moon_alt, moon_az = _moon_altitude_azimuth(dt, phase, lat_deg)
    tz_enum = tageszeit_from_altitude(sun_alt, hour, sunrise, sunset, noon)

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
        tageszeit=tz_enum,
        sunrise_hour=sunrise,
        sunset_hour=sunset,
        solar_noon_hour=noon,
    )


def sky_colors(state: CelestialState) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Sky gradient (top, bottom) from Tageszeit / sun altitude."""
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
    """Full-scene brightness/saturation/tint from solar altitude."""
    alt = state.sun_alt_deg
    if alt > 20:
        return SceneGrade(1.0, 1.0, 0.0, 0.0, 0.0)
    if alt > 5:
        t = (alt - 5) / 15.0
        return SceneGrade(
            brightness=0.92 + 0.08 * t,
            saturation=1.05 - 0.05 * t,
            tint_r=0.06 * (1 - t),
            tint_g=0.02 * (1 - t),
            tint_b=-0.03 * (1 - t),
        )
    if alt > 0:
        t = alt / 5.0
        return SceneGrade(
            brightness=0.78 + 0.14 * t,
            saturation=1.1,
            tint_r=0.12 * (1 - 0.5 * t),
            tint_g=0.04 * (1 - 0.5 * t),
            tint_b=-0.06 * (1 - 0.5 * t),
        )
    if alt > -6:
        t = (alt + 6) / 6.0
        return SceneGrade(
            brightness=0.45 + 0.33 * t,
            saturation=0.55 + 0.45 * t,
            tint_r=0.04 * t,
            tint_g=0.0,
            tint_b=0.10 * (1 - t) + 0.02 * t,
        )
    if alt > -12:
        t = (alt + 12) / 6.0
        return SceneGrade(
            brightness=0.28 + 0.17 * t,
            saturation=0.35 + 0.20 * t,
            tint_r=-0.02,
            tint_g=0.0,
            tint_b=0.12,
        )
    return SceneGrade(0.22, 0.30, -0.03, 0.0, 0.14)


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


def format_hour(h: float) -> str:
    """Format float hour as HH:MM."""
    h = h % 24.0
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}:{mm:02d}"
