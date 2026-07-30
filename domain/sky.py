"""Tageszeitenberechnung: Sonne, Mondphase, Himmelsfarben, Szenen-Grade."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from zoneinfo import ZoneInfo

SYNODIC_MONTH = 29.530588853
REF_NEW_MOON = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
DEFAULT_LAT_DEG = 51.0
DEFAULT_LON_DEG = 10.0


class Tageszeit(str, Enum):
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
    return 23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))


def _equation_of_time(doy: int) -> float:
    B = math.radians(360.0 / 365.0 * (doy - 81))
    return 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)


def _hour_angle(dt: datetime, lon_deg: float = DEFAULT_LON_DEG) -> float:
    doy = _day_of_year(dt)
    eot = _equation_of_time(doy)
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    lon_corr = (lon_deg - 15.0) / 15.0
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
    doy = _day_of_year(dt)
    decl = math.radians(_solar_declination(doy))
    lat = math.radians(lat_deg)
    cos_ha = max(-1.0, min(1.0, -math.tan(lat) * math.tan(decl)))
    ha = math.degrees(math.acos(cos_ha))
    eot = _equation_of_time(doy) / 60.0
    lon_corr = (lon_deg - 15.0) / 15.0
    noon = 12.0 - eot - lon_corr
    return noon - ha / 15.0, noon, noon + ha / 15.0


def tageszeit_from_altitude(
    sun_alt_deg: float,
    hour: float,
    sunrise: float,
    sunset: float,
    noon: float,
) -> Tageszeit:
    # Night only after astronomical twilight (~-18°); civil stays "dämmerung"
    if sun_alt_deg < -18:
        return Tageszeit.NACHT
    if sun_alt_deg < -4:
        return Tageszeit.MORGENDÄMMERUNG if hour < noon else Tageszeit.ABENDDÄMMERUNG
    if sun_alt_deg < 2:
        return Tageszeit.SONNENAUFGANG if hour < noon else Tageszeit.SONNENUNTERGANG
    if sun_alt_deg >= 40 or abs(hour - noon) < 1.5:
        return Tageszeit.MITTAG
    if hour < noon:
        return Tageszeit.VORMITTAG
    return Tageszeit.NACHMITTAG


def moon_phase_at(dt: datetime) -> tuple[float, float]:
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
    fake = dt.replace(
        hour=int(moon_hour) % 24,
        minute=int((moon_hour % 1) * 60),
        second=0,
        microsecond=0,
    )
    alt, _ = _solar_altitude_azimuth(fake, lat_deg)
    return alt, moon_az


def celestial_at(
    dt: datetime | None = None,
    *,
    tz: str = "Europe/Berlin",
    lat_deg: float = DEFAULT_LAT_DEG,
    lon_deg: float = DEFAULT_LON_DEG,
) -> CelestialState:
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
        is_day=sun_alt > -6.0,  # civil twilight still counts as "day-ish"
        local_time=dt,
        tageszeit=tz_enum,
        sunrise_hour=sunrise,
        sunset_hour=sunset,
        solar_noon_hour=noon,
    )


def sky_colors(state: CelestialState) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Sky gradient — extended bright twilight.
    Full night sky only below ~-15° (well after civil dusk / before civil dawn).
    """
    alt = state.sun_alt_deg
    if alt > 10:
        return (90, 170, 230), (180, 220, 245)
    if alt > 0:
        t = alt / 10.0
        return _lerp((255, 150, 90), (90, 170, 230), t), _lerp(
            (255, 200, 130), (180, 220, 245), t
        )
    if alt > -6:
        # Civil twilight — still warm / bright, not night
        t = (alt + 6) / 6.0
        return _lerp((120, 90, 140), (255, 150, 90), t), _lerp(
            (160, 120, 100), (255, 200, 130), t
        )
    if alt > -12:
        # Nautical twilight
        t = (alt + 12) / 6.0
        return _lerp((40, 45, 90), (120, 90, 140), t), _lerp(
            (60, 55, 90), (160, 120, 100), t
        )
    if alt > -18:
        # Astronomical twilight → night
        t = (alt + 18) / 6.0
        return _lerp((12, 16, 42), (40, 45, 90), t), _lerp(
            (28, 32, 58), (60, 55, 90), t
        )
    return (10, 14, 40), (25, 30, 55)


def scene_grade(state: CelestialState) -> SceneGrade:
    """
    Full-scene grade with longer, brighter twilight.

    Evening stays readable past sunset; morning brightens early in civil dawn.
    Deep night floor is milder so it never goes pitch-black too soon.
    """
    alt = state.sun_alt_deg
    if alt > 12:
        return SceneGrade(1.0, 1.0, 0.0, 0.0, 0.0)
    if alt > 0:
        # Day → soft golden, still nearly full brightness
        t = alt / 12.0
        return SceneGrade(
            brightness=0.90 + 0.10 * t,
            saturation=1.05 - 0.05 * t,
            tint_r=0.05 * (1 - t),
            tint_g=0.02 * (1 - t),
            tint_b=-0.02 * (1 - t),
        )
    if alt > -6:
        # Civil twilight — clearly lit scene
        t = (alt + 6) / 6.0  # 0 at -6°, 1 at 0°
        return SceneGrade(
            brightness=0.72 + 0.18 * t,
            saturation=0.85 + 0.15 * t,
            tint_r=0.08 * (1 - 0.3 * t),
            tint_g=0.02 * (1 - t),
            tint_b=0.04 * (1 - t),
        )
    if alt > -12:
        # Nautical twilight
        t = (alt + 12) / 6.0
        return SceneGrade(
            brightness=0.50 + 0.22 * t,
            saturation=0.55 + 0.30 * t,
            tint_r=0.02,
            tint_g=0.0,
            tint_b=0.08 * (1 - 0.5 * t),
        )
    if alt > -18:
        # Astronomical twilight
        t = (alt + 18) / 6.0
        return SceneGrade(
            brightness=0.38 + 0.12 * t,
            saturation=0.40 + 0.15 * t,
            tint_r=-0.01,
            tint_g=0.0,
            tint_b=0.10,
        )
    # Deep night — still not pitch black
    return SceneGrade(0.36, 0.38, -0.02, 0.0, 0.12)


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
    h = h % 24.0
    hh = int(h)
    mm = int((h - hh) * 60)
    return f"{hh:02d}:{mm:02d}"
