"""Local weather via Open-Meteo (no API key)."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeatherSnapshot:
    temperature_c: float
    weather_code: int
    cloud_cover: float       # 0–100
    precipitation: float     # mm
    wind_speed_kmh: float
    wind_dir_deg: float
    humidity: float          # 0–100
    is_day: bool

    @property
    def is_rainy(self) -> bool:
        return self.precipitation > 0.2 or self.weather_code in {
            51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99
        }

    @property
    def is_foggy(self) -> bool:
        return self.weather_code in {45, 48}

    @property
    def is_snowy(self) -> bool:
        return self.weather_code in {71, 73, 75, 77, 85, 86}

    @property
    def sky_cloud_factor(self) -> float:
        return max(0.0, min(1.0, self.cloud_cover / 100.0))


def fetch_weather(lat: float, lon: float, timeout: float = 10.0) -> WeatherSnapshot:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        "&current=temperature_2m,relative_humidity_2m,is_day,precipitation,"
        "weather_code,cloud_cover,wind_speed_10m,wind_direction_10m"
        "&timezone=Europe%2FBerlin"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "vector-toon-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    c = data["current"]
    return WeatherSnapshot(
        temperature_c=float(c.get("temperature_2m", 15.0)),
        weather_code=int(c.get("weather_code", 0)),
        cloud_cover=float(c.get("cloud_cover", 30.0)),
        precipitation=float(c.get("precipitation", 0.0)),
        wind_speed_kmh=float(c.get("wind_speed_10m", 5.0)),
        wind_dir_deg=float(c.get("wind_direction_10m", 0.0)),
        humidity=float(c.get("relative_humidity_2m", 60.0)),
        is_day=bool(c.get("is_day", 1)),
    )


def fetch_elevations(
    coords: list[tuple[float, float]], timeout: float = 15.0
) -> list[float]:
    """coords as list of (lat, lon). Returns elevation metres."""
    if not coords:
        return []
    lats = ",".join(f"{lat:.5f}" for lat, _ in coords)
    lons = ",".join(f"{lon:.5f}" for _, lon in coords)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    req = urllib.request.Request(url, headers={"User-Agent": "vector-toon-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return [float(e) for e in data.get("elevation", [0.0] * len(coords))]
