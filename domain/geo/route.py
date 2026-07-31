"""Walking routes through Germany via OSRM (OpenStreetMap routing)."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RoutePoint:
    lon: float
    lat: float
    distance_m: float  # cumulative along route from start
    elev_m: float = 0.0


@dataclass
class GeoRoute:
    """Polyline route with cumulative distances and interpolated heading."""

    points: list[RoutePoint]
    total_m: float
    start_name: str = ""
    end_name: str = ""
    source: str = "osrm"

    def sample(self, distance_m: float) -> tuple[float, float, float]:
        """Return (lon, lat, heading_deg_cw_from_north) at distance along route."""
        if not self.points:
            return 10.0, 51.0, 90.0
        d = max(0.0, min(distance_m, self.total_m))
        # find segment
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            if b.distance_m >= d:
                seg = max(1e-6, b.distance_m - a.distance_m)
                t = (d - a.distance_m) / seg
                lon = a.lon + (b.lon - a.lon) * t
                lat = a.lat + (b.lat - a.lat) * t
                heading = _bearing_deg(a.lat, a.lon, b.lat, b.lon)
                return lon, lat, heading
        last = self.points[-1]
        prev = self.points[-2] if len(self.points) > 1 else last
        return last.lon, last.lat, _bearing_deg(prev.lat, prev.lon, last.lat, last.lon)

    def elevation_at(self, distance_m: float) -> float:
        if not self.points:
            return 0.0
        d = max(0.0, min(distance_m, self.total_m))
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            if b.distance_m >= d:
                seg = max(1e-6, b.distance_m - a.distance_m)
                t = (d - a.distance_m) / seg
                return a.elev_m + (b.elev_m - a.elev_m) * t
        return self.points[-1].elev_m


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 → 2, degrees clockwise from north."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def fetch_walking_route(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    via: list[tuple[float, float]] | None = None,
    profile: str = "foot",
    timeout: float = 20.0,
) -> GeoRoute:
    """
    Fetch a walking route via public OSRM.

    Coordinates are (lon, lat) WGS84 — note lon first (OSRM convention).
    start/end also accepted as (lat, lon) if you pass latlon= style helpers.
    """
    coords = [start] + list(via or []) + [end]
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in coords)
    url = (
        f"https://router.project-osrm.org/route/v1/{profile}/{coord_str}"
        f"?overview=full&geometries=geojson&steps=false"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "vector-toon-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())

    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM routing failed: {data.get('code', data)}")

    geometry = data["routes"][0]["geometry"]["coordinates"]  # [[lon,lat], ...]
    points: list[RoutePoint] = []
    cum = 0.0
    prev = None
    for lon, lat in geometry:
        if prev is not None:
            cum += _haversine_m(prev[1], prev[0], lat, lon)
        points.append(RoutePoint(lon=lon, lat=lat, distance_m=cum))
        prev = (lon, lat)

    return GeoRoute(points=points, total_m=cum, source="osrm")


def demo_route_frankfurt_heidelberg() -> GeoRoute:
    """Fallback demo: approx foot path Frankfurt → Heidelberg corridor."""
    # (lon, lat) waypoints along a plausible SW Germany path
    waypoints = [
        (8.6821, 50.1109),   # Frankfurt
        (8.6500, 49.9500),
        (8.5800, 49.8000),
        (8.5500, 49.6500),
        (8.6700, 49.5000),
        (8.6900, 49.4100),   # Heidelberg area
    ]
    try:
        return fetch_walking_route(waypoints[0], waypoints[-1], via=waypoints[1:-1])
    except Exception:
        # offline fallback polyline
        points: list[RoutePoint] = []
        cum = 0.0
        prev = None
        for lon, lat in waypoints:
            if prev is not None:
                cum += _haversine_m(prev[1], prev[0], lat, lon)
            points.append(RoutePoint(lon=lon, lat=lat, distance_m=cum))
            prev = (lon, lat)
        return GeoRoute(points=points, total_m=cum, source="fallback", start_name="Frankfurt", end_name="Heidelberg")
