"""OSM water features along a walking route.

Detects rivers/lakes/canals near the path so the cartoon can show:
  - bridge when the route crosses water
  - water strip left of the road in the background
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WaterHit:
    """Water presence relative to the route at a given distance."""

    distance_m: float
    kind: str          # river | stream | canal | lake | water
    side: str          # cross | left | right | near
    width_hint_m: float = 30.0


@dataclass(frozen=True, slots=True)
class WaterClimate:
    """Soft water climate at a sample point (no geometries drawn 1:1)."""

    left: float = 0.0       # 0..1 water presence to the left of travel
    right: float = 0.0
    cross: float = 0.0      # 0..1 likelihood of crossing
    kind: str = ""          # dominant kind
    width_hint_m: float = 0.0


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _destination(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    R = 6371000.0
    δ = dist_m / R
    θ = math.radians(bearing_deg)
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)
    φ2 = math.asin(math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ))
    λ2 = λ1 + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ1),
        math.cos(δ) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540.0) % 360.0 - 180.0


def _classify_kind(tags: dict) -> str:
    ww = (tags.get("waterway") or "").lower()
    nat = (tags.get("natural") or "").lower()
    lu = (tags.get("landuse") or "").lower()
    water = (tags.get("water") or "").lower()
    if ww in {"river"}:
        return "river"
    if ww in {"canal"}:
        return "canal"
    if ww in {"stream", "drain", "ditch"}:
        return "stream"
    if nat == "water" or lu == "reservoir" or water in {"lake", "pond", "basin", "oxbow"}:
        return "lake" if water in {"lake", "oxbow", ""} or nat == "water" else "water"
    if ww:
        return "stream"
    return "water"


def _width_hint(tags: dict, kind: str) -> float:
    for key in ("width", "est_width"):
        raw = tags.get(key)
        if raw:
            try:
                return max(3.0, min(400.0, float(str(raw).replace(",", ".").split()[0])))
            except ValueError:
                pass
    return {
        "river": 40.0,
        "canal": 25.0,
        "stream": 8.0,
        "lake": 120.0,
        "water": 40.0,
    }.get(kind, 20.0)


def fetch_water_climate(
    lat: float,
    lon: float,
    heading_deg: float,
    radius_m: float = 280.0,
    timeout: float = 12.0,
) -> WaterClimate:
    """Query OSM water near point; classify left / right / cross relative to heading."""
    query = f"""
    [out:json][timeout:10];
    (
      way["waterway"~"river|stream|canal|drain"](around:{radius_m},{lat},{lon});
      way["natural"="water"](around:{radius_m},{lat},{lon});
      way["landuse"="reservoir"](around:{radius_m},{lat},{lon});
      relation["natural"="water"](around:{radius_m},{lat},{lon});
      relation["waterway"~"river|canal"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"User-Agent": "vector-toon-pipeline/1.0"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return WaterClimate()

    elements = payload.get("elements", [])
    if not elements:
        return WaterClimate()

    left = right = cross = 0.0
    best_kind = ""
    best_w = 0.0
    best_score = 0.0

    for el in elements:
        tags = el.get("tags") or {}
        kind = _classify_kind(tags)
        width = _width_hint(tags, kind)
        # center from overpass `out center`
        c = el.get("center") or {}
        clat = c.get("lat")
        clon = c.get("lon")
        if clat is None or clon is None:
            continue
        # bearing from route point → water center
        brg = _bearing_deg(lat, lon, clat, clon)
        # relative angle to travel heading (0 = ahead, +90 = right, -90 = left)
        rel = (brg - heading_deg + 180.0) % 360.0 - 180.0
        # rough distance via simple equirectangular
        dx = math.radians(clon - lon) * math.cos(math.radians(lat)) * 6371000.0
        dy = math.radians(clat - lat) * 6371000.0
        dist = math.hypot(dx, dy)

        # score by proximity and size
        prox = max(0.0, 1.0 - dist / max(radius_m, 1.0))
        size = min(1.0, width / 80.0)
        score = prox * (0.55 + 0.45 * size)

        # side classification
        if abs(rel) < 35.0 and dist < 90.0:
            # water roughly ahead / under path → crossing
            cross = max(cross, min(1.0, score * 1.3))
            side_score = cross
        elif -150.0 <= rel <= -20.0:
            left = max(left, min(1.0, score))
            side_score = left
        elif 20.0 <= rel <= 150.0:
            right = max(right, min(1.0, score * 0.7))
            side_score = right
        else:
            # behind or far ahead — mild near
            if dist < 120.0:
                cross = max(cross, score * 0.5)
            side_score = score * 0.4

        if side_score >= best_score:
            best_score = side_score
            best_kind = kind
            best_w = width

    return WaterClimate(
        left=left,
        right=right,
        cross=cross,
        kind=best_kind or "water",
        width_hint_m=best_w,
    )


def sample_water_along_route(
    route,
    step_m: float = 400.0,
    radius_m: float = 260.0,
    timeout: float = 10.0,
) -> list[WaterHit]:
    """Walk the route and emit WaterHits where water is significant."""
    hits: list[WaterHit] = []
    if not getattr(route, "points", None):
        return hits
    total = float(getattr(route, "total_m", 0.0) or 0.0)
    d = 0.0
    while d <= total:
        lon, lat, heading = route.sample(d)
        climate = fetch_water_climate(lat, lon, heading, radius_m=radius_m, timeout=timeout)
        if climate.cross >= 0.35:
            hits.append(WaterHit(d, climate.kind or "river", "cross", climate.width_hint_m or 30.0))
        if climate.left >= 0.30:
            hits.append(WaterHit(d, climate.kind or "water", "left", climate.width_hint_m or 40.0))
        if climate.right >= 0.45:
            hits.append(WaterHit(d, climate.kind or "water", "right", climate.width_hint_m or 30.0))
        d += step_m
    return hits
