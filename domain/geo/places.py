"""Official German place names via BKG WFS GN-DE (Geographische Namen).

Primary source for Ortsschilder (Zeichen 310/311). Settlements are queried
by bounding box + minimum population (ewz), names resolved from Endonym
features. Nominatim is only used as a last-resort fallback.

Service: https://sgx.geodatenzentrum.de/wfs_gnde
Licence: Datenlizenz Deutschland – Namensnennung 2.0
"""

from __future__ import annotations

import math
import re
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

WFS_URL = "https://sgx.geodatenzentrum.de/wfs_gnde"
USER_AGENT = "vector-toon-pipeline/1.0 (educational; BKG GN-DE client)"

DEFAULT_MIN_EWZ = 1000
DEFAULT_MAX_KM = 4.0
POINT_BBOX_DEG = 0.06


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    lat: float
    lon: float
    ewz: int = 0
    nnid: str = ""

    def distance_km(self, lat: float, lon: float) -> float:
        return _haversine_km(self.lat, self.lon, lat, lon)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _http_get(url: str, timeout: float = 8.0) -> bytes | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/xml, text/xml, */*"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _http_post_xml(url: str, body: bytes, timeout: float = 12.0) -> bytes | None:
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/xml",
            "Accept": "application/gml+xml, application/xml, */*",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


@lru_cache(maxsize=512)
def _endonym_name(end_id: str, timeout: float = 6.0) -> str | None:
    if not end_id:
        return None
    if end_id.startswith("http"):
        m = re.search(r"[?&#]ID=((?:End_|Lan_|Obj_|Sta_|Spr_)[A-Za-z0-9_]+)", end_id)
        if not m:
            m = re.search(r"#((?:End_|Lan_)[A-Za-z0-9_]+)", end_id)
        end_id = m.group(1) if m else end_id
    params = urllib.parse.urlencode(
        {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "STOREDQUERY_ID": "urn:ogc:def:query:OGC-WFS::GetFeatureById",
            "ID": end_id,
        }
    )
    data = _http_get(f"{WFS_URL}?{params}", timeout=timeout)
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    m = re.search(r"<gn:name>([^<]+)</gn:name>", text)
    if m:
        name = m.group(1).strip()
        return name or None
    return None


def _parse_gn_objects(xml: str) -> list[tuple[str, float, float, int, str]]:
    out: list[tuple[str, float, float, int, str]] = []
    parts = re.split(r"<gn:GnObjekt[\s>]", xml)
    for part in parts[1:]:
        nnid_m = re.search(r"<gn:nnid>([^<]+)", part)
        lat_m = re.search(r"<gn:geoBreite>([^<]+)", part)
        lon_m = re.search(r"<gn:geoLaenge>([^<]+)", part)
        ewz_m = re.search(r"<gn:ewz>([^<]+)", part)
        href_m = re.search(r'hatEndonym[^>]*xlink:href="([^"]+)"', part)
        if not (nnid_m and lat_m and lon_m):
            continue
        try:
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            ewz = int(ewz_m.group(1)) if ewz_m else 0
        except ValueError:
            continue
        href = href_m.group(1).replace("&", "&") if href_m else ""
        out.append((nnid_m.group(1), lat, lon, ewz, href))
    return out


def fetch_settlements_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    min_ewz: int = DEFAULT_MIN_EWZ,
    count: int = 40,
    timeout: float = 12.0,
) -> list[Place]:
    """Query GN-DE for populated places inside a WGS84 bbox."""
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<wfs:GetFeature service="WFS" version="2.0.0" count="{count}"
  xmlns:wfs="http://www.opengis.net/wfs/2.0"
  xmlns:fes="http://www.opengis.net/fes/2.0"
  xmlns:gml="http://www.opengis.net/gml/3.2"
  xmlns:gn="http://www.geodatenzentrum.de/gnde">
  <wfs:Query typeNames="gn:GnObjekt" srsName="EPSG:4326">
    <fes:Filter>
      <fes:And>
        <fes:PropertyIsGreaterThan>
          <fes:ValueReference>ewz</fes:ValueReference>
          <fes:Literal>{int(min_ewz)}</fes:Literal>
        </fes:PropertyIsGreaterThan>
        <fes:BBOX>
          <fes:ValueReference>box</fes:ValueReference>
          <gml:Envelope srsName="EPSG:4326">
            <gml:lowerCorner>{min_lon:.6f} {min_lat:.6f}</gml:lowerCorner>
            <gml:upperCorner>{max_lon:.6f} {max_lat:.6f}</gml:upperCorner>
          </gml:Envelope>
        </fes:BBOX>
      </fes:And>
    </fes:Filter>
  </wfs:Query>
</wfs:GetFeature>""".encode("utf-8")

    data = _http_post_xml(WFS_URL, body, timeout=timeout)
    if not data:
        return []
    text = data.decode("utf-8", errors="replace")
    places: list[Place] = []
    seen: set[str] = set()
    for nnid, lat, lon, ewz, href in _parse_gn_objects(text):
        if nnid in seen:
            continue
        # Only objects whose point lies inside the requested bbox
        # (state/region polygons often intersect without being local)
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        if ewz > 2_500_000:
            continue
        name = _endonym_name(href) if href else None
        if not name:
            continue
        seen.add(nnid)
        places.append(Place(name=name, lat=lat, lon=lon, ewz=ewz, nnid=nnid))
    return places


@dataclass
class PlaceIndex:
    """In-memory index of GN-DE settlements along a corridor."""

    places: list[Place] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _loaded_cells: set[tuple[int, int]] = field(default_factory=set, repr=False)
    cell_deg: float = 0.15
    min_ewz: int = DEFAULT_MIN_EWZ

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(math.floor(lat / self.cell_deg)), int(math.floor(lon / self.cell_deg)))

    def ensure_around(self, lat: float, lon: float, timeout: float = 12.0) -> None:
        cell = self._cell(lat, lon)
        needed = [cell]
        with self._lock:
            if cell in self._loaded_cells:
                needed = [
                    (cell[0] + di, cell[1] + dj)
                    for di in (-1, 0, 1)
                    for dj in (-1, 0, 1)
                ]
            missing = [c for c in needed if c not in self._loaded_cells]
        if not missing:
            return
        for ci, cj in missing:
            min_lat = ci * self.cell_deg
            min_lon = cj * self.cell_deg
            max_lat = min_lat + self.cell_deg
            max_lon = min_lon + self.cell_deg
            found = fetch_settlements_bbox(
                min_lon, min_lat, max_lon, max_lat,
                min_ewz=self.min_ewz, count=50, timeout=timeout,
            )
            with self._lock:
                if (ci, cj) in self._loaded_cells:
                    continue
                existing = {p.nnid for p in self.places if p.nnid}
                for p in found:
                    if p.nnid not in existing:
                        self.places.append(p)
                        existing.add(p.nnid)
                self._loaded_cells.add((ci, cj))

    def nearest(
        self, lat: float, lon: float, max_km: float = DEFAULT_MAX_KM
    ) -> Place | None:
        best: Place | None = None
        best_d = max_km
        with self._lock:
            snapshot = list(self.places)
        for p in snapshot:
            d = p.distance_km(lat, lon)
            if d < best_d:
                best_d = d
                best = p
        return best


_GLOBAL_INDEX = PlaceIndex()


def reverse_place_name(
    lat: float,
    lon: float,
    *,
    timeout: float = 10.0,
    max_km: float = DEFAULT_MAX_KM,
    min_ewz: int = DEFAULT_MIN_EWZ,
    allow_nominatim_fallback: bool = False,
) -> str | None:
    """Nearest official settlement name from BKG GN-DE, or None."""
    qlat = round(lat, 3)
    qlon = round(lon, 3)
    return _reverse_cached(qlat, qlon, timeout, max_km, min_ewz, allow_nominatim_fallback)


@lru_cache(maxsize=512)
def _reverse_cached(
    lat: float,
    lon: float,
    timeout: float,
    max_km: float,
    min_ewz: int,
    allow_nominatim_fallback: bool,
) -> str | None:
    try:
        if _GLOBAL_INDEX.min_ewz != min_ewz:
            _GLOBAL_INDEX.min_ewz = min_ewz
        _GLOBAL_INDEX.ensure_around(lat, lon, timeout=timeout)
        place = _GLOBAL_INDEX.nearest(lat, lon, max_km=max_km)
        if place is not None:
            return place.name
    except Exception:
        pass

    try:
        d = POINT_BBOX_DEG
        found = fetch_settlements_bbox(
            lon - d, lat - d, lon + d, lat + d,
            min_ewz=min_ewz, count=20, timeout=timeout,
        )
        if found:
            found.sort(key=lambda p: p.distance_km(lat, lon))
            if found[0].distance_km(lat, lon) <= max_km:
                return found[0].name
    except Exception:
        pass

    if allow_nominatim_fallback:
        return _nominatim_fallback(lat, lon, timeout=min(timeout, 4.0))
    return None


def _nominatim_fallback(lat: float, lon: float, timeout: float = 4.0) -> str | None:
    params = urllib.parse.urlencode(
        {
            "lat": f"{lat:.5f}",
            "lon": f"{lon:.5f}",
            "format": "json",
            "zoom": 12,
            "addressdetails": 1,
        }
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "de"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json

            data = json.loads(resp.read().decode())
    except Exception:
        return None
    addr = data.get("address") or {}
    for key in ("village", "town", "city", "municipality", "hamlet"):
        name = addr.get(key)
        if name:
            return str(name)
    return None


def preload_corridor(
    points: Iterable[tuple[float, float]],
    *,
    min_ewz: int = DEFAULT_MIN_EWZ,
    timeout: float = 12.0,
) -> int:
    """Warm the global index along (lat, lon) samples. Returns place count."""
    _GLOBAL_INDEX.min_ewz = min_ewz
    for lat, lon in points:
        try:
            _GLOBAL_INDEX.ensure_around(lat, lon, timeout=timeout)
        except Exception:
            continue
    return len(_GLOBAL_INDEX.places)
