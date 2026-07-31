"""Unified German administrative boundaries for Ortsschilder.

Nationwide backbone: BKG VG250 Gemeinden (compiled from Länder data).
Optional Land-specific WFS adapters can refine geometry where available.

Licence note (show in stream about / README when online):
  Geobasisdaten: © GeoBasis-DE / BKG (Jahr)
  VG250 — Datenlizenz Deutschland – Namensnennung – Version 2.0
  https://www.govdata.de/dl-de/by-2-0
  Bei Länder-Adaptern zusätzlich Quellenvermerk des jeweiligen Landes.
"""

from __future__ import annotations

import math
import re
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

VG250_WFS = "https://sgx.geodatenzentrum.de/wfs_vg250"
USER_AGENT = "vector-toon-pipeline/1.0 (educational; boundaries)"
DEFAULT_TIMEOUT = 15.0

ATTRIBUTION_SHORT = (
    "Geobasisdaten © GeoBasis-DE / BKG — VG250 (dl-de/by-2-0)"
)
ATTRIBUTION_LONG = (
    "Verwaltungsgrenzen: © GeoBasis-DE / BKG, Verwaltungsgebiete 1:250 000 (VG250). "
    "Lizenz: Datenlizenz Deutschland – Namensnennung – Version 2.0 "
    "(https://www.govdata.de/dl-de/by-2-0). "
    "Ortsnamen: © GeoBasis-DE / BKG, GN-DE (dl-de/by-2-0)."
)


@dataclass(frozen=True, slots=True)
class Gemeinde:
    """One municipality polygon (land surface preferred)."""

    name: str
    ags: str
    bez: str = ""
    rings: tuple[tuple[tuple[float, float], ...], ...] = ()
    min_lon: float = 0.0
    min_lat: float = 0.0
    max_lon: float = 0.0
    max_lat: float = 0.0
    source: str = "vg250"

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.rings or not self.rings[0]:
            return ((self.min_lon + self.max_lon) / 2, (self.min_lat + self.max_lat) / 2)
        xs = [p[0] for p in self.rings[0]]
        ys = [p[1] for p in self.rings[0]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def contains(self, lon: float, lat: float) -> bool:
        if not (self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat):
            return False
        if not self.rings:
            return False
        return _point_in_ring(lon, lat, self.rings[0])

    def distance_km(self, lon: float, lat: float) -> float:
        if self.contains(lon, lat):
            return 0.0
        cx, cy = self.centroid
        return _haversine_km(lat, lon, cy, cx)


@dataclass(frozen=True, slots=True)
class PlaceSegment:
    """Stretch of a route inside one Gemeinde — for Ortseingang/Ortsausgang."""

    name: str
    ags: str
    enter_m: float
    exit_m: float
    source: str = "vg250"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _point_in_ring(x: float, y: float, ring: Sequence[tuple[float, float]]) -> bool:
    n = len(ring)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
        ):
            inside = not inside
        j = i
    return inside


def _http_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/xml, text/xml, */*"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def _parse_poslist(text: str) -> tuple[tuple[float, float], ...]:
    nums = [float(x) for x in text.split()]
    if len(nums) < 6:
        return ()
    return tuple((nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2))


def _parse_vg250_gem(xml: str) -> list[Gemeinde]:
    out: list[Gemeinde] = []
    parts = re.split(r"<vg250:vg250_gem[\s>]", xml)
    for part in parts[1:]:
        gen = re.search(r"<vg250:gen>([^<]+)", part)
        ags = re.search(r"<vg250:ags>([^<]+)", part)
        bez = re.search(r"<vg250:bez>([^<]+)", part)
        if not gen or not ags:
            continue
        rings: list[tuple[tuple[float, float], ...]] = []
        for pl in re.findall(r"<gml:posList[^>]*>([^<]+)</gml:posList>", part):
            ring = _parse_poslist(pl)
            if len(ring) >= 3:
                rings.append(ring)
        if not rings:
            continue
        all_pts = [p for r in rings for p in r]
        lons = [p[0] for p in all_pts]
        lats = [p[1] for p in all_pts]
        out.append(
            Gemeinde(
                name=gen.group(1).strip(),
                ags=ags.group(1).strip(),
                bez=(bez.group(1).strip() if bez else ""),
                rings=tuple(rings),
                min_lon=min(lons),
                min_lat=min(lats),
                max_lon=max(lons),
                max_lat=max(lats),
                source="vg250",
            )
        )
    return out


def fetch_gemeinden_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    *,
    count: int = 30,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Gemeinde]:
    """Query BKG VG250 Gemeinden intersecting a WGS84 bbox."""
    params = urllib.parse.urlencode(
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": "vg250:vg250_gem",
            "srsName": "EPSG:4326",
            "count": str(count),
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat},EPSG:4326",
        }
    )
    data = _http_get(f"{VG250_WFS}?{params}", timeout=timeout)
    if not data:
        return []
    return _parse_vg250_gem(data.decode("utf-8", errors="replace"))


@dataclass
class BoundaryIndex:
    """Cell-cached nationwide Gemeinde index (VG250 backbone)."""

    gemeinden: list[Gemeinde] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _loaded: set[tuple[int, int]] = field(default_factory=set, repr=False)
    cell_deg: float = 0.2

    def _cell(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(math.floor(lat / self.cell_deg)), int(math.floor(lon / self.cell_deg)))

    def ensure_around(self, lat: float, lon: float, timeout: float = DEFAULT_TIMEOUT) -> None:
        cell = self._cell(lat, lon)
        with self._lock:
            if cell in self._loaded:
                return
        ci, cj = cell
        min_lat = ci * self.cell_deg
        min_lon = cj * self.cell_deg
        found = fetch_gemeinden_bbox(
            min_lon,
            min_lat,
            min_lon + self.cell_deg,
            min_lat + self.cell_deg,
            count=40,
            timeout=timeout,
        )
        with self._lock:
            if cell in self._loaded:
                return
            known = {g.ags for g in self.gemeinden}
            for g in found:
                if g.ags not in known:
                    self.gemeinden.append(g)
                    known.add(g.ags)
            self._loaded.add(cell)

    def at(self, lon: float, lat: float, timeout: float = DEFAULT_TIMEOUT) -> Gemeinde | None:
        self.ensure_around(lat, lon, timeout=timeout)
        with self._lock:
            snap = list(self.gemeinden)
        for g in snap:
            if g.contains(lon, lat):
                return g
        best: Gemeinde | None = None
        best_d = 15.0
        for g in snap:
            d = g.distance_km(lon, lat)
            if d < best_d:
                best_d = d
                best = g
        return best

    def name_at(self, lon: float, lat: float, timeout: float = DEFAULT_TIMEOUT) -> str | None:
        g = self.at(lon, lat, timeout=timeout)
        return g.name if g else None


_GLOBAL = BoundaryIndex()


def gemeinde_at(lon: float, lat: float, *, timeout: float = DEFAULT_TIMEOUT) -> Gemeinde | None:
    return _GLOBAL.at(lon, lat, timeout=timeout)


def gemeinde_name(lon: float, lat: float, *, timeout: float = DEFAULT_TIMEOUT) -> str | None:
    return _GLOBAL.name_at(lon, lat, timeout=timeout)


def segments_along_route(
    samples: Iterable[tuple[float, float, float]],
    *,
    timeout: float = DEFAULT_TIMEOUT,
    min_length_m: float = 800.0,
) -> list[PlaceSegment]:
    """Enter/exit segments from (lon, lat, distance_m) samples along a route."""
    raw: list[tuple[str, str, float, float]] = []
    cur_name: str | None = None
    cur_ags = ""
    enter = 0.0
    last_d = 0.0
    for lon, lat, dist_m in samples:
        last_d = dist_m
        g = gemeinde_at(lon, lat, timeout=timeout)
        name = g.name if g else None
        ags = g.ags if g else ""
        if name == cur_name:
            continue
        if cur_name is not None:
            raw.append((cur_name, cur_ags, enter, dist_m))
        cur_name = name
        cur_ags = ags
        enter = dist_m
    if cur_name is not None:
        raw.append((cur_name, cur_ags, enter, last_d))

    out: list[PlaceSegment] = []
    for name, ags, a, b in raw:
        if (b - a) < min_length_m:
            continue
        out.append(PlaceSegment(name=name, ags=ags, enter_m=a, exit_m=b, source="vg250"))
    return out


def preload_route_points(
    points: Iterable[tuple[float, float]],
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Warm cache along (lon, lat) samples."""
    for lon, lat in points:
        try:
            _GLOBAL.ensure_around(lat, lon, timeout=timeout)
        except Exception:
            continue
    return len(_GLOBAL.gemeinden)


LAND_ADAPTERS: dict[str, str] = {
    "DE-NW": "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg",
    "DE-HE": "https://basisdienste.geoportal.hessen.de/ogc/borders",
    "DE-BY": "https://geoservices.bayern.de/wfs/v1/ogc_atkis_basisdlm.cgi",
    "DE-BW": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_INSP_BW_Verwaltungseinheiten_ATKIS_BasisDLM",
    "DE-BB": "https://isk.geobasis-bb.de/ows/vg_wfs",
    "DE-BE": "https://isk.geobasis-bb.de/ows/vg_wfs",
    "DE-SH": "https://dienste.gdi-sh.de/WFS_SH_ATKIS_BDLM_VWG_OpenGBD",
    "DE-HH": "https://geodienste.hamburg.de/WFS_HH_ALKIS_vereinfacht",
}


@lru_cache(maxsize=1)
def attribution_text() -> str:
    return ATTRIBUTION_LONG
