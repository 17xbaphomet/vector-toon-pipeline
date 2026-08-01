"""Coarse landscape character from real-world density + building height signals.

Primary online source: OpenStreetMap Overpass
  - landuse / natural / building counts → mood climate
  - building height | building:levels → mean/max height for scale
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum

_METERS_PER_LEVEL = 3.0
_DEFAULT_HEIGHT = {
    "urban": 14.0, "suburban": 9.0, "village": 7.0, "industrial": 12.0,
    "farmland": 6.0, "forest": 5.0, "water": 5.0, "open": 5.0,
}


class LandCover(str, Enum):
    URBAN = "urban"
    SUBURBAN = "suburban"
    VILLAGE = "village"
    FARMLAND = "farmland"
    FOREST = "forest"
    WATER = "water"
    INDUSTRIAL = "industrial"
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class LanduseSample:
    cover: LandCover
    building_density: float
    forest_density: float
    farm_density: float
    industrial_density: float
    population_hint: float
    mean_building_height_m: float = 0.0
    max_building_height_m: float = 0.0
    tall_share: float = 0.0
    height_sample_count: int = 0

    def to_mood_name(self) -> str:
        bd, fd, farm, ind = (
            self.building_density, self.forest_density,
            self.farm_density, self.industrial_density,
        )
        if self.cover == LandCover.URBAN or bd > 0.55 or (bd > 0.4 and ind > 0.25):
            return "stadt"
        if self.cover == LandCover.INDUSTRIAL or (ind > 0.4 and bd > 0.2):
            return "stadt"
        if self.cover == LandCover.VILLAGE or (0.12 <= bd <= 0.55 and farm > 0.12):
            return "dorf"
        if self.cover == LandCover.SUBURBAN and bd > 0.2:
            return "dorf" if farm > 0.1 else "stadt"
        if self.cover == LandCover.FOREST or (fd > 0.45 and bd < 0.2):
            return "wald"
        return "offenland"

    def climate_tuple(self):
        return (float(self.building_density), float(self.forest_density),
                float(self.farm_density), float(self.industrial_density))

    def effective_mean_height_m(self) -> float:
        if self.mean_building_height_m > 0.5 and self.height_sample_count >= 2:
            return self.mean_building_height_m
        return _DEFAULT_HEIGHT.get(self.cover.value, 7.0)

    def building_scale_mul(self, base_m: float = 8.0) -> float:
        h = self.effective_mean_height_m()
        return max(0.7, min(2.4, (h / max(1.0, base_m)) ** 0.65))

    def skyline_scale_mul(self) -> float:
        mx = self.max_building_height_m or self.effective_mean_height_m()
        return max(0.85, min(1.85, (max(12.0, mx) / 15.0) ** 0.4))


_HEIGHT_RE = re.compile(r"^\\s*([0-9]+(?:[.,][0-9]+)?)\\s*(m|meter|metres|meters)?\\s*$", re.IGNORECASE)

_BUILDING_TYPE_HEIGHT = {
    "yes": 8.0, "house": 7.0, "detached": 7.5, "semidetached_house": 7.0,
    "terrace": 8.0, "residential": 10.0, "apartments": 16.0, "bungalow": 4.5,
    "cabin": 4.0, "garage": 3.0, "shed": 3.0, "farm": 7.0, "barn": 8.0,
    "commercial": 12.0, "retail": 9.0, "office": 18.0, "industrial": 12.0,
    "warehouse": 10.0, "school": 10.0, "hospital": 14.0, "church": 14.0,
    "hotel": 18.0, "university": 14.0, "factory": 12.0, "construction": 8.0,
}


def _parse_height_m(tags: dict):
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        m = _HEIGHT_RE.match(str(raw).replace(",", "."))
        if m:
            try:
                val = float(m.group(1))
                if 1.5 <= val <= 400.0:
                    return val
            except ValueError:
                pass
        try:
            val = float(str(raw).replace(",", ".").split()[0])
            if 1.5 <= val <= 400.0:
                return val
        except (ValueError, IndexError):
            pass
    levels = tags.get("building:levels") or tags.get("levels")
    if levels is not None:
        try:
            n = float(str(levels).split(";")[0].replace(",", "."))
            if 0.5 <= n <= 120.0:
                return n * _METERS_PER_LEVEL
        except ValueError:
            pass
    btype = str(tags.get("building", "")).lower().strip()
    return _BUILDING_TYPE_HEIGHT.get(btype)


def fetch_osm_landuse(lat: float, lon: float, radius_m: float = 400.0, timeout: float = 12.0) -> LanduseSample:
    query = f"""
    [out:json][timeout:10];
    (
      way["landuse"](around:{radius_m},{lat},{lon});
      way["natural"](around:{radius_m},{lat},{lon});
      way["building"](around:{radius_m},{lat},{lon});
      relation["landuse"](around:{radius_m},{lat},{lon});
      node["building"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "vector-toon-pipeline/1.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return LanduseSample(cover=LandCover.OPEN, building_density=0.05, forest_density=0.15,
                             farm_density=0.3, industrial_density=0.0, population_hint=0.1)
    elements = payload.get("elements", [])
    buildings = forests = farms = industrial = water = 0
    heights = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if "building" in tags:
            buildings += 1
            h = _parse_height_m(tags)
            if h is not None:
                heights.append(h)
        lu = tags.get("landuse", "")
        nat = tags.get("natural", "")
        if lu in {"forest", "wood"} or nat in {"wood", "forest"}:
            forests += 1
        if lu in {"farmland", "meadow", "orchard", "vineyard", "grass"}:
            farms += 1
        if lu in {"industrial", "commercial", "retail"}:
            industrial += 1
        if lu == "residential":
            buildings += 1
        if nat == "water" or lu == "reservoir":
            water += 1
    bd = min(1.0, buildings / 25.0)
    fd = min(1.0, forests / 12.0)
    farm = min(1.0, farms / 12.0)
    ind = min(1.0, industrial / 8.0)
    if bd > 0.55:
        cover = LandCover.URBAN
    elif bd > 0.25:
        cover = LandCover.SUBURBAN if ind < 0.2 else LandCover.INDUSTRIAL
    elif bd > 0.12 and farm > 0.1:
        cover = LandCover.VILLAGE
    elif fd > 0.4:
        cover = LandCover.FOREST
    elif farm > 0.35:
        cover = LandCover.FARMLAND
    elif water > forests and water > 2:
        cover = LandCover.WATER
    else:
        cover = LandCover.OPEN
    if heights:
        mean_h = sum(heights) / len(heights)
        max_h = max(heights)
        tall = sum(1 for h in heights if h >= 15.0) / len(heights)
        n_h = len(heights)
    else:
        mean_h = max_h = tall = 0.0
        n_h = 0
    return LanduseSample(
        cover=cover, building_density=bd, forest_density=fd, farm_density=farm,
        industrial_density=ind, population_hint=min(1.0, bd * 0.8 + ind * 0.3),
        mean_building_height_m=mean_h, max_building_height_m=max_h,
        tall_share=tall, height_sample_count=n_h,
    )
