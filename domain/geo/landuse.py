"""Map real-world landuse / density signals → RegionMood + feature biases.

Primary online source: OpenStreetMap Overpass (practical for live routes).
BKG LBM-DE / VG25 can be plugged in offline later as a LanduseProvider.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum


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
    building_density: float   # 0–1
    forest_density: float     # 0–1
    farm_density: float       # 0–1
    industrial_density: float # 0–1
    population_hint: float    # 0–1 relative

    def to_mood_name(self) -> str:
        if self.cover == LandCover.URBAN or self.building_density > 0.55:
            return "stadt"
        if self.cover == LandCover.VILLAGE or (
            self.building_density > 0.2 and self.farm_density > 0.15
        ):
            return "dorf"
        if self.cover == LandCover.FOREST or self.forest_density > 0.45:
            return "wald"
        if self.cover == LandCover.INDUSTRIAL or self.industrial_density > 0.35:
            return "stadt"
        return "offenland"


def fetch_osm_landuse(lat: float, lon: float, radius_m: float = 400.0, timeout: float = 12.0) -> LanduseSample:
    """Query Overpass around a point for landuse/building signals."""
    query = f"""
    [out:json][timeout:10];
    (
      way["landuse"](around:{radius_m},{lat},{lon});
      way["natural"](around:{radius_m},{lat},{lon});
      way["building"](around:{radius_m},{lat},{lon});
      relation["landuse"](around:{radius_m},{lat},{lon});
    );
    out tags;
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
        return LanduseSample(
            cover=LandCover.OPEN,
            building_density=0.05,
            forest_density=0.15,
            farm_density=0.3,
            industrial_density=0.0,
            population_hint=0.1,
        )

    elements = payload.get("elements", [])
    n = max(1, len(elements))
    buildings = forests = farms = industrial = water = 0
    for el in elements:
        tags = el.get("tags", {})
        if "building" in tags:
            buildings += 1
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

    return LanduseSample(
        cover=cover,
        building_density=bd,
        forest_density=fd,
        farm_density=farm,
        industrial_density=ind,
        population_hint=min(1.0, bd * 0.8 + ind * 0.3),
    )
