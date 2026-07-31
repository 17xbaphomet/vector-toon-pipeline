"""Place names for German Ortsschilder (Zeichen 310/311).

Uses Nominatim reverse geocoding (OSM) with a small cache.
Falls back to None so callers can keep synthetic names.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from functools import lru_cache


@lru_cache(maxsize=256)
def reverse_place_name(lat: float, lon: float, timeout: float = 4.0) -> str | None:
    """Nearest city/town/village name, or None on failure."""
    # quantize to ~100 m so nearby samples share cache
    qlat = round(lat, 3)
    qlon = round(lon, 3)
    return _reverse_uncached(qlat, qlon, timeout)


def _reverse_uncached(lat: float, lon: float, timeout: float) -> str | None:
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
        headers={
            "User-Agent": "vector-toon-pipeline/1.0 (educational; contact: local)",
            "Accept-Language": "de",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return None
    addr = data.get("address") or {}
    for key in (
        "village",
        "town",
        "city",
        "municipality",
        "city_district",
        "suburb",
        "hamlet",
        "county",
    ):
        name = addr.get(key)
        if name:
            return str(name)
    # last resort: first token of display_name
    disp = data.get("display_name") or ""
    if disp:
        return disp.split(",")[0].strip()
    return None
