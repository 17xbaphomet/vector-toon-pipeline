"""Geo context along a route: position, weather, landuse, heading."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .landuse import LanduseSample, fetch_osm_landuse
from .route import GeoRoute
from .weather import WeatherSnapshot, fetch_elevations, fetch_weather


@dataclass(frozen=True, slots=True)
class GeoSample:
    distance_m: float
    lon: float
    lat: float
    heading_deg: float       # clockwise from north — walking direction
    elevation_m: float
    weather: WeatherSnapshot | None
    landuse: LanduseSample | None

    @property
    def mood_name(self) -> str:
        if self.landuse is None:
            return "offenland"
        return self.landuse.to_mood_name()


@dataclass
class GeoContext:
    """Caches weather/landuse along a route; samples by distance."""

    route: GeoRoute
    weather_refresh_m: float = 2000.0
    landuse_refresh_m: float = 600.0
    _weather_cache: dict[int, WeatherSnapshot] = field(default_factory=dict)
    _landuse_cache: dict[int, LanduseSample] = field(default_factory=dict)
    _elev_loaded: bool = False

    def enrich_elevations(self, step_m: float = 500.0) -> None:
        if self._elev_loaded or not self.route.points:
            return
        coords: list[tuple[float, float]] = []
        indices: list[int] = []
        d = 0.0
        while d <= self.route.total_m:
            lon, lat, _ = self.route.sample(d)
            coords.append((lat, lon))
            indices.append(len(coords) - 1)
            d += step_m
        try:
            elevs = fetch_elevations(coords)
        except Exception:
            elevs = [0.0] * len(coords)
        # stamp nearest route points
        for pt in self.route.points:
            idx = min(int(pt.distance_m / step_m), len(elevs) - 1)
            object.__setattr__(pt, "elev_m", elevs[idx])  # RoutePoint is frozen — rebuild if needed
        # RoutePoint is frozen: rebuild points list
        from .route import RoutePoint

        new_pts = []
        for pt in self.route.points:
            idx = min(int(pt.distance_m / step_m), len(elevs) - 1)
            new_pts.append(
                RoutePoint(lon=pt.lon, lat=pt.lat, distance_m=pt.distance_m, elev_m=elevs[idx])
            )
        self.route.points = new_pts
        self._elev_loaded = True

    def sample(self, distance_m: float, *, fetch_live: bool = True) -> GeoSample:
        lon, lat, heading = self.route.sample(distance_m)
        elev = self.route.elevation_at(distance_m)

        weather = None
        landuse = None
        if fetch_live:
            w_key = int(distance_m // self.weather_refresh_m)
            if w_key not in self._weather_cache:
                try:
                    self._weather_cache[w_key] = fetch_weather(lat, lon)
                except Exception:
                    pass
            weather = self._weather_cache.get(w_key)

            l_key = int(distance_m // self.landuse_refresh_m)
            if l_key not in self._landuse_cache:
                try:
                    self._landuse_cache[l_key] = fetch_osm_landuse(lat, lon)
                except Exception:
                    pass
            landuse = self._landuse_cache.get(l_key)

        return GeoSample(
            distance_m=distance_m,
            lon=lon,
            lat=lat,
            heading_deg=heading,
            elevation_m=elev,
            weather=weather,
            landuse=landuse,
        )
