"""Geo context along a route: position, weather, landuse, heading."""

from __future__ import annotations

from dataclasses import dataclass, field

from .landuse import LanduseSample, fetch_osm_landuse
from .water import WaterClimate, fetch_water_climate
from .route import GeoRoute
from .weather import WeatherSnapshot, fetch_elevations, fetch_weather


@dataclass(frozen=True, slots=True)
class GeoSample:
    distance_m: float
    lon: float
    lat: float
    heading_deg: float
    elevation_m: float
    weather: WeatherSnapshot | None
    landuse: LanduseSample | None
    water: WaterClimate | None = None

    @property
    def mood_name(self) -> str:
        if self.landuse is None:
            return "offenland"
        return self.landuse.to_mood_name()


@dataclass
class GeoContext:
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
        d = 0.0
        while d <= self.route.total_m:
            lon, lat, _ = self.route.sample(d)
            coords.append((lat, lon))
            d += step_m
        try:
            elevs = fetch_elevations(coords)
        except Exception:
            elevs = [0.0] * len(coords)
        from .route import RoutePoint
        new_pts = []
        for pt in self.route.points:
            idx = min(int(pt.distance_m / step_m), len(elevs) - 1)
            new_pts.append(RoutePoint(lon=pt.lon, lat=pt.lat, distance_m=pt.distance_m, elev_m=elevs[idx]))
        self.route.points = new_pts
        self._elev_loaded = True

    def sample(self, distance_m: float, *, fetch_live: bool = True) -> GeoSample:
        lon, lat, heading = self.route.sample(distance_m)
        elev = self.route.elevation_at(distance_m)
        weather = None
        landuse = None
        water = None
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
            if not hasattr(self, "_water_cache"):
                self._water_cache = {}
            if l_key not in self._water_cache:
                try:
                    self._water_cache[l_key] = fetch_water_climate(lat, lon, heading, radius_m=280.0)
                except Exception:
                    self._water_cache[l_key] = WaterClimate()
            water = self._water_cache.get(l_key)
        return GeoSample(
            distance_m=distance_m, lon=lon, lat=lat, heading_deg=heading,
            elevation_m=elev, weather=weather, landuse=landuse, water=water,
        )
