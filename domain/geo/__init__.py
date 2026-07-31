"""Geo-driven Germany walk: routes, weather, elevation, landuse context."""

from .context import GeoContext, GeoSample
from .route import GeoRoute, fetch_walking_route
from .weather import WeatherSnapshot, fetch_weather

__all__ = [
    "GeoContext",
    "GeoSample",
    "GeoRoute",
    "fetch_walking_route",
    "WeatherSnapshot",
    "fetch_weather",
]
