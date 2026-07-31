# Geo-driven Route through Germany

Ziel: Route durch Deutschland mit echten Ortsdaten fuer Hintergrund, Objekte, Wetter und Himmel.

## Architektur

```
OSRM (Fussroute) --> GeoRoute (lon/lat/distance/heading)
                         |
         +---------------+---------------+
         v               v               v
   Open-Meteo        Overpass/OSM     Open-Meteo
   Weather           Landuse          Elevation
         |               |               |
         +---------------+---------------+
                         v
                   GeoSample
              (mood, weather, heading,
               lat/lon, elev)
                         |
          +--------------+--------------+
          v              v              v
   RegionMood      Feature weights   heading_project()
   biases          (Stadt/Dorf/...)  Sonne/Mond/Sterne
```

## Datenquellen

| Bedarf | Live-API (jetzt) | Offline / BKG (spaeter) |
|--------|------------------|-------------------------|
| Route | OSRM foot | eigene GPX |
| Wetter | Open-Meteo Forecast | Cache |
| Hoehe | Open-Meteo Elevation (DEM 90 m) | BKG DGM200, OpenDTM-DE |
| Bebauung / Landnutzung | OSM Overpass | **LBM-DE 2021** (BKG, CC-BY, GeoPackage) |
| Demographie | OSM population hint | **VG25** Verwaltungsgebiete |
| Himmel | `heading_project(view_az=heading)` | — |

### BKG Open Data (https://gdz.bkg.bund.de)

- **LBM-DE 2021**: Landbedeckung + Versiegelung/Vegetation, GeoPackage, CC-BY-4.0 (~2 GB)
- **VG25**: Verwaltungsgebiete 1:25 000
- **DGM200 / DGM1000**: freies Gelaendemodell (grob)

LBM-DE eignet sich als lokaler Tile-Cache (RTree), nicht als Live-Download pro Frame.

## Python-API

```python
from domain.geo import GeoContext
from domain.geo.route import demo_route_frankfurt_heidelberg, fetch_walking_route

route = demo_route_frankfurt_heidelberg()
# route = fetch_walking_route((8.68, 50.11), (8.69, 49.41))  # lon, lat

ctx = GeoContext(route)
ctx.enrich_elevations()
sample = ctx.sample(distance_m=1500.0)
print(sample.lat, sample.lon, sample.heading_deg, sample.mood_name)
```

Himmel in Laufrichtung:

```python
from domain.celestial import heading_project, project_stars
xy = heading_project(alt, az, w, h, view_az_deg=sample.heading_deg)
stars = project_stars(dt, w, h, lat_deg=sample.lat, lon_deg=sample.lon,
                      view_az_deg=sample.heading_deg)
```

## Attribution

- OpenStreetMap-Mitwirkende (Route, Landuse)
- Open-Meteo (CC BY 4.0)
- Bei BKG: (c) GeoBasis-DE / BKG (CC-BY-4.0)
