# Geo-driven Route through Germany

## Designprinzip (wichtig)

**Keine 1:1-Abbildung** der realen Gebaeude oder Felder.

Geo-Daten steuern nur ein **grobes Landschaftsklima**:

- Bebauungsdichte hoch → mehr Haeuser / Stadt-Mood
- Waldanteil hoch → mehr Baeume / Wald-Mood
- Agrar / offen → Felder, Bauernhoefe, Offenland
- Wetter / Hoehe / Laufrichtung → Himmel, Grade, leichte Props

Die konkreten Objekte entstehen weiterhin **organisch und zufaellig**
(wie bisher ueber Tags + Soft-Region-Bias). Die Route fuehlt sich
"nach Ort an", ohne ein Kartenabbild zu sein.

Spaeter (optional): **Landmarken-Vorausschau** — markante Gebaeude /
Wahrzeichen einige hundert Meter voraus per OSM/Wikidata finden und
**stilisiert grob** nachbilden (Silhouette, grober Typ), nicht fotorealistisch.

## Architektur

```
OSRM (Fussroute) --> GeoRoute (lon/lat/distance/heading)
                         |
         +---------------+---------------+
         v               v               v
   Open-Meteo        Overpass/OSM     Open-Meteo
   Weather           density only     Elevation
         |               |               |
         +---------------+---------------+
                         v
                   GeoSample
         (mood bias, weather, heading,
          density 0..1 — no object coords)
                         |
          +--------------+--------------+
          v              v              v
   RegionMood      Feature spawn     heading_project()
   probability     weights only      Sonne/Mond/Sterne
```

## Datenquellen

| Bedarf | Live-API (jetzt) | Offline / BKG (spaeter) |
|--------|------------------|-------------------------|
| Route | OSRM foot | eigene GPX |
| Wetter | Open-Meteo Forecast | Cache |
| Hoehe | Open-Meteo Elevation (DEM 90 m) | BKG DGM200, OpenDTM-DE |
| Dichte / Charakter | OSM Overpass (Zaehlungen) | **LBM-DE 2021** Versiegelung/Vegetation |
| Demographie (Hint) | OSM | **VG25** |
| Himmel | `heading_project(view_az=heading)` | — |
| Landmarken (spaeter) | OSM + Wikidata voraus | — |

### BKG Open Data (https://gdz.bkg.bund.de)

- **LBM-DE 2021**: Landbedeckung + Versiegelungs-/Vegetationsgrad — ideal als
  **Dichte-Signal**, nicht als Objektliste
- **VG25**: Verwaltungsgebiete (Ortsschilder, Regionsnamen)
- **DGM200 / DGM1000**: grobes Relief

## Python-API

```python
from domain.geo import GeoContext
from domain.geo.route import demo_route_frankfurt_heidelberg

route = demo_route_frankfurt_heidelberg()
ctx = GeoContext(route)
ctx.enrich_elevations()
sample = ctx.sample(distance_m=1500.0)
# sample.mood_name  -> "stadt" | "dorf" | "wald" | "offenland"
# sample.landuse.building_density  -> 0..1  (nur Bias, keine Positionen)
```

## Landmarken (Roadmap)

1. Entlang der Route alle N Meter OSM `tourism=attraction|artwork`,
   `historic=*`, `building=church|castle`, `name=*` im Vorausschau-Fenster
   (z.B. 800–2000 m voraus) abfragen.
2. Treffer priorisieren (Wikidata prominence / Wikipedia).
3. Stilisierte Silhouette oder generischer Typ-Prop (Kirche, Turm, Schloss)
   mit Name-Schild einblenden — **keine** genaue Fassadenkopie.

## Attribution

- OpenStreetMap-Mitwirkende (Route, Dichte)
- Open-Meteo (CC BY 4.0)
- Bei BKG: (c) GeoBasis-DE / BKG (CC-BY-4.0)
