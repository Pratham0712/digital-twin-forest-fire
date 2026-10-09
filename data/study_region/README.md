# Study region boundary — Bandipur Tiger Reserve

The application looks for the boundary in this order:

1. **OFFICIAL**: `bandipur_official.geojson` (or `.json`, `.kml`, `.shp` with its `.shx`/`.dbf`/`.prj`).
   Put the official notified Tiger Reserve boundary (core + buffer, ≈1,456 km²) here.
   It is picked up automatically and labelled **OFFICIAL**. Shapefiles in a projected CRS
   (for example UTM 43N) are converted to WGS84 using their `.prj`.
2. **APPROXIMATE / PROVISIONAL**: `bandipur_provisional.geojson` (shipped).
   It is only the latitude/longitude **extent** published by NTCA, so it is a box that
   over-covers the reserve. It is for development and testing only, and everything derived
   from it is labelled APPROXIMATE / PROVISIONAL.

An OpenStreetMap National Park polygon must not be used as the complete Tiger Reserve.
