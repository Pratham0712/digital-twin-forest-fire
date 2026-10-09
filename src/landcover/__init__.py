"""
Land-cover fusion for the local fire-spread simulation.

    ESA WorldCover (broad land-cover classes, 10 m)      worldcover.py
  + OpenStreetMap (roads, buildings, water as vectors)  osm_vectors.py
  + Sentinel-2 NDVI (vegetation condition, 10 m)        sentinel2_ndvi.py
  + optional Dynamic World (cached probabilities)       dynamic_world.py
        -> fusion.py (per-cell fractions, documented thresholds)
        -> provider.py (source hierarchy, fallbacks, caching, provenance)
        -> fuel_map.LandCover on exactly the CA grid

Google satellite imagery is only the visual base map; it is never classified.
"""
