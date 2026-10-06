"""
regions.py - the shared set of preset regions this project supports
(cross-question 11/12: "we're using the concept of different regions...
we can consider only 5 regions but for those 5 we should handle correctly").

Pulled out of src/dashboard/dashboard_common.py so that non-dashboard code
(the background scheduler in particular) can use the exact same region
definitions without importing Streamlit/Plotly/Folium - those are dashboard
rendering dependencies that a headless background process has no business
needing just to know where "California" is.

The 5 regions: Karnataka/Western Ghats (the validated, trained-on region)
plus 4 presets spanning genuinely different fire climates, plus "Custom"
(any user-supplied lat/lon box) handled separately in the dashboard sidebar -
that custom option is the 5th "region" in the sense the review will ask
about: an untrained location the system still produces a live prediction
for, using the same FWI + CA pipeline, just without a region-specific
validated model.
"""
from config.config import REGION, RegionConfig

REGION_PRESETS = {
    "Karnataka / Western Ghats": REGION,
    # Curated set of India's other officially fire-prone states (per Forest
    # Survey of India fire-alert data) - replaces the earlier non-Indian
    # placeholder regions (California/Australia/Mediterranean) now that the
    # project's scope is India-wide rather than a generic architecture demo.
    # Each is real geography with real FIRMS/weather coverage; NONE are
    # "validated" yet in the sense Karnataka is - that requires pulling real
    # historical data and training/evaluating a model per region (see
    # scripts/build_india_regions_dataset.py) before removing this caveat.
    "Uttarakhand": RegionConfig(name="Uttarakhand", min_lat=28.7, max_lat=31.5,
                                 min_lon=77.5, max_lon=81.0,
                                 grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Himachal Pradesh": RegionConfig(name="Himachal Pradesh", min_lat=30.3, max_lat=33.3,
                                      min_lon=75.5, max_lon=79.1,
                                      grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Odisha": RegionConfig(name="Odisha", min_lat=17.8, max_lat=22.6,
                            min_lon=81.4, max_lon=87.5,
                            grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Chhattisgarh": RegionConfig(name="Chhattisgarh", min_lat=17.8, max_lat=24.1,
                                  min_lon=80.2, max_lon=84.4,
                                  grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Madhya Pradesh": RegionConfig(name="Madhya Pradesh", min_lat=21.0, max_lat=26.9,
                                    min_lon=74.0, max_lon=82.8,
                                    grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Maharashtra": RegionConfig(name="Maharashtra", min_lat=15.6, max_lat=22.0,
                                 min_lon=72.6, max_lon=80.9,
                                 grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Andhra Pradesh / Telangana": RegionConfig(name="Andhra Pradesh / Telangana",
                                                 min_lat=12.6, max_lat=19.9,
                                                 min_lon=76.8, max_lon=84.8,
                                                 grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
    "Assam": RegionConfig(name="Assam", min_lat=24.1, max_lat=28.2,
                           min_lon=89.7, max_lon=96.0,
                           grid_resolution_deg=0.5, weather_grid_resolution_deg=1.5),
}
