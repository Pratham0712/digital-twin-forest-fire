"""Primary study region (Bandipur Tiger Reserve): OFFICIAL vs APPROXIMATE
boundary, GeoJSON / KML / shapefile support, boundary crossing and separate
Bandipur / REGIONAL EXTENSION statistics. The boundary never stops the fire."""
import json
import struct

import numpy as np
import pytest

from config.config import STUDY_REGION
from src.geo import study_region as sr
from src.geo.study_region import APPROXIMATE, OFFICIAL, StudyRegion, load_study_region, read_kml, read_shapefile
from src.simulation.fuel_map import FUEL, LandCover
from src.simulation.local_spread import FocusArea, SimulationDomain, run_local_spread

LAT, LON = 11.6667, 76.6333
HOT = {"ffmc": 97.0, "ndvi": 0.8, "bui": 80.0, "risk_score": 0.9, "zone_id": 1, "fwi": 60.0}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    sr._CACHE.clear()
    monkeypatch.setattr(STUDY_REGION, "official_boundary_stem", str(tmp_path / "bandipur_official"))
    yield tmp_path
    sr._CACHE.clear()


def test_shipped_boundary_is_labelled_approximate_not_official():
    r = load_study_region()
    assert r is not None and r.status == APPROXIMATE and not r.official
    assert "APPROXIMATE" in r.label and r.notes and "not the notified" in r.notes[0]
    assert r.contains(LAT, LON)                                       # the default focus lies inside
    assert r.stated_area_km2 == pytest.approx(1456.309)
    assert abs(r.area_km2 - r.stated_area_km2) > 500                   # it is an extent box, and says so


def test_official_geojson_takes_precedence(_fresh):
    poly = [[76.60, 11.65], [76.66, 11.65], [76.66, 11.69], [76.60, 11.69], [76.60, 11.65]]
    (_fresh / "bandipur_official.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"source": "Test Dept."},
                                                    "geometry": {"type": "Polygon", "coordinates": [poly]}}]}))
    r = load_study_region()
    assert r.status == OFFICIAL and r.official and r.source == "Test Dept." and "OFFICIAL" in r.label
    assert r.contains(11.66, 76.62) and not r.contains(11.70, 76.62)


def test_kml_with_hole(_fresh):
    kml = """<kml><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>
    76.60,11.60 76.70,11.60 76.70,11.70 76.60,11.70 76.60,11.60</coordinates></LinearRing></outerBoundaryIs>
    <innerBoundaryIs><LinearRing><coordinates>76.64,11.64 76.66,11.64 76.66,11.66 76.64,11.66 76.64,11.64
    </coordinates></LinearRing></innerBoundaryIs></Polygon></Placemark></kml>"""
    p = _fresh / "bandipur_official.kml"
    p.write_text(kml)
    r = load_study_region()
    assert r.status == OFFICIAL and r.contains(11.62, 76.62) and not r.contains(11.65, 76.65)
    assert read_kml(p)[0][1]                                           # the hole was read


def _write_shp(path, rings):
    """Minimal ESRI polygon shapefile writer (one record)."""
    pts = [p for r in rings for p in r]
    parts = np.cumsum([0] + [len(r) for r in rings[:-1]]).tolist()
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    rec = struct.pack("<i4d2i", 5, min(xs), min(ys), max(xs), max(ys), len(rings), len(pts))
    rec += struct.pack(f"<{len(parts)}i", *parts) + b"".join(struct.pack("<2d", *p) for p in pts)
    body = struct.pack(">2i", 1, len(rec) // 2) + rec
    hdr = struct.pack(">7i", 9994, 0, 0, 0, 0, 0, (100 + len(body)) // 2) + struct.pack("<2i", 1000, 5)
    hdr += struct.pack("<8d", min(xs), min(ys), max(xs), max(ys), 0, 0, 0, 0)
    path.write_bytes(hdr + body)


def test_shapefile_geographic_and_projected(_fresh):
    ring = [(76.60, 11.60), (76.60, 11.70), (76.70, 11.70), (76.70, 11.60), (76.60, 11.60)]   # clockwise
    p = _fresh / "bandipur_official.shp"
    _write_shp(p, [ring])
    polys = read_shapefile(p)
    assert len(polys) == 1 and StudyRegion("x", OFFICIAL, "t", polys).contains(11.65, 76.65)
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.warp import transform
    xs, ys = transform("EPSG:4326", "EPSG:32643", [q[0] for q in ring], [q[1] for q in ring])
    _write_shp(p, [list(zip(xs, ys))])
    p.with_suffix(".prj").write_text(CRS.from_epsg(32643).to_wkt())
    polys = read_shapefile(p)
    r = StudyRegion("x", OFFICIAL, "t", polys)
    assert r.contains(11.65, 76.65) and not r.contains(11.75, 76.65)
    assert r.bbox == pytest.approx((11.60, 76.60, 11.70, 76.70), abs=1e-6)


def _run(region, wind_from=270.0, dur=120):
    f = FocusArea(name="t", lat=LAT, lon=LON, width_m=250.0, height_m=250.0, cell_m=25.0)
    d = SimulationDomain(f, 40)
    lc = LandCover(np.full((d.n_rows, d.n_cols), FUEL, np.int8), "fused", "fixture", 0, {}, status="full",
                   fuel_load=np.full((d.n_rows, d.n_cols), 0.9))
    return run_local_spread(f, HOT, 10.0, wind_from, domain=d, land_cover=lc, n_ignition=2, placement="Centre",
                            seed=4, duration_minutes=dur, base_spread_prob=1.2, study_region=region), d


def test_boundary_crossing_detected_and_statistics_kept_separate():
    # boundary 100 m east of the ignition, fire pushed east by a west wind
    east = LON + 100 / (111320 * np.cos(np.radians(LAT)))
    region = StudyRegion("Bandipur Tiger Reserve", APPROXIMATE, "test", [(np.array(
        [[76.5, 11.5], [east, 11.5], [east, 11.8], [76.5, 11.8], [76.5, 11.5]]), [])])
    res, d = _run(region)
    info = res.study_region
    assert info["crossed"] and info["crossing_step"] > 0 and info["crossing_lon"] > east
    assert not info["ignition_outside_region"]
    fin = res.final
    assert fin["burned_ha_regional_extension"] > 0 and fin["burned_ha_bandipur"] > 0
    assert fin["burned_ha_bandipur"] + fin["burned_ha_regional_extension"] == pytest.approx(fin["burned_ha"])
    burned = res.ignition_step >= 0
    assert (burned & ~res.inside_region).any()                        # the fire continued past the boundary
    early = [m for m in res.metrics if m["step"] < info["crossing_step"]]
    assert all(m["fire_area_ha_regional_extension"] == 0 for m in early)


def test_no_crossing_when_the_fire_stays_inside():
    region = StudyRegion("Bandipur Tiger Reserve", APPROXIMATE, "test", [(np.array(
        [[76.0, 11.0], [77.0, 11.0], [77.0, 12.0], [76.0, 12.0], [76.0, 11.0]]), [])])
    res, _ = _run(region, dur=30)
    assert not res.study_region["crossed"] and res.study_region["crossing_step"] is None
    assert res.final["burned_ha_regional_extension"] == 0


def test_split_stats_helper():
    region = load_study_region()
    f = FocusArea(name="t", lat=LAT, lon=LON, width_m=250.0, height_m=250.0, cell_m=25.0)
    d = SimulationDomain(f, 2)
    aff = np.zeros((d.n_rows, d.n_cols), bool)
    aff[:3, :3] = True
    st = sr.split_stats(region, d, aff, 25.0)
    assert st["inside_cells"] == 9 and st["outside_cells"] == 0 and st["region"] == APPROXIMATE


def test_regional_twin_statistics_are_limited_to_the_study_region():
    import pandas as pd
    from types import SimpleNamespace
    region = StudyRegion("Bandipur Tiger Reserve", APPROXIMATE, "test", [(np.array(
        [[76.2, 11.6], [76.9, 11.6], [76.9, 11.9], [76.2, 11.9], [76.2, 11.6]]), [])])
    grid = pd.DataFrame({"zone_id": ["A", "B", "C"], "latitude": [11.65, 11.75, 12.5],
                         "longitude": [76.35, 76.65, 76.65]})
    alerts = [SimpleNamespace(zone_id="B", severity="HIGH"), SimpleNamespace(zone_id="C", severity="EXTREME")]
    hot = pd.DataFrame({"latitude": [11.7, 12.6], "longitude": [76.5, 76.6]})
    z = sr.zone_summary(region, grid, np.array([0.2, 0.5, 0.95]), alerts, hot)
    assert z["zones"] == 2 and z["peak_risk"] == pytest.approx(0.5) and z["mean_risk"] == pytest.approx(0.35)
    assert z["alerts"] == 1 and z["detections"] == 1                  # zone C / the 2nd detection are outside
