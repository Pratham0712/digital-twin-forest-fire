"""Land-cover FUSION on the CA grid (OFFLINE FIXTURE TESTS).

WorldCover-format and NDVI rasters are synthetic fixtures; OpenStreetMap features
are Overpass-format dicts built here. Covers classification per class, fractional
and mixed cells, road classes (D4), no over-blocking, fuel load from NDVI,
confidence from source agreement, degraded / unavailable modes and provenance."""
import numpy as np
import pytest

from config.config import LAND_COVER
from src.landcover import osm_vectors, provider, sentinel2_ndvi, worldcover
from src.landcover.fusion import FUEL_SRC_CLASS_DEFAULT, FUEL_SRC_NDVI
from src.landcover.grid import GridSpec
from src.simulation.fuel_map import BUILT, FUEL, NON_FUEL, ROAD, UNKNOWN, WATER
from tests.landcover_fixtures import ndvi_loader, paint, unavailable_loader, wc_loader

LAT, LON, CELL, N = 11.6667, 76.6333, 25.0, 20
DLAT = CELL / 111320.0
DLON = CELL / (111320.0 * np.cos(np.radians(LAT)))
SPEC = GridSpec(LAT + N / 2 * DLAT, LAT - N / 2 * DLAT, LON - N / 2 * DLON, LON + N / 2 * DLON, N, N, CELL)


def ll(c, r):
    """lat/lon of fractional (col, row) of SPEC."""
    return {"lat": SPEC.north - r * DLAT, "lon": SPEC.west + c * DLON}


def box(c0, r0, c1, r1, tags):
    return {"type": "way", "tags": tags, "geometry": [ll(c0, r0), ll(c1, r0), ll(c1, r1), ll(c0, r1), ll(c0, r0)]}


def line(pts, tags):
    return {"type": "way", "tags": tags, "geometry": [ll(*p) for p in pts]}


def cell_rect(c0, r0, c1, r1):
    """lat/lon rectangle of cells [r0:r1, c0:c1] (for painting rasters)."""
    return SPEC.north - r1 * DLAT, SPEC.north - r0 * DLAT, SPEC.west + c0 * DLON, SPEC.west + c1 * DLON


def fused(monkeypatch, wc=wc_loader(10), osm=None, ndvi=None):
    monkeypatch.setattr(worldcover, "load_window", wc)
    monkeypatch.setattr(sentinel2_ndvi, "load_window", ndvi or unavailable_loader())
    if osm is None:
        monkeypatch.setattr(osm_vectors, "load_osm", lambda spec, allow_fetch=True:
                            (None, {"status": "unavailable", "error": "test: no OSM"}))
    else:
        monkeypatch.setattr(osm_vectors, "load_osm", lambda spec, allow_fetch=True:
                            ({"fetched_utc": "2026-10-01T00:00:00Z", "elements": osm}, {"status": "cached",
                                                                                         "retrieved": "2026-10-01"}))
    provider.clear_memory_cache()
    return provider.grid_land_cover(SPEC, allow_fetch=False)


def painter(*patches):
    def p(w):
        for (c0, r0, c1, r1), v in patches:
            s, n, we, e = cell_rect(c0, r0, c1, r1)
            paint(w, s, n, we, e, v)
    return p


# 2, 8 ─ WorldCover -> exact CA grid, vegetation classes ─────────────────────
def test_02_08_worldcover_to_ca_grid_and_vegetation(monkeypatch):
    lc = fused(monkeypatch, wc_loader(10, painter(((0, 0, 5, 20), 20), ((5, 0, 10, 20), 30), ((10, 0, 15, 20), 40))),
               osm=[])
    assert lc.classes.shape == SPEC.shape and lc.fuel_load.shape == SPEC.shape
    assert (lc.classes == FUEL).all()                                   # tree, shrub, grass, crop all burnable
    assert lc.fractions["shrub"][:, :5].min() == pytest.approx(1.0) and lc.fractions["grass"][:, 5:10].min() == 1
    d = LAND_COVER.default_fuel_load
    assert lc.fuel_load[3, 2] == pytest.approx(d[20]) and lc.fuel_load[3, 7] == pytest.approx(d[30])
    assert lc.fuel_load[3, 12] == pytest.approx(d[40]) and lc.fuel_load[3, 17] == pytest.approx(d[10])
    assert (lc.fuel_source[lc.classes == FUEL] == FUEL_SRC_CLASS_DEFAULT).all()
    assert lc.status == "full"


# 4, 5, 7 ─ water, building, bare ─────────────────────────────────────────
def test_04_05_07_water_built_bare_from_worldcover(monkeypatch):
    lc = fused(monkeypatch, wc_loader(10, painter(((2, 2, 6, 6), 80), ((10, 2, 14, 6), 50), ((2, 12, 6, 16), 60))),
               osm=[])
    assert (lc.classes[2:6, 2:6] == WATER).all() and (lc.classes[2:6, 10:14] == BUILT).all()
    assert (lc.classes[12:16, 2:6] == NON_FUEL).all() and lc.classes[10, 10] == FUEL
    assert (lc.fuel_load[lc.classes != FUEL] == 0).all()               # 18: non-burnable fuel load = 0


def test_03_05_osm_overlay_building_and_water_on_vegetation(monkeypatch):
    osm = [box(2, 2, 6, 6, {"building": "yes"}), box(10, 10, 15, 15, {"natural": "water"})]
    lc = fused(monkeypatch, osm=osm)
    assert (lc.classes[2:6, 2:6] == BUILT).all() and (lc.classes[10:15, 10:15] == WATER).all()
    assert lc.classes[8, 8] == FUEL and lc.fuel_load[3, 3] == 0


# 6 + D4 ─ road classes ────────────────────────────────────────────────────
def test_06_major_road_is_a_continuous_barrier(monkeypatch):
    lc = fused(monkeypatch, osm=[line([(0.3, 0.2), (19.7, 19.8)], {"highway": "trunk"})])
    road = lc.classes == ROAD
    assert road.sum() >= 20 and (np.diag(road)).all()
    from tests.test_fuel_map import _reachable
    free = lc.classes == FUEL
    assert not _reachable(free, (0, 19))[(19, 0)] or not _reachable(free, (19, 0))[(0, 19)] or \
        not _reachable(free, (0, 19))[19, 0]


@pytest.mark.parametrize("hw,width", [("track", None), ("service", None), ("unclassified", None),
                                      ("tertiary", None), ("unclassified", "6")])
def test_06_d4_minor_road_or_track_does_not_block_a_vegetation_cell(monkeypatch, hw, width):
    tags = {"highway": hw, **({"width": width} if width else {})}
    lc = fused(monkeypatch, osm=[line([(-1, 10.5), (21, 10.5)], tags)])          # E-W through row 10
    if hw == "track":                                                            # not fetched/mapped as a barrier
        assert (lc.classes == FUEL).all()
        return
    assert (lc.classes[10] == FUEL).all()                                        # the 25 m cells stay burnable
    assert 0.0 < lc.fractions["road"][10, 5] < LAND_COVER.road_block_fraction    # fractional coverage only
    assert lc.fuel_load[10, 5] < lc.fuel_load[3, 5]                             # less fuel in the road cells


def test_major_road_classes_are_configurable(monkeypatch):
    monkeypatch.setattr(LAND_COVER, "road_block_classes", ("trunk", "primary", "secondary", "tertiary"))
    lc = fused(monkeypatch, osm=[line([(-1, 10.5), (21, 10.5)], {"highway": "tertiary"})])
    assert (lc.classes[10] == ROAD).all()


def test_wide_road_area_fraction_blocks(monkeypatch):
    lc = fused(monkeypatch, osm=[line([(-1, 10.5), (21, 10.5)], {"highway": "unclassified", "width": "30"})])
    assert (lc.classes[10] == ROAD).all()                                        # >= road_block_fraction of the cell


# 8 (D-over-blocking) ─ trees next to a road / house are not rejected ────────
def test_no_over_blocking_near_roads_houses_and_landuse(monkeypatch):
    osm = [line([(-1, 10.5), (21, 10.5)], {"highway": "service"}),
           box(4.2, 4.2, 4.6, 4.6, {"building": "yes"}),                         # 10 m hut inside one 25 m cell
           box(12, 0, 20, 8, {"landuse": "commercial"})]                         # land-use polygon over trees
    lc = fused(monkeypatch, osm=osm)
    assert lc.classes[9, 5] == FUEL and lc.classes[11, 5] == FUEL and lc.classes[10, 5] == FUEL
    assert lc.classes[4, 4] == FUEL and 0 < lc.fractions["built"][4, 4] < 0.5
    assert (lc.classes[0:8, 12:20] == FUEL).all()                                # WorldCover sees trees there
    lc2 = fused(monkeypatch, wc_loader(10, painter(((12, 0, 20, 8), 50))), osm=osm)
    assert (lc2.classes[1:7, 13:19] == BUILT).all()                              # built where WorldCover agrees


# 9, 10 ─ mixed cells and fractions ─────────────────────────────────────────
def test_09_10_fractions_and_mixed_cell_rules(monkeypatch):
    # OSM polygons give exact sub-cell coverage (5 x 5 points per 25 m cell):
    # cell (5, 5): 40 % water (2 of 5 point columns), rest tree -> FUEL with reduced load
    # cell (5, 12): 60 % water -> WATER
    osm = [box(5, 5, 5.4, 6, {"natural": "water"}), box(12, 5, 12.6, 6, {"natural": "water"})]
    lc = fused(monkeypatch, osm=osm)
    f = lc.fractions
    total = f["water"] + f["built"] + f["road"] + f["bare"] + f["vegetation"] + f["unknown"]
    assert np.allclose(total, 1.0)
    assert f["water"][5, 5] == pytest.approx(0.4) and lc.classes[5, 5] == FUEL
    assert lc.fuel_load[5, 5] == pytest.approx(0.6 * LAND_COVER.default_fuel_load[10])
    assert f["water"][5, 12] == pytest.approx(0.6) and lc.classes[5, 12] == WATER


def test_mixed_cell_without_vegetation_is_never_fuel(monkeypatch):
    osm = [box(5, 5, 5.4, 6, {"natural": "water"}), box(5.4, 5, 6, 6, {"building": "yes"})]   # 40 % / 60 %
    lc = fused(monkeypatch, osm=osm)
    assert lc.classes[5, 5] == BUILT
    osm = [box(5, 5, 5.4, 6, {"natural": "water"}), box(5.4, 5, 5.8, 6, {"building": "yes"})]  # 40/40/20 tree
    lc = fused(monkeypatch, osm=osm)
    assert lc.classes[5, 5] == WATER        # 80 % non-burnable >= 70 %: dominant class, ties -> water precedence


# 16, 18, 19 ─ NDVI -> fuel load ───────────────────────────────────────────
def test_16_19_ndvi_fuel_load_varies_spatially(monkeypatch):
    def grad(w):
        for k, v in enumerate((0.2, 0.4, 0.6, 0.8)):
            s, n, we, e = cell_rect(5 * k, 0, 5 * k + 5, 20)
            paint(w, s, n, we, e, v)
    lc = fused(monkeypatch, osm=[], ndvi=ndvi_loader(0.5, grad))
    b, d = LAND_COVER.ndvi_bare, LAND_COVER.ndvi_dense
    for k, v in enumerate((0.2, 0.4, 0.6, 0.8)):
        assert lc.fuel_load[10, 5 * k + 2] == pytest.approx(np.clip((v - b) / (d - b), 0, 1), abs=1e-5)
    row = lc.fuel_load[10]
    assert np.all(np.diff(row) >= -1e-6) and row[-1] > row[0]           # increases with NDVI, cell by cell
    assert (lc.fuel_source == FUEL_SRC_NDVI).all()
    assert "Sentinel-2" in lc.label or "NDVI" in lc.label


def test_18_water_and_building_have_zero_fuel_even_with_high_ndvi(monkeypatch):
    lc = fused(monkeypatch, wc_loader(10, painter(((2, 2, 6, 6), 80))), osm=[box(10, 10, 14, 14, {"building": "y"})],
               ndvi=ndvi_loader(0.9))
    assert (lc.fuel_load[2:6, 2:6] == 0).all() and (lc.fuel_load[10:14, 10:14] == 0).all()
    assert lc.fuel_load[17, 17] == pytest.approx(1.0)


# confidence (source agreement only) ────────────────────────────────────────
def test_confidence_from_source_agreement(monkeypatch):
    osm = [box(2, 2, 6, 6, {"natural": "water"}),            # OSM water where WorldCover also has water -> HIGH
           box(10, 2, 14, 6, {"natural": "water"}),          # OSM water where WorldCover has pure trees -> LOW
           box(2, 12, 6, 16, {"building": "yes"})]           # OSM building, WorldCover built-up -> HIGH
    lc = fused(monkeypatch, wc_loader(10, painter(((2, 2, 6, 6), 80), ((2, 12, 6, 16), 50), ((14, 14, 18, 18), 80))),
               osm=osm, ndvi=ndvi_loader(0.7))
    c = lc.confidence
    assert (c[3:5, 3:5] == 3).all() and (c[3:5, 11:13] == 1).all() and (c[13:15, 3:5] == 3).all()
    assert (c[15:17, 15:17] == 2).all()                      # WorldCover water only (OSM silent) -> MEDIUM
    assert c[9, 9] == 3                                      # pure tree + Sentinel-2 NDVI agrees -> HIGH
    lc2 = fused(monkeypatch, osm=[])                         # WorldCover only -> MEDIUM, never invented HIGH
    assert (lc2.confidence == 2).all()


# degraded / unavailable (D6) ─────────────────────────────────────────────
def test_osm_only_mode_marks_unmapped_cells_unverified_not_fuel(monkeypatch):
    lc = fused(monkeypatch, wc=unavailable_loader("WorldCover offline"),
               osm=[box(2, 2, 6, 6, {"natural": "water"}), line([(-1, 10.5), (21, 10.5)], {"highway": "primary"})])
    assert lc.status == "degraded" and provider.DEGRADED_TITLE in lc.label
    assert (lc.classes[2:6, 2:6] == WATER).all() and (lc.classes[10] == ROAD).all()
    assert (lc.classes[15:20, 0:20] == UNKNOWN).all() and not (lc.classes == FUEL).any()
    assert lc.non_fuel[15, 15] and not lc.non_burnable(allow_unverified=True)[15, 15]
    assert any("UNVERIFIED" in n for n in lc.notes) and lc.confidence[15, 15] == 0


def test_worldcover_only_mode_is_degraded_but_classified(monkeypatch):
    lc = fused(monkeypatch, osm=None)
    assert lc.status == "degraded" and (lc.classes == FUEL).all()
    assert any("OpenStreetMap unavailable" in n for n in lc.notes)


def test_no_source_is_land_cover_unavailable(monkeypatch):
    lc = fused(monkeypatch, wc=unavailable_loader("WorldCover offline"), osm=None)
    assert lc.status == "unavailable" and not lc.available
    assert lc.label.startswith(provider.UNAVAILABLE_TITLE) and "ESA WorldCover" in lc.label and \
        "OpenStreetMap" in lc.label
    assert (lc.classes == UNKNOWN).all() and lc.non_fuel.all()            # never silently all-fuel


# 11 ─ provenance ──────────────────────────────────────────────────────────
def test_11_provenance(monkeypatch):
    lc = fused(monkeypatch, osm=[box(2, 2, 6, 6, {"natural": "water"})], ndvi=ndvi_loader(0.6))
    p = lc.provenance
    assert p["fusion_version"] == LAND_COVER.fusion_version and p["rules_version"] == LAND_COVER.rules_version
    assert p["bounds"] == SPEC.bounds and p["cell_m"] == CELL and p["grid"] == [N, N]
    assert set(lc.sources) == {"worldcover", "osm", "sentinel2", "dynamic_world"}
    assert "OFFLINE TEST FIXTURE" in lc.label and "OpenStreetMap (1 mapped feature, retrieved 2026-10-01)" in lc.label
    assert p["worldcover_version"] == LAND_COVER.worldcover_version
    assert lc.source_mask[3, 3] & 1 and lc.source_mask[3, 3] & 2 and lc.source_mask[3, 3] & 4


def test_fused_result_is_cached_in_memory(monkeypatch):
    a = fused(monkeypatch, osm=[])
    b = provider.grid_land_cover(SPEC, allow_fetch=False)
    assert a is b
