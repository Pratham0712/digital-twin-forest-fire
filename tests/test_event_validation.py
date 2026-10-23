"""Event-level spread validation: the pieces that must be right for the numbers to mean anything."""
import numpy as np
import pandas as pd
import pytest

from src.simulation import event_validation as ev


def test_auc_perfect_random_and_ties():
    truth = np.array([0, 0, 0, 1, 1], bool)
    cand = np.ones(5, bool)
    assert ev.auc(np.array([1, 2, 3, 4, 5.0]), truth, cand) == 1.0
    assert ev.auc(np.array([5, 4, 3, 2, 1.0]), truth, cand) == 0.0
    assert ev.auc(np.zeros(5), truth, cand) == pytest.approx(0.5)      # all tied = chance
    assert ev.auc(np.arange(5.0), np.zeros(5, bool), cand) is None     # nothing to rank


def test_growth_direction_reads_the_compass():
    prev = np.zeros((40, 40), bool)
    prev[20, 18:22] = True
    east = prev.copy(); east[20, 22:30] = True
    north = prev.copy(); north[10:19, 20] = True
    assert abs(ev.growth_direction(prev, east) - 90.0) < 5
    d = ev.growth_direction(prev, north)
    assert min(d, 360 - d) < 5
    assert ev.growth_direction(prev, prev) is None                     # no growth, no direction


def test_ros_automaton_runs_downwind_first_at_pixel_scale():
    seeds = np.zeros((41, 41), bool)
    seeds[20, 20] = True
    wx = {"ffmc": 92.0, "bui": 80.0, "wind": 4.0}
    t = ev.ca_arrival(seeds, wx, wind_from=270.0)                      # wind from the west -> fire runs east
    assert np.isfinite(t[20, 26]) and np.isfinite(t[20, 14])
    assert t[20, 26] < t[20, 14]


def test_isotropic_arrival_is_symmetric_enough():
    seeds = np.zeros((41, 41), bool)
    seeds[20, 20] = True
    t = ev.ca_arrival_iso(seeds, {"ffmc": 92.0, "bui": 80.0, "wind": 3.0})
    assert abs(t[20, 25] - t[20, 15]) < 0.25 * max(t[20, 25], t[20, 15])


def test_best_radius_and_tau_pick_the_matching_scale():
    seeds = np.zeros((31, 31), bool)
    seeds[15, 15] = True
    dist = ev.dist_to_seeds(seeds)
    truth = (dist <= 3) & ~seeds
    assert ev.best_radius(dist, seeds, truth) == 3
    assert ev.best_radius(dist, seeds, np.zeros_like(truth)) is None


def test_weather_lookup_refuses_far_zones():
    t = pd.DataFrame({"latitude": [12.0], "longitude": [76.0], "date": ["2025-03-01"],
                      "ffmc": [90.0], "bui": [50.0], "wx_wind_speed_ms": [3.0]})
    w = ev.WeatherLookup([t])
    assert w.at(12.05, 76.05, "2025-03-01")["ffmc"] == 90.0
    assert w.at(20.0, 80.0, "2025-03-01") is None                      # hundreds of km away: no honest weather
    assert w.at(12.0, 76.0, "2025-03-02") is None                      # no data that day
