import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/research/btc_eth_lead_lag_controls.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("lead_lag_controls", SOURCE)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)
sys.path.remove(str(SOURCE.parent))


@pytest.fixture
def protocol():
    return {
        "minimum_prior_fit_windows": 50,
        "horizons_seconds": [1],
        "forecast_controls": {
            "models": {"own_momentum": [0, 1], "cross_coin": [0, 1, 2], "past_shifted_leader": [0, 1, 3]}
        },
        "uncertainty": {"seed": 20261106, "bootstrap_draws": 2000, "percentiles": [2.5, 97.5]},
    }


def test_forecast_moments_match_direct_error_and_prior_day_fit(protocol):
    rng = np.random.default_rng(1)
    features = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    target = features @ np.array([0.01, 0.1, 0.4, 0.0])
    moment = controls.moments(features, target)
    prior = {("eth", 1): {**moment, "windows": 50, "dates": {"2026-10-04", "2026-10-05"}, "last_day": "2026-10-05"}}
    fitted = controls.forecast_fits(prior, "2026-10-06", protocol)
    columns, beta = fitted[("eth", 1)]["models"]["cross_coin"]
    computed = (
        moment["square"] - 2 * beta @ moment["cross"][columns] + beta @ moment["gram"][np.ix_(columns, columns)] @ beta
    )
    direct = np.mean((target - features[:, columns] @ beta) ** 2)
    assert computed == pytest.approx(direct, abs=1e-14)
    assert beta == pytest.approx([0.01, 0.1, 0.4])
    with pytest.raises(ValueError, match="current or future"):
        controls.forecast_fits(prior, "2026-10-05", protocol)


def test_degenerate_forecast_is_unknown_instead_of_fabricated(protocol):
    features = np.ones((100, 4))
    prior = {
        ("eth", 1): {
            **controls.moments(features, np.ones(100)),
            "windows": 50,
            "dates": {"2026-10-04", "2026-10-05"},
            "last_day": "2026-10-05",
        }
    }
    assert controls.forecast_fits(prior, "2026-10-06", protocol) == {}


def test_residual_fit_uses_earlier_days_and_requires_variance(protocol):
    x = np.linspace(-2, 2, 100)
    y = 0.3 + 2 * x + np.sin(np.arange(100)) * 0.1
    stats = {
        "sums": np.array([len(x), x.sum(), y.sum(), x @ x, x @ y, y @ y]),
        "windows": 50,
        "dates": {"2026-10-04", "2026-10-05"},
        "last_day": "2026-10-05",
    }
    fit = controls.residual_fits({(0, 0): stats}, "2026-10-06", protocol)[(0, 0)]
    assert fit["alpha"] == pytest.approx(0.3, abs=0.01)
    assert fit["beta"] == pytest.approx(2, abs=0.01)
    assert fit["sigma"] > 0
    assert fit["last_fit_day"] < "2026-10-06"
    with pytest.raises(ValueError, match="current or future"):
        controls.residual_fits({(0, 0): stats}, "2026-10-05", protocol)
    stats["windows"] = 49
    assert controls.residual_fits({(0, 0): stats}, "2026-10-06", protocol) == {}


def test_first_event_is_not_reselected_to_avoid_missing_future():
    books = {
        ("eth", "up"): {
            "mid": np.array([0.5, 0.51, 0.52, np.nan, 0.54]),
            "bid": np.array([0.49, 0.50, 0.51, np.nan, 0.53]),
            "ask": np.array([0.51, 0.52, 0.53, np.nan, 0.55]),
            "valid": np.array([True, True, True, False, True]),
        }
    }
    event = controls.event_record(
        {"window_start": 1791158400},
        "eth",
        "signal",
        np.array([False, True, True, False, False]),
        np.ones(5),
        books,
        1,
        [2],
    )
    assert event["elapsed_seconds"] == 1
    assert event["midpoint_responses"]["2"] is None


def test_past_shift_never_wraps_future_values():
    assert controls.previous(np.array([1, 2, 3, 4]), 2) == pytest.approx([np.nan, np.nan, 1, 2], nan_ok=True)


def test_uncertainty_resamples_days_and_keeps_dependence(protocol):
    interval = controls.day_interval(
        {"d1": np.array([1000.0, 1000]), "d2": np.array([-1000.0, 1000]), "d3": np.array([1000.0, 1000])}, protocol
    )
    json.dumps(interval, allow_nan=False)
    assert interval["days"] == 3
    assert interval["estimate"] == pytest.approx(1 / 3)
    assert interval["bootstrap_interval"][0] < 0 < interval["bootstrap_interval"][1]
    assert controls.day_interval({"d1": np.array([1000.0, 1000])}, protocol)["bootstrap_interval"] is None


def test_matching_reports_common_support_and_unique_windows(protocol):
    base = {
        "window_start": 1791158400,
        "day": "2026-10-05",
        "target": "eth",
        "time_bucket": 0,
        "probability_band": 1,
        "spread_band": 0,
    }
    events = [
        {**base, "group": "F3_confirmation", "midpoint_responses": {"1": 0.02}},
        {**base, "group": "F3_own_momentum", "midpoint_responses": {"1": 0.01}},
        {
            **base,
            "day": "2026-10-06",
            "window_start": 1791244800,
            "group": "F3_confirmation",
            "midpoint_responses": {"1": 0.05},
        },
    ]
    result = controls.matched_events(events, protocol)["F3_confirmation_vs_F3_own_momentum:1"]
    assert result["estimate"] == pytest.approx(0.01)
    assert result["unique_windows_in_common_strata"] == 1
    assert result["observations_in_common_strata"] == 2
    assert result["usable_group_observations"] == 3
    assert result["bootstrap_interval"] is None
