import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/research/btc_eth_lead_lag_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("lead_lag_diagnostics", SOURCE)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def test_asof_preserves_empty_latest_state_and_staleness():
    grid = np.array([-1, 0, 49, 50, 51, 100, 201])
    result = diagnostics.asof_book(
        np.array([0, 50, 100]),
        np.array([0.4, np.nan, 0.5]),
        np.array([0.42, np.nan, 0.52]),
        np.array([True, False, True]),
        grid,
        100,
    )
    assert result["valid"].tolist() == [False, True, True, False, False, True, False]
    assert np.isnan(result["bid"][[0, 3, 4, 6]]).all()
    assert result["bid"][2] == 0.4
    assert result["bid"][5] == 0.5


def test_quote_reader_uses_actual_depth_and_stable_latest_duplicate(tmp_path):
    path = tmp_path / "snapshots.parquet"
    table = pa.table(
        {
            "token_id": ["token"] * 4,
            "observed_at_us": [100, 0, 50, 50],
            "best_bid": [0.5, 0.4, 0.41, 0.42],
            "best_ask": [0.52, 0.42, 0.43, 0.44],
            "bids_price": [[0.5], [0.4], [0.41], []],
            "bids_size": [[5.0], [5.0], [5.0], []],
            "asks_price": [[0.52], [0.42], [0.43], []],
            "asks_size": [[5.0], [5.0], [5.0], []],
        }
    )
    pq.write_table(table, path)
    stat = path.stat()
    result = diagnostics.read_book(
        {"file": str(path), "token_id": "token"},
        {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "rows": 4},
        np.array([0, 50, 75, 100]),
        100,
    )
    assert result["valid"].tolist() == [True, False, False, True]
    assert np.isnan(result["mid"][1:3]).all()


def test_training_guard_rejects_validation_and_holdout():
    plan = {
        "data": {"training_start": "2026-10-04T00:00:00+00:00", "training_end_exclusive": "2026-10-18T00:00:00+00:00"}
    }
    diagnostics.guard_training_window(
        {"partition": "development", "window_start": 1791158400, "window_end": 1791159300}, plan
    )
    for partition, start in [("development", 1792281600), ("holdout", 1792972800)]:
        with pytest.raises(ValueError, match="outside"):
            diagnostics.guard_training_window(
                {"partition": partition, "window_start": start, "window_end": start + 900}, plan
            )


def test_lag_direction_and_constant_series_are_reported_honestly():
    leader = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
    follower = np.concatenate([np.zeros(2), leader[:-2]])
    assert diagnostics.correlation(leader[:-2], follower[2:]) == pytest.approx(1)
    assert diagnostics.correlation(follower[:-2], leader[2:]) < 0.5
    assert diagnostics.correlation(np.zeros(10), np.zeros(10)) is None


def test_empty_quote_is_captured_data_but_cannot_be_traded():
    grid = np.array([0, 50, 100])
    result = diagnostics.asof_book(
        np.array([0, 50, 100]),
        np.array([0.4, np.nan, np.nan]),
        np.array([0.42, np.nan, np.nan]),
        np.array([True, False, False]),
        grid,
        100,
    )
    assert result["fresh"].tolist() == [True, True, True]
    assert result["valid"].tolist() == [True, False, False]


def test_full_capture_quality_survives_nontradable_expiry():
    plan = {
        "causality": {"decision_interval_seconds": 0.125},
        "quality": {
            "full15m_coverage_fraction_min": 0.8,
            "first_captured_elapsed_seconds_max": 5,
            "last_captured_elapsed_seconds_min": 897,
            "max_capture_gap_seconds": 5,
        },
        "trial_policy": {"training_diagnostic_lags_seconds": [0.125, 0.5, 1, 2, 5, 10]},
        "families": [
            {"diagnostic_leader_move_min": 0.05, "diagnostic_follower_move_max": 0.02},
            {"diagnostic_raw_gap_min": 0.1},
            {"diagnostic_both_move_min": 0.02},
        ],
    }
    length = 7200
    valid = np.arange(length) < 4000
    book = {
        "fresh": np.ones(length, dtype=bool),
        "valid": valid,
        "mid": np.where(valid, 0.5, np.nan),
        "rows": length,
    }
    books = {(coin, side): book for coin in ["btc", "eth"] for side in ["up", "down"]}
    result = diagnostics.window_diagnostics(
        {"window_start": 1791158400, "window_end": 1791159300, "native_replay_single_source": True},
        books,
        plan,
        {},
    )
    assert result["eligible_full15m"] is True
    assert result["paired_coverage_fraction"] == 1
    assert result["all_four_tradable_fraction"] == pytest.approx(4000 / 7200)
    assert all(event is None for event in result["family_diagnostic_first_events"].values())
