"""Read-only BTC/ETH book diagnostics; all predictive evaluation stays in native backtests."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLUMNS = [
    "token_id",
    "observed_at_us",
    "best_bid",
    "best_ask",
    "bids_price",
    "bids_size",
    "asks_price",
    "asks_size",
]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def guard_training_window(window: dict, plan: dict) -> None:
    start = int(datetime.fromisoformat(plan["data"]["training_start"]).timestamp())
    end = int(
        datetime.fromisoformat(plan["data"]["training_end_exclusive"]).timestamp()
    )
    if (
        window["partition"] != "development"
        or not start <= window["window_start"] < window["window_end"] <= end
    ):
        raise ValueError(
            "Diagnostic input is outside the preregistered training partition"
        )


def read_book(item: dict, expected: dict, grid: np.ndarray, max_age_us: int) -> dict:
    path = Path(item["file"])
    stat = path.stat()
    if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
        raise ValueError(f"Frozen source changed: {path}")
    table = pq.read_table(
        path,
        columns=COLUMNS,
        filters=[("token_id", "=", str(item["token_id"]))],
        use_threads=False,
    )
    ts = table["observed_at_us"].to_numpy()
    if len(ts) != expected["rows"]:
        raise ValueError(f"Frozen row count changed: {path}")
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    bid = table["best_bid"].to_numpy()[order]
    ask = table["best_ask"].to_numpy()[order]
    valid = (
        np.isfinite(bid)
        & np.isfinite(ask)
        & (bid > 0)
        & (ask < 1)
        & (bid < ask)
        & (ts >= grid[0])
    )
    for side, reference in [("bids", bid), ("asks", ask)]:
        prices = table[side + "_price"].combine_chunks()
        sizes = table[side + "_size"].combine_chunks()
        offsets = prices.offsets.to_numpy()
        size_offsets = sizes.offsets.to_numpy()
        has = (np.diff(offsets) > 0) & (np.diff(size_offsets) > 0)
        top = np.full(len(table), np.nan)
        size = np.full(len(table), np.nan)
        top[has] = prices.values.to_numpy()[offsets[:-1][has]]
        size[has] = sizes.values.to_numpy()[size_offsets[:-1][has]]
        top, size = top[order], size[order]
        valid &= (
            np.isfinite(size)
            & (size > 0)
            & np.isclose(top, reference, atol=1e-9, rtol=0)
        )
    sampled = asof_book(ts, bid, ask, valid, grid, max_age_us)
    sampled["rows"] = len(table)
    return sampled


def asof_book(ts, bid, ask, valid, grid, max_age_us):
    out_bid = np.full(len(grid), np.nan)
    out_ask = np.full(len(grid), np.nan)
    visible = np.zeros(len(grid), dtype=bool)
    fresh = np.zeros(len(grid), dtype=bool)
    age = np.full(len(grid), np.nan)
    if len(ts):
        indices = np.searchsorted(ts, grid, side="right") - 1
        clipped = np.maximum(indices, 0)
        age = grid - ts[clipped]
        fresh = (
            (indices >= 0) & (age >= 0) & (age <= max_age_us) & (ts[clipped] >= grid[0])
        )
        visible = fresh & valid[clipped]
        out_bid[visible], out_ask[visible] = (
            bid[clipped[visible]],
            ask[clipped[visible]],
        )
    return {
        "bid": out_bid,
        "ask": out_ask,
        "mid": (out_bid + out_ask) / 2,
        "valid": visible,
        "fresh": fresh,
        "age_us": age,
    }


def correlation(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3:
        return None
    x, y = a[valid], b[valid]
    x, y = x - np.mean(x), y - np.mean(y)
    scale = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / scale, -1, 1)) if scale > 0 else None


def window_diagnostics(window: dict, books: dict, plan: dict, labels: dict) -> dict:
    step = plan["causality"]["decision_interval_seconds"]
    covered = np.logical_and.reduce([book["fresh"] for book in books.values()])
    all_tradable = np.logical_and.reduce([book["valid"] for book in books.values()])
    valid = books[("btc", "up")]["valid"] & books[("eth", "up")]["valid"]
    indices = np.flatnonzero(covered)
    first = float(indices[0] * step) if len(indices) else None
    last = float(indices[-1] * step) if len(indices) else None
    gap = (
        float(max(0, np.max(np.diff(indices)) - 1) * step) if len(indices) > 1 else None
    )
    quality = plan["quality"]
    eligible = bool(
        len(indices)
        and np.mean(covered) >= quality["full15m_coverage_fraction_min"]
        and first <= quality["first_captured_elapsed_seconds_max"]
        and last >= quality["last_captured_elapsed_seconds_min"]
        and gap is not None
        and gap <= quality["max_capture_gap_seconds"]
    )
    result = {
        "window_start": window["window_start"],
        "window_end": window["window_end"],
        "native_single_source": window["native_replay_single_source"],
        "eligible_full15m": eligible,
        "eligibility_basis": "captured feed coverage, including explicit empty books",
        "paired_coverage_fraction": float(np.mean(covered)),
        "all_four_tradable_fraction": float(np.mean(all_tradable)),
        "up_pair_feature_fraction": float(np.mean(valid)),
        "first_captured_elapsed_seconds": first,
        "last_captured_elapsed_seconds": last,
        "max_internal_capture_gap_seconds": gap,
        "five_minute_capture_fractions": [
            float(np.mean(part)) for part in np.array_split(covered, 3)
        ],
        "five_minute_all_four_tradable_fractions": [
            float(np.mean(part)) for part in np.array_split(all_tradable, 3)
        ],
        "input_rows": sum(book["rows"] for book in books.values()),
        "settlement_targets": labels,
        "lag_correlations": {},
        "family_diagnostic_first_events": {},
    }
    if not eligible:
        return result
    btc, eth = books[("btc", "up")]["mid"].copy(), books[("eth", "up")]["mid"].copy()
    btc[~valid], eth[~valid] = np.nan, np.nan
    result["up_mid_level_correlation"] = correlation(btc, eth)
    logits = [
        np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        for p in [btc, eth]
    ]
    increments = [np.diff(logit, prepend=np.nan) for logit in logits]
    for leader, follower, x, y in [
        ("btc", "eth", increments[0], increments[1]),
        ("eth", "btc", increments[1], increments[0]),
    ]:
        result["lag_correlations"][leader + "_to_" + follower] = {
            str(lag): correlation(x[: -round(lag / step)], y[round(lag / step) :])
            for lag in plan["trial_policy"]["training_diagnostic_lags_seconds"]
        }
    lookback = round(5 / step)
    changes = [
        np.concatenate(
            [np.full(lookback, np.nan), values[lookback:] - values[:-lookback]]
        )
        for values in [btc, eth]
    ]
    events = []
    f1 = plan["families"][0]
    for index, (leader, follower) in enumerate([("btc", "eth"), ("eth", "btc")]):
        mask = (
            (np.abs(changes[index]) >= f1["diagnostic_leader_move_min"])
            & (np.abs(changes[1 - index]) <= f1["diagnostic_follower_move_max"])
            & valid
        )
        direction_executable = np.where(
            changes[index] > 0,
            books[(follower, "up")]["valid"],
            books[(follower, "down")]["valid"],
        )
        mask &= direction_executable
        ticks = np.flatnonzero(mask)
        if len(ticks):
            tick = int(ticks[0])
            events.append(
                {
                    "elapsed_seconds": tick * step,
                    "leader": leader,
                    "follower": follower,
                    "direction": "up" if changes[index][tick] > 0 else "down",
                }
            )
    result["family_diagnostic_first_events"]["F1"] = (
        min(
            events,
            key=lambda event: (event["elapsed_seconds"], event["leader"] != "btc"),
        )
        if events
        else None
    )
    for family, mask in [
        (
            "F2",
            (np.abs(btc - eth) >= plan["families"][1]["diagnostic_raw_gap_min"])
            & valid,
        ),
        (
            "F3",
            (changes[0] * changes[1] > 0)
            & (
                np.minimum(np.abs(changes[0]), np.abs(changes[1]))
                >= plan["families"][2]["diagnostic_both_move_min"]
            )
            & valid,
        ),
    ]:
        ticks = np.flatnonzero(mask)
        result["family_diagnostic_first_events"][family] = (
            {"elapsed_seconds": float(ticks[0] * step)} if len(ticks) else None
        )
    return result


def summarize(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["eligible_full15m"]]
    levels = [
        row["up_mid_level_correlation"]
        for row in eligible
        if row["up_mid_level_correlation"] is not None
    ]
    lag_summary = {}
    for direction in ["btc_to_eth", "eth_to_btc"]:
        lag_summary[direction] = {}
        for lag in ["0.125", "0.5", "1", "2", "5", "10"]:
            values = [
                row["lag_correlations"][direction][lag]
                for row in eligible
                if row["lag_correlations"][direction][lag] is not None
            ]
            lag_summary[direction][lag] = {
                "windows": len(values),
                "window_weighted_mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
    targets = [
        row["settlement_targets"]
        for row in rows
        if all(
            row["settlement_targets"].get(coin) in ["up", "down"]
            for coin in ["btc", "eth"]
        )
    ]
    agreement = sum(target["btc"] == target["eth"] for target in targets)
    btc_up = sum(target["btc"] == "up" for target in targets)
    eth_up = sum(target["eth"] == "up" for target in targets)
    expected_agreement = (
        (btc_up / len(targets)) * (eth_up / len(targets))
        + (1 - btc_up / len(targets)) * (1 - eth_up / len(targets))
        if targets
        else None
    )
    return {
        "windows_processed": len(rows),
        "eligible_full15m_windows": len(eligible),
        "eligible_native_single_source_windows": sum(row["native_single_source"] for row in eligible),
        "ineligible_retained": len(rows) - len(eligible),
        "input_rows": sum(row["input_rows"] for row in rows),
        "level_correlation_window_mean": float(np.mean(levels)) if levels else None,
        "level_correlation_window_median": float(np.median(levels)) if levels else None,
        "lag_correlations": lag_summary,
        "family_diagnostic_signal_windows": {
            family: sum(
                row["family_diagnostic_first_events"].get(family) is not None
                for row in eligible
            )
            for family in ["F1", "F2", "F3"]
        },
        "settlement_agreement": {
            "known_paired_windows": len(targets),
            "unknown_paired_windows": len(rows) - len(targets),
            "same_direction_windows": agreement,
            "different_direction_windows": len(targets) - agreement,
            "same_direction_fraction": agreement / len(targets) if targets else None,
            "btc_up_windows": btc_up,
            "eth_up_windows": eth_up,
            "independent_marginal_agreement_baseline": expected_agreement,
        },
        "independent_count_unit": "window, with remaining economic dependence across adjacent windows",
        "native_fills_or_profit_computed": False,
        "strategy_candidates_frozen": 0,
        "significance_claimed": False,
    }


def run(job_path: Path) -> None:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = Path(job["output_directory"])
    progress_path = output / (job["job_id"] + "_status.json")
    try:
        plan_path = Path(job["plan_path"])
        if hashlib.sha256(plan_path.read_bytes()).hexdigest() != job["plan_sha256"]:
            raise ValueError("Research plan fingerprint changed")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != job["analysis_source_sha256"]
        ):
            raise ValueError("Diagnostic source fingerprint changed")
        for window in job["windows"]:
            guard_training_window(window, plan)
        rows = []
        step_us = round(plan["causality"]["decision_interval_seconds"] * 1e6)
        for window in job["windows"]:
            grid = np.arange(
                window["window_start"] * 1_000_000,
                window["window_end"] * 1_000_000,
                step_us,
                dtype=np.int64,
            )
            books = {
                (item["coin"], item["side"]): read_book(
                    item,
                    job["files"][item["file"]],
                    grid,
                    int(plan["causality"]["max_quote_age_seconds"] * 1e6),
                )
                for item in window["canonical_tokens"]
            }
            if set(books) != {
                (coin, side) for coin in ["btc", "eth"] for side in ["up", "down"]
            }:
                raise ValueError("Expected exactly four actual token books")
            rows.append(
                window_diagnostics(
                    window,
                    books,
                    plan,
                    job["settlement_labels"].get(str(window["window_start"]), {}),
                )
            )
            if len(rows) % 8 == 0 or len(rows) == len(job["windows"]):
                write_json(
                    progress_path,
                    {
                        "job_id": job["job_id"],
                        "status": "running",
                        "updated_at": datetime.now(UTC).isoformat(),
                        "windows_done": len(rows),
                        "windows_total": len(job["windows"]),
                        "last_window_start": window["window_start"],
                        "holdout_files_read": 0,
                    },
                )
        result = {
            "job_id": job["job_id"],
            "completed_at": datetime.now(UTC).isoformat(),
            "analysis_source_sha256": job["analysis_source_sha256"],
            "plan_sha256": job["plan_sha256"],
            "manifest_sha256": job["manifest_sha256"],
            "capture_clock_limitation": "Apparent lag can reflect provider sampling; this descriptive screen does not establish executable alpha.",
            "holdout_files_read": 0,
            "validation_files_read": 0,
            "summary": summarize(rows),
            "windows": rows,
        }
        result_path = output / (job["job_id"] + "_result.json")
        write_json(result_path, result)
        write_json(
            progress_path,
            {
                "job_id": job["job_id"],
                "status": "completed",
                "updated_at": result["completed_at"],
                "windows_done": len(rows),
                "windows_total": len(rows),
                "result_path": str(result_path),
                "holdout_files_read": 0,
                "validation_files_read": 0,
            },
        )
    except Exception:
        write_json(
            progress_path,
            {
                "job_id": job["job_id"],
                "status": "failed",
                "updated_at": datetime.now(UTC).isoformat(),
                "error_traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    run(parser.parse_args().job)
