"""Chronological training controls for BTC/ETH; no orders, fills or portfolio PnL."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import btc_eth_lead_lag_diagnostics as diagnostics


def previous(values, ticks):
    result = np.full(len(values), np.nan)
    result[ticks:] = values[:-ticks]
    return result


def moments(features, target):
    count = len(target)
    return {
        "rows": count,
        "gram": features.T @ features / count,
        "cross": features.T @ target / count,
        "square": float(target @ target / count),
    }


def forecast_fits(prior, day, protocol):
    fitted = {}
    for key, stats in prior.items():
        if stats["last_day"] >= day:
            raise ValueError("Forecast fit includes current or future UTC day")
        if len(stats["dates"]) < 2 or stats["windows"] < protocol["minimum_prior_fit_windows"]:
            continue
        models = {}
        for name, columns in protocol["forecast_controls"]["models"].items():
            gram = stats["gram"][np.ix_(columns, columns)]
            if np.linalg.matrix_rank(gram) != len(columns):
                break
            coefficients = np.linalg.solve(gram, stats["cross"][columns])
            models[name] = (columns, coefficients)
        if len(models) == len(protocol["forecast_controls"]["models"]):
            fitted[key] = {"models": models, "last_fit_day": stats["last_day"], "fit_windows": stats["windows"]}
    return fitted


def residual_fits(prior, day, protocol):
    fitted = {}
    for key, stats in prior.items():
        if stats["last_day"] >= day:
            raise ValueError("Residual fit includes current or future UTC day")
        if len(stats["dates"]) < 2 or stats["windows"] < protocol["minimum_prior_fit_windows"]:
            continue
        count, sx, sy, sxx, sxy, syy = stats["sums"]
        xx = sxx - sx * sx / count
        xy = sxy - sx * sy / count
        yy = syy - sy * sy / count
        if xx <= 0 or count <= 2:
            continue
        beta = xy / xx
        alpha = (sy - beta * sx) / count
        variance = (yy - beta * xy) / (count - 2)
        if variance <= 1e-16:
            continue
        fitted[key] = {
            "alpha": float(alpha),
            "beta": float(beta),
            "sigma": float(np.sqrt(variance)),
            "last_fit_day": stats["last_day"],
            "fit_windows": stats["windows"],
        }
    return fitted


def event_record(window, target, group, mask, direction, books, step, horizons, fit_day=None):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    index = int(indices[0])
    side = "up" if direction[index] > 0 else "down"
    book = books[(target, side)]
    response = {}
    for horizon in horizons:
        later = index + round(horizon / step)
        response[str(horizon)] = (
            float(book["mid"][later] - book["mid"][index]) if later < len(mask) and book["valid"][later] else None
        )
    up = float(books[(target, "up")]["mid"][index])
    spread = float(book["ask"][index] - book["bid"][index])
    return {
        "window_start": window["window_start"],
        "day": datetime.fromtimestamp(window["window_start"], UTC).date().isoformat(),
        "target": target,
        "group": group,
        "side": side,
        "elapsed_seconds": index * step,
        "time_bucket": int(index * step // 300),
        "probability_band": int(np.searchsorted([0.3, 0.7], up, side="right")),
        "spread_band": int(np.searchsorted([0.02, 0.05], spread, side="right")),
        "current_spread": spread,
        "last_fit_day": fit_day,
        "midpoint_responses": response,
    }


def day_interval(totals, protocol):
    days = sorted(totals)
    values = np.asarray([totals[day] for day in days], dtype=float)
    if not len(days):
        return {"days": 0, "estimate": None, "bootstrap_interval": None}
    estimate = float(values[:, 0].sum() / values[:, 1].sum())
    interval = None
    if len(days) >= 3:
        rng = np.random.default_rng(protocol["uncertainty"]["seed"])
        sampled = values[rng.integers(0, len(days), (protocol["uncertainty"]["bootstrap_draws"], len(days)))].sum(
            axis=1
        )
        estimates = sampled[:, 0] / sampled[:, 1]
        interval = np.percentile(estimates, protocol["uncertainty"]["percentiles"]).tolist()
    return {
        "days": len(days),
        "estimate": estimate,
        "bootstrap_interval": interval,
        "positive_days": int(sum(value[0] > 0 for value in totals.values())),
        "daily": {
            day: {"estimate": float(value[0] / value[1]), "weight": float(value[1])} for day, value in totals.items()
        },
    }


def summarize_forecasts(rows, protocol):
    output = {}
    for target in ["btc", "eth"]:
        for horizon in protocol["horizons_seconds"]:
            selected = [row for row in rows if row["target"] == target and row["horizon_seconds"] == horizon]
            key = f"{target}:{horizon}"
            entry = {"windows": len(selected), "models": {}, "lift_vs_controls": {}}
            for model in protocol["forecast_controls"]["models"]:
                totals = defaultdict(lambda: np.zeros(2))
                for row in selected:
                    totals[row["day"]] += [row["mse"][model], 1]
                entry["models"][model] = day_interval(totals, protocol)
            for control in ["own_momentum", "past_shifted_leader"]:
                totals = defaultdict(lambda: np.zeros(2))
                for row in selected:
                    totals[row["day"]] += [row["mse"][control] - row["mse"]["cross_coin"], 1]
                entry["lift_vs_controls"][control] = day_interval(totals, protocol)
            output[key] = entry
    return output


def matched_events(events, protocol):
    output = {}
    comparisons = [
        ("F2_adjusted_residual", "F2_raw_gap"),
        ("F3_confirmation", "F3_own_momentum"),
        ("F3_confirmation", "F3_disagreement"),
    ]
    for treatment, control in comparisons:
        for horizon in protocol["horizons_seconds"]:
            strata = defaultdict(lambda: {treatment: [], control: []})
            selected_total = 0
            for event in events:
                if event["group"] not in [treatment, control]:
                    continue
                value = event["midpoint_responses"][str(horizon)]
                if value is None:
                    continue
                selected_total += 1
                key = tuple(event[name] for name in ["day", "target", "time_bucket", "probability_band", "spread_band"])
                strata[key][event["group"]].append((event["window_start"], value))
            totals = defaultdict(lambda: np.zeros(2))
            windows = set()
            common = 0
            matched_observations = 0
            for key, groups in strata.items():
                left, right = groups[treatment], groups[control]
                if not left or not right:
                    continue
                weight = min(len(left), len(right))
                difference = float(np.mean([value for _, value in left]) - np.mean([value for _, value in right]))
                totals[key[0]] += [difference * weight, weight]
                windows.update(start for start, _ in left + right)
                common += 1
                matched_observations += len(left) + len(right)
            output[f"{treatment}_vs_{control}:{horizon}"] = {
                **day_interval(totals, protocol),
                "common_strata": common,
                "unique_windows_in_common_strata": len(windows),
                "usable_group_observations": selected_total,
                "observations_in_common_strata": matched_observations,
                "units": "difference in actual chosen token midpoint response;no fills,costs orPnL",
            }
    return output


def aggregate_lag_uncertainty(rows, protocol):
    result = {}
    for direction in ["btc_to_eth", "eth_to_btc"]:
        for horizon in protocol["horizons_seconds"]:
            totals = defaultdict(lambda: np.zeros(2))
            for row in rows:
                value = row["lag_correlations"][direction].get(str(horizon))
                if value is None:
                    continue
                day = datetime.fromtimestamp(row["window_start"], UTC).date().isoformat()
                totals[day] += [value, 1]
            result[f"{direction}:{horizon}"] = day_interval(totals, protocol)
    return result


def run(job_path):
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = Path(job["output_directory"])
    status_path = output / (job["job_id"] + "_status.json")
    try:
        loaded = {}
        for key in ["training_request", "diagnostics_result", "controls_protocol", "base_protocol"]:
            path = Path(job[key + "_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != job[key + "_sha256"]:
                raise ValueError(f"Frozen {key} fingerprint changed")
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != job["analysis_source_sha256"]:
            raise ValueError("Controls source fingerprint changed")
        if hashlib.sha256(Path(diagnostics.__file__).read_bytes()).hexdigest() != job["base_source_sha256"]:
            raise ValueError("Base diagnostic source fingerprint changed")
        request, baseline = loaded["training_request"], loaded["diagnostics_result"]
        protocol, plan = loaded["controls_protocol"], loaded["base_protocol"]
        if baseline["plan_sha256"] != request["plan_sha256"] or request["plan_sha256"] != job["base_protocol_sha256"]:
            raise ValueError("Training request and diagnostic protocol differ")
        if (
            baseline["analysis_source_sha256"] != job["base_source_sha256"]
            or baseline["manifest_sha256"] != request["manifest_sha256"]
        ):
            raise ValueError("Training diagnostic fingerprints differ")
        for window in request["windows"]:
            diagnostics.guard_training_window(window, plan)
        eligible_rows = [row for row in baseline["windows"] if row["eligible_full15m"] and row["native_single_source"]]
        eligible = {row["window_start"] for row in eligible_rows}
        windows = sorted(
            [window for window in request["windows"] if window["window_start"] in eligible],
            key=lambda item: item["window_start"],
        )
        if len(windows) != len(eligible) or len(eligible_rows) != len(eligible):
            raise ValueError("Duplicate or missing control windows")
        step = protocol["decision_interval_seconds"]
        lookback = round(protocol["lookback_seconds"] / step)
        shifted = round(protocol["shifted_leader_seconds"] / step)
        prior_forecast, prior_residual = {}, {}
        fitted, residual_models = {}, {}
        forecasts, events, residual_models_by_day = [], [], {}
        current_day = None
        warmup_end = (
            datetime.fromisoformat(plan["data"]["training_start"]).timestamp()
            + protocol["warmup_calendar_days"] * 86400
        )
        horizons = protocol["horizons_seconds"]
        for count, window in enumerate(windows, 1):
            day = datetime.fromtimestamp(window["window_start"], UTC).date().isoformat()
            if day != current_day:
                current_day = day
                fitted = forecast_fits(prior_forecast, day, protocol) if window["window_start"] >= warmup_end else {}
                residual_models = (
                    residual_fits(prior_residual, day, protocol) if window["window_start"] >= warmup_end else {}
                )
                residual_models_by_day[day] = {str(key): model for key, model in residual_models.items()}
            grid = np.arange(
                window["window_start"] * 1_000_000, window["window_end"] * 1_000_000, round(step * 1e6), dtype=np.int64
            )
            books = {
                (item["coin"], item["side"]): diagnostics.read_book(
                    item, request["files"][item["file"]], grid, round(plan["causality"]["max_quote_age_seconds"] * 1e6)
                )
                for item in window["canonical_tokens"]
            }
            all_valid = np.logical_and.reduce([book["valid"] for book in books.values()])
            up = {coin: books[(coin, "up")]["mid"] for coin in ["btc", "eth"]}
            changes = {coin: up[coin] - previous(up[coin], lookback) for coin in up}
            present = all_valid.copy()
            for coin in up:
                present &= (up[coin] >= protocol["probability_min"]) & (up[coin] <= protocol["probability_max"])
            for target, leader in [("btc", "eth"), ("eth", "btc")]:
                features = np.column_stack(
                    [np.ones(len(grid)), changes[target], changes[leader], previous(changes[leader], shifted)]
                )
                feature_valid = present & np.isfinite(features).all(axis=1)
                for horizon in horizons:
                    ticks = round(horizon / step)
                    future = np.full(len(grid), np.nan)
                    future[:-ticks] = up[target][ticks:] - up[target][:-ticks]
                    mask = feature_valid & np.isfinite(future)
                    if not np.any(mask):
                        continue
                    moment = moments(features[mask], future[mask])
                    key = (target, horizon)
                    if key in fitted:
                        fit = fitted[key]
                        mse = {}
                        for model, (columns, coefficients) in fit["models"].items():
                            square = (
                                moment["square"]
                                - 2 * coefficients @ moment["cross"][columns]
                                + coefficients @ moment["gram"][np.ix_(columns, columns)] @ coefficients
                            )
                            if square < -1e-12:
                                raise ValueError("Negative forecast squared error from sufficient moments")
                            mse[model] = float(max(0, square))
                        forecasts.append(
                            {
                                "window_start": window["window_start"],
                                "day": day,
                                "target": target,
                                "horizon_seconds": horizon,
                                "rows": moment["rows"],
                                "last_fit_day": fit["last_fit_day"],
                                "fit_windows": fit["fit_windows"],
                                "mse": mse,
                            }
                        )
                    if key not in prior_forecast:
                        prior_forecast[key] = {
                            "gram": np.zeros((4, 4)),
                            "cross": np.zeros(4),
                            "windows": 0,
                            "dates": set(),
                        }
                    stats = prior_forecast[key]
                    stats["gram"] += moment["gram"]
                    stats["cross"] += moment["cross"]
                    stats["windows"] += 1
                    stats["dates"].add(day)
                    stats["last_day"] = day
                own, other = changes[target], changes[leader]
                active = present & np.isfinite(own) & np.isfinite(other)
                trend = np.abs(own) >= plan["families"][2]["diagnostic_both_move_min"]
                other_moves = np.abs(other) >= plan["families"][2]["diagnostic_both_move_min"]
                for group, condition in [
                    ("F3_own_momentum", trend),
                    ("F3_confirmation", trend & other_moves & (own * other > 0)),
                    ("F3_disagreement", trend & other_moves & (own * other < 0)),
                ]:
                    event = event_record(window, target, group, active & condition, np.sign(own), books, step, horizons)
                    if event is not None:
                        events.append(event)
            with np.errstate(divide="ignore", invalid="ignore"):
                logs = {coin: np.log(value / (1 - value)) for coin, value in up.items()}
            volatility = np.maximum(
                np.abs(logs["btc"] - previous(logs["btc"], lookback)),
                np.abs(logs["eth"] - previous(logs["eth"], lookback)),
            )
            time_bucket = np.floor(np.arange(len(grid)) * step / 300).astype(int)
            volatility_bucket = np.searchsorted(
                protocol["residual_controls"]["volatility_edges"], volatility, side="right"
            )
            zscore = np.full(len(grid), np.nan)
            base = present & np.isfinite(volatility)
            for time_group in range(3):
                for vol_group in range(3):
                    key = (time_group, vol_group)
                    mask = base & (time_bucket == time_group) & (volatility_bucket == vol_group)
                    if key in residual_models:
                        fit = residual_models[key]
                        zscore[mask] = (logs["eth"][mask] - fit["alpha"] - fit["beta"] * logs["btc"][mask]) / fit[
                            "sigma"
                        ]
                    sample = mask & (
                        np.arange(len(grid)) % round(protocol["residual_controls"]["sampling_fit_seconds"] / step) == 0
                    )
                    x, y = logs["btc"][sample], logs["eth"][sample]
                    if not len(x):
                        continue
                    if key not in prior_residual:
                        prior_residual[key] = {"sums": np.zeros(6), "windows": 0, "dates": set()}
                    stats = prior_residual[key]
                    stats["sums"] += [len(x), x.sum(), y.sum(), x @ x, x @ y, y @ y]
                    stats["windows"] += 1
                    stats["dates"].add(day)
                    stats["last_day"] = day
            residual_event = event_record(
                window,
                "eth",
                "F2_adjusted_residual",
                base & (np.abs(zscore) >= protocol["residual_controls"]["z_threshold"]),
                -np.sign(zscore),
                books,
                step,
                horizons,
                max((fit["last_fit_day"] for fit in residual_models.values()), default=None),
            )
            if residual_event is not None:
                events.append(residual_event)
            if window["window_start"] >= warmup_end:
                gap = up["eth"] - up["btc"]
                raw_event = event_record(
                    window,
                    "eth",
                    "F2_raw_gap",
                    base & (np.abs(gap) >= plan["families"][1]["diagnostic_raw_gap_min"]),
                    -np.sign(gap),
                    books,
                    step,
                    horizons,
                )
                if raw_event is not None:
                    events.append(raw_event)
            if count % 8 == 0 or count == len(windows):
                diagnostics.write_json(
                    status_path,
                    {
                        "job_id": job["job_id"],
                        "status": "running",
                        "phase": "chronological_training_controls",
                        "updated_at": datetime.now(UTC).isoformat(),
                        "windows_done": count,
                        "windows_total": len(windows),
                        "last_day": day,
                        "validation_files_read": 0,
                        "holdout_files_read": 0,
                    },
                )
        events_summary = {
            group: {
                "events": sum(event["group"] == group for event in events),
                "unique_windows": len({event["window_start"] for event in events if event["group"] == group}),
            }
            for group in sorted({event["group"] for event in events})
        }
        result = {
            "job_id": job["job_id"],
            "completed_at": datetime.now(UTC).isoformat(),
            "analysis_source_sha256": job["analysis_source_sha256"],
            "base_source_sha256": job["base_source_sha256"],
            "controls_protocol_sha256": job["controls_protocol_sha256"],
            "training_request_sha256": job["training_request_sha256"],
            "base_protocol_sha256": job["base_protocol_sha256"],
            "manifest_sha256": request["manifest_sha256"],
            "windows_analyzed": len(windows),
            "excluded_window_starts": [
                window["window_start"] for window in request["windows"] if window["window_start"] not in eligible
            ],
            "forecast_models": summarize_forecasts(forecasts, protocol),
            "event_counts": events_summary,
            "matched_event_comparisons": matched_events(events, protocol),
            "lag_day_uncertainty": aggregate_lag_uncertainty(eligible_rows, protocol),
            "forecast_windows": forecasts,
            "events": events,
            "residual_models_by_day": residual_models_by_day,
            "validation_files_read": 0,
            "holdout_files_read": 0,
            "native_fills_or_profit_computed": False,
            "limitations": [
                "Observed capture clock can differ from exchange clock.",
                "Midpoint responses and prediction errors are not executable net profits.",
                "Matching and own-momentum controls are observational, not randomized causal identification.",
                "Multiple horizons and variants are diagnostic probes, not independent candidate successes.",
                "Only14trainingdays;day bootstrap cannot establish long-regime robustness.",
                "First events can overlap acrossgroups;unique windows are never additive.",
            ],
        }
        result_path = output / (job["job_id"] + "_result.json")
        diagnostics.write_json(result_path, result)
        diagnostics.write_json(
            status_path,
            {
                "job_id": job["job_id"],
                "status": "completed",
                "updated_at": result["completed_at"],
                "windows_done": len(windows),
                "windows_total": len(windows),
                "result_path": str(result_path),
                "validation_files_read": 0,
                "holdout_files_read": 0,
            },
        )
    except Exception:
        diagnostics.write_json(
            status_path,
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
