"""Tests for MarketDataViewSource — the _BookSource adapter over MarketDataView.

The adapter is what the matching engine + discovery grid consume. It must
present the engine's _BookSource protocol (snapshot_at + iter_snapshots) and
faithfully delegate to the view's point-in-time book access.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.book_source import MarketDataViewSource
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.view import MarketDataView


def _write(path, token_id, ticks):
    n = len(ticks)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array([t[0] for t in ticks], pa.int64()),
            "sequence": pa.array(list(range(n)), pa.int64()),
            "best_bid": pa.array([t[1] for t in ticks], pa.float64()),
            "best_ask": pa.array([t[2] for t in ticks], pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[t[1], t[1] - 0.01] for t in ticks], pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0, 5.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[t[2], t[2] + 0.01] for t in ticks], pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0, 5.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )
    pq.write_table(table, str(path))


def _view(tmp_path, token_ticks, base, span_us):
    by_token = {}
    for tok, ticks in token_ticks.items():
        f = tmp_path / f"snapshots__{tok}.parquet"
        _write(f, tok, ticks)
        by_token[tok] = TokenCoverage(tok, files=(str(f),), start_us=base, end_us=base + span_us)
    cov = CoverageMap(by_token=by_token, requested=tuple(token_ticks), window_start_us=base, window_end_us=base + span_us)
    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + span_us) / 1e6, tz=timezone.utc)
    return MarketDataView(coverage=cov, start=start, end=end)


@pytest.mark.asyncio
async def test_snapshot_at_point_in_time(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {"up": [(base, 0.40, 0.42), (base + 10_000_000, 0.55, 0.57)]}, base, 60_000_000)
    src = MarketDataViewSource(view)

    def at(sec):
        return datetime.fromtimestamp((base + sec * 1_000_000) / 1e6, tz=timezone.utc)

    assert await src.snapshot_at(token_id="up", ts=at(-1)) is None      # before first
    s0 = await src.snapshot_at(token_id="up", ts=at(5))
    assert s0 is not None and s0.bids[0].price == 0.40                  # full ladder preserved
    assert len(s0.bids) == 2 and len(s0.asks) == 2
    s1 = await src.snapshot_at(token_id="up", ts=at(15))
    assert s1 is not None and s1.bids[0].price == 0.55


@pytest.mark.asyncio
async def test_iter_snapshots_global_order(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {
        "up": [(base + 0, 0.40, 0.42), (base + 20_000_000, 0.45, 0.47)],
        "down": [(base + 10_000_000, 0.58, 0.60), (base + 30_000_000, 0.55, 0.57)],
    }, base, 60_000_000)
    src = MarketDataViewSource(view)
    times = [int(s.observed_at.timestamp() * 1e6) async for s in src.iter_snapshots()]
    assert times == sorted(times)
    assert len(times) == 4


@pytest.mark.asyncio
async def test_truncation_surface_present(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {"up": [(base, 0.4, 0.42)]}, base, 60_000_000)
    src = MarketDataViewSource(view)
    # The engine reads .truncated on the book source; the adapter exposes it.
    assert src.truncated is False
    assert src.truncation_reason is None


@pytest.mark.asyncio
async def test_execution_stream_memory_tracks_overlapping_history(tmp_path):
    import tracemalloc

    base = 1_700_000_000_000_000
    token_ticks = {
        f"token_{i}": [(base + i * 1_000_000 + j, 0.4, 0.42) for j in range(512)]
        for i in range(32)
    }
    view = _view(tmp_path, token_ticks, base, 33_000_000)
    source = MarketDataViewSource(view)
    stream = source.iter_snapshots()
    tracemalloc.start()
    try:
        first = await anext(stream)
        _, peak = tracemalloc.get_traced_memory()
        assert first.token_id == "token_0"
        assert peak < 4 * 1024 * 1024
    finally:
        await stream.aclose()
        tracemalloc.stop()
    assert view.loaded_token_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("statistics", [True, False])
async def test_lazy_merge_matches_eager_history_with_unsorted_overlapping_files(tmp_path, statistics):
    from services.marketdata.book import load_book_series, us_from_dt

    base = 1_700_000_000_000_000
    files = []
    for name, token, ticks in [
        ("a1", "a", [(base + 30, 0.43, 0.45), (base + 10, 0.40, 0.42), (base - 10, 0.3, 0.32)]),
        ("a2", "a", [(base + 10, 0.41, 0.43), (base + 20, 0.42, 0.44), (base + 110, 0.5, 0.52)]),
        ("b", "b", [(base + 20, 0.55, 0.57), (base + 10, 0.56, 0.58)]),
    ]:
        path = tmp_path / f"{name}.parquet"
        _write(path, token, ticks)
        table = pq.read_table(path)
        pq.write_table(table, path, row_group_size=2, write_statistics=statistics)
        files.append(str(path))
    coverage = CoverageMap(
        by_token={
            "b": TokenCoverage("b", files=(files[2],)),
            "a": TokenCoverage("a", files=tuple(files[:2])),
        },
        requested=("b", "a"),
    )
    start = datetime.fromtimestamp(base / 1e6, tz=timezone.utc)
    end = datetime.fromtimestamp((base + 100) / 1e6, tz=timezone.utc)
    view = MarketDataView(coverage=coverage, start=start, end=end)
    expected = []
    for token in coverage.covered_tokens:
        series, _ = load_book_series(token, coverage.files_for(token), start_us=base, end_us=base + 100)
        expected.extend(snapshot for _, snapshot in series.iter_range(base, base + 100))
    expected.sort(key=lambda snapshot: snapshot.observed_at)
    source = MarketDataViewSource(view)
    actual = [snapshot async for snapshot in source.iter_snapshots()]
    def fields(snapshot):
        return (
            snapshot.token_id, snapshot.observed_at, snapshot.sequence,
            snapshot.spread_bps, snapshot.trade_price, snapshot.trade_size,
            snapshot.trade_side, snapshot.best_bid, snapshot.best_ask,
            snapshot.bids, snapshot.asks,
        )

    assert [fields(snapshot) for snapshot in actual] == [fields(snapshot) for snapshot in expected]
    assert [snapshot.sequence for snapshot in actual if snapshot.token_id == "a"] == [1, 0, 1, 0]
    assert all(len(snapshot.bids) == 2 and len(snapshot.asks) == 2 for snapshot in actual)
    assert view.loaded_token_count() == 0
    assert view.dataset_snapshot().total_rows == 2
    as_of = await source.snapshot_at(token_id="a", ts=datetime.fromtimestamp((base + 15) / 1e6, tz=timezone.utc))
    assert us_from_dt(as_of.observed_at) == base + 10
    assert as_of.best_bid == 0.41


@pytest.mark.asyncio
async def test_closing_execution_stream_releases_overlapping_histories(tmp_path):
    base = 1_700_000_000_000_000
    view = _view(tmp_path, {
        "a": [(base, 0.4, 0.42), (base + 20, 0.41, 0.43)],
        "b": [(base, 0.5, 0.52), (base + 20, 0.51, 0.53)],
    }, base, 100)
    stream = MarketDataViewSource(view).iter_snapshots()
    await anext(stream)
    await stream.aclose()
    assert view.loaded_token_count() == 0
    assert view.dataset_snapshot().total_rows == 2
