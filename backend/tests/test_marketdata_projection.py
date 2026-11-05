"""Tests for the crypto_update projection (Phase 2b).

Covers the pure reconstruction logic (timestamp parsing, market liveness, market-dict
shape + preservation of missing sides) and an end-to-end projection over a
hand-built MarketDataView (no DB) asserting event shape + gap-fill exclusion.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from services.external_data.parquet_schema import SNAPSHOT_SCHEMA
from services.marketdata.coverage import CoverageMap, TokenCoverage
from services.marketdata.projection import (
    ProjectedMarket,
    _market_dict,
    _parse_iso_us,
    project_crypto_update_events,
)
from services.marketdata.view import MarketDataView


def _us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def test_parse_iso_us():
    dt = datetime(2026, 5, 28, 19, 13, tzinfo=timezone.utc)
    assert _parse_iso_us(dt.isoformat().replace("+00:00", "Z")) == _us(dt)
    assert _parse_iso_us(None) is None
    assert _parse_iso_us("nope") is None


def _market():
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    return ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="btc-updown-5m-1",
        title="BTC Up or Down", coin="btc", timeframe="5m",
        start_us=_us(base), end_us=_us(base) + 300_000_000,
        up_token="UP", down_token="DOWN", price_to_beat=80000.0,
    )


def test_market_alive_at_and_key():
    m = _market()
    base = _us(datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc))
    assert m.market_key() == "0xabc"
    assert m.alive_at(base) is True
    assert m.alive_at(base - 1) is False
    assert m.alive_at(base + 300_000_000) is False  # end exclusive
    assert set(m.tokens) == {"UP", "DOWN"}


def test_market_dict_two_sided():
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    md = _market_dict(m, tick=tick, up_bid=0.60, up_ask=0.62, down_bid=0.38, down_ask=0.40)
    assert md is not None
    assert md["clob_token_ids"] == ["UP", "DOWN"]
    assert md["up_price"] == pytest.approx(0.61)
    assert md["down_price"] == pytest.approx(0.39)
    assert md["seconds_left"] == pytest.approx(240, abs=2)
    assert md["is_live"] is True
    assert md["condition_id"] == "0xabc"


def test_market_dict_preserves_missing_side() -> None:
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    # An absent DOWN book must not become a synthetic executable quote.
    md = _market_dict(m, tick=tick, up_bid=0.60, up_ask=0.62, down_bid=None, down_ask=None)
    assert md is not None
    assert md["down_price"] is None


def test_market_dict_none_without_book():
    m = _market()
    tick = datetime(2026, 5, 28, 21, 1, tzinfo=timezone.utc)
    assert _market_dict(m, tick=tick, up_bid=None, up_ask=None, down_bid=None, down_ask=None) is None


# ── end-to-end projection over a hand-built view (no DB) ────────────────
def _write_snapshot_file(path, token_id, ticks):
    n = len(ticks)
    table = pa.table(
        {
            "token_id": pa.array([token_id] * n, pa.string()),
            "observed_at_us": pa.array([t[0] for t in ticks], pa.int64()),
            "sequence": pa.array(list(range(n)), pa.int64()),
            "best_bid": pa.array([t[1] for t in ticks], pa.float64()),
            "best_ask": pa.array([t[2] for t in ticks], pa.float64()),
            "spread_bps": pa.array([None] * n, pa.float64()),
            "bids_price": pa.array([[t[1]] for t in ticks], pa.list_(pa.float64())),
            "bids_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "asks_price": pa.array([[t[2]] for t in ticks], pa.list_(pa.float64())),
            "asks_size": pa.array([[10.0]] * n, pa.list_(pa.float64())),
            "trade_price": pa.array([None] * n, pa.float64()),
            "trade_size": pa.array([None] * n, pa.float64()),
            "trade_side": pa.array([None] * n, pa.string()),
        },
        schema=SNAPSHOT_SCHEMA,
    )
    pq.write_table(table, str(path))


@pytest.mark.asyncio
async def test_project_crypto_update_events_end_to_end(tmp_path):
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    base_us = _us(base)
    # UP/DOWN book files, 1s cadence for 10s
    up_ticks = [(base_us + i * 1_000_000, 0.60, 0.62) for i in range(10)]
    dn_ticks = [(base_us + i * 1_000_000, 0.38, 0.40) for i in range(10)]
    fu = tmp_path / "snapshots__UP.parquet"
    fd = tmp_path / "snapshots__DOWN.parquet"
    _write_snapshot_file(fu, "UP", up_ticks)
    _write_snapshot_file(fd, "DOWN", dn_ticks)

    start = base
    end = datetime.fromtimestamp((base_us + 10_000_000) / 1e6, tz=timezone.utc)
    cov = CoverageMap(
        by_token={
            "UP": TokenCoverage("UP", files=(str(fu),), start_us=base_us, end_us=base_us + 10_000_000),
            "DOWN": TokenCoverage("DOWN", files=(str(fd),), start_us=base_us, end_us=base_us + 10_000_000),
        },
        requested=("UP", "DOWN"), window_start_us=base_us, window_end_us=base_us + 10_000_000,
    )
    view = MarketDataView(coverage=cov, start=start, end=end)

    m = ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="btc-updown-5m-1",
        title="BTC Up or Down", coin="btc", timeframe="5m",
        start_us=base_us, end_us=base_us + 300_000_000,
        up_token="UP", down_token="DOWN", price_to_beat=80000.0,
    )

    events, stats = await project_crypto_update_events(
        start=start, end=end, cadence_seconds=2.0, view=view, markets=[m],
    )
    assert stats["events"] > 0
    ev0 = events[0]
    md = ev0.payload["markets"][0]
    assert md["condition_id"] == "0xabc"
    assert md["up_price"] == pytest.approx(0.61)
    assert md["down_price"] == pytest.approx(0.39)
    # cadence 2s over a 10s window -> ~5-6 events
    assert 4 <= stats["events"] <= 6


@pytest.mark.asyncio
async def test_projection_excludes_recorded_markets(tmp_path):
    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    base_us = _us(base)
    fu = tmp_path / "snapshots__UP.parquet"
    _write_snapshot_file(fu, "UP", [(base_us, 0.6, 0.62)])
    cov = CoverageMap(
        by_token={"UP": TokenCoverage("UP", files=(str(fu),), start_us=base_us, end_us=base_us + 5_000_000)},
        requested=("UP",), window_start_us=base_us, window_end_us=base_us + 5_000_000,
    )
    view = MarketDataView(coverage=cov, start=base, end=datetime.fromtimestamp((base_us + 5_000_000) / 1e6, tz=timezone.utc))
    m = ProjectedMarket(
        market_id="111", condition_id="0xabc", slug="s", title="t", coin="btc", timeframe="5m",
        start_us=base_us, end_us=base_us + 300_000_000, up_token="UP", down_token="DOWN", price_to_beat=None,
    )
    events, stats = await project_crypto_update_events(
        start=base, end=datetime.fromtimestamp((base_us + 5_000_000) / 1e6, tz=timezone.utc),
        cadence_seconds=1.0, view=view, markets=[m], exclude_market_keys={"0xabc"},
    )
    assert stats["markets_active"] == 0
    assert stats["events"] == 0


@pytest.mark.asyncio
async def test_scanner_projection_preserves_ladders_and_clears_empty_books() -> None:
    from datetime import timedelta
    from types import SimpleNamespace
    from typing import Any

    from services import strategy_sdk
    from services.backtest.book_replay import BookSnapshot
    from services.marketdata.projection import project_market_data_refresh_events
    from services.strategy_sdk import StrategySDK
    from utils import utcnow as clock

    base = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)

    async def book_at(token_id: str, tick: datetime, **kwargs: Any) -> BookSnapshot:
        if tick > base or token_id == "DOWN":
            return BookSnapshot(token_id, tick)
        return BookSnapshot(token_id, tick, bids_raw=((0.49, 50),), asks_raw=((0.5, 2), (0.6, 3)))

    events, _ = await project_market_data_refresh_events(
        start=base, end=base + timedelta(seconds=1), cadence_seconds=1,
        markets=[_market()], view=SimpleNamespace(book_at=book_at),
    )
    assert len(events) == 2
    assert events[0].markets[0]["down_price"] is None
    assert events[0].prices["UP"]["asks"] == ((0.5, 2), (0.6, 3))
    market = SimpleNamespace(clob_token_ids=["UP"])
    old_clock = clock.set_replay_clock_us(_us(base))
    old_books = strategy_sdk.set_replay_books(events[0].prices)
    try:
        depth = StrategySDK.get_order_book_depth(market, size_usd=2.8)
        assert depth["vwap_price"] == pytest.approx(2.8 / 5)
        from services.strategy_inputs import hydrate_market

        first_market = hydrate_market(events[0].markets[0])
        assert first_market.outcome_prices == [0.495, 0.0]
        assert StrategySDK.get_live_price(first_market, events[0].prices, "YES") == pytest.approx(0.495)
        assert StrategySDK.get_live_price(first_market, events[0].prices, "NO") == 0.0
        empty_market = hydrate_market(events[1].markets[0])
        assert empty_market.outcome_prices == [0.0, 0.0]
        assert events[1].prices["UP"]["asks"] == ()
        assert events[1].prices["DOWN"]["bids"] == ()
        strategy_sdk.set_replay_books(events[1].prices)
        assert StrategySDK.get_live_price(first_market, events[0].prices, "YES") == 0.0
        assert StrategySDK.get_live_price(first_market, events[0].prices, "NO") == 0.0
        assert StrategySDK.get_best_bid_ask(market) is None
        assert StrategySDK.get_order_book_depth(market) is None
    finally:
        strategy_sdk.restore_replay_books(old_books)
        clock.restore_replay_clock(old_clock)


@pytest.mark.parametrize(
    "quotes,expected",
    [
        ((0.49, 0.50, 0.30, 0.40), [0.495, 0.35]),
        ((0.49, 0.50, None, None), [0.495, 0.0]),
        ((None, None, 0.30, 0.40), [0.0, 0.35]),
        ((None, 0.50, None, None), [0.0, 0.0]),
        ((0.49, None, None, None), [0.0, 0.0]),
        ((None, None, None, None), [0.0, 0.0]),
    ],
)
def test_projected_catalog_hydrates_independent_prices_without_complements(
    quotes: tuple[float | None, ...], expected: list[float]
) -> None:
    from services.marketdata.projection import _catalog_market_dict
    from services.strategy_inputs import hydrate_market

    tick = datetime(2026, 5, 28, 21, 0, tzinfo=timezone.utc)
    row = _catalog_market_dict(_market(), tick=tick, up_bid=quotes[0], up_ask=quotes[1], down_bid=quotes[2], down_ask=quotes[3])
    market = hydrate_market(row)
    assert market.outcome_prices == expected
    assert [token.price for token in market.tokens] == expected
    assert market.end_date == datetime(2026, 5, 28, 21, 5, tzinfo=timezone.utc)
