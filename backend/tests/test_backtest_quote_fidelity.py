from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, AsyncIterator, Sequence
from unittest.mock import AsyncMock

import pytest

from services import strategy_sdk
from services.backtest import BacktestConfig, BacktestEngine, InMemoryBookReplay, LatencyModel, TradeIntent
from services.backtest.book_replay import BookSnapshot, PriceLevel
from services.backtest.latency_model import LatencyProfile
from services.backtest.matching_engine import Fill
from services.backtest.portfolio import Portfolio, Position
from services.strategies.base import ExitDecision
from services.strategy_backtester import _build_per_tick_prices_grid
from services.strategy_sdk import StrategySDK
from utils import utcnow as clock


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def snapshot(
    seconds: float, bids: Sequence[tuple[float, float]] = (), asks: Sequence[tuple[float, float]] = ()
) -> BookSnapshot:
    return BookSnapshot(
        token_id="token",
        observed_at=T0 + timedelta(seconds=seconds),
        bids=tuple(PriceLevel(*level) for level in bids),
        asks=tuple(PriceLevel(*level) for level in asks),
    )


async def price_grid(
    monkeypatch: pytest.MonkeyPatch, tape: list[BookSnapshot], ticks: list[float]
) -> dict[str, list[dict[str, Any] | None]]:
    from services.marketdata.book_source import MarketDataViewSource
    from services.marketdata.view import MarketDataView

    monkeypatch.setattr(MarketDataView, "build", AsyncMock(return_value=object()))

    async def iter_snapshots(self: MarketDataViewSource) -> AsyncIterator[BookSnapshot]:
        for item in tape:
            yield item

    monkeypatch.setattr(MarketDataViewSource, "iter_snapshots", iter_snapshots)
    return await _build_per_tick_prices_grid(
        token_ids=["token"],
        ticks=[T0 + timedelta(seconds=t) for t in ticks],
        start_dt=T0,
        end_dt=T0 + timedelta(seconds=10),
    )


@pytest.mark.asyncio
async def test_grid_preserves_empty_and_one_sided_states_without_lookahead(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = await price_grid(
        monkeypatch,
        [
            snapshot(0, [(0.96, 5)], [(0.97, 10)]),
            snapshot(0.129),
            snapshot(0.2, asks=[(0.98, 3)]),
            snapshot(0.3, [(0.97, 7)], [(0.98, 9)]),
        ],
        [0.1, 0.129, 0.15, 0.2, 0.25, 0.3],
    )
    states = grid["token"]
    assert states[0]["best_bid"] == 0.96
    for state in states[1:3]:
        assert state["observed_at"] == T0 + timedelta(seconds=0.129)
        assert state["best_bid"] == 0 and state["best_ask"] == 0
        assert state["bids"] == () and state["asks"] == ()
        assert state["bid_depth"] == 0 and state["ask_depth"] == 0
    for state in states[3:5]:
        assert state["best_bid"] == 0 and state["best_ask"] == 0.98
        assert state["bids"] == ()
    assert states[5]["best_bid"] == 0.97


@pytest.mark.asyncio
async def test_replay_sdk_uses_actual_ladder_for_levels_and_vwap(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = await price_grid(
        monkeypatch,
        [
            snapshot(0, [(0.49, 50)], [(0.5, 2), (0.6, 3)]),
            snapshot(1),
        ],
        [0, 1],
    )
    market = SimpleNamespace(clob_token_ids=["token"])
    old_clock = clock.set_replay_clock_us(int(T0.timestamp() * 1_000_000))
    previous = strategy_sdk.set_replay_books({"token": grid["token"][0]})
    try:
        assert StrategySDK.get_book_levels(market, max_levels=1) == [{"price": 0.5, "size": 2}]
        assert StrategySDK.get_book_levels(market, max_levels=10) == [
            {"price": 0.5, "size": 2},
            {"price": 0.6, "size": 3},
        ]
        assert StrategySDK.get_book_imbalance(market, levels=1) == pytest.approx((50 - 2) / 52)
        assert StrategySDK.get_book_imbalance(market, levels=2) == pytest.approx((50 - 5) / 55)
        estimate = StrategySDK.get_order_book_depth(market, size_usd=2.8)
        assert estimate["vwap_price"] == pytest.approx(2.8 / 5)
        assert estimate["available_liquidity"] == pytest.approx(2.8)
        assert estimate["slippage_bps"] > 0
        strategy_sdk.set_replay_books({"token": grid["token"][1]})
        assert StrategySDK.get_best_bid_ask(market) is None
        assert StrategySDK.get_book_levels(market) is None
        assert StrategySDK.get_order_book_depth(market) is None
    finally:
        strategy_sdk.restore_replay_books(previous)
        clock.restore_replay_clock(old_clock)


@pytest.mark.parametrize("delay", [0.5, 500, 1000, 2000, 20000])
def test_equal_latency_quantiles_are_constant_and_do_not_consume_rng(delay: float) -> None:
    import random

    profile = LatencyProfile.from_quantiles(p50_ms=delay, p95_ms=delay)
    rng = random.Random(42)
    initial_rng = rng.getstate()
    for _ in range(50):
        assert profile.sample(rng) == pytest.approx(delay, abs=1e-9)
    assert rng.getstate() == initial_rng
    assert profile.quantile(0.01) == pytest.approx(delay)
    assert profile.quantile(0.99) == pytest.approx(delay)


@pytest.mark.parametrize("p50,p95", [(0, 0), (-1, 2), (2, 1), (float("nan"), 2), (1, float("inf"))])
def test_invalid_latency_quantiles_are_rejected(p50: float, p95: float) -> None:
    with pytest.raises(ValueError):
        LatencyProfile.from_quantiles(p50_ms=p50, p95_ms=p95)


@pytest.mark.asyncio
async def test_exit_receives_executable_quotes_signal_context_and_replay_clock() -> None:
    received = []
    market = SimpleNamespace(clob_token_ids=["token"])

    class Strategy:
        config = {"horizon_seconds": 60}
        _replay_now_us = None

        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            received.append((position, dict(market_state), clock.utcnow(), StrategySDK.get_best_bid_ask(market)))
            if market_state.get("best_bid", 0) >= 0.99:
                return ExitDecision(action="close", reason="executable target", close_price=0.99)
            return ExitDecision(action="hold", reason="below executable target")

    strategy = Strategy()
    engine = BacktestEngine(
        config=BacktestConfig(
            latency=LatencyModel.deterministic(submit_ms=500, cancel_ms=500),
            final_close_at_last_mid=False,
        ),
        strategy=strategy,
    )
    context = {"signal": {"threshold": 0.99}, "exit_within_seconds": 60}
    intent = TradeIntent(
        intent_id="entry",
        emitted_at=T0,
        token_id="token",
        side="BUY",
        size=5,
        limit_price=0.97,
        tif="IOC",
        meta={"strategy_context": context},
    )
    await engine.run(
        book_source=InMemoryBookReplay(
            [
                snapshot(0, [(0.95, 20)], [(0.96, 20)]),
                snapshot(0.6, [(0.95, 20)], [(0.96, 20)]),
                snapshot(1, [(0.983, 20)], [(0.998, 20)]),
                snapshot(2),
                snapshot(3, [(0.99, 20)], [(0.999, 20)]),
                snapshot(3.6, [(0.99, 20)], [(0.999, 20)]),
            ]
        ),
        trade_intents=[intent],
    )
    wide = next(row for row in received if row[1].get("observed_at") == T0 + timedelta(seconds=1))
    pos, state, now, sdk_quotes = wide
    assert state["best_bid"] == 0.983 and state["best_ask"] == 0.998
    assert state["current_price"] > 0.99
    assert sdk_quotes == (0.983, 0.998)
    assert now == state["observed_at"]
    assert pos.signal_emitted_at == T0
    assert pos.opened_at > pos.signal_emitted_at
    assert pos.strategy_context == context
    assert pos.config == strategy.config
    pos.strategy_context["signal"]["threshold"] = 0
    assert context["signal"]["threshold"] == 0.99
    assert engine.portfolio.closed_position_count() == 1
    exits = [order for order in engine.matching.all_orders() if order.meta.get("role") == "exit"]
    assert exits and min(order.submitted_at for order in exits) == T0 + timedelta(seconds=3)
    assert strategy._replay_now_us is None
    assert strategy_sdk._replay_books is None


def test_initial_signal_context_survives_partial_fills_and_resets_on_reopen() -> None:
    portfolio = Portfolio()
    first_context = {"signal": {"id": "first"}}

    def fill(order_id: str, side: str, size: float, seconds: float) -> Fill:
        return Fill(
            order_id=order_id,
            token_id="token",
            side=side,
            price=0.5,
            size=size,
            fee_usd=0,
            occurred_at=T0 + timedelta(seconds=seconds),
            fill_index=0,
        )

    portfolio.apply_fill(
        fill("entry", "BUY", 2, 1), strategy_slug=None, signal_emitted_at=T0, strategy_context=first_context
    )
    portfolio.apply_fill(
        fill("entry", "BUY", 3, 2), strategy_slug=None, signal_emitted_at=T0, strategy_context=first_context
    )
    portfolio.apply_fill(
        fill("scale", "BUY", 1, 3),
        strategy_slug=None,
        signal_emitted_at=T0 + timedelta(seconds=2),
        strategy_context={"signal": {"id": "scale"}},
    )
    first = portfolio.positions[("token", "BUY", None)]
    first_context["signal"]["id"] = "mutated"
    assert first.signal_emitted_at == T0
    assert first.strategy_context == {"signal": {"id": "first"}}
    portfolio.apply_fill(fill("close", "SELL", 6, 4), strategy_slug=None)
    portfolio.apply_fill(
        fill("reopen", "BUY", 2, 6),
        strategy_slug=None,
        signal_emitted_at=T0 + timedelta(seconds=5),
        strategy_context={"signal": {"id": "second"}},
    )
    reopened = portfolio.positions[("token", "BUY", None)]
    assert reopened is not first
    assert reopened.signal_emitted_at == T0 + timedelta(seconds=5)
    assert reopened.strategy_context == {"signal": {"id": "second"}}


def test_exit_callback_failure_restores_clock_and_book_contexts() -> None:
    previous_book = {"sentinel": {"bid": 0.1, "ask": 0.2}}
    previous_clock_us = int((T0 - timedelta(seconds=10)).timestamp() * 1_000_000)

    class Strategy:
        _replay_now_us = previous_clock_us
        config = {}

        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            assert clock.utcnow() == market_state["observed_at"]
            raise RuntimeError("deliberate callback failure")

    strategy = Strategy()
    engine = BacktestEngine(strategy=strategy)
    position = Position(token_id="token", side="BUY", entry_price=0.5, size=2, opened_at=T0)
    old_clock = clock.set_replay_clock_us(previous_clock_us)
    old_books = strategy_sdk.set_replay_books(previous_book)
    try:
        assert engine._call_should_exit(position, snapshot(1, [(0.5, 2)], [(0.6, 2)])) is None
        assert clock.replay_clock_us() == previous_clock_us
        assert strategy._replay_now_us == previous_clock_us
        assert strategy_sdk._replay_books is previous_book
    finally:
        strategy_sdk.restore_replay_books(old_books)
        clock.restore_replay_clock(old_clock)


def test_raw_ladder_access_keeps_lazy_price_levels_unmaterialised() -> None:
    bids = ((0.49, 50),)
    asks = ((0.5, 2), (0.6, 3))
    book = BookSnapshot("token", T0, top_bid=0.49, top_ask=0.5, bids_raw=bids, asks_raw=asks)
    assert book.levels("bid") is bids
    assert book.levels("ask") is asks
    assert book._bids is None and book._asks is None


def test_replay_sdk_does_not_invent_ladders_from_aggregate_depth() -> None:
    market = SimpleNamespace(clob_token_ids=["token"])
    old_books = strategy_sdk.set_replay_books(
        {
            "token": {"bid": 0.49, "ask": 0.5, "bid_depth": 50, "ask_depth": 100},
        }
    )
    try:
        assert StrategySDK.get_book_levels(market) is None
        assert StrategySDK.get_order_book_depth(market) is None
        assert StrategySDK.get_book_imbalance(market) is None
    finally:
        strategy_sdk.restore_replay_books(old_books)


@pytest.mark.asyncio
async def test_exit_sdk_retains_other_tokens_and_clears_their_empty_books() -> None:
    other_market = SimpleNamespace(clob_token_ids=["other"])
    received = []

    class Strategy:
        config = {}

        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            received.append(
                (StrategySDK.get_best_bid_ask(other_market), StrategySDK.get_order_book_depth(other_market))
            )
            return ExitDecision(action="hold", reason="inspect historical books")

    engine = BacktestEngine(strategy=Strategy())
    position = Position(token_id="token", side="BUY", entry_price=0.5, size=2, opened_at=T0)
    await engine._on_snapshot(BookSnapshot("other", T0, bids_raw=((0.4, 3),), asks_raw=((0.6, 5),)))
    engine._call_should_exit(position, snapshot(0.25, [(0.5, 2)], [(0.6, 2)]))
    quotes, depth = received[-1]
    assert quotes == (0.4, 0.6)
    assert depth["staleness_ms"] == pytest.approx(250)
    await engine._on_snapshot(BookSnapshot("other", T0 + timedelta(seconds=0.3)))
    engine._call_should_exit(position, snapshot(0.4, [(0.5, 2)], [(0.6, 2)]))
    assert received[-1] == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", ["crypto.update.dispatch", "polymarket.catalog.snapshot"])
@pytest.mark.parametrize("projected", [False, True])
async def test_events_are_delivered_at_first_tick_after_observation(
    monkeypatch: pytest.MonkeyPatch, topic: str, projected: bool
) -> None:
    from services import strategy_backtester
    from services.marketdata import projection
    from services.recorded_event_bus import backtest_bridge

    times = [T0 + timedelta(milliseconds=ms) for ms in (100, 150, 250)]
    envelopes = [
        SimpleNamespace(topic=topic, observed_at_us=int(at.timestamp() * 1_000_000), source="test", payload={})
        for at in times
    ]

    async def replay(**kwargs: Any) -> AsyncIterator[SimpleNamespace]:
        for envelope in [] if projected else envelopes:
            yield envelope

    projected_events = [SimpleNamespace(timestamp=at) for at in times]

    async def project(**kwargs: Any) -> tuple[list[SimpleNamespace], dict[str, int]]:
        return (projected_events if projected else []), {}

    monkeypatch.setattr(backtest_bridge, "replay_events_for_strategy", replay)
    monkeypatch.setattr(projection, "project_crypto_update_events", project)
    monkeypatch.setattr(projection, "project_market_data_refresh_events", project)
    monkeypatch.setattr(strategy_backtester, "_CATALOG_TICK_GRID_CACHE", OrderedDict())
    ticks = [T0 + timedelta(milliseconds=ms) for ms in (0, 100, 200)]
    bins = [[], [], []]
    count = await strategy_backtester._replay_bus_events_into_tick_grid(
        strategy=SimpleNamespace(subscriptions=[topic]),
        start_dt=T0,
        ticks=ticks,
        actual_interval=0.1,
        n_ticks=3,
        events_by_tick=bins,
        candidate_token_ids=[],
        catalog_markets=[],
    )
    assert count == 2
    assert bins[0] == []
    assert [event.timestamp for event in bins[1]] == times[:1]
    assert [event.timestamp for event in bins[2]] == times[1:2]
    assert all(event.timestamp <= ticks[i] for i, events in enumerate(bins) for event in events)


@pytest.mark.asyncio
async def test_persisted_signals_use_emission_time_for_window_and_entry() -> None:
    from sqlalchemy.dialects import postgresql
    from services.strategy_backtester import _load_opps_from_trade_signals

    def signal(emitted: datetime | None, created: datetime) -> SimpleNamespace:
        return SimpleNamespace(
            id="signal",
            strategy_type="unit_replay",
            status="active",
            created_at=created,
            payload_json={
                **({"signal_emitted_at": emitted.isoformat()} if emitted else {}),
                "positions_to_take": [{"token_id": "token", "side": "BUY", "price": 0.5}],
            },
        )

    emitted = T0 + timedelta(seconds=0.15)
    rows = [
        signal(emitted, T0 + timedelta(days=1)),
        signal(None, T0 + timedelta(seconds=0.2)),
        signal(T0 - timedelta(seconds=1), T0 + timedelta(seconds=0.1)),
    ]
    rows.append(signal(None, T0 + timedelta(seconds=0.3)))
    rows[-1].payload_json["signal_emitted_at"] = ""
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows)))
    )
    opps = await _load_opps_from_trade_signals(
        session=session,
        slug="unit_replay",
        start_dt=T0,
        end_dt=T0 + timedelta(seconds=1),
        max_rows=100,
    )
    assert [opp.detected_at for opp in opps] == [emitted, T0 + timedelta(seconds=0.2), T0 + timedelta(seconds=0.3)]
    statement = session.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert sql.count("coalesce(") == 3
    assert "signal_emitted_at" in sql and "timezone(" in sql and "nullif(" in sql


def test_signal_evaluation_uses_emission_time_for_remaining_resolution_window() -> None:
    from services.strategy_backtester import _backtest_evaluate_opportunity

    emitted = T0 + timedelta(seconds=30)
    payload = {"resolution_date": (T0 + timedelta(seconds=60)).isoformat(), "signal_emitted_at": emitted.isoformat()}
    underlying = SimpleNamespace(created_at=T0 - timedelta(seconds=30), payload_json=payload)
    opportunity = SimpleNamespace(detected_at=emitted, _underlying_signal=underlying)
    received = []

    class Strategy:
        config = {}

        def evaluate(self, signal: Any, context: dict[str, Any]) -> SimpleNamespace:
            resolution = datetime.fromisoformat(signal.payload_json["resolution_date"])
            received.append((resolution - datetime.now(timezone.utc)).total_seconds())
            return SimpleNamespace(decision="selected")

    result = _backtest_evaluate_opportunity(
        strategy=Strategy(),
        opp=opportunity,
        pdata={},
        initial_capital_usd=100,
    )
    assert result is not None
    assert received == pytest.approx([30], abs=0.5)
    assert underlying.payload_json is payload
    assert payload["resolution_date"] == (T0 + timedelta(seconds=60)).isoformat()


@pytest.mark.parametrize(
    "books",
    [
        {},
        {"token": {}},
        {"token": {"best_bid": 0.49, "bids": ((0.49, 2),)}},
        {"token": {"best_ask": 0.5, "asks": ((0.5, 2),)}},
    ],
)
def test_replay_prices_never_fall_back_to_catalog_or_live_runtime(
    monkeypatch: pytest.MonkeyPatch, books: dict[str, dict[str, Any]]
) -> None:
    from unittest.mock import Mock
    from services import market_runtime

    runtime = Mock(side_effect=AssertionError("live runtime must not be read in replay"))
    monkeypatch.setattr(market_runtime, "get_market_runtime", runtime)
    market = SimpleNamespace(clob_token_ids=["token"], outcome_prices=[0.9], yes_price=0.9)
    prior = strategy_sdk.set_replay_books(books)
    try:
        assert StrategySDK.get_live_price(market, {"token": {"mid": 0.9}}) == 0.0
        assert StrategySDK.get_ws_mid_price("token") is None
        assert StrategySDK.get_ws_spread_bps("token") is None
        assert StrategySDK.get_ws_mid_price("missing") is None
        expected_ask = float(books.get("token", {}).get("best_ask") or 0.0)
        assert StrategySDK.get_ask_priced_fill(market, {}) == (expected_ask, 0.0, expected_ask > 0)
        expected_bid = 0.49 if books.get("token", {}).get("best_bid") else 0.0
        assert StrategySDK.get_bid_priced_exit(market, {}) == (expected_bid, 0.0, expected_bid > 0)
        runtime.assert_not_called()
    finally:
        strategy_sdk.restore_replay_books(prior)


def test_exit_sdk_price_helpers_and_resolution_time_use_historical_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock
    from services import market_runtime

    runtime = Mock(side_effect=AssertionError("live runtime must not be read in replay"))
    monkeypatch.setattr(market_runtime, "get_market_runtime", runtime)
    market = SimpleNamespace(clob_token_ids=["token"], outcome_prices=[0.9], end_date=T0 + timedelta(seconds=61))
    values = []

    class Strategy:
        config = {}

        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            values.append(
                (
                    StrategySDK.get_ws_mid_price("token"),
                    StrategySDK.get_ws_spread_bps("token"),
                    StrategySDK.get_live_price(market, {"token": {"mid": 0.9}}),
                    StrategySDK.time_to_resolution(market),
                    StrategySDK.time_to_resolution(market, now_ms=int(T0.timestamp() * 1000)),
                    StrategySDK.get_ask_priced_fill(market, {}),
                    StrategySDK.get_bid_priced_exit(market, {}),
                )
            )
            return ExitDecision(action="hold", reason="historical context")

    engine = BacktestEngine(strategy=Strategy())
    position = Position(token_id="token", side="BUY", entry_price=0.5, size=2, opened_at=T0)
    engine._call_should_exit(position, snapshot(1, [(0.49, 2)], [(0.5, 3)]))
    mid, spread, price, remaining, explicit, buy, sell = values[0]
    assert mid == price == pytest.approx(0.495)
    assert spread == pytest.approx(0.01 / 0.495 * 10000)
    assert remaining == 60 and explicit == 61
    assert buy == (0.5, 0.495, True) and sell == (0.49, 0.495, True)
    runtime.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("submit_ms", [500, 1000, 2000])
@pytest.mark.parametrize("laddered", [False, True])
async def test_pending_exits_never_sell_more_than_the_held_position(submit_ms: int, laddered: bool) -> None:
    class Strategy:
        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            if market_state["observed_at"] >= T0 + timedelta(seconds=3):
                return ExitDecision(action="close", reason="close at bid", close_price=0.5)
            return ExitDecision(action="hold", reason="before exit")

        def resolve_exit_policy(self, decision: ExitDecision, close_trigger: str) -> Any:
            if laddered:
                return StrategySDK.build_ladder_exit_policy(
                    levels=2, step_ticks=1, offset_ticks=1, chunk_size=5,
                    distribution="equal", escalation_after_seconds=None,
                )
            return None

    engine = BacktestEngine(
        config=BacktestConfig(
            latency=LatencyModel.deterministic(submit_ms=submit_ms, cancel_ms=submit_ms),
            final_close_at_last_mid=False,
        ),
        strategy=Strategy(),
    )
    await engine.run(
        book_source=InMemoryBookReplay([
            snapshot(i / 20, [(0.5, 20)], [(0.6, 20)]) for i in range(161)
        ]),
        trade_intents=[TradeIntent(
            intent_id="entry", emitted_at=T0, token_id="token",
            side="BUY", size=10, limit_price=0.6, tif="IOC",
        )],
    )
    sells = [fill for fill in engine.matching.all_fills() if fill.side == "SELL"]
    assert sum(fill.size for fill in sells) == pytest.approx(10)
    exits = [order for order in engine.matching.all_orders() if order.meta.get("role") in {"exit", "exit_child"}]
    assert len(exits) == (2 if laddered else 1)
    assert engine.portfolio.closed_position_count() == 1


@pytest.mark.asyncio
async def test_partial_ioc_exit_retries_only_the_remaining_position() -> None:
    class Strategy:
        def should_exit(self, position: SimpleNamespace, market_state: dict[str, Any]) -> ExitDecision:
            if market_state["observed_at"] >= T0 + timedelta(seconds=1):
                return ExitDecision(action="close", reason="close at bid", close_price=0.5)
            return ExitDecision(action="hold", reason="before exit")

    engine = BacktestEngine(
        config=BacktestConfig(
            latency=LatencyModel.deterministic(submit_ms=500, cancel_ms=500),
            final_close_at_last_mid=False,
        ),
        strategy=Strategy(),
    )
    await engine.run(
        book_source=InMemoryBookReplay([
            snapshot(i / 20, [(0.5, 5)], [(0.6, 20)]) for i in range(61)
        ]),
        trade_intents=[TradeIntent(
            intent_id="entry", emitted_at=T0, token_id="token",
            side="BUY", size=10, limit_price=0.6, tif="IOC",
        )],
    )
    exits = [order for order in engine.matching.all_orders() if order.meta.get("role") == "exit"]
    assert [order.size for order in exits] == [10, 5]
    assert [order.filled_size for order in exits] == [5, 5]
    assert exits[1].submitted_at >= exits[0].venue_received_at
    assert engine.portfolio.closed_position_count() == 1
    assert engine.portfolio.open_position_count() == 0
