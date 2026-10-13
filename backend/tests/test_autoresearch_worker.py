"""Behavioral tests for the scheduled autoresearch worker loop."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import workers.autoresearch_worker as worker


class _FakeLLMManager:
    def is_available(self) -> bool:
        return True


@pytest.fixture()
def fake_ai(monkeypatch):
    async def _init():
        return _FakeLLMManager()

    import services.ai as ai_pkg

    monkeypatch.setattr(ai_pkg, "initialize_ai", _init)


@pytest.mark.asyncio
async def test_run_experiment_consumes_stream_and_returns_terminal_event(
    monkeypatch, fake_ai
):
    events = [
        {"event": "experiment_start", "data": {}},
        {"event": "iteration_start", "data": {}},
        {"event": "done", "data": {"best_score": 1.5}},
    ]

    async def _fake_stream(strategy_id, settings_override=None):
        for e in events:
            yield e

    from services.autoresearch_service import autoresearch_service

    monkeypatch.setattr(
        autoresearch_service, "run_strategy_params_stream", _fake_stream
    )

    async def _fake_settings():
        return {"mode": "params", "interval_seconds": 60}

    monkeypatch.setattr(worker, "_load_settings", _fake_settings)

    final = await worker._run_experiment("strategy-1")
    assert final == "done"


@pytest.mark.asyncio
async def test_run_experiment_routes_code_mode_without_trader(monkeypatch, fake_ai):
    calls: dict = {}

    async def _fake_code_stream(trader_id, strategy_id, settings_override=None):
        calls["trader_id"] = trader_id
        calls["strategy_id"] = strategy_id
        yield {"event": "done", "data": {}}

    from services.autoresearch_service import autoresearch_service

    monkeypatch.setattr(
        autoresearch_service, "run_code_evolution_stream", _fake_code_stream
    )

    async def _fake_settings():
        return {"mode": "code", "interval_seconds": 60}

    monkeypatch.setattr(worker, "_load_settings", _fake_settings)

    final = await worker._run_experiment("strategy-2")
    assert final == "done"
    assert calls["trader_id"] is None
    assert calls["strategy_id"] == "strategy-2"


@pytest.mark.asyncio
async def test_start_loop_gate_interval_and_round_robin(monkeypatch, fake_ai):
    monkeypatch.setattr(worker, "_POLL_SECONDS", 0.01)

    async def _fake_settings():
        return {"mode": "params", "interval_seconds": 60}

    monkeypatch.setattr(worker, "_load_settings", _fake_settings)

    strategies = ["alpha", "beta"]
    monkeypatch.setattr(worker, "_enabled_strategy_ids", lambda: _list())

    async def _list():
        return list(strategies)

    ran: list[str] = []

    async def _fake_run(strategy_id):
        ran.append(strategy_id)
        return "done"

    monkeypatch.setattr(worker, "_run_experiment", _fake_run)

    async def _not_busy(_strategy_id):
        return False

    monkeypatch.setattr(worker, "_has_running_experiment_elsewhere", _not_busy)

    task = asyncio.create_task(worker.start_loop())
    # Poll is 0.01s; the 60s interval must block the second experiment.
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ran == ["alpha"], "interval gate must prevent back-to-back runs"


@pytest.mark.asyncio
async def test_start_loop_disabled_interval_never_runs(monkeypatch, fake_ai):
    monkeypatch.setattr(worker, "_POLL_SECONDS", 0.01)

    async def _fake_settings():
        return {"mode": "params", "interval_seconds": 0}

    monkeypatch.setattr(worker, "_load_settings", _fake_settings)

    async def _fake_run(strategy_id):
        raise AssertionError("must not run when interval <= 0")

    monkeypatch.setattr(worker, "_run_experiment", _fake_run)
    monkeypatch.setattr(worker, "_enabled_strategy_ids", lambda: _empty())

    async def _empty():
        return ["alpha"]

    task = asyncio.create_task(worker.start_loop())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_has_running_experiment_elsewhere_freshness(monkeypatch, fake_ai):
    from datetime import timedelta

    def _make_session(updated_values):
        class _Result:
            def scalars(self):
                class _S:
                    def all(self):
                        return list(updated_values)

                return _S()

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, *_a, **_kw):
                return _Result()

        return _FakeSession

    from utils.utcnow import utcnow

    naive_fresh = utcnow().replace(tzinfo=None) - timedelta(minutes=5)
    aware_fresh = utcnow() - timedelta(minutes=5)
    stale = utcnow() - timedelta(hours=3)

    cases = [
        ([naive_fresh], True),
        ([aware_fresh], True),
        ([stale], False),
        ([stale, aware_fresh], True),
        ([None], False),
        ([], False),
    ]
    for updated_values, expected in cases:
        monkeypatch.setattr(
            worker, "AsyncSessionLocal", _make_session(updated_values)
        )
        got = await worker._has_running_experiment_elsewhere("strategy-1")
        assert got is expected, f"{updated_values!r} -> {got}, want {expected}"


@pytest.mark.asyncio
async def test_start_loop_skips_strategy_busy_elsewhere(monkeypatch, fake_ai):
    monkeypatch.setattr(worker, "_POLL_SECONDS", 0.01)

    async def _fake_settings():
        return {"mode": "params", "interval_seconds": 60}

    async def _busy_list():
        return ["alpha", "beta"]

    monkeypatch.setattr(worker, "_load_settings", _fake_settings)
    monkeypatch.setattr(worker, "_enabled_strategy_ids", _busy_list)

    async def _fake_busy(strategy_id):
        return strategy_id == "alpha"

    monkeypatch.setattr(worker, "_has_running_experiment_elsewhere", _fake_busy)

    async def _fake_run(strategy_id):
        raise AssertionError(f"must not run while busy elsewhere: {strategy_id}")

    monkeypatch.setattr(worker, "_run_experiment", _fake_run)

    task = asyncio.create_task(worker.start_loop())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
