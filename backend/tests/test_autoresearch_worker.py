"""Scheduler behavior with controllable time, AI readiness and worker controls."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import workers.autoresearch_worker as worker
from services import ai
from services.autoresearch_service import autoresearch_service
from utils.utcnow import utcnow


async def _until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.fixture
def scheduler(monkeypatch):
    state = SimpleNamespace(
        settings={"mode": "params", "interval_seconds": 60},
        control={"is_enabled": True, "is_paused": False, "requested_run_at": None},
        revision=utcnow(),
        now=utcnow(),
        available=True,
        snapshots=[],
    )
    session = AsyncMock()

    async def read_revision(*_args):
        return state.revision

    session.scalar.side_effect = read_revision
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(worker, "AsyncSessionLocal", lambda: context)
    monkeypatch.setattr(worker, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(worker, "utcnow", lambda: state.now)
    monkeypatch.setattr(worker, "_load_settings", AsyncMock(side_effect=lambda: dict(state.settings)))
    monkeypatch.setattr(worker, "read_worker_control", AsyncMock(side_effect=lambda *_a, **_kw: dict(state.control)))
    monkeypatch.setattr(worker, "_enabled_strategy_ids", AsyncMock(return_value=["strategy-a", "strategy-b"]))
    monkeypatch.setattr(worker, "clear_worker_run_request", AsyncMock())
    state.run = AsyncMock(return_value="done")
    monkeypatch.setattr(worker, "_run_experiment", state.run)

    async def snapshot(_session, name, **values):
        assert name == "autoresearch"
        state.snapshots.append(values)

    monkeypatch.setattr(worker, "write_worker_snapshot", snapshot)
    state.manager = SimpleNamespace(is_available=lambda: state.available, initialize=AsyncMock(return_value=True))
    monkeypatch.setattr(ai, "get_llm_manager", lambda: state.manager)
    monkeypatch.setattr(ai, "initialize_ai", AsyncMock(return_value=state.manager))
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["params", "code"])
async def test_run_experiment_consumes_and_closes_stream(monkeypatch, mode):
    calls = []
    closed = asyncio.Event()

    async def stream(*args, **kwargs):
        calls.append((args, kwargs))
        try:
            yield {"event": "experiment_start", "data": {}}
            yield {"event": "done", "data": {"status": "completed"}}
        finally:
            closed.set()

    monkeypatch.setattr(autoresearch_service, "run_strategy_params_stream", stream)
    monkeypatch.setattr(autoresearch_service, "run_code_evolution_stream", stream)
    assert await worker._run_experiment("strategy-a", mode) == "completed"
    assert closed.is_set()
    expected = [(("strategy-a",), {})] if mode == "params" else [((), {"trader_id": None, "strategy_id": "strategy-a"})]
    assert calls == expected


@pytest.mark.asyncio
async def test_interval_gate_and_round_robin(scheduler):
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: len(scheduler.snapshots) >= 4)
        assert scheduler.run.await_args_list[0].args == ("strategy-a", "params")
        assert scheduler.run.await_count == 1
        scheduler.now += timedelta(seconds=61)
        await _until(lambda: scheduler.run.await_count == 2)
        assert scheduler.run.await_args_list[1].args == ("strategy-b", "params")
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_disabled_schedule_writes_snapshot_without_initializing_ai(scheduler):
    scheduler.settings["interval_seconds"] = 0
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: len(scheduler.snapshots) >= 2)
        scheduler.run.assert_not_awaited()
        scheduler.manager.initialize.assert_not_awaited()
        assert scheduler.snapshots[-1]["current_activity"] == "Scheduled research disabled"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("control_key", ["is_paused", "is_enabled"])
async def test_control_changes_cancel_work_and_preserve_heartbeats(scheduler, monkeypatch, control_key):
    cancelled = asyncio.Event()

    async def blocked(*_args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(worker, "_run_experiment", blocked)
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: len(scheduler.snapshots) >= 3)
        assert all(snapshot["stats"]["experiment_active"] for snapshot in scheduler.snapshots)
        scheduler.control[control_key] = control_key == "is_paused"
        await asyncio.wait_for(cancelled.wait(), 2)
        await _until(lambda: not scheduler.snapshots[-1]["stats"]["experiment_active"])
        assert scheduler.snapshots[-1]["running"] is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unavailable_ai_is_retried_until_ready(scheduler):
    scheduler.available = False

    async def initialize():
        if scheduler.manager.initialize.await_count >= 2:
            scheduler.available = True
        return True

    scheduler.manager.initialize.side_effect = initialize
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: scheduler.run.await_count == 1)
        assert scheduler.manager.initialize.await_count == 2
        assert any(snapshot["current_activity"] == "Waiting for AI configuration" for snapshot in scheduler.snapshots)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_transient_ai_initialization_failure_recovers(scheduler):
    scheduler.manager.initialize.side_effect = [RuntimeError("temporary settings read failure"), True]
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: scheduler.run.await_count == 1)
        assert scheduler.manager.initialize.await_count == 2
        assert scheduler.snapshots[-1]["last_error"] is None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_ai_configuration_refreshes_during_schedule_wait(scheduler):
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: len(scheduler.snapshots) >= 3)
        assert scheduler.run.await_count == 1
        scheduler.revision += timedelta(seconds=1)
        await _until(lambda: scheduler.manager.initialize.await_count == 2)
        assert scheduler.run.await_count == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_ai_configuration_refresh_waits_until_active_experiment_finishes(scheduler):
    finished = asyncio.Event()

    async def run(*_args):
        await finished.wait()
        return "done"

    scheduler.run.side_effect = run
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: scheduler.run.await_count == 1)
        assert scheduler.manager.initialize.await_count == 1
        scheduler.revision += timedelta(seconds=1)
        snapshot_count = len(scheduler.snapshots)
        await _until(lambda: len(scheduler.snapshots) >= snapshot_count + 2)
        assert scheduler.manager.initialize.await_count == 1
        assert scheduler.snapshots[-1]["stats"]["experiment_active"] is True

        finished.set()
        await _until(lambda: scheduler.manager.initialize.await_count == 2)
        assert scheduler.run.await_count == 1
        assert scheduler.snapshots[-1]["current_activity"] == "Waiting for next experiment"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manual_run_request_works_with_scheduling_disabled(scheduler):
    scheduler.settings["interval_seconds"] = 0
    scheduler.control["requested_run_at"] = scheduler.now
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: scheduler.run.await_count == 1)
        worker.clear_worker_run_request.assert_awaited_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_configuration_refresh_with_available_old_provider_retries(scheduler):
    scheduler.manager.initialize.side_effect = [False, True]
    task = asyncio.create_task(worker.start_loop())
    try:
        await _until(lambda: scheduler.run.await_count == 1)
        assert scheduler.manager.initialize.await_count == 2
        assert scheduler.snapshots[0]["current_activity"] == "Waiting for AI configuration"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
