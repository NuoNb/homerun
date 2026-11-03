import asyncio
import os
import sys
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from sqlalchemy import MetaData, select, text, update

from models.database import AppSettings, AutoresearchExperiment, AutoresearchIteration, Strategy, Trader, WorkerControl
from services import autoresearch_runtime as runtime
from services import autoresearch_service as service_module
from services.autoresearch_service import AutoresearchService
from tests.postgres_test_db import build_postgres_session_factory
from utils.utcnow import utcnow

pytestmark = pytest.mark.db


@pytest_asyncio.fixture
async def experiment_db(monkeypatch):
    metadata = MetaData()
    Trader.__table__.to_metadata(metadata)
    Strategy.__table__.to_metadata(metadata)
    AppSettings.__table__.to_metadata(metadata)
    WorkerControl.__table__.to_metadata(metadata)
    AutoresearchExperiment.__table__.to_metadata(metadata)
    AutoresearchIteration.__table__.to_metadata(metadata)
    engine, factory = await build_postgres_session_factory(metadata, "autoresearch_runtime")
    monkeypatch.setattr(runtime, "async_engine", engine)
    monkeypatch.setattr(runtime, "AsyncSessionLocal", factory)
    monkeypatch.setattr(service_module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(runtime, "_CONTROL_POLL_SECONDS", 0.01)
    monkeypatch.setattr(runtime, "_HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(
        service_module, "load_autoresearch_settings", AsyncMock(return_value=service_module._defaults())
    )
    async with factory() as session:
        session.add(Trader(id="test-trader", name="Test trader"))
        session.add(AppSettings(id="default"))
        session.add(
            Strategy(
                id="test-strategy",
                slug="test-strategy",
                name="Test strategy",
                source_code="test source",
                config={"threshold": 0.5},
                config_schema={"param_fields": [{"key": "threshold", "type": "float", "min": 0, "max": 1}]},
            )
        )
        await session.commit()
    try:
        yield factory
    finally:
        await engine.dispose()


async def _drain(stream):
    return [event async for event in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("other_mode", ["code", "strategy_params"])
async def test_manual_and_scheduled_starts_share_atomic_ownership(experiment_db, monkeypatch, other_mode):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def baseline(**_kwargs):
        entered.set()
        await release.wait()
        return {"execution": {"success": True}}

    monkeypatch.setattr("services.backtest.unified_runner.run_unified_backtest", baseline)
    monkeypatch.setattr(
        service_module, "_augment_with_walk_forward_gate", AsyncMock(side_effect=lambda **kw: kw["result"])
    )
    worker_service = AutoresearchService()
    api_service = AutoresearchService()
    worker = asyncio.create_task(
        _drain(worker_service.run_strategy_params_stream("test-strategy", {"max_iterations": 0}))
    )
    await asyncio.wait_for(entered.wait(), 3)
    try:
        if other_mode == "code":
            second = api_service.run_code_evolution_stream(None, "test-strategy", {"max_iterations": 0})
        else:
            second = api_service.run_strategy_params_stream("test-strategy", {"max_iterations": 0})
        events = await asyncio.wait_for(_drain(second), 3)
        assert events[0]["event"] == "error"
        assert "already running" in events[0]["data"]["error"]
        async with experiment_db() as session:
            rows = (await session.scalars(select(AutoresearchExperiment))).all()
            assert len(rows) == 1
            assert rows[0].status == "running"
    finally:
        release.set()
        await asyncio.wait_for(worker, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["code", "strategy_params", "params"])
async def test_remote_stop_cancels_inflight_work_before_auto_apply(experiment_db, monkeypatch, mode):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    applied = []

    async def blocking_stream(*args):
        control = args[-2]
        try:
            entered.set()
            await asyncio.Event().wait()
            async with experiment_db() as apply_session:
                if await control.lock_for_apply(apply_session):
                    applied.append(True)
            yield {"event": "done", "data": {}}
        finally:
            cancelled.set()

    private_name = {
        "code": "_run_code_evolution_stream",
        "strategy_params": "_run_strategy_params_stream",
        "params": "_run_trader_params_stream",
    }[mode]
    monkeypatch.setattr(AutoresearchService, private_name, blocking_stream)
    worker_service = AutoresearchService()
    api_service = AutoresearchService()
    if mode == "code":
        stream = worker_service.run_code_evolution_stream(None, "test-strategy")
        stop = api_service.stop_strategy_experiment
        stop_id = "test-strategy"
    elif mode == "strategy_params":
        stream = worker_service.run_strategy_params_stream("test-strategy")
        stop = api_service.stop_strategy_params_experiment
        stop_id = "test-strategy"
    else:
        stream = worker_service.run_experiment_stream("test-trader")
        stop = api_service.stop_experiment
        stop_id = "test-trader"
    task = asyncio.create_task(_drain(stream))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        response = await stop(stop_id)
        assert response["stopped"] is True
        assert response["experiment_id"]
        events = await asyncio.wait_for(task, 3)
        assert cancelled.is_set()
        assert applied == []
        assert events[-1]["data"]["status"] == "paused"
        async with experiment_db() as session:
            row = await session.get(AutoresearchExperiment, response["experiment_id"])
            assert row.status == "paused"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_database_owner_keeps_old_running_row_alive(experiment_db):
    async with runtime.claim_experiment(
        trader_id=None, strategy_id="test-strategy", mode="strategy_params", settings={}
    ) as control:
        async with experiment_db() as session:
            await session.execute(
                update(AutoresearchExperiment)
                .where(AutoresearchExperiment.id == control.experiment_id)
                .values(updated_at=utcnow() - timedelta(hours=2))
            )
            await session.commit()
        response = await AutoresearchService().get_strategy_params_experiment_status("test-strategy")
        assert response["status"] == "running"
        control.final_status = "completed"


@pytest.mark.asyncio
async def test_orphan_and_failure_status_remain_terminal(experiment_db):
    async with experiment_db() as session:
        session.add(
            AutoresearchExperiment(
                id="crashed", strategy_id="test-strategy", mode="strategy_params", name="Crashed", status="running"
            )
        )
        await session.commit()
    service = AutoresearchService()
    first = await service.get_strategy_params_experiment_status("test-strategy")
    second = await service.get_strategy_params_experiment_status("test-strategy")
    assert first["status"] == second["status"] == "failed"


@pytest.mark.asyncio
async def test_error_and_cancelled_streams_release_claim(experiment_db, monkeypatch):
    async def failing_stream(*args):
        yield {"event": "error", "data": {"error": "Backtest failed"}}

    monkeypatch.setattr(AutoresearchService, "_run_strategy_params_stream", failing_stream)
    events = await _drain(AutoresearchService().run_strategy_params_stream("test-strategy"))
    assert events[0]["event"] == "error"
    async with experiment_db() as session:
        assert (await session.scalar(select(AutoresearchExperiment.status))) == "failed"
    async with runtime.claim_experiment(
        trader_id=None, strategy_id="test-strategy", mode="code", settings={}
    ) as next_run:
        assert next_run is not None
        next_run.final_status = "completed"


@pytest.mark.asyncio
async def test_stop_without_running_experiment_reports_no_stop(experiment_db):
    assert await AutoresearchService().stop_strategy_params_experiment("test-strategy") == {
        "stopped": False,
        "experiment_id": None,
    }


@pytest.mark.asyncio
async def test_simultaneous_starts_claim_only_one_experiment(experiment_db):
    barrier = asyncio.Event()
    claimed = asyncio.Event()
    rejected = asyncio.Event()
    release = asyncio.Event()

    async def contender(mode):
        await barrier.wait()
        async with runtime.claim_experiment(
            trader_id=None, strategy_id="test-strategy", mode=mode, settings={}
        ) as control:
            if control is None:
                rejected.set()
                return False
            claimed.set()
            await release.wait()
            control.final_status = "completed"
            return True

    tasks = [asyncio.create_task(contender(mode)) for mode in ("code", "strategy_params")]
    barrier.set()
    try:
        await asyncio.wait_for(claimed.wait(), 3)
        await asyncio.wait_for(rejected.wait(), 3)
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert sorted(results) == [False, True]
    async with experiment_db() as session:
        assert len((await session.scalars(select(AutoresearchExperiment))).all()) == 1


@pytest.mark.asyncio
async def test_claim_is_visible_in_an_independent_process(experiment_db):
    script = """
import asyncio, os, sys
import asyncpg
async def main():
    connection = await asyncpg.connect(os.environ["HOMERUN_AUTORESEARCH_TEST_DATABASE_URL"])
    try:
        acquired = await connection.fetchval("SELECT pg_try_advisory_lock($1::bigint)", int(sys.argv[1]))
        print("claimed" if acquired else "busy")
    finally:
        await connection.close()
asyncio.run(main())
"""
    env = dict(os.environ)
    env["HOMERUN_AUTORESEARCH_TEST_DATABASE_URL"] = (
        experiment_db.kw["bind"].url.set(drivername="postgresql").render_as_string(hide_password=False)
    )
    key = runtime.experiment_lock_keys(None, "test-strategy")[0]

    async def probe():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            str(key),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 0, error.decode()
        return output.decode().strip()

    async with runtime.claim_experiment(
        trader_id=None, strategy_id="test-strategy", mode="code", settings={}
    ) as control:
        assert await probe() == "busy"
        control.final_status = "completed"
    assert await probe() == "claimed"


@pytest.mark.asyncio
async def test_stopping_serializes_with_auto_apply_transaction(experiment_db):
    async with runtime.claim_experiment(
        trader_id=None, strategy_id="test-strategy", mode="strategy_params", settings={}
    ) as control:
        async with experiment_db() as session:
            assert await control.lock_for_apply(session)
            stop = asyncio.create_task(AutoresearchService().stop_strategy_params_experiment("test-strategy"))
            try:
                await asyncio.sleep(0.03)
                assert not stop.done()
                strategy = await session.get(Strategy, "test-strategy")
                strategy.config = {"threshold": 0.6}
                await session.commit()
                response = await asyncio.wait_for(stop, 3)
                assert response["stopped"] is True
            finally:
                stop.cancel()
                await asyncio.gather(stop, return_exceptions=True)
        async with experiment_db() as session:
            assert await control.lock_for_apply(session) is False
            assert (await session.get(Strategy, "test-strategy")).config == {"threshold": 0.6}


@pytest.mark.asyncio
async def test_database_ownership_loss_cancels_inflight_work(experiment_db):
    blocked = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        try:
            blocked.set()
            await asyncio.Event().wait()
            yield {"event": "done", "data": {}}
        finally:
            cancelled.set()

    with pytest.raises(RuntimeError, match="lost database ownership"):
        async with runtime.claim_experiment(
            trader_id=None, strategy_id="test-strategy", mode="code", settings={}
        ) as control:
            task = asyncio.create_task(_drain(control.consume(work())))
            try:
                await asyncio.wait_for(blocked.wait(), 3)
                async with experiment_db() as session:
                    pid = await session.scalar(
                        text(
                            "SELECT pid FROM pg_locks WHERE locktype='advisory' AND granted "
                            "AND database=(SELECT oid FROM pg_database WHERE datname=current_database())"
                        )
                    )
                    assert pid is not None
                    await session.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
                    await session.commit()
                await asyncio.wait_for(task, 3)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert cancelled.is_set()
    async with experiment_db() as session:
        assert await session.scalar(select(AutoresearchExperiment.status)) == "failed"


@pytest.mark.asyncio
async def test_worker_controls_and_interval_api_share_persisted_state(experiment_db, monkeypatch):
    from api import routes_workers

    monkeypatch.setattr(routes_workers, "_worker_detail", AsyncMock(return_value={"worker_name": "autoresearch"}))
    async with experiment_db() as session:
        await routes_workers.pause_worker("autoresearch", session)
        row = await session.get(WorkerControl, "autoresearch")
        assert row.is_paused is True
        await routes_workers.set_worker_master_enabled("autoresearch", False, session)
        assert row.is_enabled is False
        await routes_workers.set_worker_run_interval("autoresearch", 0, session)
        await session.refresh(row)
        controls = await routes_workers._generic_worker_controls(session)
        assert controls["autoresearch"]["interval_seconds"] == 0
        assert controls["autoresearch"]["is_paused"] is True
        assert controls["autoresearch"]["is_enabled"] is False
        app_settings = await session.get(AppSettings, "default")
        await session.refresh(app_settings)
        assert app_settings.autoresearch_interval_seconds == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [1, 59, 7201])
async def test_worker_interval_api_rejects_unsafe_research_cadence(experiment_db, interval):
    from api import routes_workers
    from fastapi import HTTPException

    async with experiment_db() as session:
        with pytest.raises(HTTPException) as raised:
            await routes_workers.set_worker_run_interval("autoresearch", interval, session)
    assert raised.value.status_code == 400


@pytest.mark.asyncio
async def test_stream_keeps_context_and_cleanup_in_the_same_task():
    from contextvars import ContextVar

    context = ContextVar("autoresearch_stream_context", default=None)
    closed = asyncio.Event()

    async def source():
        token = context.set("owned")
        try:
            yield {"event": "progress", "data": {}}
            await asyncio.sleep(0)
            assert context.get() == "owned"
            yield {"event": "done", "data": {}}
        finally:
            context.reset(token)
            closed.set()

    control = runtime.AutoresearchRun("context-test")
    events = await _drain(control.consume(source()))
    assert events[-1]["event"] == "done"
    assert control.final_status == "completed"
    assert closed.is_set()
    assert context.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["code", "strategy_params", "params"])
async def test_each_public_stream_persists_completed_baseline_run(experiment_db, monkeypatch, mode):
    monkeypatch.setattr(
        "services.backtest.unified_runner.run_unified_backtest",
        AsyncMock(return_value={"execution": {"success": True}}),
    )
    monkeypatch.setattr(
        service_module, "_augment_with_walk_forward_gate", AsyncMock(side_effect=lambda **kw: kw["result"])
    )
    monkeypatch.setattr(service_module, "_load_opportunity_history", AsyncMock(return_value=[{"id": "history-item"}]))
    monkeypatch.setattr(service_module, "_evaluate_params_sync", lambda *_args: (1.0, {"trade_count": 0}))
    service = AutoresearchService()
    if mode == "code":
        stream = service.run_code_evolution_stream(None, "test-strategy", {"max_iterations": 0})
    elif mode == "strategy_params":
        stream = service.run_strategy_params_stream("test-strategy", {"max_iterations": 0})
    else:
        stream = service.run_experiment_stream("test-trader", {"max_iterations": 0})
    events = await _drain(stream)
    assert events[0]["event"] == "experiment_start"
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["status"] == "completed"
    async with experiment_db() as session:
        row = await session.get(AutoresearchExperiment, events[0]["data"]["experiment_id"])
        assert row.status == "completed"
        assert row.finished_at is not None
        assert row.mode == mode


@pytest.mark.asyncio
async def test_trader_scoped_code_run_excludes_both_scopes(experiment_db):
    async with runtime.claim_experiment(
        trader_id="test-trader", strategy_id="test-strategy", mode="code", settings={}
    ) as control:
        async with runtime.claim_experiment(
            trader_id=None, strategy_id="test-strategy", mode="strategy_params", settings={}
        ) as strategy_run:
            assert strategy_run is None
        async with runtime.claim_experiment(
            trader_id="test-trader", strategy_id=None, mode="params", settings={}
        ) as trader_run:
            assert trader_run is None
        control.final_status = "completed"
