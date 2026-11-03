from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.autoresearch_service import AutoresearchService, load_autoresearch_settings
from utils.utcnow import utcnow


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_status", ["failed", "completed", "paused"])
async def test_terminal_status_is_stable_across_repeated_reads(stored_status):
    service = AutoresearchService()
    row = SimpleNamespace(id="terminal-experiment", status=stored_status, updated_at=utcnow())
    session = AsyncMock()

    first = await service._resolve_runtime_status(session, row)
    second = await service._resolve_runtime_status(session, row)

    assert (first, second, row.status) == (stored_status, stored_status, stored_status)
    session.scalar.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_orphan_stays_failed_after_first_status_query():
    row = SimpleNamespace(
        id="orphan-experiment",
        status="running",
        trader_id=None,
        strategy_id="dynamic-strategy",
        updated_at=utcnow(),
        finished_at=None,
    )
    session = AsyncMock()
    session.scalar.return_value = True
    service = AutoresearchService()

    first = await service._resolve_runtime_status(session, row)
    second = await service._resolve_runtime_status(session, row)

    assert (first, second, row.status) == ("failed", "failed", "failed")
    assert row.finished_at is not None
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_status_uses_database_owner_instead_of_timestamp_age():
    from datetime import timedelta

    row = SimpleNamespace(
        id="owned-experiment",
        status="running",
        trader_id=None,
        strategy_id="dynamic-strategy",
        updated_at=utcnow() - timedelta(hours=2),
    )
    session = AsyncMock()
    session.scalar.return_value = False

    assert await AutoresearchService()._resolve_runtime_status(session, row) == "running"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_status_does_not_overwrite_concurrently_completed_experiment():
    row = SimpleNamespace(id="finished-experiment", status="running", trader_id=None, strategy_id="dynamic-strategy")
    session = AsyncMock()
    session.scalar.return_value = True

    async def refresh(_row):
        _row.status = "completed"

    session.refresh.side_effect = refresh
    assert await AutoresearchService()._resolve_runtime_status(session, row) == "completed"
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row, expected",
    [
        (SimpleNamespace(autoresearch_interval_seconds=0), 0),
        (SimpleNamespace(autoresearch_interval_seconds=None), 600),
        (SimpleNamespace(), 600),
    ],
)
async def test_load_autoresearch_settings_interval_semantics(monkeypatch, row, expected):
    session = AsyncMock()
    session.execute.return_value = SimpleNamespace(scalar_one_or_none=lambda: row)
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr("services.autoresearch_service.AsyncSessionLocal", lambda: context)

    assert (await load_autoresearch_settings())["interval_seconds"] == expected
