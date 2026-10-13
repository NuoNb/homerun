import pytest

from services.autoresearch_service import AutoresearchService, _resolve_autoresearch_status


class _FakeSession:
    def __init__(self) -> None:
        self.committed = False

    async def commit(self) -> None:
        self.committed = True


class _FakeExperiment:
    def __init__(self, experiment_id: str, status: str) -> None:
        self.id = experiment_id
        self.status = status
        self.finished_at = None
        self.updated_at = None


def test_resolve_autoresearch_status_requires_matching_active_experiment() -> None:
    assert _resolve_autoresearch_status("exp-1", "running", "exp-1") == "running"
    assert _resolve_autoresearch_status("exp-1", "running", None) == "failed"
    assert _resolve_autoresearch_status("exp-1", "completed", None) == "completed"


@pytest.mark.asyncio
async def test_resolve_runtime_status_marks_orphaned_running_experiment_failed() -> None:
    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")

    status = await service._resolve_runtime_status(session, row, "strategy:tail-end-carry")

    assert status == "failed"
    assert row.status == "failed"
    assert row.finished_at is not None
    assert row.updated_at is not None
    assert session.committed


@pytest.mark.asyncio
async def test_resolve_runtime_status_keeps_owned_active_experiment_running() -> None:
    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    service._active_experiments["strategy:tail-end-carry"] = "exp-1"
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")

    status = await service._resolve_runtime_status(session, row, "strategy:tail-end-carry")

    assert status == "running"
    assert row.status == "running"
    assert row.finished_at is None
    assert not session.committed


@pytest.mark.asyncio
async def test_load_autoresearch_settings_interval_semantics(monkeypatch) -> None:
    """0 means scheduling disabled; only a missing column falls back to 600."""
    from services.autoresearch_service import load_autoresearch_settings

    class _Result:
        def __init__(self, row):
            self._row = row

        def scalar_one_or_none(self):
            return self._row

    class _FakeSession:
        def __init__(self, row):
            self._row = row

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, *_a, **_kw):
            return _Result(self._row)

    class _RowWithZeroInterval:
        autoresearch_interval_seconds = 0

    class _RowWithoutColumn:
        pass

    monkeypatch.setattr(
        "services.autoresearch_service.AsyncSessionLocal",
        lambda: _FakeSession(_RowWithZeroInterval()),
    )
    settings = await load_autoresearch_settings()
    assert settings["interval_seconds"] == 0

    monkeypatch.setattr(
        "services.autoresearch_service.AsyncSessionLocal",
        lambda: _FakeSession(_RowWithoutColumn()),
    )
    settings = await load_autoresearch_settings()
    assert settings["interval_seconds"] == 600


@pytest.mark.asyncio
async def test_resolve_runtime_status_keeps_fresh_cross_process_row_running() -> None:
    """A running row refreshed recently belongs to another process — the
    in-memory lock check alone must not mark it failed."""
    from datetime import timedelta

    from utils.utcnow import utcnow

    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")
    row.updated_at = utcnow() - timedelta(minutes=5)

    status = await service._resolve_runtime_status(session, row, "strategy:slug")

    assert status == "running"
    assert row.status == "running"
    assert not session.committed


@pytest.mark.asyncio
async def test_resolve_runtime_status_fails_stale_cross_process_row() -> None:
    """A running row untouched past the grace window is a crashed orphan."""
    from datetime import timedelta

    from utils.utcnow import utcnow

    service = AutoresearchService()
    service._active_experiments.clear()
    service._stop_flags.clear()
    session = _FakeSession()
    row = _FakeExperiment("exp-1", "running")
    row.updated_at = utcnow() - timedelta(hours=2)

    status = await service._resolve_runtime_status(session, row, "strategy:slug")

    assert status == "failed"
    assert row.status == "failed"
    assert session.committed
