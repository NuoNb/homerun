"""Worker loop that auto-triggers scheduled autoresearch experiments.

The ``autoresearch_interval_seconds`` AppSettings column is documented
as the "auto-trigger interval" but nothing ever consumed it —
experiments could only be started from the BacktestStudio UI or by an
AI agent calling ``start_param_iteration``.  This worker closes that
gap.  It lives on the jobs plane (each iteration is an LLM proposal +
unified backtest — the same class of CPU/GIL load as backtest_worker,
so it must stay off the trading plane), polls the hot-reloadable
setting, and when the interval has elapsed runs one headless experiment
for the next enabled strategy (round-robin cursor).  ``interval <= 0``
disables scheduling entirely.

One experiment at a time per process — same bounding rationale as
backtest_worker.  Mode follows the global autoresearch ``mode``
setting: "params" iterates ``Strategy.config`` (bounds-clamped,
auto-applied on improvement), "code" evolves ``Strategy.source_code``
with version snapshots + walk-forward gate.  Experiments, iterations,
and auto-apply are persisted by autoresearch_service itself, so
scheduled runs show up in the UI's experiment/fleet views like manual
ones.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select

from models.database import AsyncSessionLocal, AutoresearchExperiment, Strategy
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("workers.autoresearch")

_POLL_SECONDS = 30.0
# An experiment row counts as "running elsewhere" only while it stays
# fresh.  The service's in-memory ``_active_experiments`` lock is
# per-process — the API process (SSE/MCP/UI) can't see this worker's
# lock and vice versa, so the shared DB row is the only cross-process
# truth.  A crashed process leaves ``status='running'`` frozen; its
# ``updated_at`` goes stale (every live iteration refreshes it) and the
# row stops blocking.  Generous margin: a healthy experiment writes at
# least every few minutes.
_RUNNING_EXPERIMENT_FRESH_SECONDS = 3600.0


async def _load_settings() -> dict:
    from services.autoresearch_service import load_autoresearch_settings

    return await load_autoresearch_settings()


async def _enabled_strategy_ids() -> list[str]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(Strategy.id).where(Strategy.enabled.is_(True)).order_by(Strategy.id.asc())
            )
        ).scalars().all()
    return [str(row) for row in rows]


async def _has_running_experiment_elsewhere(strategy_id: str) -> bool:
    """Cross-process duplicate guard.

    The service locks per strategy in memory only, so an experiment
    started from the API (SSE endpoint, MCP tool, BacktestStudio) is
    invisible to this worker process.  Concurrent experiments on the
    same strategy double the LLM spend and race on auto-apply, so check
    the shared experiment row before starting our own.
    """
    cutoff = utcnow().timestamp() - _RUNNING_EXPERIMENT_FRESH_SECONDS
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(AutoresearchExperiment.updated_at).where(
                    AutoresearchExperiment.strategy_id == strategy_id,
                    AutoresearchExperiment.status == "running",
                )
            )
        ).scalars().all()
    for updated in rows:
        if updated is None:
            continue
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if updated.timestamp() >= cutoff:
            return True
    return False


async def _run_experiment(strategy_id: str) -> str:
    """Consume one experiment stream headless; return the terminal event."""
    from services.autoresearch_service import autoresearch_service

    settings = await _load_settings()
    mode = str(settings.get("mode") or "params")
    if mode == "code":
        stream = autoresearch_service.run_code_evolution_stream(
            trader_id=None,
            strategy_id=strategy_id,
        )
    else:
        stream = autoresearch_service.run_strategy_params_stream(strategy_id)

    final_event = "done"
    async for event_dict in stream:
        event_type = str(event_dict.get("event") or "")
        if event_type:
            final_event = event_type
        if event_type == "error":
            data = event_dict.get("data") or {}
            logger.warning(
                "Autoresearch experiment errored",
                strategy_id=strategy_id,
                mode=mode,
                error=data.get("error"),
            )
    return final_event


async def start_loop() -> None:
    logger.info("Autoresearch worker starting (poll=%.0fs)", _POLL_SECONDS)

    # Autoresearch drives the LLM proposer — without AI the experiments
    # fail immediately with "AI subsystem not initialized".  Mirrors the
    # reverse-engineer / news worker init pattern; treat failure as
    # recoverable (next poll tick retries).
    try:
        from services.ai import initialize_ai

        llm_manager = await initialize_ai()
        logger.info(
            "AI initialized in autoresearch worker (available=%s)",
            llm_manager.is_available(),
        )
    except Exception:
        logger.exception("AI init failed in autoresearch worker")

    last_finished_at: datetime | None = None
    cursor = 0
    while True:
        await asyncio.sleep(_POLL_SECONDS)
        try:
            settings = await _load_settings()
            interval = int(settings.get("interval_seconds") or 0)
        except Exception:
            logger.exception("Autoresearch worker: settings read failed")
            continue
        if interval <= 0:
            continue
        if last_finished_at is not None and (utcnow() - last_finished_at).total_seconds() < interval:
            continue

        try:
            strategy_ids = await _enabled_strategy_ids()
        except Exception:
            logger.exception("Autoresearch worker: strategy list read failed")
            continue
        if not strategy_ids:
            continue

        strategy_id = strategy_ids[cursor % len(strategy_ids)]
        cursor += 1
        try:
            if await _has_running_experiment_elsewhere(strategy_id):
                logger.info(
                    "Autoresearch worker: experiment already running elsewhere",
                    strategy_id=strategy_id,
                )
                last_finished_at = utcnow()
                continue
        except Exception:
            logger.exception(
                "Autoresearch worker: duplicate-experiment check failed",
                strategy_id=strategy_id,
            )
        started_at = utcnow()
        try:
            final_event = await _run_experiment(strategy_id)
        except Exception:
            logger.exception("Autoresearch worker: experiment crashed", strategy_id=strategy_id)
            final_event = "crashed"
        last_finished_at = utcnow()
        logger.info(
            "Autoresearch worker: experiment finished",
            strategy_id=strategy_id,
            final_event=final_event,
            duration_seconds=round((utcnow() - started_at).total_seconds(), 1),
            next_attempt_in_seconds=interval,
        )
