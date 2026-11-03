"""Schedule autoresearch on the jobs plane with shared controls and heartbeats."""

from __future__ import annotations

import asyncio
from datetime import datetime

from sqlalchemy import select

from models.database import AppSettings, AsyncSessionLocal, Strategy
from services.worker_state import clear_worker_run_request, read_worker_control, write_worker_snapshot
from utils.logger import get_logger
from utils.utcnow import utcnow

logger = get_logger("workers.autoresearch")

_POLL_SECONDS = 30.0


async def _load_settings() -> dict:
    from services.autoresearch_service import load_autoresearch_settings

    return await load_autoresearch_settings()


async def _enabled_strategy_ids() -> list[str]:
    async with AsyncSessionLocal() as session:
        rows = (
            (await session.execute(select(Strategy.id).where(Strategy.enabled.is_(True)).order_by(Strategy.id.asc())))
            .scalars()
            .all()
        )
    return [str(row) for row in rows]


async def _run_experiment(strategy_id: str, mode: str) -> str:
    from services.autoresearch_service import autoresearch_service

    if mode == "code":
        stream = autoresearch_service.run_code_evolution_stream(trader_id=None, strategy_id=strategy_id)
    else:
        stream = autoresearch_service.run_strategy_params_stream(strategy_id)
    final_event = "done"
    try:
        async for event_dict in stream:
            event_type = str(event_dict.get("event") or "")
            if event_type == "error":
                final_event = "error"
                logger.warning(
                    "Autoresearch experiment errored",
                    strategy_id=strategy_id,
                    mode=mode,
                    error=(event_dict.get("data") or {}).get("error"),
                )
            elif event_type == "done" and final_event != "error":
                final_event = str((event_dict.get("data") or {}).get("status") or "done")
    finally:
        await stream.aclose()
    return final_event


async def start_loop() -> None:
    from services.ai import get_llm_manager, initialize_ai

    active_task: asyncio.Task | None = None
    active_strategy_id: str | None = None
    active_mode: str | None = None
    last_finished_at: datetime | None = None
    last_run_at: datetime | None = None
    ai_revision = None
    previous_manager = None
    last_error = None
    final_event = None
    cursor = 0
    try:
        while True:
            try:
                if active_task is not None and active_task.done():
                    try:
                        final_event = active_task.result()
                    except Exception as exc:
                        final_event = "crashed"
                        last_error = str(exc)
                        logger.warning("Autoresearch experiment crashed", strategy_id=active_strategy_id, exc_info=exc)
                    active_task = None
                    active_strategy_id = None
                    active_mode = None
                    last_finished_at = utcnow()

                settings = await _load_settings()
                interval = int(settings["interval_seconds"])
                async with AsyncSessionLocal() as session:
                    control = await read_worker_control(session, "autoresearch", default_interval=600)
                    revision = await session.scalar(select(AppSettings.updated_at).where(AppSettings.id == "default"))

                enabled = bool(control["is_enabled"])
                paused = bool(control["is_paused"])
                requested = control.get("requested_run_at") is not None
                manager = None
                ai_ready = False
                if enabled and not paused and active_task is None and (interval > 0 or requested):
                    try:
                        manager = get_llm_manager()
                    except RuntimeError:
                        manager = await initialize_ai()
                    ai_ready = True
                    if manager is not previous_manager or revision != ai_revision or not manager.is_available():
                        ai_ready = await manager.initialize()
                        if ai_ready:
                            previous_manager = manager
                            ai_revision = revision

                if not enabled or paused:
                    if active_task is not None:
                        active_task.cancel()
                        await asyncio.gather(active_task, return_exceptions=True)
                        active_task = None
                        active_strategy_id = None
                        active_mode = None
                        last_finished_at = utcnow()
                    activity = "Disabled" if not enabled else "Paused"
                elif active_task is not None:
                    activity = f"Researching {active_strategy_id} ({active_mode})"
                elif interval <= 0 and not requested:
                    activity = "Scheduled research disabled"
                elif (
                    not requested
                    and last_finished_at is not None
                    and (utcnow() - last_finished_at).total_seconds() < interval
                ):
                    activity = "Waiting for next experiment"
                else:
                    if not ai_ready or not manager.is_available():
                        activity = "Waiting for AI configuration"
                        last_error = "AI configuration unavailable"
                    else:
                        strategy_ids = await _enabled_strategy_ids()
                        if strategy_ids:
                            active_strategy_id = strategy_ids[cursor % len(strategy_ids)]
                            cursor += 1
                            active_mode = str(settings["mode"])
                            active_task = asyncio.create_task(_run_experiment(active_strategy_id, active_mode))
                            last_run_at = utcnow()
                            last_error = None
                            activity = f"Researching {active_strategy_id} ({active_mode})"
                            if requested:
                                async with AsyncSessionLocal() as session:
                                    await clear_worker_run_request(session, "autoresearch")
                        else:
                            activity = "No enabled strategies"

                async with AsyncSessionLocal() as session:
                    await write_worker_snapshot(
                        session,
                        "autoresearch",
                        running=enabled and not paused and (interval > 0 or active_task is not None),
                        enabled=enabled,
                        current_activity=activity,
                        interval_seconds=int(_POLL_SECONDS),
                        last_run_at=last_run_at,
                        last_error=last_error,
                        stats={
                            "strategy_id": active_strategy_id,
                            "mode": active_mode,
                            "experiment_active": active_task is not None,
                            "schedule_interval_seconds": interval,
                            "final_event": final_event,
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Autoresearch scheduler tick failed", exc_info=exc)
            await asyncio.sleep(_POLL_SECONDS)
    finally:
        if active_task is not None:
            active_task.cancel()
            await asyncio.gather(active_task, return_exceptions=True)
