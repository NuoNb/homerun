"""Database ownership and cancellation shared by every autoresearch entry point."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, AsyncIterator

from sqlalchemy import or_, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from models.database import AsyncSessionLocal, AutoresearchExperiment, async_engine
from utils.logger import get_logger
from utils.retry import run_with_db_retries
from utils.utcnow import utcnow

logger = get_logger(__name__)

_CONTROL_POLL_SECONDS = 1.0
_HEARTBEAT_SECONDS = 30.0


def experiment_lock_keys(trader_id: str | None, strategy_id: str | None) -> list[int]:
    scopes = []
    if trader_id is not None:
        scopes.append(f"trader:{trader_id}")
    if strategy_id is not None:
        scopes.append(f"strategy:{strategy_id}")
    return sorted(
        int.from_bytes(hashlib.blake2b(f"homerun:autoresearch:{scope}".encode(), digest_size=8).digest(), signed=True)
        for scope in scopes
    )


@dataclass
class AutoresearchRun:
    experiment_id: str
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    final_status: str = "failed"
    control_error: Exception | None = None
    producer: asyncio.Task | None = field(default=None, init=False, repr=False)

    async def lock_for_apply(self, session) -> bool:
        status = await session.scalar(
            select(AutoresearchExperiment.status)
            .where(AutoresearchExperiment.id == self.experiment_id)
            .with_for_update()
        )
        if status != "running":
            self.stop_requested.set()
        return status == "running" and not self.stop_requested.is_set()

    async def consume(self, stream: AsyncGenerator[dict[str, Any], None]) -> AsyncGenerator[dict[str, Any], None]:
        events: asyncio.Queue = asyncio.Queue(maxsize=1)

        async def produce() -> None:
            try:
                async for event in stream:
                    await events.put(event)
            finally:
                await stream.aclose()

        self.producer = asyncio.create_task(produce())
        stopped = asyncio.create_task(self.stop_requested.wait())
        pending = None
        failed = False
        try:
            while not self.stop_requested.is_set():
                pending = asyncio.create_task(events.get())
                done, _ = await asyncio.wait((self.producer, pending, stopped), return_when=asyncio.FIRST_COMPLETED)
                if stopped in done:
                    break
                if self.producer in done and not pending.done() and events.empty():
                    self.producer.result()
                    self.final_status = "failed" if failed else "completed"
                    return
                event = await pending
                if event.get("event") == "error":
                    failed = True
                if event.get("event") == "done":
                    self.final_status = "failed" if failed else "completed"
                    event = {**event, "data": {**event.get("data", {}), "status": self.final_status}}
                yield event
            if self.control_error is not None:
                raise RuntimeError("Autoresearch lost database ownership") from self.control_error
            self.final_status = "paused"
        except Exception:
            self.final_status = "failed"
            raise
        except (asyncio.CancelledError, GeneratorExit):
            if not failed and self.final_status != "completed":
                self.final_status = "paused"
            raise
        finally:
            tasks = [self.producer, stopped]
            if pending is not None:
                tasks.append(pending)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.producer = None
        yield {
            "event": "done",
            "data": {"experiment_id": self.experiment_id, "status": "paused", "early_stop_reason": "stopped"},
        }


@asynccontextmanager
async def claim_experiment(
    *,
    trader_id: str | None,
    strategy_id: str | None,
    mode: str,
    settings: dict[str, Any],
) -> AsyncIterator[AutoresearchRun | None]:
    keys = experiment_lock_keys(trader_id, strategy_id)
    async with async_engine.connect() as connection:
        acquired = []
        monitor = None
        run = None
        try:
            for key in keys:
                locked = await connection.scalar(
                    text("SELECT pg_try_advisory_lock(CAST(:key AS bigint))"), {"key": key}
                )
                await connection.commit()
                if not locked:
                    yield None
                    return
                acquired.append(key)

            run = AutoresearchRun(str(uuid.uuid4()))

            async def create_record(_attempt: int) -> None:
                now = utcnow()
                scopes = []
                if trader_id is not None:
                    scopes.append(AutoresearchExperiment.trader_id == trader_id)
                if strategy_id is not None:
                    scopes.append(AutoresearchExperiment.strategy_id == strategy_id)
                async with AsyncSessionLocal() as session:
                    await session.execute(
                        update(AutoresearchExperiment)
                        .where(
                            AutoresearchExperiment.status == "running",
                            AutoresearchExperiment.id != run.experiment_id,
                            or_(*scopes),
                        )
                        .values(status="failed", finished_at=now, updated_at=now)
                    )
                    await session.execute(
                        pg_insert(AutoresearchExperiment)
                        .values(
                            id=run.experiment_id,
                            trader_id=trader_id,
                            strategy_id=strategy_id,
                            name=f"{mode}-{now.strftime('%Y%m%d-%H%M')}",
                            mode=mode,
                            status="running",
                            settings_json=settings,
                            started_at=now,
                            created_at=now,
                            updated_at=now,
                        )
                        .on_conflict_do_nothing(index_elements=[AutoresearchExperiment.id])
                    )
                    await session.commit()

            await run_with_db_retries(create_record)

            async def watch_control() -> None:
                last_heartbeat = time.monotonic()
                try:
                    while True:
                        await asyncio.sleep(_CONTROL_POLL_SECONDS)
                        status = await connection.scalar(
                            select(AutoresearchExperiment.status).where(AutoresearchExperiment.id == run.experiment_id)
                        )
                        if time.monotonic() - last_heartbeat >= _HEARTBEAT_SECONDS and status == "running":
                            await connection.execute(
                                update(AutoresearchExperiment)
                                .where(
                                    AutoresearchExperiment.id == run.experiment_id,
                                    AutoresearchExperiment.status == "running",
                                )
                                .values(updated_at=utcnow())
                            )
                            last_heartbeat = time.monotonic()
                        await connection.commit()
                        if status != "running":
                            run.stop_requested.set()
                            if run.producer is not None:
                                run.producer.cancel()
                            return
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    run.control_error = exc
                    run.stop_requested.set()
                    if run.producer is not None:
                        run.producer.cancel()
                    logger.warning("Autoresearch ownership check failed", experiment_id=run.experiment_id, exc_info=exc)

            monitor = asyncio.create_task(watch_control())
            yield run
        finally:
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            try:
                if run is not None:

                    async def finish_record(_attempt: int) -> None:
                        async with AsyncSessionLocal() as session:
                            await session.execute(
                                update(AutoresearchExperiment)
                                .where(
                                    AutoresearchExperiment.id == run.experiment_id,
                                    AutoresearchExperiment.status == "running",
                                )
                                .values(status=run.final_status, finished_at=utcnow(), updated_at=utcnow())
                            )
                            await session.commit()

                    await run_with_db_retries(finish_record)
            finally:
                try:
                    await connection.rollback()
                    for key in reversed(acquired):
                        await connection.execute(text("SELECT pg_advisory_unlock(CAST(:key AS bigint))"), {"key": key})
                    await connection.commit()
                except BaseException:
                    await connection.invalidate()
                    raise
