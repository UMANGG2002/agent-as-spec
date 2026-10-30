"""The `relay` process role: publishes the transactional outbox to Redis
Streams and keeps time partitions ahead of the calendar.

    python -m app.relay_main

Run 2+ replicas for availability; `FOR UPDATE SKIP LOCKED` lets them share
the work without double-publishing a batch.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from sqlalchemy import text

from app.config import settings
from app.core.logging import configure_logging
from app.core.outbox import relay_once
from app.db import engine, init_db, system_session

logger = logging.getLogger(__name__)


async def maintain_partitions() -> None:
    if engine.dialect.name != "postgresql":
        return
    async with system_session() as session:
        for parent in ("outbox_events", "audit_log"):
            await session.execute(text("SELECT ensure_monthly_partitions(:p)"), {"p": parent})


async def main() -> None:
    configure_logging(settings.log_level, json=settings.log_json)
    await init_db()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    last_maintenance = 0.0
    while not stop.is_set():
        try:
            if time.monotonic() - last_maintenance > 86_400:
                await maintain_partitions()
                last_maintenance = time.monotonic()
            async with system_session() as session:
                published = await relay_once(session)
            if not published:
                await asyncio.sleep(0.2)
        except Exception:  # noqa: BLE001 - keep relaying
            logger.exception("outbox relay iteration failed")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
