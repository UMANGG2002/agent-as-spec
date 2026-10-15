"""Shared async SQLAlchemy engine/session. SQLite by default (per the
architecture doc's "runs fully local" scope decision); point DATABASE_URL at
Postgres for anything beyond the minimal build -- nothing else changes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    # Import model modules so their tables register on Base.metadata before create_all.
    from app.marketplace import models as _marketplace_models  # noqa: F401
    from app.registry import store as _registry_store  # noqa: F401
    from app.knowledge import store as _knowledge_store  # noqa: F401
    from app.runtime import approvals as _approvals  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
