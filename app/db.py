"""Database engines and sessions.

  - Postgres (staging/production): asyncpg with a bounded pool, pre-ping,
    recycle, a per-transaction statement timeout, and optional read-replica
    routing (`ReadSessionLocal`) for read-heavy endpoints. Behind PgBouncer
    (transaction pooling) set DB_BEHIND_PGBOUNCER=true: asyncpg's prepared
    statement cache is turned off and every setting is `SET LOCAL`, which
    PgBouncer can safely multiplex.
  - SQLite (local/test): unchanged developer experience.

Row-level security: on Postgres every transaction begins with
`set_config('app.tenant_id', <tenant>, true)` -- taken from the request's
tenant context (`app.core.context`) or an explicit `tenant_session(...)` --
and the RLS policies in migrations/sql/0002_rls.sql make Postgres itself
refuse rows of any other tenant. A transaction with no tenant bound sees
nothing tenant-scoped; background jobs that legitimately work across
tenants (outbox relay, reapers) use `system_session()`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import settings
from app.core import context

logger = logging.getLogger(__name__)


def _engine_kwargs(url: str) -> dict:
    if not url.startswith("postgresql"):
        return {"echo": False}
    connect_args: dict = {"server_settings": {"application_name": f"agent-as-spec-{settings.service_role}"}}
    if settings.db_behind_pgbouncer:
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_cache_size"] = 0
        connect_args.pop("server_settings")  # PgBouncer rejects unknown startup params
    return {
        "echo": False,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_timeout": settings.db_pool_timeout_seconds,
        "pool_recycle": settings.db_pool_recycle_seconds,
        "pool_pre_ping": True,
        "connect_args": connect_args,
    }


engine = create_async_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

read_engine = (
    create_async_engine(settings.database_read_url, **_engine_kwargs(settings.database_read_url))
    if settings.database_read_url
    else engine
)
ReadSessionLocal = async_sessionmaker(read_engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


SYSTEM = "__system__"


@event.listens_for(Session, "after_begin")
def _scope_transaction(session: Session, transaction, connection) -> None:  # noqa: ANN001
    """Bind tenant + timeouts to every new transaction (Postgres only)."""
    if connection.dialect.name != "postgresql":
        return
    tenant = session.info.get("tenant_id") or context.current_tenant()
    connection.exec_driver_sql(f"SET LOCAL statement_timeout = {int(settings.db_statement_timeout_ms)}")
    if not settings.db_enforce_rls:
        return
    if tenant == SYSTEM:
        connection.execute(text("SELECT set_config('app.role', 'system', true)"))
    elif tenant:
        connection.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant})


def _all_models() -> None:
    """Import every module that declares tables so they register on Base."""
    from app.core import audit as _audit  # noqa: F401
    from app.core import outbox as _outbox  # noqa: F401
    from app.evals import store as _evals_store  # noqa: F401
    from app.knowledge import store as _knowledge_store  # noqa: F401
    from app.marketplace import models as _marketplace_models  # noqa: F401
    from app.registry import store as _registry_store  # noqa: F401
    from app.runtime import approvals as _approvals  # noqa: F401
    from app.sessions import models as _session_models  # noqa: F401
    from app.specs import store as _specs_store  # noqa: F401
    from app.tenancy import models as _tenancy_models  # noqa: F401


async def init_db() -> None:
    """Local/test: create tables directly. Postgres: apply the versioned SQL
    migrations instead (app/core/migrate.py), which also install RLS,
    full-text indexes and partitioning."""
    _all_models()
    if engine.dialect.name == "postgresql":
        from app.core import migrate

        if settings.database_migration_url and settings.database_migration_url != settings.database_url:
            owner = create_async_engine(settings.database_migration_url)
            try:
                await migrate.apply(owner)
            finally:
                await owner.dispose()
        else:
            await migrate.apply(engine)
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_sqlite_add_missing_columns)


def _sqlite_add_missing_columns(sync_conn) -> None:  # noqa: ANN001
    """Local dev convenience: `create_all` never alters an existing table, so a
    marketplace.db created by an older version lacks newer columns ("no such
    column: capabilities.search_text"). Add whatever is missing as a nullable
    column. Additive only -- never drops/renames; Postgres uses real migrations."""
    from sqlalchemy import inspect

    insp = inspect(sync_conn)
    existing_tables = set(insp.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in have:
                continue
            ddl_type = col.type.compile(dialect=sync_conn.dialect)
            sync_conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {ddl_type}')
            logger.warning("sqlite: added missing column %s.%s (upgrading an older local DB)", table.name, col.name)


@asynccontextmanager
async def session_scope(*, tenant_id: str | None = None) -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        if tenant_id:
            session.info["tenant_id"] = tenant_id
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def tenant_session(tenant_id: str):
    return session_scope(tenant_id=tenant_id)


def system_session():
    """Cross-tenant maintenance work only (outbox relay, reapers, migrations)."""
    return session_scope(tenant_id=SYSTEM)


async def ping() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
