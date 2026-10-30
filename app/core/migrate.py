"""Versioned Postgres migrations (Flyway-style), safe to run from every pod
at start-up:

  - a Postgres advisory lock serialises concurrent runners -- the first pod
    migrates, the others wait and then find nothing to do;
  - bootstrap: when the app connects as a different role than the one
    migrating (DATABASE_URL vs DATABASE_MIGRATION_URL), that least-privilege
    role is created if missing (NOSUPERUSER NOBYPASSRLS) and, after every
    run, granted exactly the table privileges it needs -- so no init scripts
    have to be mounted into the database container;
  - step 0001 creates any missing base table from the ORM metadata
    (partitioned tables excluded -- they're created by 0003's DDL);
  - every later step is a reviewed SQL file in migrations/sql, applied once,
    in order, inside a transaction, and recorded in `schema_migrations`
    with its checksum -- an applied file that was edited afterwards is
    refused instead of silently diverging.

Schema changes to existing tables are always a new SQL file, never an
edit to the ORM alone.

    python -m app.core.migrate        # apply (also runs on API start-up)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)
SQL_DIR = Path(__file__).resolve().parents[2] / "migrations" / "sql"
LOCK_ID = 727_101  # arbitrary, stable advisory-lock key


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _app_role(engine: AsyncEngine):
    """(user, password, database) the application connects as, if it is a
    different Postgres role from the one running migrations."""
    from sqlalchemy.engine import make_url

    from app.config import settings

    app_url = make_url(settings.database_url)
    if not app_url.drivername.startswith("postgresql") or not app_url.username:
        return None
    if app_url.username == engine.url.username:
        return None
    return app_url.username, app_url.password or "", app_url.database or ""


async def _bootstrap_role(conn, role) -> None:  # noqa: ANN001
    user, password, database = role
    exists = (await conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": user})).scalar()
    if not exists:
        await conn.exec_driver_sql(
            f"CREATE ROLE {_ident(user)} LOGIN PASSWORD {_literal(password)} NOSUPERUSER NOBYPASSRLS")
        logger.info("created application role %s", user)
    if database:
        await conn.exec_driver_sql(f"GRANT CONNECT ON DATABASE {_ident(database)} TO {_ident(user)}")


async def _grant_app_role(conn, role) -> None:  # noqa: ANN001
    user = _ident(role[0])
    for stmt in (
        f"GRANT USAGE ON SCHEMA public TO {user}",
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {user}",
        f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {user}",
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {user}",
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO {user}",
        # ADK's session service and mem0 create their own tables at first use.
        f"GRANT CREATE ON SCHEMA public TO {user}",
        f"REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM {user}",
    ):
        await conn.exec_driver_sql(stmt)


def _files() -> list[tuple[str, str, str]]:
    out = []
    for path in sorted(SQL_DIR.glob("*.sql")):
        body = path.read_text()
        out.append((path.stem, body, hashlib.sha256(body.encode()).hexdigest()))
    return out


async def apply(engine: AsyncEngine) -> list[str]:
    from app.db import Base, _all_models

    _all_models()
    applied: list[str] = []
    async with engine.connect() as conn:
        await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": LOCK_ID})
        await conn.commit()  # the advisory lock is session-level: it outlives this commit
        role = _app_role(engine)
        try:
            if role:
                async with conn.begin():
                    await _bootstrap_role(conn, role)
            async with conn.begin():
                await conn.execute(text(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, "
                    "checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"))
                await conn.execute(text("SELECT set_config('app.role', 'system', true)"))
                tables = [t for t in Base.metadata.sorted_tables if not t.info.get("partitioned")]
                await conn.run_sync(lambda sync: Base.metadata.create_all(sync, tables=tables))
            done = {r[0]: r[1] for r in (await conn.execute(text(
                "SELECT version, checksum FROM schema_migrations"))).all()}
            await conn.commit()
            for version, body, checksum in _files():
                if version in done:
                    if done[version] != checksum:
                        raise RuntimeError(f"migration {version} was modified after being applied")
                    continue
                async with conn.begin():
                    await conn.execute(text("SELECT set_config('app.role', 'system', true)"))
                    raw = await conn.get_raw_connection()
                    await raw.driver_connection.execute(body)  # multi-statement, simple protocol
                    await conn.execute(text("INSERT INTO schema_migrations (version, checksum) VALUES (:v, :c)"),
                                       {"v": version, "c": checksum})
                applied.append(version)
                logger.info("applied migration %s", version)
            if role:
                async with conn.begin():
                    await _grant_app_role(conn, role)
        finally:
            await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": LOCK_ID})
            await conn.commit()
    return applied


async def _main() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import settings
    from app.core.logging import configure_logging

    configure_logging(settings.log_level, json=settings.log_json)
    # Migrate as the OWNER role; the app role may not create tables.
    owner = create_async_engine(settings.database_migration_url or settings.database_url)
    try:
        print("applied:", await apply(owner) or "nothing to do")
    finally:
        await owner.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
