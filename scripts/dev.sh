#!/usr/bin/env bash
# Run Agent-as-Spec locally with real Postgres + Redis (+ mem0), without
# building any Docker image. Only Postgres and Redis run in containers; the
# API, runner and relay run on your machine with uvicorn / python.
#
#   ./scripts/dev.sh all       infra + migrate + api + runner + relay (Ctrl+C stops them)
#   ./scripts/dev.sh infra     start Postgres + Redis only
#   ./scripts/dev.sh migrate   apply migrations (creates the least-privilege app role)
#   ./scripts/dev.sh api       uvicorn with --reload, in the foreground
#   ./scripts/dev.sh runner    queue runner, in the foreground
#   ./scripts/dev.sh relay     outbox relay, in the foreground
#   ./scripts/dev.sh worker    the 42-capability test worker on :9100
#   ./scripts/dev.sh test      the test suite against this Postgres + Redis
#   ./scripts/dev.sh status    what is running
#   ./scripts/dev.sh down      stop Postgres + Redis (data kept)
#   ./scripts/dev.sh reset     stop and DELETE the Postgres + Redis data
#
# Environment knobs: DOCKER="sudo docker", SKIP_INFRA=1 (you run Postgres and
# Redis yourself), MEMORY_BACKEND=mem0, RUNNER_MODE=inline, PG_PORT, REDIS_PORT,
# API_PORT. Defaults live in scripts/dev.env.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/dev.env

DOCKER="${DOCKER:-docker}"
COMPOSE="$DOCKER compose -f docker-compose.infra.yml"
if [ -x .venv/bin/python ]; then PY=".venv/bin/python"; else PY="${PYTHON:-python3}"; fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

check_python_deps() {
  "$PY" - <<'EOF' || die "Missing Python packages. Run:  uv sync --all-extras   (then re-run this script)"
import importlib.util, sys
missing = [m for m in ("fastapi", "uvicorn", "asyncpg", "redis", "google.adk") if importlib.util.find_spec(m) is None]
if missing:
    print("missing:", ", ".join(missing)); sys.exit(1)
EOF
  if [ "$MEMORY_BACKEND" = "mem0" ]; then
    "$PY" -c "import mem0" 2>/dev/null || die "MEMORY_BACKEND=mem0 needs the memory extra:  uv sync --extra memory"
    if [ -z "${LLM_BASE_URL:-}" ] && ! grep -qE '^LLM_BASE_URL=.+' .env 2>/dev/null; then
      echo "WARNING: mem0 needs an embedding model. Set LLM_BASE_URL and LLM_API_KEY (in .env or the shell)," >&2
      echo "         otherwise memory writes will fail. Use MEMORY_BACKEND=mock to skip real memory." >&2
    fi
  fi
}

cmd_infra() {
  if [ "${SKIP_INFRA:-0}" = "1" ]; then echo "SKIP_INFRA=1: expecting Postgres on :$PG_PORT and Redis on :$REDIS_PORT"; return; fi
  command -v "${DOCKER%% *}" >/dev/null 2>&1 || die "docker not found. Install it, or run your own Postgres 16 + pgvector and Redis 7 and use SKIP_INFRA=1."
  say "Starting Postgres (pgvector) and Redis"
  $COMPOSE up -d --wait postgres redis || die "could not start the containers. If Docker needs sudo:  DOCKER='sudo docker' ./scripts/dev.sh all"
}

cmd_migrate() {
  say "Applying migrations"
  "$PY" -m app.core.migrate
}

cmd_api()    { say "API on http://127.0.0.1:$API_PORT/ui/"; SERVICE_ROLE=api exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir app; }
cmd_runner() { say "Runner"; SERVICE_ROLE=runner exec "$PY" -m app.sessions.runner; }
cmd_relay()  { say "Outbox relay"; SERVICE_ROLE=relay exec "$PY" -m app.relay_main; }
cmd_worker() { say "Test worker on :9100 (register it in the UI as http://127.0.0.1:9100)"; exec "$PY" scripts/run_worker.py --no-register --host 127.0.0.1; }

cmd_test() {
  # The suite DROPs the public schema and flushes its Redis DB, so it must
  # never run against the dev database: use a separate database + Redis DB 15.
  export DATABASE_URL="${DATABASE_URL%/agentspec}/agentspec_test"
  export DATABASE_MIGRATION_URL="${DATABASE_MIGRATION_URL%/agentspec}/agentspec_test"
  export REDIS_URL="${REDIS_URL%/0}/15"
  say "Running tests on database agentspec_test and Redis DB 15 (your dev data is untouched)"
  "$PY" - <<'EOF2'
import asyncio, os
import asyncpg
url = os.environ["DATABASE_MIGRATION_URL"].replace("postgresql+asyncpg", "postgresql")
admin = url.rsplit("/", 1)[0] + "/postgres"
async def main():
    conn = await asyncpg.connect(admin)
    if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname='agentspec_test'"):
        await conn.execute("CREATE DATABASE agentspec_test")
    await conn.close()
asyncio.run(main())
EOF2
  exec "$PY" -m pytest -q "$@"
}

cmd_status() {
  if [ "${SKIP_INFRA:-0}" != "1" ]; then $COMPOSE ps || true; fi
  if curl -fsS "http://127.0.0.1:$API_PORT/ready" >/dev/null 2>&1; then echo "api: ready on :$API_PORT"; else echo "api: not responding on :$API_PORT"; fi
}

cmd_all() {
  check_python_deps
  cmd_infra
  cmd_migrate
  say "Starting api, runner and relay (Ctrl+C stops all three)"
  trap 'echo; echo "stopping..."; kill 0 2>/dev/null; exit 0' INT TERM
  ( SERVICE_ROLE=api "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir app 2>&1 | sed -u 's/^/[api]    /' ) &
  ( SERVICE_ROLE=runner "$PY" -m app.sessions.runner 2>&1 | sed -u 's/^/[runner] /' ) &
  ( SERVICE_ROLE=relay "$PY" -m app.relay_main 2>&1 | sed -u 's/^/[relay]  /' ) &
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$API_PORT/ready" >/dev/null 2>&1; then
      printf '\n\033[1;32mReady:\033[0m  UI http://127.0.0.1:%s/ui/   |   docs http://127.0.0.1:%s/docs\n' "$API_PORT" "$API_PORT"
      echo "Log in with the dev login. For tools to chat with:  ./scripts/dev.sh worker   (second terminal)"
      break
    fi
    sleep 1
  done
  wait
}

case "${1:-all}" in
  all)     cmd_all ;;
  infra)   cmd_infra ;;
  migrate) check_python_deps; cmd_migrate ;;
  api)     check_python_deps; cmd_api ;;
  runner)  check_python_deps; cmd_runner ;;
  relay)   check_python_deps; cmd_relay ;;
  worker)  check_python_deps; cmd_worker ;;
  test)    shift; cmd_test "$@" ;;
  status)  cmd_status ;;
  down)    $COMPOSE down ;;
  reset)   $COMPOSE down -v ;;
  -h|--help|help) sed -n '2,22p' "$0" ;;
  *) die "unknown command '$1' (try: ./scripts/dev.sh help)" ;;
esac
