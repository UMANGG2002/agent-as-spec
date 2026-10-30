# Run it locally without building Docker images

Only Postgres (with pgvector) and Redis run in containers, and both are public images that are pulled, not built. The API, the runner and the relay run on your machine with uvicorn and Python, so a code change reloads in a second.

## One-time setup

```bash
uv sync --all-extras          # installs the app, mem0, fastmcp and the test tools
chmod +x scripts/dev.sh       # once
```

You need Docker with Compose v2 (`docker compose`). If Docker needs sudo on your machine, prefix every command with `DOCKER="sudo docker"`.

## Start everything

```bash
./scripts/dev.sh all
```

This does four things in order:

1. Starts Postgres 16 + pgvector and Redis 7 (`docker-compose.infra.yml`) and waits until both are healthy.
2. Runs the migrations as the Postgres owner. This also creates the least-privilege `agentspec_app` role the app connects as, so row-level security is enforced exactly as in production.
3. Starts three processes with prefixed logs: `[api]` (uvicorn with auto-reload), `[runner]` and `[relay]`.
4. Prints the URLs once `/ready` answers.

Open <http://127.0.0.1:8000/ui/> and use the dev login. API docs are at <http://127.0.0.1:8000/docs>. Press Ctrl+C to stop all three processes. The containers keep running, and your data stays.

## Give the agents tools to use

In a second terminal:

```bash
./scripts/dev.sh worker
```

This starts the 42-capability test worker on port 9100. In the UI's Agent Builder tab, click **Register all** with the URL `http://127.0.0.1:9100`. The API runs on your machine, so `127.0.0.1` is correct here (inside Docker it would be `host.docker.internal`).

## What each command does

| Command | What it does |
|---|---|
| `./scripts/dev.sh all` | infra, migrate, api, runner, relay |
| `./scripts/dev.sh infra` | Postgres and Redis only |
| `./scripts/dev.sh migrate` | apply migrations |
| `./scripts/dev.sh api` / `runner` / `relay` | one process in the foreground, for debugging in its own terminal |
| `./scripts/dev.sh worker` | the test worker with 42 capabilities |
| `./scripts/dev.sh test` | the test suite, on a separate `agentspec_test` database and Redis DB 15, so dev data is never touched |
| `./scripts/dev.sh status` | containers and API health |
| `./scripts/dev.sh down` | stop the containers, keep the data |
| `./scripts/dev.sh reset` | stop the containers and delete the data |

## Settings

Defaults live in `scripts/dev.env`. Your `.env` is never modified, and it still supplies the model settings (`GOOGLE_API_KEY`, or `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL_NAME`). Anything you export in the shell wins over `dev.env`.

| Setting | Default | Notes |
|---|---|---|
| `MEMORY_BACKEND` | `mock` | `mem0` uses self-hosted mem0 on the Postgres above. It needs an embedding model, so set `LLM_BASE_URL` and `LLM_API_KEY`. The script warns if they are missing. |
| `RUNNER_MODE` | `queue` | `inline` runs agents inside the API process, with no runner needed. |
| `PG_PORT`, `REDIS_PORT`, `API_PORT` | 5432, 6379, 8000 | Change them if a port is taken. |
| `SKIP_INFRA` | off | Set to `1` if you run Postgres and Redis yourself (the script then expects the default passwords in `dev.env`). |

Examples:

```bash
MEMORY_BACKEND=mem0 ./scripts/dev.sh all
PG_PORT=5433 REDIS_PORT=6380 ./scripts/dev.sh all
RUNNER_MODE=inline ./scripts/dev.sh api      # API only, no runner process
```

## Lightest option: no Docker at all

```bash
uv run uvicorn app.main:app --reload
```

With nothing configured this uses SQLite, in-memory Redis and the inline runner. It is the quickest way to try the UI, but it does not exercise Postgres row-level security, Redis Streams or the separate runner. Use `dev.sh all` when you want those.

## Troubleshooting

- **`permission denied` on the Docker socket:** run `DOCKER="sudo docker" ./scripts/dev.sh all`.
- **Port already in use:** set `PG_PORT` or `REDIS_PORT` as above.
- **`Missing Python packages`:** run `uv sync --all-extras` in the project folder.
- **Old `marketplace.db` errors:** that file is only used in the no-Docker mode and is upgraded automatically.
- **Start clean:** `./scripts/dev.sh reset`, then `./scripts/dev.sh all`.
