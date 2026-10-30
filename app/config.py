"""Central settings. Everything here has a safe local-dev default so the
scaffold runs with zero external services configured, per the architecture
doc's "runs fully local" assumption."""

from __future__ import annotations

from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _GatewayLiteLlm(**kwargs: Any) -> Any:
    """Build the `LiteLlm` instance for the custom gateway, with capability
    detection pinned instead of guessed.

    ADK decides whether a model supports combining `output_schema` and
    `tools` in a single turn via `LiteLlm.capabilities.output_schema_and_tools`
    (see `google/adk/models/lite_llm.py`'s `_resolve_capabilities`). For a
    custom model id it doesn't recognize (like a gateway-registered alias
    such as) on a generic OpenAI-compatible route, it can't
    look up real model info, so it falls back to *assuming* native support
    rather than leaving it unknown.

    When that assumption is wrong, ADK skips the one thing that reliably
    tells the model when to stop: its own documented workaround of injecting
    an explicit `set_model_response` tool with an instruction to call it when
    done (see `google/adk/flows/llm_flows/prompt/_schema.py`). Instead it
    just sets a raw structured-output schema on every turn and hopes the
    model naturally produces a schema-conformant, tool-call-free response on
    its own. If it never does, `LlmAgent`'s run loop -- a bare `while True`
    with no iteration cap (see `base_llm_flow.py`'s `run_async`) -- never
    sees a final response and keeps calling the model forever. That is what
    caused the planner agent's infinite-loop-of-LLM-calls symptom: paired
    with `PlanReActPlanner`'s own, different, free-text finishing protocol
    (a `/*FINAL_ANSWER*/` tag) layered on top, the model had three
    conflicting ideas of how to signal "I'm done" and satisfied none of
    them reliably.

    Declaring `output_schema_and_tools=False` here forces ADK onto its safer,
    explicit path regardless of what this unverified gateway model actually
    supports. Needs `google-adk[extensions]` installed (pulls in `litellm`);
    only imported here, lazily, so the rest of the app still runs fine
    without it when only Gemini is used."""
    from google.adk.models.lite_llm import LiteLlm, LlmCapabilities

    class _PinnedCapabilityLiteLlm(LiteLlm):
        @property
        def capabilities(self) -> LlmCapabilities:
            return LlmCapabilities(output_schema_and_tools=False)

    return _PinnedCapabilityLiteLlm(**kwargs)


class Settings(BaseSettings):
    """Every knob, with a safe local default. `environment` picks the
    profile: `local` (zero external services -- SQLite, an in-process Redis
    stand-in, mocked memory) or `staging`/`production` (Postgres, Redis,
    mem0, asymmetric JWT keys -- and `_validate_profile` refuses to start if
    any dev-only default is still in place)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"  # local | test | staging | production
    service_role: str = "api"  # api | runner | relay -- which process this is
    log_level: str = "INFO"
    log_json: bool = False  # JSON lines in staging/production (forced on there)

    # --- Postgres ---------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///./marketplace.db"
    # Owner role used ONLY to run migrations (DDL, RLS policies). The app
    # itself connects with DATABASE_URL as a least-privilege role, so RLS
    # applies to it. Defaults to DATABASE_URL (single-role dev setups).
    database_migration_url: str | None = None
    # Optional read replica for read-heavy paths (catalog search, listings).
    database_read_url: str | None = None
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout_seconds: float = 5.0
    db_pool_recycle_seconds: int = 1800
    db_statement_timeout_ms: int = 5000
    # PgBouncer in transaction mode can't keep server-side prepared
    # statements across transactions; set true when running behind it.
    db_behind_pgbouncer: bool = False
    # Row-level security: every tenant-scoped transaction runs
    # `SET LOCAL app.tenant_id = ...` so Postgres itself refuses cross-tenant
    # rows (see migrations/sql/0002_rls.sql). Postgres only.
    db_enforce_rls: bool = True
    # ADK session store (sessions + events). Defaults to database_url; a
    # separate database/cluster is recommended at scale (append-heavy).
    session_database_url: str | None = None

    # Local only: a human-readable JSON mirror of the capability registry.
    capabilities_json_path: str = "./.data/capabilities.json"

    # --- Redis --------------------------------------------------------------
    # "memory://" = in-process stand-in (fakeredis): local/test only, not
    # shared across processes. Anything else is a real Redis URL.
    redis_url: str = "memory://"
    redis_cluster: bool = False
    redis_max_connections: int = 200
    redis_socket_timeout_seconds: float = 2.0

    # --- Caching ------------------------------------------------------------
    cache_l1_max_items: int = 50_000
    cache_default_ttl_seconds: int = 60
    cache_ttl_jitter_ratio: float = 0.1
    compiled_agent_cache_size: int = 512

    # --- Sessions, runners, streaming ----------------------------------------
    # inline: the API process runs invocations itself (local dev);
    # queue: the API enqueues and separate runner workers execute (prod).
    runner_mode: str = "inline"
    work_queue_shards: int = 16
    runner_concurrency: int = 32  # concurrent invocations per runner process
    runner_visibility_timeout_ms: int = 60_000  # reclaim a crashed worker's job after this
    session_lease_ttl_ms: int = 30_000
    # Cap on events loaded per session turn (compaction summaries keep older
    # context available); bounds DB reads for very long conversations.
    session_max_events_loaded: int = 1_000
    invocation_timeout_seconds: int = 600
    stream_maxlen: int = 5_000
    stream_retention_seconds: int = 86_400
    stream_heartbeat_seconds: float = 15.0
    stream_block_ms: int = 5_000

    # --- Tenancy limits (defaults; per-tenant overrides in the tenants table)
    default_tenant_tier: str = "standard"
    rate_limit_enabled: bool = True
    max_request_body_bytes: int = 1_048_576
    max_message_chars: int = 32_000
    load_shed_max_inflight: int = 2_000  # per API process; 503 above this

    # --- Memory (mem0) ------------------------------------------------------
    # mock = deterministic in-process memory (default for local/test);
    # mem0 = self-hosted mem0 OSS with pgvector in Postgres.
    memory_backend: str = "auto"  # auto | mock | mem0 | off
    memory_search_timeout_seconds: float = 0.8
    memory_top_k: int = 5
    mem0_pgvector_dsn: str | None = None  # defaults to database_url (sync DSN)
    mem0_collection: str = "agent_memories"
    mem0_embedding_model: str = "text-embedding-3-small"
    mem0_embedding_dims: int = 1536
    mem0_llm_model: str | None = None  # defaults to llm_model_name

    # --- Context compaction (ADK EventsCompactionConfig) ----------------------
    context_compaction_enabled: bool = True
    context_compaction_token_threshold: int = 60_000
    context_compaction_retain_events: int = 12
    context_compaction_interval: int | None = 20  # also compact every N invocations
    context_compaction_overlap: int = 2
    compaction_model_name: str | None = None  # cheaper gateway model for summaries

    # -- evaluation (app/evals) ----------------------------------------------------------
    eval_judge_model_name: str | None = None  # a different gateway model for the LLM judge
    eval_judge_samples: int = 1  # >1 (odd): median of N judge samples per rubric
    eval_on_approval: bool = False  # run the tenant's golden sets when a spec version is approved
    eval_online_sample_rate: float = 0.0  # fraction of finished production runs scored by the judge (0 = off)
    eval_gate_max_pass_rate_drop: float = 0.0  # regression gate: allowed pass-rate drop vs baseline
    eval_gate_max_judge_drop: float = 0.05  # regression gate: allowed mean-judge-score drop
    eval_concurrency: int = 4

    # --- Security -----------------------------------------------------------
    allow_custom_callbacks: bool = False
    custom_callback_module_prefixes: list[str] = ["app.callbacks."]
    # HS256 with jwt_signing_key in local/test. Staging/production: RS256 or
    # ES256 with a key ring -- JWT_PRIVATE_KEYS_JSON='{"k2026a": "-----BEGIN..."}'
    # plus JWT_ACTIVE_KID; public keys are published at /.well-known/jwks.json.
    jwt_signing_key: str = "dev-only-change-me"
    jwt_algorithm: str = "HS256"
    jwt_private_keys_json: str | None = None
    jwt_public_keys_json: str | None = None
    jwt_active_kid: str | None = None
    jwt_issuer: str = "capability-marketplace"
    jwt_audience: str = "agent-as-spec"
    access_token_ttl_seconds: int = 900
    delegated_token_ttl_seconds: int = 120
    # Vault envelope encryption: versioned data keys, newest first. Rotate by
    # prepending a key; old ciphertexts still decrypt. Use a KMS in production.
    vault_keys_json: str | None = None
    cors_allow_origins: list[str] = []
    # Outbound calls to capability endpoints: private/link-local/metadata
    # addresses are refused unless listed here (SSRF guard). Local/test allow
    # loopback so the test worker works.
    outbound_allow_private_networks: bool = False
    outbound_allowed_hosts: list[str] = []

    # --- Outbound resilience --------------------------------------------------
    tool_call_timeout_seconds: float = 30.0
    tool_call_max_retries: int = 2
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_seconds: float = 30.0
    capability_max_concurrency: int = 64  # per capability, per process (bulkhead)
    mcp_pool_idle_seconds: float = 300.0

    # --- Models ---------------------------------------------------------------
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model_name: str | None = None

    # --- Observability --------------------------------------------------------
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "capability-marketplace"
    otel_trace_sample_ratio: float = 1.0  # set ~0.01-0.05 at 100k rps
    otel_console_export: bool = False  # local debugging only

    composio_api_key: str | None = None

    @property
    def is_production_like(self) -> bool:
        return self.environment in ("staging", "production")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def effective_memory_backend(self) -> str:
        if self.memory_backend != "auto":
            return self.memory_backend
        return "mem0" if self.is_production_like else "mock"

    @model_validator(mode="after")
    def _validate_profile(self) -> "Settings":
        if self.environment in ("local", "test"):
            # Loopback test workers are fine locally.
            if "outbound_allow_private_networks" not in self.model_fields_set:
                self.outbound_allow_private_networks = True
            return self
        problems = []
        if not self.is_postgres:
            problems.append("DATABASE_URL must be Postgres")
        if self.redis_url.startswith("memory://"):
            problems.append("REDIS_URL must point at a real Redis")
        if self.jwt_algorithm.upper().startswith("HS") or not self.jwt_private_keys_json or not self.jwt_active_kid:
            problems.append("use RS256/ES256 with JWT_PRIVATE_KEYS_JSON + JWT_ACTIVE_KID")
        if not self.vault_keys_json:
            problems.append("VAULT_KEYS_JSON is required")
        if self.runner_mode != "queue":
            problems.append("RUNNER_MODE must be 'queue'")
        if problems:
            raise ValueError(f"unsafe {self.environment} configuration: " + "; ".join(problems))
        self.log_json = True
        return self

    @property
    def planner_is_live(self) -> bool:
        """Whether we have what we need to call a real model, vs. run the
        deterministic offline planner used by tests and the demo script."""
        return bool(self.google_api_key) or bool(self.llm_api_key and self.llm_base_url)

    @property
    def live_model(self) -> "str | Any":
        """What to hand ADK's `model=` parameter (`LlmAgent`/`Agent` accept
        either a plain model-name string or a `BaseLlm` instance): a
        `LiteLlm` wrapper pointed at `llm_base_url` when a custom gateway is
        configured, otherwise the plain Gemini model string. The `openai/`
        prefix is LiteLLM's own convention for "treat this as a generic
        OpenAI-compatible chat-completions endpoint at a custom api_base"
        -- the part after it (`llm_model_name`) is whatever model id *that
        endpoint* expects, e.g. an alias your gateway registered such as
        "", not a real OpenAI model name. Needs
        `google-adk[extensions]` installed (pulls in the `litellm` package);
        only imported here, lazily, so the rest of the app still runs fine
        without it when only Gemini is used."""
        if self.llm_api_key and self.llm_base_url:
            return _GatewayLiteLlm(
                model=f"openai/{self.llm_model_name}",
                api_key=self.llm_api_key,
                api_base=self.llm_base_url,
            )
        return self.gemini_model


    @property
    def compaction_model(self) -> "str | Any | None":
        """Model for context-compaction summaries: a cheaper gateway alias if
        configured, else None (ADK then uses the agent's own model)."""
        if self.compaction_model_name and self.llm_api_key and self.llm_base_url:
            return _GatewayLiteLlm(
                model=f"openai/{self.compaction_model_name}", api_key=self.llm_api_key, api_base=self.llm_base_url
            )
        return None


settings = Settings()
