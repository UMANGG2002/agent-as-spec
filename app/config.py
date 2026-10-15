"""Central settings. Everything here has a safe local-dev default so the
scaffold runs with zero external services configured, per the architecture
doc's "runs fully local" assumption."""

from __future__ import annotations

from typing import Any

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
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "local"

    database_url: str = "sqlite+aiosqlite:///./marketplace.db"
    redis_url: str = "redis://localhost:6379/0"
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "capability_marketplace"
    # Where approved AgentSpecs live when nothing's listening on mongo_url --
    # see app/specs/store.py's module docstring for why this has to be a
    # shared file, not a plain in-memory dict.
    agent_specs_fallback_path: str = "./.data/agent_specs_fallback.json"
    # A human-readable JSON mirror of the capability registry (whose real
    # source of truth is the `capabilities` SQL table -- see
    # app/registry/store.py). Written alongside every `save()`, purely so the
    # registry can be read with a text editor instead of a DB client; nothing
    # ever reads this file back.
    capabilities_json_path: str = "./.data/capabilities.json"
    # A spec's runtime config can name a `custom` callback by dotted import
    # path (see app/factory/callback_registry.py). That's arbitrary code
    # execution chosen by whoever approves a spec, so it's off by default
    # and, when on, limited to modules under these prefixes -- set e.g.
    # ALLOW_CUSTOM_CALLBACKS=true and
    # CUSTOM_CALLBACK_MODULE_PREFIXES='["my_company.callbacks."]'.
    allow_custom_callbacks: bool = False
    custom_callback_module_prefixes: list[str] = ["app.callbacks."]

    jwt_signing_key: str = "dev-only-change-me"
    jwt_issuer: str = "capability-marketplace"
    access_token_ttl_seconds: int = 900
    delegated_token_ttl_seconds: int = 120

    # Two ways to get a live model, either of which turns on `planner_is_live`:
    #  - Google's native Gemini API: GOOGLE_API_KEY (+ optionally GEMINI_MODEL)
    #  - Any OpenAI-compatible endpoint via LiteLLM (self-hosted gateway, a
    #    proxy in front of several providers, etc.): LLM_API_KEY + LLM_BASE_URL
    #    (+ LLM_MODEL_NAME, the model id *that endpoint* expects, not Google's).
    # If both are set, the LiteLLM one wins -- it's the one that needs two
    # values set together on purpose, so it's never accidentally live.
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model_name: str| None = None
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "capability-marketplace"

    composio_api_key: str | None = None

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


settings = Settings()
