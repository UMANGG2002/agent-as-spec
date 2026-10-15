"""One TracerProvider + one MeterProvider for the whole process, per the
architecture doc's Observability section. Every other layer (auth, guardrails,
tool wrapper, redis stream writer) pulls its tracer/meter from here rather than
configuring its own exporter.
"""

from __future__ import annotations

import logging

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import settings

_configured = False


class _CurrentStdout:
    """The console exporters' `out` stream. OpenTelemetry's default binds
    whatever `sys.stdout` is at construction time and keeps writing to it
    from background export threads -- so if that stream is later swapped
    and closed (pytest's output capture does exactly this; so can a
    process shutting down), every later export raises "I/O operation on
    closed file" and dumps a traceback. Resolving `sys.stdout` at write
    time, and dropping output once nothing usable is left, keeps console
    telemetry a best-effort debug aid instead of a source of noise."""

    def write(self, text: str) -> int:
        import sys

        try:
            return sys.stdout.write(text)
        except (ValueError, AttributeError):
            return 0

    def flush(self) -> None:
        import sys

        try:
            sys.stdout.flush()
        except (ValueError, AttributeError):
            pass


def configure_telemetry() -> None:
    """Idempotent: call once at process start (app.main imports this)."""
    global _configured
    if _configured:
        return

    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})

    tracer_provider = TracerProvider(resource=resource)
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        span_exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
    else:
        span_exporter = ConsoleSpanExporter(out=_CurrentStdout())
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        metric_exporter = OTLPMetricExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/metrics")
    else:
        metric_exporter = ConsoleMetricExporter(out=_CurrentStdout())
    reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    logging.getLogger(__name__).info(
        "telemetry configured (otlp=%s)", bool(settings.otel_exporter_otlp_endpoint)
    )
    _configured = True


tracer = trace.get_tracer("capability_marketplace")
meter = metrics.get_meter("capability_marketplace")

invocations_total = meter.create_counter(
    "invocations_total", description="Capability invocations, by tenant/capability/status"
)
guardrail_blocks_total = meter.create_counter(
    "guardrail_blocks_total", description="Guardrail blocks, by policy/stage"
)
tool_latency_ms = meter.create_histogram("tool_latency_ms", unit="ms")
model_latency_ms = meter.create_histogram("model_latency_ms", unit="ms")
