"""OpenTelemetry setup and the instruments used by the request path (ADR-0009)."""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

TRACER_NAME = "modelmesh"


class Instruments:
    def __init__(self, meter: metrics.Meter) -> None:
        self.requests = meter.create_counter("mm.requests")
        self.request_duration = meter.create_histogram("mm.request.duration", unit="ms")
        self.attempt_duration = meter.create_histogram("mm.attempt.duration", unit="ms")
        self.ttft = meter.create_histogram("mm.ttft", unit="ms")
        self.tokens = meter.create_counter("mm.tokens")
        self.cost = meter.create_counter("mm.cost.estimated_usd", unit="USD")
        self.fallbacks = meter.create_counter("mm.fallbacks")
        self.cache_hits = meter.create_counter("mm.cache.hits")
        self.router_latency = meter.create_histogram("mm.router.decision_latency", unit="ms")


@dataclass
class Telemetry:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    tracer: trace.Tracer
    instruments: Instruments

    def shutdown(self) -> None:
        self.tracer_provider.shutdown()
        self.meter_provider.shutdown()


def build_telemetry(
    *,
    service_name: str,
    otlp_endpoint: str | None,
    metrics_enabled: bool = False,
    span_processors: list[SpanProcessor] | None = None,
    metric_readers: list[MetricReader] | None = None,
    sample_ratio: float = 1.0,
) -> Telemetry:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource, sampler=ParentBased(TraceIdRatioBased(sample_ratio)))
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    for processor in span_processors or []:
        provider.add_span_processor(processor)

    readers: list[MetricReader] = list(metric_readers or [])
    if otlp_endpoint and metrics_enabled:
        metric_exporter = OTLPMetricExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/metrics")
        readers.append(PeriodicExportingMetricReader(metric_exporter))
    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    return Telemetry(
        tracer_provider=provider,
        meter_provider=meter_provider,
        tracer=provider.get_tracer(TRACER_NAME),
        instruments=Instruments(meter_provider.get_meter(TRACER_NAME)),
    )


def current_trace_id() -> str:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else "0" * 32
