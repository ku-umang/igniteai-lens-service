import sys
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from prometheus_client import start_http_server

from core.config import settings

# Logger will be set after avoiding circular import
logger = None


def _get_logger():
    """Get logger instance, avoiding circular import."""
    global logger
    if logger is None:
        # Import here to avoid circular dependency
        from core.logging import get_logger

        logger = get_logger(__name__)
    return logger


# Global instances
tracer: Optional[trace.Tracer] = None
_trace_provider: Optional[TracerProvider] = None
_span_processors: List[BatchSpanProcessor] = []
_meter_provider: Optional[MeterProvider] = None
_logger_provider: Optional[LoggerProvider] = None
_log_processors: List[BatchLogRecordProcessor] = []
_logging_handler: Optional[LoggingHandler] = None


class SafeConsoleSpanExporter(SpanExporter):
    """A console span exporter that handles I/O errors gracefully."""

    def __init__(self):
        self._shutdown = False

    def export(self, spans) -> SpanExportResult:
        """Export spans to console with error handling."""
        if self._shutdown:
            return SpanExportResult.FAILURE

        try:
            for span in spans:
                # Simple span output - avoid complex formatting that might fail
                span_context = span.get_span_context()
                span_dict = {
                    "name": span.name,
                    "trace_id": f"{span_context.trace_id:032x}" if span_context else "unknown",
                    "span_id": f"{span_context.span_id:016x}" if span_context else "unknown",
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                }

                sys.stdout.write(f"Span: {span_dict}\n")
                sys.stdout.flush()

            return SpanExportResult.SUCCESS

        except (ValueError, OSError, AttributeError) as e:
            # I/O operation on closed file or similar errors
            _get_logger().debug("Console exporter I/O error", error=str(e))
            return SpanExportResult.FAILURE
        except Exception as e:
            _get_logger().warning("Unexpected error in console exporter", error=str(e))
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any pending spans."""
        if self._shutdown:
            return False
        try:
            if not sys.stdout.closed:
                sys.stdout.flush()
            return True
        except Exception:
            return False


class SafeConsoleLogExporter(LogExporter):
    """A console log exporter that handles I/O errors gracefully."""

    def __init__(self):
        self._shutdown = False

    def export(self, batch) -> None:
        """Export log records to console with error handling."""
        if self._shutdown:
            return

        try:
            for log_record in batch:
                # Simple log output - avoid complex formatting that might fail
                # Access LogRecord attributes correctly
                log_dict = {
                    "timestamp": getattr(log_record, "timestamp", None),
                    "severity": getattr(log_record, "severity_text", None),
                    "body": str(getattr(log_record, "body", "")),
                    "trace_id": f"{trace_id:032x}" if (trace_id := getattr(log_record, "trace_id", None)) else None,
                    "span_id": f"{span_id:016x}" if (span_id := getattr(log_record, "span_id", None)) else None,
                }

                sys.stdout.write(f"Log: {log_dict}\n")
                sys.stdout.flush()

        except (ValueError, OSError, AttributeError) as e:
            # I/O operation on closed file or similar errors
            _get_logger().debug("Console log exporter I/O error", error=str(e))
        except Exception as e:
            _get_logger().warning("Unexpected error in console log exporter", error=str(e))

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True

    def force_flush(self, _timeout_millis: int = 30000) -> bool:
        """Force flush any pending logs."""
        if self._shutdown:
            return False
        try:
            if not sys.stdout.closed:
                sys.stdout.flush()
            return True
        except Exception:
            return False


def create_resource() -> Resource:
    """Create OpenTelemetry resource with service information."""
    return Resource.create(
        {
            "service.name": settings.OTEL_SERVICE_NAME,
            "service.version": settings.OTEL_SERVICE_VERSION,
            "service.environment": settings.ENVIRONMENT.value,
        }
    )


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing with Jaeger support."""
    global tracer, _trace_provider, _span_processors

    # Create resource
    resource = create_resource()

    # Configure sampling
    sampler = TraceIdRatioBased(rate=settings.TRACE_SAMPLING_RATE)

    # Set up trace provider with sampling
    provider = TracerProvider(resource=resource, sampler=sampler)

    _trace_provider = provider

    # Configure OTLP exporter for Jaeger (if enabled)
    if settings.JAEGER_ENABLED:
        try:
            otlp_endpoint = f"http://{settings.JAEGER_AGENT_HOST}:4317"
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(otlp_processor)
            trace.set_tracer_provider(provider)
            _span_processors.append(otlp_processor)
            _get_logger().info("OTLP gRPC trace exporter configured", endpoint=otlp_endpoint)
        except ImportError:
            _get_logger().debug("OTLP gRPC exporter not available")
        except Exception as e:
            _get_logger().warning("Failed to configure OTLP gRPC exporter", error=str(e))

    # Fallback to console exporter for development if no exporters configured
    if not _span_processors and settings.ENVIRONMENT == "development":
        try:
            console_exporter = SafeConsoleSpanExporter()
            console_processor = BatchSpanProcessor(
                console_exporter,
                export_timeout_millis=1000,
                max_export_batch_size=64,
                schedule_delay_millis=500,
            )
            provider.add_span_processor(console_processor)
            _span_processors.append(console_processor)
            _get_logger().info("Console trace exporter configured for development")
        except Exception as e:
            _get_logger().warning("Failed to configure console exporter", error=str(e))

    # Create global tracer
    tracer = trace.get_tracer(__name__)
    _get_logger().info(
        "OpenTelemetry tracing configured", sampling_rate=settings.TRACE_SAMPLING_RATE, exporters_count=len(_span_processors)
    )


def setup_metrics() -> None:
    """Configure OpenTelemetry metrics with Prometheus."""
    global _meter_provider

    if not settings.ENABLE_METRICS:
        return

    # Create resource
    resource = create_resource()

    # Set up Prometheus metric reader
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    set_meter_provider(provider)
    _meter_provider = provider

    # Start Prometheus HTTP server
    try:
        start_http_server(settings.METRICS_PORT)
        _get_logger().info("Prometheus metrics server started", port=settings.METRICS_PORT)
    except Exception as e:
        _get_logger().error("Failed to start Prometheus metrics server", error=str(e))


def setup_logging() -> None:
    """Configure OpenTelemetry logging with OTLP support."""
    global _logger_provider, _log_processors, _logging_handler

    if not settings.JAEGER_LOGS_ENABLED:
        return

    # Create resource
    resource = create_resource()

    # Set up logger provider
    provider = LoggerProvider(resource=resource)
    set_logger_provider(provider)
    _logger_provider = provider

    # Configure OTLP exporter for logs (if Jaeger is enabled)
    if settings.JAEGER_ENABLED:
        try:
            otlp_endpoint = f"http://{settings.JAEGER_AGENT_HOST}:4317"
            otlp_log_exporter = OTLPLogExporter(endpoint=otlp_endpoint, insecure=True)
            otlp_log_processor = BatchLogRecordProcessor(otlp_log_exporter)
            provider.add_log_record_processor(otlp_log_processor)
            _log_processors.append(otlp_log_processor)
            _get_logger().info("OTLP gRPC log exporter configured", endpoint=otlp_endpoint)
        except ImportError:
            _get_logger().debug("OTLP gRPC log exporter not available")
        except Exception as e:
            _get_logger().warning("Failed to configure OTLP gRPC log exporter", error=str(e))

    # Add console exporter for development if no exporters configured
    if not _log_processors and settings.ENVIRONMENT == "development":
        try:
            console_log_exporter = SafeConsoleLogExporter()
            console_log_processor = BatchLogRecordProcessor(
                console_log_exporter,
                export_timeout_millis=1000,
                max_export_batch_size=64,
                schedule_delay_millis=500,
            )
            provider.add_log_record_processor(console_log_processor)
            _log_processors.append(console_log_processor)
            _get_logger().info("Console log exporter configured for development")
        except Exception as e:
            _get_logger().warning("Failed to configure console log exporter", error=str(e))

    # Create logging handler that can be used by Python logging
    _logging_handler = LoggingHandler(logger_provider=provider)

    _get_logger().info("OpenTelemetry logging configured", exporters_count=len(_log_processors))


def get_logging_handler() -> Optional[LoggingHandler]:
    """Get the OpenTelemetry logging handler."""
    return _logging_handler


def get_logger_provider():
    """Get the OpenTelemetry logger provider."""
    return _logger_provider


def instrument_app(app: Any) -> None:
    """Instrument FastAPI application with OpenTelemetry."""
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls="/health,/metrics",  # Exclude health check and metrics endpoints
        )
        _get_logger().info("FastAPI instrumentation enabled")

        # Instrument SQLAlchemy
        SQLAlchemyInstrumentor().instrument()
        _get_logger().info("SQLAlchemy instrumentation enabled")

        # Instrument Redis
        RedisInstrumentor().instrument()
        _get_logger().info("Redis instrumentation enabled")

    except Exception as e:
        _get_logger().error("Failed to instrument application", error=str(e))


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance."""
    if tracer is None:
        setup_tracing()
    return tracer or trace.get_tracer(__name__)


def _add_trace_attributes(span, attributes: Optional[Dict[str, Any]], args, kwargs):
    """Add attributes to span for traced functions."""
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    if args:
        span.set_attribute("function.args_count", len(args))
    if kwargs:
        span.set_attribute("function.kwargs_count", len(kwargs))


def trace_function(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function calls."""
    import asyncio
    from functools import wraps

    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with get_tracer().start_as_current_span(span_name) as span:
                    _add_trace_attributes(span, attributes, args, kwargs)
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        span.set_attribute("function.error", str(e))
                        raise

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with get_tracer().start_as_current_span(span_name) as span:
                    _add_trace_attributes(span, attributes, args, kwargs)
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        span.set_attribute("function.error", str(e))
                        raise

            return sync_wrapper

    return decorator


# Alias for backwards compatibility
trace_async_function = trace_function


def shutdown_observability() -> None:
    """Shutdown all observability components gracefully."""
    global _trace_provider, _meter_provider, _logger_provider, _logging_handler

    _get_logger().info("Shutting down observability components")

    # Shutdown log processors first
    for processor in _log_processors:
        try:
            # Force flush any pending logs first
            processor.force_flush(timeout_millis=1000)

            # Shutdown the processor
            processor.shutdown()
            _get_logger().debug("Log processor shutdown completed")
        except Exception as e:
            _get_logger().warning("Error shutting down log processor", error=str(e))

    # Shutdown span processors
    for processor in _span_processors:
        try:
            # Force flush any pending spans first
            processor.force_flush(timeout_millis=1000)

            # Shutdown the processor
            processor.shutdown()
            _get_logger().debug("Span processor shutdown completed")
        except Exception as e:
            _get_logger().warning("Error shutting down span processor", error=str(e))

    # Clear the processors lists
    _log_processors.clear()
    _span_processors.clear()

    # Reset global providers
    _trace_provider = None
    _meter_provider = None
    _logger_provider = None
    _logging_handler = None

    _get_logger().info("Observability components shutdown completed")


def init_observability() -> None:
    """Initialize all observability components."""
    try:
        setup_tracing()
        setup_logging()
        setup_metrics()
        _get_logger().info("Observability initialized")
    except Exception as e:
        _get_logger().error("Failed to initialize observability", error=str(e))
        # Don't let observability failures prevent app startup
        _get_logger().warning("Application will continue without full observability")
