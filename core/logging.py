import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import structlog
from opentelemetry import trace

from core.config import settings
from core.observability import get_logging_handler

# Context variable to store correlation ID across async requests
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current request context."""
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current request context."""
    correlation_id_var.set(correlation_id)


def add_correlation_id(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add correlation ID to log entries."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def add_service_info(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add service information to log entries."""
    event_dict["service"] = settings.OTEL_SERVICE_NAME
    event_dict["version"] = settings.OTEL_SERVICE_VERSION
    event_dict["environment"] = settings.ENVIRONMENT.value
    return event_dict


def get_trace_id() -> Optional[str]:
    """Get the current trace ID from OpenTelemetry context."""
    try:
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().trace_id != trace.INVALID_TRACE_ID:
            # Convert trace ID to hex string (32 characters, zero-padded)
            return f"{current_span.get_span_context().trace_id:032x}"
    except Exception:
        pass
    return None


def get_span_id() -> Optional[str]:
    """Get the current span ID from OpenTelemetry context."""
    try:
        current_span = trace.get_current_span()
        if current_span and current_span.get_span_context().span_id != trace.INVALID_SPAN_ID:
            # Convert span ID to hex string (16 characters, zero-padded)
            return f"{current_span.get_span_context().span_id:016x}"
    except Exception:
        pass
    return None


def add_trace_context(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add trace and span IDs to log entries."""
    trace_id = get_trace_id()
    span_id = get_span_id()

    if trace_id:
        event_dict["trace_id"] = trace_id
    if span_id:
        event_dict["span_id"] = span_id

    return event_dict


def add_otel_logging(_logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Send log entries to OpenTelemetry logging."""
    # Get the OpenTelemetry logging handler
    otel_handler = get_logging_handler()
    if not otel_handler:
        return event_dict

    try:
        # Convert structlog level to standard logging level
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }

        # Get log level from event_dict or method_name
        log_level = event_dict.get("level", method_name)
        if isinstance(log_level, str):
            log_level = log_level.lower()

        numeric_level = level_map.get(log_level, logging.INFO)

        # Create a log record
        record = logging.LogRecord(
            name=event_dict.get("logger", "structlog"),
            level=numeric_level,
            pathname="",
            lineno=0,
            msg=event_dict.get("event", ""),
            args=(),
            exc_info=None,
        )

        # Add additional attributes to the record
        record.correlation_id = event_dict.get("correlation_id")
        record.service = event_dict.get("service")
        record.version = event_dict.get("version")
        record.environment = event_dict.get("environment")

        # Add any extra fields as attributes
        for key, value in event_dict.items():
            if key not in ["event", "level", "logger", "correlation_id", "service", "version", "environment", "timestamp"]:
                setattr(record, key, value)

        # Send to OpenTelemetry
        otel_handler.emit(record)

    except Exception:
        # Don't let OpenTelemetry errors break regular logging
        pass

    return event_dict


def configure_logging() -> None:
    """Configure structured logging with structlog."""
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.value),
    )

    # Common processors for all loggers
    common_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        add_trace_context,  # Add trace and span IDs to logs
        add_service_info,
        add_otel_logging,  # Send logs to OpenTelemetry
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        # JSON formatter for production
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable formatter for development
        renderer = structlog.dev.ConsoleRenderer(colors=settings.ENVIRONMENT == "development")

    # Configure structlog
    structlog.configure(
        processors=common_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Add OpenTelemetry handler to root logger if available
    # This ensures that direct Python logging calls also get sent to OpenTelemetry
    try:
        otel_handler = get_logging_handler()
        if otel_handler:
            root_logger = logging.getLogger()
            root_logger.addHandler(otel_handler)
    except ImportError:
        # OpenTelemetry may not be fully initialized yet
        pass

    # Configure specific loggers
    # Silence noisy third-party loggers in production
    if settings.ENVIRONMENT == "production":
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)


class CentralizedLogger:
    """Centralized logger with OpenTelemetry integration for both structured logging and tracing."""

    def __init__(self, name: str = __name__):
        self.name = name
        self.logger = structlog.get_logger(name)
        self.tracer = trace.get_tracer(name)

    def _log_with_trace(self, level: str, event: str, **kwargs):
        """Log with OpenTelemetry trace context and span attributes."""
        span = trace.get_current_span()

        if span and span.is_recording():
            self._add_span_event(span, level, event, **kwargs)
            self._add_span_attributes(span, **kwargs)
            if level in ["error", "critical"]:
                self._set_span_status_on_error(span, event)

        # Log using structlog
        try:
            log_method = getattr(self.logger, level)
            log_method(event, **kwargs)
        except Exception:
            # Fallback to standard logging if structlog fails
            std_logger = logging.getLogger(self.name)
            getattr(std_logger, level, std_logger.info)(f"{event}: {kwargs}")

    def _add_span_event(self, span, level: str, event: str, **kwargs):
        """Add event to span for Jaeger UI visibility."""
        event_attributes = {
            "level": level.upper(),
            "logger": self.name,
            "timestamp": int(time.time() * 1000),
            "event_name": event,
        }

        # Add kwargs as event attributes (exclude exc_info)
        for key, value in kwargs.items():
            if key != "exc_info":
                event_attributes[key] = str(value) if isinstance(value, (dict, list)) else value

        span.add_event(f"[{level.upper()}] {event}", attributes=event_attributes)

    def _add_span_attributes(self, span, **kwargs):
        """Add attributes to span for searchability."""
        for key, value in kwargs.items():
            if key == "exc_info":
                continue

            attr_key = f"log.{key}"
            safe_value = self._convert_to_safe_attribute(value)

            try:
                span.set_attribute(attr_key, safe_value)
            except Exception:
                pass  # Skip attributes that fail to set

    @staticmethod
    def _convert_to_safe_attribute(value):
        """Convert value to safe span attribute."""
        if value is None:
            return "null"

        if isinstance(value, (str, bool, float)):
            return value if isinstance(value, str) and len(str(value)) <= 500 else str(value)[:500]

        if isinstance(value, int):
            # Handle large integers that might cause protobuf issues
            return str(value) if (value > 2**63 - 1 or value < -(2**63)) else value

        if isinstance(value, (dict, list)):
            import json

            return json.dumps(value, default=str)[:500]

        return str(value)[:500]

    @staticmethod
    def _set_span_status_on_error(span, event: str):
        """Set span status for error levels."""
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, event))
        except Exception:
            pass

    def debug(self, event: str, **kwargs):
        """Log debug message with OpenTelemetry integration."""
        self._log_with_trace("debug", event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log info message with OpenTelemetry integration."""
        self._log_with_trace("info", event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning message with OpenTelemetry integration."""
        self._log_with_trace("warning", event, **kwargs)

    def warn(self, event: str, **kwargs):
        """Alias for warning to match standard logging interface."""
        self.warning(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error message with OpenTelemetry integration."""
        self._log_with_trace("error", event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical message with OpenTelemetry integration."""
        self._log_with_trace("critical", event, **kwargs)

    def exception(self, event: str, **kwargs):
        """Log exception with traceback and OpenTelemetry integration."""
        # Add exc_info=True to capture the exception traceback
        kwargs["exc_info"] = True
        self._log_with_trace("error", event, **kwargs)

        # Record the exception in the current span
        span = trace.get_current_span()
        if span and span.is_recording():
            _, exc_value, _ = sys.exc_info()
            if exc_value:
                span.record_exception(exc_value)

    def with_context(self, **kwargs):
        """Return a new logger instance with added context."""
        # Create a new bound logger with context
        bound_logger = self.logger.bind(**kwargs)

        # Create new CentralizedLogger instance with the bound logger
        new_logger = CentralizedLogger(self.name)
        new_logger.logger = bound_logger
        return new_logger

    def bind(self, **kwargs):
        """Alias for with_context to match structlog interface."""
        return self.with_context(**kwargs)


def get_logger(name: str) -> CentralizedLogger:
    """Get a configured centralized logger instance."""
    return CentralizedLogger(name)
