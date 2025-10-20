"""Metrics middleware for tracking HTTP request/response metrics."""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.api.metrics import (
    http_exceptions_total,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_requests_in_progress,
    http_requests_total,
    http_response_size_bytes,
)
from core.logging import get_logger

logger = get_logger(__name__)

# Endpoints to exclude from metrics collection
EXCLUDED_ENDPOINTS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP request/response metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        # Skip metrics collection for excluded endpoints
        if request.url.path in EXCLUDED_ENDPOINTS:
            return await call_next(request)

        # Normalize endpoint path (replace path params with placeholders)
        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method

        # Track requests in progress
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        # Record request size
        request_size = request.headers.get("content-length")
        if request_size:
            try:
                http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(int(request_size))
            except ValueError:
                pass

        # Record start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Record response metrics
            duration = time.time() - start_time
            status_code = response.status_code

            # Record metrics
            http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

            # Record response size
            response_size = response.headers.get("content-length")
            if response_size:
                try:
                    http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(int(response_size))
                except ValueError:
                    pass

            return response

        except Exception as exc:
            # Record exception metrics
            duration = time.time() - start_time
            exception_type = type(exc).__name__

            http_exceptions_total.labels(method=method, endpoint=endpoint, exception_type=exception_type).inc()
            http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

            # Re-raise the exception to be handled by exception handlers
            raise

        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

    @staticmethod
    def _normalize_endpoint(path: str) -> str:
        """Normalize endpoint path by replacing UUIDs and IDs with placeholders.

        This prevents cardinality explosion in metrics by grouping similar endpoints.

        Examples:
            /api/v1/sessions/123e4567-e89b-12d3-a456-426614174000 -> /api/v1/sessions/{id}
            /api/v1/users/42 -> /api/v1/users/{id}

        """
        import re

        # Replace UUIDs with {id}
        path = re.sub(
            r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "/{id}",
            path,
            flags=re.IGNORECASE,
        )

        # Replace numeric IDs with {id}
        path = re.sub(r"/\d+", "/{id}", path)

        return path
