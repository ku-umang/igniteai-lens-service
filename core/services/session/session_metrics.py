"""Session-specific business metrics for Prometheus monitoring."""

import time
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram

from core.logging import get_logger

logger = get_logger(__name__)

# Session operation metrics
session_operations_total = Counter(
    "session_operations_total",
    "Total number of session operations",
    ["operation", "status"],
)

session_operation_duration_seconds = Histogram(
    "session_operation_duration_seconds",
    "Time spent on session operations",
    ["operation"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Session state metrics
sessions_active_total = Gauge(
    "sessions_active_total",
    "Number of active sessions by status",
    ["status"],
)

sessions_by_tenant = Gauge(
    "sessions_by_tenant",
    "Number of sessions per tenant",
    ["tenant_id"],
)

# Database query metrics for sessions
session_db_query_duration_seconds = Histogram(
    "session_db_query_duration_seconds",
    "Duration of database queries for session operations",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

session_db_queries_total = Counter(
    "session_db_queries_total",
    "Total number of database queries for sessions",
    ["query_type", "status"],
)


class SessionMetrics:
    """Session metrics collector and reporter."""

    def __init__(self):
        self._operation_timers: Dict[str, float] = {}

    def record_operation_start(self, operation: str, operation_id: str) -> None:
        """Record the start of a session operation."""
        self._operation_timers[operation_id] = time.time()
        logger.debug("Session operation started", operation=operation, operation_id=operation_id)

    def record_operation_complete(
        self,
        operation: str,
        operation_id: str,
        success: bool = True,
    ) -> None:
        """Record completion of a session operation."""
        status = "success" if success else "error"
        session_operations_total.labels(operation=operation, status=status).inc()

        # Record duration if timer exists
        if operation_id in self._operation_timers:
            duration = time.time() - self._operation_timers[operation_id]
            session_operation_duration_seconds.labels(operation=operation).observe(duration)
            del self._operation_timers[operation_id]

        logger.debug(
            "Session operation completed",
            operation=operation,
            operation_id=operation_id,
            success=success,
        )

    def record_session_created(self, tenant_id: str, count: Optional[int] = None, operation_id: Optional[str] = None) -> None:
        """Record session creation.

        Args:
            tenant_id: Tenant identifier
            count: Optional actual count from database. If provided, sets the gauge to this value.
                   If not provided, increments the gauge (legacy behavior, may cause drift).
            operation_id: Optional operation identifier

        """
        self.record_operation_complete("create", operation_id or "session_create", success=True)
        if count is not None:
            self.set_tenant_sessions(tenant_id, count)
        else:
            self.increment_tenant_sessions(tenant_id)

    def record_session_deleted(self, tenant_id: str, count: Optional[int] = None, operation_id: Optional[str] = None) -> None:
        """Record session deletion.

        Args:
            tenant_id: Tenant identifier
            count: Optional actual count from database. If provided, sets the gauge to this value.
                   If not provided, decrements the gauge (legacy behavior, may cause negative values).
            operation_id: Optional operation identifier

        """
        self.record_operation_complete("delete", operation_id or "session_delete", success=True)
        if count is not None:
            self.set_tenant_sessions(tenant_id, count)
        else:
            self.decrement_tenant_sessions(tenant_id)

    def record_session_updated(self, operation_id: Optional[str] = None, success: bool = True) -> None:
        """Record session update."""
        self.record_operation_complete("update", operation_id or "session_update", success=success)

    def record_session_list(self, count: int, operation_id: Optional[str] = None, success: bool = True) -> None:
        """Record session list operation."""
        self.record_operation_complete("list", operation_id or "session_list", success=success)
        logger.debug("Session list operation", count=count)

    def record_session_get(self, operation_id: Optional[str] = None, success: bool = True) -> None:
        """Record session get operation."""
        self.record_operation_complete("get", operation_id or "session_get", success=success)

    def update_active_sessions(self, status: str, count: int) -> None:
        """Update the gauge for active sessions by status."""
        sessions_active_total.labels(status=status).set(count)
        logger.debug("Updated active sessions count", status=status, count=count)

    def increment_tenant_sessions(self, tenant_id: str) -> None:
        """Increment session count for a tenant."""
        sessions_by_tenant.labels(tenant_id=tenant_id).inc()

    def decrement_tenant_sessions(self, tenant_id: str) -> None:
        """Decrement session count for a tenant."""
        sessions_by_tenant.labels(tenant_id=tenant_id).dec()

    def set_tenant_sessions(self, tenant_id: str, count: int) -> None:
        """Set the exact session count for a tenant based on actual database count.

        This is the preferred method to ensure metrics accuracy and avoid negative values.

        Args:
            tenant_id: Tenant identifier
            count: Actual number of sessions from database

        """
        sessions_by_tenant.labels(tenant_id=tenant_id).set(count)
        logger.debug("Set tenant sessions count", tenant_id=tenant_id, count=count)

    def record_db_query(
        self,
        query_type: str,
        duration: float,
        success: bool = True,
    ) -> None:
        """Record database query metrics for session operations."""
        status = "success" if success else "error"
        session_db_queries_total.labels(query_type=query_type, status=status).inc()
        session_db_query_duration_seconds.labels(query_type=query_type).observe(duration)

        logger.debug(
            "Session DB query recorded",
            query_type=query_type,
            duration=duration,
            success=success,
        )


# Global metrics instance
session_metrics = SessionMetrics()
