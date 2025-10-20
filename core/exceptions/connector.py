"""Connector-specific exceptions for data source connections."""

from typing import Any

from core.exceptions.base import IgniteLensBaseError


class ConnectorError(IgniteLensBaseError):
    """Base exception for all connector-related errors."""

    def __init__(
        self,
        message: str,
        connector_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize connector error.

        Args:
            message: Human-readable error message
            connector_key: The connector type that failed
            details: Additional error context

        """
        super().__init__(message, details=details)
        self.connector_key = connector_key


class ConnectionTestFailedError(ConnectorError):
    """Raised when connection test fails.

    This indicates the connection parameters are invalid or
    the remote service is unreachable.
    """

    def __init__(
        self,
        message: str,
        connector_key: str | None = None,
        host: str | None = None,
        port: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize connection test failure.

        Args:
            message: Human-readable error message
            connector_key: The connector type that failed
            host: Host that was attempted
            port: Port that was attempted
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.host = host
        self.port = port


class QueryExecutionError(ConnectorError):
    """Raised when query execution fails.

    This could be due to syntax errors, permission issues,
    or runtime errors in the query.
    """

    def __init__(
        self,
        message: str,
        connector_key: str | None = None,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize query execution error.

        Args:
            message: Human-readable error message
            connector_key: The connector type that failed
            query: The query that failed (may be truncated)
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.query = query


class UnsupportedConnectorError(ConnectorError):
    """Raised when requested connector type is not supported."""

    def __init__(
        self,
        connector_key: str,
        available_connectors: list[str] | None = None,
    ) -> None:
        """Initialize unsupported connector error.

        Args:
            connector_key: The unsupported connector type
            available_connectors: List of supported connector types

        """
        available = ", ".join(available_connectors) if available_connectors else "none"
        message = f"Connector '{connector_key}' is not supported. Available connectors: {available}"
        super().__init__(message, connector_key=connector_key)
        self.available_connectors = available_connectors or []


class ConnectionPoolExhaustedError(ConnectorError):
    """Raised when connection pool has no available connections."""

    def __init__(
        self,
        message: str = "Connection pool exhausted, no available connections",
        connector_key: str | None = None,
        pool_size: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize pool exhausted error.

        Args:
            message: Human-readable error message
            connector_key: The connector type
            pool_size: Maximum pool size
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.pool_size = pool_size


class ConnectionTimeoutError(ConnectorError):
    """Raised when connection attempt times out."""

    def __init__(
        self,
        message: str,
        connector_key: str | None = None,
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize connection timeout error.

        Args:
            message: Human-readable error message
            connector_key: The connector type
            timeout_seconds: The timeout value that was exceeded
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.timeout_seconds = timeout_seconds


class QueryTimeoutError(ConnectorError):
    """Raised when query execution times out."""

    def __init__(
        self,
        message: str,
        connector_key: str | None = None,
        query: str | None = None,
        timeout_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize query timeout error.

        Args:
            message: Human-readable error message
            connector_key: The connector type
            query: The query that timed out (may be truncated)
            timeout_seconds: The timeout value that was exceeded
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.query = query
        self.timeout_seconds = timeout_seconds


class InvalidCredentialsError(ConnectorError):
    """Raised when provided credentials are invalid."""

    def __init__(
        self,
        message: str = "Authentication failed with provided credentials",
        connector_key: str | None = None,
        username: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize invalid credentials error.

        Args:
            message: Human-readable error message
            connector_key: The connector type
            username: Username that failed (for logging/debugging)
            details: Additional error context

        """
        super().__init__(message, connector_key=connector_key, details=details)
        self.username = username
