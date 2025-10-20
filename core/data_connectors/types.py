"""Type definitions for data connectors."""

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, TypedDict


class QueryResult(TypedDict):
    """Result of a query execution.

    Attributes:
        columns: List of column names in order
        rows: Sequence of row tuples/dicts
        row_count: Number of rows returned
        execution_time_ms: Query execution time in milliseconds
        metadata: Additional metadata (query plan, warnings, etc.)

    """

    columns: list[str]
    rows: Sequence[tuple[Any, ...] | dict[str, Any]]
    row_count: int
    execution_time_ms: float
    metadata: dict[str, Any]


class ConnectionConfig(Protocol):
    """Protocol for connection configuration.

    Defines the expected structure for connector configuration.
    """

    host: str
    port: int
    database: str
    username: str | None
    password: str | None
    ssl_enabled: bool
    timeout: int


class ConnectorCapability(str, Enum):
    """Capabilities that a connector may support."""

    STREAMING = "streaming"
    TRANSACTIONS = "transactions"
    PREPARED_STATEMENTS = "prepared_statements"
    POOLING = "pooling"
    BULK_INSERT = "bulk_insert"
    BULK_UPDATE = "bulk_update"
    ASYNC_CURSOR = "async_cursor"


class ConnectionStatus(str, Enum):
    """Connection status states."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"


class ConnectorMetadata(TypedDict):
    """Metadata about a connector's capabilities and configuration.

    Attributes:
        connector_key: Unique identifier for the connector type
        version: Connector version
        capabilities: List of supported capabilities
        max_pool_size: Maximum connection pool size
        default_timeout: Default query timeout in seconds
        supports_transactions: Whether transactions are supported

    """

    connector_key: str
    version: str
    capabilities: list[ConnectorCapability]
    max_pool_size: int
    default_timeout: int
    supports_transactions: bool


class QueryMetadata(TypedDict, total=False):
    """Optional metadata about query execution.

    Attributes:
        query_plan: Query execution plan (if available)
        warnings: Any warnings generated during execution
        affected_rows: Number of rows affected (for DML)
        last_insert_id: Last inserted ID (if applicable)
        query_id: Unique identifier for the query
        timestamp: When the query was executed

    """

    query_plan: str | None
    warnings: list[str]
    affected_rows: int | None
    last_insert_id: int | None
    query_id: str | None
    timestamp: datetime


class PoolConfig(TypedDict, total=False):
    """Configuration for connection pooling.

    Attributes:
        min_size: Minimum number of connections in pool
        max_size: Maximum number of connections in pool
        max_queries: Maximum queries per connection before recycling
        max_inactive_connection_lifetime: Max seconds a connection can be idle
        timeout: Timeout for acquiring connection from pool
        command_timeout: Timeout for individual commands

    """

    min_size: int
    max_size: int
    max_queries: int
    max_inactive_connection_lifetime: float
    timeout: float
    command_timeout: float
