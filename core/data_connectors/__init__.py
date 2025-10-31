"""Data connectors module for connecting to various data sources."""

from core.data_connectors.base import BaseConnector
from core.data_connectors.duckdb import DuckDBConnector
from core.data_connectors.factory import (
    ConnectorRegistry,
    create_connector,
    create_connector_from_dict,
    get_registry,
    register_connector,
)
from core.data_connectors.mysql import MySQLConnector
from core.data_connectors.postgresql import PostgreSQLConnector
from core.data_connectors.sqlite import SQLiteConnector
from core.data_connectors.types import (
    ConnectionStatus,
    ConnectorCapability,
    ConnectorMetadata,
    PoolConfig,
    QueryMetadata,
    QueryResult,
)

__all__ = [
    # Base connector
    "BaseConnector",
    # Concrete connectors
    "PostgreSQLConnector",
    "MySQLConnector",
    "SQLiteConnector",
    "DuckDBConnector",
    # Factory functions
    "create_connector",
    "create_connector_from_dict",
    "register_connector",
    "get_registry",
    "ConnectorRegistry",
    # Types
    "QueryResult",
    "QueryMetadata",
    "PoolConfig",
    "ConnectorMetadata",
    "ConnectorCapability",
    "ConnectionStatus",
]
