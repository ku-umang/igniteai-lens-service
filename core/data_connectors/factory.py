"""Connector factory with registry pattern for creating data source connectors."""

from typing import Any, Type
from uuid import UUID

from core.data_connectors.base import BaseConnector
from core.data_connectors.duckdb import DuckDBConnector
from core.data_connectors.mysql import MySQLConnector
from core.data_connectors.postgresql import PostgreSQLConnector
from core.data_connectors.sqlite import SQLiteConnector
from core.data_connectors.types import PoolConfig
from core.exceptions.connector import UnsupportedConnectorError
from core.integrations.schema import DataSourceResponse
from core.logging import get_logger

logger = get_logger(__name__)


class ConnectorRegistry:
    """Registry for data source connectors.

    Maintains a mapping of connector keys to connector classes.
    Supports dynamic registration and lookup.
    """

    def __init__(self) -> None:
        """Initialize the connector registry."""
        self._connectors: dict[str, Type[BaseConnector]] = {}
        self._register_default_connectors()

    def _register_default_connectors(self) -> None:
        """Register built-in connectors."""
        self.register(PostgreSQLConnector.CONNECTOR_KEY, PostgreSQLConnector)
        self.register(MySQLConnector.CONNECTOR_KEY, MySQLConnector)
        self.register(SQLiteConnector.CONNECTOR_KEY, SQLiteConnector)
        self.register(DuckDBConnector.CONNECTOR_KEY, DuckDBConnector)

        logger.info(
            "Registered default connectors",
            extra={"connectors": list(self._connectors.keys())},
        )

    def register(self, connector_key: str, connector_class: Type[BaseConnector]) -> None:
        """Register a connector class.

        Args:
            connector_key: Unique identifier for the connector (e.g., 'postgresql')
            connector_class: Connector class that extends BaseConnector

        """
        if connector_key in self._connectors:
            logger.warning(
                "Overwriting existing connector",
                extra={
                    "connector_key": connector_key,
                    "old_class": self._connectors[connector_key].__name__,
                    "new_class": connector_class.__name__,
                },
            )

        self._connectors[connector_key] = connector_class
        logger.debug(
            "Registered connector",
            extra={
                "connector_key": connector_key,
                "connector_class": connector_class.__name__,
            },
        )

    def unregister(self, connector_key: str) -> None:
        """Unregister a connector.

        Args:
            connector_key: Connector key to unregister

        """
        if connector_key in self._connectors:
            del self._connectors[connector_key]
            logger.debug(
                "Unregistered connector",
                extra={"connector_key": connector_key},
            )

    def get(self, connector_key: str) -> Type[BaseConnector] | None:
        """Get a connector class by key.

        Args:
            connector_key: Connector key to lookup

        Returns:
            Connector class or None if not found

        """
        return self._connectors.get(connector_key)

    def list_connectors(self) -> list[str]:
        """List all registered connector keys.

        Returns:
            List of connector keys

        """
        return list(self._connectors.keys())

    def is_registered(self, connector_key: str) -> bool:
        """Check if a connector is registered.

        Args:
            connector_key: Connector key to check

        Returns:
            True if connector is registered

        """
        return connector_key in self._connectors


# Global registry instance
_registry = ConnectorRegistry()


def get_registry() -> ConnectorRegistry:
    """Get the global connector registry.

    Returns:
        Global ConnectorRegistry instance

    """
    return _registry


def create_connector(
    datasource: DataSourceResponse,
    pool_config: PoolConfig | None = None,
) -> BaseConnector:
    """Create a connector from a DataSourceResponse.

    This factory function examines the connector_key field and instantiates
    the appropriate connector class from the registry.

    Args:
        datasource: DataSourceResponse containing connection details
        pool_config: Optional custom pool configuration

    Returns:
        Initialized connector instance (not yet connected)

    Raises:
        UnsupportedConnectorError: If connector type is not supported

    Example:
        >>> datasource = DataSourceResponse(...)
        >>> connector = create_connector(datasource)
        >>> async with connector:
        ...     result = await connector.execute_query("SELECT * FROM users")

    """
    connector_key = datasource.connector_key.lower()

    logger.debug(
        "Creating connector",
        extra={
            "connector_key": connector_key,
            "datasource_id": str(datasource.id),
            "tenant_id": str(datasource.tenant_id),
        },
    )

    # Lookup connector class in registry
    connector_class = _registry.get(connector_key)

    if connector_class is None:
        available = _registry.list_connectors()
        logger.error(
            "Unsupported connector type",
            extra={
                "connector_key": connector_key,
                "available_connectors": available,
            },
        )
        raise UnsupportedConnectorError(
            connector_key=connector_key,
            available_connectors=available,
        )

    # Create connector instance
    try:
        connector = connector_class(  # type: ignore[call-arg]
            datasource_id=datasource.id,
            tenant_id=datasource.tenant_id,
            config=datasource.config_json,
            credentials=datasource.credentials,
            pool_config=pool_config,
        )

        logger.info(
            "Connector created successfully",
            extra={
                "connector_key": connector_key,
                "connector_class": connector_class.__name__,
                "datasource_id": str(datasource.id),
            },
        )

        return connector

    except Exception as e:
        logger.error(
            "Failed to create connector",
            extra={
                "connector_key": connector_key,
                "error": str(e),
            },
        )
        raise


def create_connector_from_dict(
    datasource_id: UUID,
    tenant_id: UUID,
    connector_key: str,
    config: dict[str, Any],
    credentials: dict[str, Any] | None = None,
    pool_config: PoolConfig | None = None,
) -> BaseConnector:
    """Create a connector from individual parameters.

    This is a convenience function when you don't have a DataSourceResponse object.

    Args:
        datasource_id: Unique identifier for the datasource
        tenant_id: Tenant identifier
        connector_key: Type of connector (e.g., 'postgresql', 'mysql')
        config: Configuration dictionary
        credentials: Credentials dictionary
        pool_config: Optional pool configuration

    Returns:
        Initialized connector instance (not yet connected)

    Raises:
        UnsupportedConnectorError: If connector type is not supported

    """
    connector_key_lower = connector_key.lower()
    connector_class = _registry.get(connector_key_lower)

    if connector_class is None:
        available = _registry.list_connectors()
        raise UnsupportedConnectorError(
            connector_key=connector_key_lower,
            available_connectors=available,
        )

    return connector_class(  # type: ignore[call-arg]
        datasource_id=datasource_id,
        tenant_id=tenant_id,
        config=config,
        credentials=credentials,
        pool_config=pool_config,
    )


def register_connector(connector_key: str, connector_class: Type[BaseConnector]) -> None:
    """Register a custom connector class.

    This allows extending the system with custom connectors.

    Args:
        connector_key: Unique identifier for the connector
        connector_class: Connector class that extends BaseConnector

    Example:
        >>> class RedisConnector(BaseConnector):
        ...     CONNECTOR_KEY = "redis"
        ...     # ... implementation
        >>>
        >>> register_connector("redis", RedisConnector)

    """
    _registry.register(connector_key, connector_class)
