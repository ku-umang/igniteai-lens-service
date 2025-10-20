"""Base connector class for all data source connectors."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from opentelemetry import trace

from core.data_connectors.types import (
    ConnectionStatus,
    ConnectorMetadata,
    PoolConfig,
    QueryResult,
)
from core.exceptions.connector import (
    ConnectionTimeoutError,
    ConnectorError,
    QueryTimeoutError,
)
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class BaseConnector(ABC):
    """Abstract base class for all data source connectors.

    This class provides the common interface and functionality for all connectors,
    including connection pooling, query execution, streaming, and observability.

    All concrete connectors must implement the abstract methods defined here.
    """

    def __init__(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
        connector_key: str,
        config: dict[str, Any],
        credentials: dict[str, Any] | None = None,
        pool_config: PoolConfig | None = None,
    ) -> None:
        """Initialize the base connector.

        Args:
            datasource_id: Unique identifier for the datasource
            tenant_id: Tenant identifier for multi-tenancy
            connector_key: Type of connector (e.g., 'postgresql', 'mysql')
            config: Configuration dictionary (non-sensitive)
            credentials: Credentials dictionary (sensitive)
            pool_config: Connection pool configuration

        """
        self.datasource_id = datasource_id
        self.tenant_id = tenant_id
        self.connector_key = connector_key
        self.config = config
        self.credentials = credentials or {}
        self.pool_config = pool_config or self._default_pool_config()

        self._pool: Any = None
        self._status = ConnectionStatus.DISCONNECTED
        self._connection_id = str(uuid4())

        logger.info(
            "Initializing connector",
            extra={
                "connector_key": connector_key,
                "datasource_id": str(datasource_id),
                "tenant_id": str(tenant_id),
                "connection_id": self._connection_id,
            },
        )

    def _default_pool_config(self) -> PoolConfig:
        """Get default pool configuration.

        Returns:
            Default PoolConfig with sensible defaults

        """
        return PoolConfig(
            min_size=1,
            max_size=10,
            max_queries=50000,
            max_inactive_connection_lifetime=300.0,
            timeout=10.0,
            command_timeout=60.0,
        )

    @abstractmethod
    async def _create_pool(self) -> Any:
        """Create connection pool for the specific database.

        Returns:
            Database-specific connection pool object

        Raises:
            ConnectorError: If pool creation fails

        """
        pass

    @abstractmethod
    async def _close_pool(self) -> None:
        """Close the connection pool and cleanup resources.

        Raises:
            ConnectorError: If pool closure fails

        """
        pass

    @abstractmethod
    async def _test_connection_impl(self) -> bool:
        """Implementation-specific connection test.

        Returns:
            True if connection is successful

        Raises:
            ConnectorError: If connection test fails

        """
        pass

    @abstractmethod
    async def _execute_query_impl(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> QueryResult:
        """Implementation-specific query execution.

        Args:
            query: SQL query to execute
            params: Query parameters (positional or named)

        Returns:
            QueryResult with columns, rows, and metadata

        Raises:
            QueryExecutionError: If query execution fails

        """
        pass

    @abstractmethod
    def get_metadata(self) -> ConnectorMetadata:
        """Get connector metadata and capabilities.

        Returns:
            ConnectorMetadata describing the connector

        """
        pass

    async def initialize(self) -> None:
        """Initialize the connector and create connection pool.

        Raises:
            ConnectorError: If initialization fails

        """
        with tracer.start_as_current_span(
            "connector.initialize",
            attributes={
                "connector.key": self.connector_key,
                "datasource.id": str(self.datasource_id),
                "tenant.id": str(self.tenant_id),
            },
        ):
            try:
                logger.info(
                    "Creating connection pool",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "pool_config": self.pool_config,
                    },
                )

                self._pool = await self._create_pool()
                self._status = ConnectionStatus.CONNECTED

                logger.info(
                    "Connection pool created successfully",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                    },
                )

            except Exception as e:
                self._status = ConnectionStatus.ERROR
                logger.error(
                    "Failed to initialize connector",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "error": str(e),
                    },
                )
                raise ConnectorError(
                    f"Failed to initialize connector: {str(e)}",
                    connector_key=self.connector_key,
                ) from e

    async def close(self) -> None:
        """Close the connector and cleanup all resources.

        Raises:
            ConnectorError: If cleanup fails

        """
        with tracer.start_as_current_span(
            "connector.close",
            attributes={
                "connector.key": self.connector_key,
                "connection.id": self._connection_id,
            },
        ):
            try:
                if self._pool is not None:
                    logger.info(
                        "Closing connection pool",
                        extra={
                            "connector_key": self.connector_key,
                            "connection_id": self._connection_id,
                        },
                    )

                    await self._close_pool()
                    self._pool = None
                    self._status = ConnectionStatus.DISCONNECTED

                    logger.info(
                        "Connection pool closed successfully",
                        extra={
                            "connector_key": self.connector_key,
                            "connection_id": self._connection_id,
                        },
                    )

            except Exception as e:
                logger.error(
                    "Error closing connector",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "error": str(e),
                    },
                )
                raise ConnectorError(
                    f"Failed to close connector: {str(e)}",
                    connector_key=self.connector_key,
                ) from e

    async def test_connection(self, timeout: float | None = None) -> bool:
        """Test the connection to the data source.

        Args:
            timeout: Maximum time to wait for connection test (seconds)

        Returns:
            True if connection is successful

        Raises:
            ConnectionTimeoutError: If connection test times out
            ConnectorError: If connection test fails

        """
        timeout = timeout or self.pool_config.get("timeout", 10.0)

        with tracer.start_as_current_span(
            "connector.test_connection",
            attributes={
                "connector.key": self.connector_key,
                "datasource.id": str(self.datasource_id),
                "timeout": timeout,
            },
        ) as span:
            try:
                self._status = ConnectionStatus.TESTING

                logger.info(
                    "Testing connection",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "timeout": timeout,
                    },
                )

                # Ensure pool is initialized
                if self._pool is None:
                    await self.initialize()

                # Run connection test with timeout
                result = await asyncio.wait_for(
                    self._test_connection_impl(),
                    timeout=timeout,
                )

                self._status = ConnectionStatus.CONNECTED if result else ConnectionStatus.ERROR
                span.set_attribute("connection.success", result)

                logger.info(
                    "Connection test completed",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "success": result,
                    },
                )

                return result

            except asyncio.TimeoutError as e:
                self._status = ConnectionStatus.ERROR
                logger.error(
                    "Connection test timed out",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "timeout": timeout,
                    },
                )
                raise ConnectionTimeoutError(
                    f"Connection test timed out after {timeout} seconds",
                    connector_key=self.connector_key,
                    timeout_seconds=timeout,
                ) from e

            except Exception as e:
                self._status = ConnectionStatus.ERROR
                logger.error(
                    "Connection test failed",
                    extra={
                        "connector_key": self.connector_key,
                        "connection_id": self._connection_id,
                        "error": str(e),
                    },
                )
                raise

    async def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> pd.DataFrame:
        """Execute a query and return the result as a pandas DataFrame.

        Args:
            query: SQL query to execute
            params: Query parameters (positional or named)
            timeout: Maximum time to wait for query execution (seconds)

        Returns:
            pandas DataFrame with query results

        Raises:
            QueryTimeoutError: If query execution times out
            QueryExecutionError: If query execution fails

        """
        timeout = timeout or self.pool_config.get("command_timeout", 60.0)
        query_id = str(uuid4())

        with tracer.start_as_current_span(
            "connector.execute_query",
            attributes={
                "connector.key": self.connector_key,
                "query.id": query_id,
                "query.length": len(query),
                "timeout": timeout,
            },
        ) as span:
            try:
                # Ensure pool is initialized
                if self._pool is None:
                    await self.initialize()

                logger.debug(
                    "Executing query",
                    extra={
                        "connector_key": self.connector_key,
                        "query_id": query_id,
                        "query_length": len(query),
                        "has_params": params is not None,
                    },
                )

                start_time = time.time()

                # Execute query with timeout
                result = await asyncio.wait_for(
                    self._execute_query_impl(query, params),
                    timeout=timeout,
                )

                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                # Convert QueryResult to DataFrame
                df = pd.DataFrame(data=result["rows"], columns=result["columns"])  # type: ignore[call-arg]

                # Store metadata as DataFrame attributes
                df.attrs["execution_time_ms"] = execution_time
                df.attrs["query_id"] = query_id
                df.attrs["metadata"] = result.get("metadata", {})

                span.set_attribute("query.rows", len(df))
                span.set_attribute("query.execution_time_ms", execution_time)

                logger.info(
                    "Query executed successfully",
                    extra={
                        "connector_key": self.connector_key,
                        "query_id": query_id,
                        "row_count": len(df),
                        "execution_time_ms": execution_time,
                    },
                )

                return df

            except asyncio.TimeoutError as e:
                logger.error(
                    "Query execution timed out",
                    extra={
                        "connector_key": self.connector_key,
                        "query_id": query_id,
                        "timeout": timeout,
                    },
                )
                raise QueryTimeoutError(
                    f"Query execution timed out after {timeout} seconds",
                    connector_key=self.connector_key,
                    query=query[:200],  # Truncate for logging
                    timeout_seconds=timeout,
                ) from e

            except Exception as e:
                logger.error(
                    "Query execution failed",
                    extra={
                        "connector_key": self.connector_key,
                        "query_id": query_id,
                        "error": str(e),
                    },
                )
                raise

    async def __aenter__(self) -> "BaseConnector":
        """Enter async context manager.

        Returns:
            Self for use in async with statement

        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        """
        await self.close()

    @property
    def status(self) -> ConnectionStatus:
        """Get current connection status.

        Returns:
            Current ConnectionStatus

        """
        return self._status

    @property
    def connection_id(self) -> str:
        """Get unique connection identifier.

        Returns:
            Connection ID string

        """
        return self._connection_id

    def transaction(self) -> Any:
        """Get transaction context manager.

        This is a base implementation that can be overridden by connectors
        that support transactions.

        Returns:
            Transaction context manager (database-specific)

        Raises:
            NotImplementedError: If transactions are not supported

        """
        raise NotImplementedError(f"Transactions not supported by {self.connector_key} connector")
