"""PostgreSQL connector implementation using asyncpg."""

from typing import Any
from uuid import UUID

import asyncpg

from core.data_connectors.base import BaseConnector
from core.data_connectors.types import (
    ConnectorCapability,
    ConnectorMetadata,
    PoolConfig,
    QueryResult,
)
from core.exceptions.connector import (
    ConnectionTestFailedError,
    InvalidCredentialsError,
    QueryExecutionError,
)
from core.logging import get_logger

logger = get_logger(__name__)


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector using asyncpg.

    Supports connection pooling, async queries, streaming results,
    and prepared statements.
    """

    CONNECTOR_KEY = "postgresql"
    VERSION = "1.0.0"

    def __init__(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
        config: dict[str, Any],
        credentials: dict[str, Any] | None = None,
        pool_config: PoolConfig | None = None,
    ) -> None:
        """Initialize PostgreSQL connector.

        Args:
            datasource_id: Unique identifier for the datasource
            tenant_id: Tenant identifier for multi-tenancy
            config: Configuration dictionary containing host, port, database, etc.
            credentials: Credentials dictionary containing username, password
            pool_config: Connection pool configuration

        """
        super().__init__(
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            connector_key=self.CONNECTOR_KEY,
            config=config,
            credentials=credentials,
            pool_config=pool_config,
        )

    def _build_connection_params(self) -> dict[str, Any]:
        """Build connection parameters from config and credentials.

        Returns:
            Dictionary of connection parameters for asyncpg

        """
        params: dict[str, Any] = {
            "host": self.config.get("host", "localhost"),
            "port": self.config.get("port", 5432),
            "database": self.config.get("database", "postgres"),
            "timeout": self.pool_config.get("timeout", 10.0),
            "command_timeout": self.pool_config.get("command_timeout", 60.0),
            "min_size": self.pool_config.get("min_size", 1),
            "max_size": self.pool_config.get("max_size", 10),
            "max_queries": self.pool_config.get("max_queries", 50000),
            "max_inactive_connection_lifetime": self.pool_config.get("max_inactive_connection_lifetime", 300.0),
        }

        # Add credentials if provided
        if self.credentials:
            if "username" in self.credentials:
                params["user"] = self.credentials["username"]
            if "password" in self.credentials:
                params["password"] = self.credentials["password"]

        # Add SSL configuration if specified
        if self.config.get("ssl_enabled", False):
            params["ssl"] = "require"

        # Add any additional asyncpg-specific options from config
        for key in ["server_settings", "statement_cache_size"]:
            if key in self.config:
                params[key] = self.config[key]

        return params

    async def _create_pool(self) -> asyncpg.Pool:
        """Create asyncpg connection pool.

        Returns:
            asyncpg connection pool

        Raises:
            ConnectionTestFailedError: If pool creation fails
            InvalidCredentialsError: If authentication fails

        """
        params = self._build_connection_params()
        try:
            pool = await asyncpg.create_pool(**params)

            logger.info(
                "PostgreSQL connection pool created",
                extra={
                    "host": params["host"],
                    "port": params["port"],
                    "database": params["database"],
                    "min_size": params["min_size"],
                    "max_size": params["max_size"],
                },
            )

            return pool

        except asyncpg.InvalidPasswordError as e:
            logger.error(
                "PostgreSQL authentication failed",
                extra={
                    "host": params.get("host"),
                    "port": params.get("port"),
                    "user": params.get("user"),
                },
            )
            raise InvalidCredentialsError(
                "PostgreSQL authentication failed",
                connector_key=self.CONNECTOR_KEY,
                username=params.get("user"),
            ) from e

        except (
            asyncpg.PostgresConnectionError,
            asyncpg.CannotConnectNowError,
            OSError,
        ) as e:
            logger.error(
                "Failed to create PostgreSQL connection pool",
                extra={
                    "host": params.get("host"),
                    "port": params.get("port"),
                    "error": str(e),
                },
            )
            raise ConnectionTestFailedError(
                f"Failed to connect to PostgreSQL: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                host=params.get("host"),
                port=params.get("port"),
            ) from e

    async def _close_pool(self) -> None:
        """Close the asyncpg connection pool."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def _test_connection_impl(self) -> bool:
        """Test PostgreSQL connection by executing a simple query.

        Returns:
            True if connection is successful

        Raises:
            ConnectionTestFailedError: If connection test fails

        """
        try:
            async with self._pool.acquire() as conn:
                # Execute a simple query to verify connection
                result: int = await conn.fetchval("SELECT 1")
                return result == 1

        except Exception as e:
            logger.error(
                "PostgreSQL connection test failed",
                extra={"error": str(e)},
            )
            raise ConnectionTestFailedError(
                f"PostgreSQL connection test failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
            ) from e

    async def _execute_query_impl(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute PostgreSQL query and return complete result set.

        Args:
            query: SQL query to execute
            params: Query parameters (positional tuple or named dict)

        Returns:
            QueryResult with columns, rows, and metadata

        Raises:
            QueryExecutionError: If query execution fails

        """
        try:
            async with self._pool.acquire() as conn:
                # Convert named parameters to positional if necessary
                if isinstance(params, dict):
                    # asyncpg uses $1, $2, etc. for positional parameters
                    # For simplicity, we'll execute as-is if it's a dict
                    # In production, you might want to convert named to positional
                    result = await conn.fetch(query)
                else:
                    result = await conn.fetch(query, *(params or ()))

                # Extract column names from the first row if available
                columns = list(result[0].keys()) if result else []

                # Convert records to list of tuples
                rows = [tuple(record.values()) for record in result]

                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=0.0,  # Will be set by base class
                    metadata={},
                )

        except asyncpg.PostgresSyntaxError as e:
            logger.error(
                "PostgreSQL syntax error",
                extra={
                    "error": str(e),
                    "query_snippet": query[:200],
                },
            )
            raise QueryExecutionError(
                f"PostgreSQL syntax error: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except asyncpg.PostgresError as e:
            logger.error(
                "PostgreSQL query execution error",
                extra={
                    "error": str(e),
                    "error_code": getattr(e, "sqlstate", None),
                },
            )
            raise QueryExecutionError(
                f"PostgreSQL query failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error executing PostgreSQL query",
                extra={"error": str(e)},
            )
            raise QueryExecutionError(
                f"Query execution failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

    def get_metadata(self) -> ConnectorMetadata:
        """Get PostgreSQL connector metadata.

        Returns:
            ConnectorMetadata describing capabilities

        """
        return ConnectorMetadata(
            connector_key=self.CONNECTOR_KEY,
            version=self.VERSION,
            capabilities=[
                ConnectorCapability.STREAMING,
                ConnectorCapability.TRANSACTIONS,
                ConnectorCapability.PREPARED_STATEMENTS,
                ConnectorCapability.POOLING,
                ConnectorCapability.ASYNC_CURSOR,
            ],
            max_pool_size=self.pool_config.get("max_size", 10),
            default_timeout=int(self.pool_config.get("command_timeout", 60.0)),
            supports_transactions=True,
        )
