"""MySQL connector implementation using aiomysql."""

from typing import Any
from uuid import UUID

import aiomysql

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


class MySQLConnector(BaseConnector):
    """MySQL database connector using aiomysql.

    Supports connection pooling, async queries, and streaming results.
    Compatible with MySQL 5.7+ and MariaDB.
    """

    CONNECTOR_KEY = "mysql"
    VERSION = "1.0.0"

    def __init__(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
        config: dict[str, Any],
        credentials: dict[str, Any] | None = None,
        pool_config: PoolConfig | None = None,
    ) -> None:
        """Initialize MySQL connector.

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
            Dictionary of connection parameters for aiomysql

        """
        params: dict[str, Any] = {
            "host": self.config.get("host", "localhost"),
            "port": self.config.get("port", 3306),
            "db": self.config.get("database", "mysql"),
            "minsize": self.pool_config.get("min_size", 1),
            "maxsize": self.pool_config.get("max_size", 10),
            "pool_recycle": int(self.pool_config.get("max_inactive_connection_lifetime", 300)),
            "charset": self.config.get("charset", "utf8mb4"),
            "autocommit": self.config.get("autocommit", True),
        }

        # Add credentials if provided
        if self.credentials:
            if "username" in self.credentials:
                params["user"] = self.credentials["username"]
            if "password" in self.credentials:
                params["password"] = self.credentials["password"]

        # Add SSL configuration if specified
        if self.config.get("ssl_enabled", False):
            ssl_config = {
                "ssl": {
                    "ca": self.config.get("ssl_ca"),
                    "cert": self.config.get("ssl_cert"),
                    "key": self.config.get("ssl_key"),
                }
            }
            # Filter out None values
            ssl_config["ssl"] = {k: v for k, v in ssl_config["ssl"].items() if v}
            if ssl_config["ssl"]:
                params.update(ssl_config)

        return params

    async def _create_pool(self) -> aiomysql.Pool:
        """Create aiomysql connection pool.

        Returns:
            aiomysql connection pool

        Raises:
            ConnectionTestFailedError: If pool creation fails
            InvalidCredentialsError: If authentication fails

        """
        params = self._build_connection_params()
        try:
            pool = await aiomysql.create_pool(**params)

            logger.info(
                "MySQL connection pool created",
                extra={
                    "host": params["host"],
                    "port": params["port"],
                    "database": params["db"],
                    "min_size": params["minsize"],
                    "max_size": params["maxsize"],
                },
            )

            return pool

        except aiomysql.OperationalError as e:
            error_code = e.args[0] if e.args else 0

            # Error 1045: Access denied (authentication failure)
            if error_code == 1045:
                logger.error(
                    "MySQL authentication failed",
                    extra={
                        "host": params.get("host"),
                        "port": params.get("port"),
                        "user": params.get("user"),
                        "error_code": error_code,
                    },
                )
                raise InvalidCredentialsError(
                    "MySQL authentication failed",
                    connector_key=self.CONNECTOR_KEY,
                    username=params.get("user"),
                ) from e

            # Other operational errors (connection refused, etc.)
            logger.error(
                "Failed to create MySQL connection pool",
                extra={
                    "host": params.get("host"),
                    "port": params.get("port"),
                    "error": str(e),
                    "error_code": error_code,
                },
            )
            raise ConnectionTestFailedError(
                f"Failed to connect to MySQL: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                host=params.get("host"),
                port=params.get("port"),
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error creating MySQL connection pool",
                extra={"error": str(e)},
            )
            raise ConnectionTestFailedError(
                f"Failed to connect to MySQL: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
            ) from e

    async def _close_pool(self) -> None:
        """Close the aiomysql connection pool."""
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            logger.info("MySQL connection pool closed")

    async def _test_connection_impl(self) -> bool:
        """Test MySQL connection by executing a simple query.

        Returns:
            True if connection is successful

        Raises:
            ConnectionTestFailedError: If connection test fails

        """
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
                    result = await cursor.fetchone()
                    return result is not None and result[0] == 1

        except Exception as e:
            logger.error(
                "MySQL connection test failed",
                extra={"error": str(e)},
            )
            raise ConnectionTestFailedError(
                f"MySQL connection test failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
            ) from e

    async def _execute_query_impl(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute MySQL query and return complete result set.

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
                async with conn.cursor(aiomysql.DictCursor) as cursor:
                    # Execute query with parameters
                    await cursor.execute(query, params)

                    # Fetch all results
                    result = await cursor.fetchall()

                    # Extract column names from cursor description
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    # Convert dict rows to tuples for consistency
                    rows = [tuple(row.values()) for row in result] if result else []

                    return QueryResult(
                        columns=columns,
                        rows=rows,
                        row_count=len(rows),
                        execution_time_ms=0.0,  # Will be set by base class
                        metadata={
                            "affected_rows": cursor.rowcount,
                            "last_insert_id": cursor.lastrowid,
                        },
                    )

        except aiomysql.ProgrammingError as e:
            logger.error(
                "MySQL syntax or programming error",
                extra={
                    "error": str(e),
                    "query_snippet": query[:200],
                },
            )
            raise QueryExecutionError(
                f"MySQL syntax error: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except aiomysql.Error as e:
            logger.error(
                "MySQL query execution error",
                extra={"error": str(e)},
            )
            raise QueryExecutionError(
                f"MySQL query failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error executing MySQL query",
                extra={"error": str(e)},
            )
            raise QueryExecutionError(
                f"Query execution failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

    def get_metadata(self) -> ConnectorMetadata:
        """Get MySQL connector metadata.

        Returns:
            ConnectorMetadata describing capabilities

        """
        return ConnectorMetadata(
            connector_key=self.CONNECTOR_KEY,
            version=self.VERSION,
            capabilities=[
                ConnectorCapability.STREAMING,
                ConnectorCapability.TRANSACTIONS,
                ConnectorCapability.POOLING,
            ],
            max_pool_size=self.pool_config.get("max_size", 10),
            default_timeout=int(self.pool_config.get("command_timeout", 60.0)),
            supports_transactions=True,
        )
