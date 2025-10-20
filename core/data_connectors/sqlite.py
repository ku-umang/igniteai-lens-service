"""SQLite connector implementation using aiosqlite."""

from typing import Any
from uuid import UUID

import aiosqlite

from core.data_connectors.base import BaseConnector
from core.data_connectors.types import (
    ConnectorCapability,
    ConnectorMetadata,
    PoolConfig,
    QueryResult,
)
from core.exceptions.connector import (
    ConnectionTestFailedError,
    QueryExecutionError,
)
from core.logging import get_logger

logger = get_logger(__name__)


class SQLiteConnector(BaseConnector):
    """SQLite database connector using aiosqlite.

    SQLite is a file-based database, so connection pooling is simpler.
    Supports async queries and streaming results.
    """

    CONNECTOR_KEY = "sqlite"
    VERSION = "1.0.0"

    def __init__(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
        config: dict[str, Any],
        credentials: dict[str, Any] | None = None,
        pool_config: PoolConfig | None = None,
    ) -> None:
        """Initialize SQLite connector.

        Args:
            datasource_id: Unique identifier for the datasource
            tenant_id: Tenant identifier for multi-tenancy
            config: Configuration dictionary containing database path
            credentials: Not used for SQLite (file-based)
            pool_config: Connection pool configuration (limited for SQLite)

        """
        super().__init__(
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            connector_key=self.CONNECTOR_KEY,
            config=config,
            credentials=credentials,
            pool_config=pool_config,
        )

    def _get_database_path(self) -> str:
        """Get the database file path from config.

        Returns:
            Absolute path to SQLite database file

        """
        # Support both 'path' and 'database' keys
        db_path: str = self.config.get("path") or self.config.get("database", ":memory:")  # type: ignore[assignment]
        return db_path

    async def _create_pool(self) -> aiosqlite.Connection:
        """Create SQLite connection.

        Note: SQLite doesn't have traditional pooling. We create a single
        connection that will be shared. For true concurrent access, each
        query can create a new connection.

        Returns:
            aiosqlite connection

        Raises:
            ConnectionTestFailedError: If connection creation fails

        """
        db_path = None
        try:
            db_path = self._get_database_path()
            timeout = self.pool_config.get("timeout", 10.0)

            conn = await aiosqlite.connect(
                db_path,
                timeout=timeout,
                check_same_thread=False,  # Allow multi-threaded access
            )

            # Enable foreign keys (disabled by default in SQLite)
            await conn.execute("PRAGMA foreign_keys = ON")

            # Set journal mode for better concurrency
            journal_mode = self.config.get("journal_mode", "WAL")
            await conn.execute(f"PRAGMA journal_mode = {journal_mode}")

            # Set row factory to return dict-like rows
            conn.row_factory = aiosqlite.Row

            logger.info(
                "SQLite connection created",
                extra={
                    "database": db_path,
                    "journal_mode": journal_mode,
                },
            )

            return conn

        except Exception as e:
            logger.error(
                "Failed to create SQLite connection",
                extra={
                    "database": db_path,
                    "error": str(e),
                },
            )
            raise ConnectionTestFailedError(
                f"Failed to connect to SQLite: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
            ) from e

    async def _close_pool(self) -> None:
        """Close the SQLite connection."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("SQLite connection closed")

    async def _test_connection_impl(self) -> bool:
        """Test SQLite connection by executing a simple query.

        Returns:
            True if connection is successful

        Raises:
            ConnectionTestFailedError: If connection test fails

        """
        try:
            cursor = await self._pool.execute("SELECT 1")
            result = await cursor.fetchone()
            await cursor.close()
            return result is not None and result[0] == 1

        except Exception as e:
            logger.error(
                "SQLite connection test failed",
                extra={"error": str(e)},
            )
            raise ConnectionTestFailedError(
                f"SQLite connection test failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
            ) from e

    async def _execute_query_impl(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute SQLite query and return complete result set.

        Args:
            query: SQL query to execute
            params: Query parameters (positional tuple or named dict)

        Returns:
            QueryResult with columns, rows, and metadata

        Raises:
            QueryExecutionError: If query execution fails

        """
        try:
            cursor = await self._pool.execute(query, params or ())
            result = await cursor.fetchall()

            # Extract column names from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Convert Row objects to tuples
            rows = [tuple(row) for row in result] if result else []

            # Get affected rows for DML operations
            affected_rows = cursor.rowcount if cursor.rowcount > 0 else 0

            await cursor.close()

            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=0.0,  # Will be set by base class
                metadata={
                    "affected_rows": affected_rows,
                    "last_insert_id": cursor.lastrowid,
                },
            )

        except aiosqlite.OperationalError as e:
            logger.error(
                "SQLite operational error",
                extra={
                    "error": str(e),
                    "query_snippet": query[:200],
                },
            )
            raise QueryExecutionError(
                f"SQLite operational error: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except aiosqlite.DatabaseError as e:
            logger.error(
                "SQLite database error",
                extra={
                    "error": str(e),
                    "query_snippet": query[:200],
                },
            )
            raise QueryExecutionError(
                f"SQLite query failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

        except Exception as e:
            logger.error(
                "Unexpected error executing SQLite query",
                extra={"error": str(e)},
            )
            raise QueryExecutionError(
                f"Query execution failed: {str(e)}",
                connector_key=self.CONNECTOR_KEY,
                query=query[:200],
            ) from e

    def get_metadata(self) -> ConnectorMetadata:
        """Get SQLite connector metadata.

        Returns:
            ConnectorMetadata describing capabilities

        """
        return ConnectorMetadata(
            connector_key=self.CONNECTOR_KEY,
            version=self.VERSION,
            capabilities=[
                ConnectorCapability.STREAMING,
                ConnectorCapability.TRANSACTIONS,
            ],
            max_pool_size=1,  # SQLite typically uses single connection
            default_timeout=int(self.pool_config.get("command_timeout", 60.0)),
            supports_transactions=True,
        )
