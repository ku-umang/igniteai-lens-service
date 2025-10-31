"""DuckDB connector implementation with S3 CSV support."""

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import duckdb

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


class DuckDBConnector(BaseConnector):
    """DuckDB connector with S3 CSV file support.

    This connector creates an in-memory DuckDB database and registers CSV files
    from S3 as views. Files are accessed using the httpfs extension with AWS
    credentials from environment variables.
    """

    CONNECTOR_KEY = "duckdb"
    VERSION = "1.0.0"

    def __init__(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
        config: dict[str, Any],
        credentials: dict[str, Any] | None = None,
        pool_config: PoolConfig | None = None,
    ) -> None:
        """Initialize DuckDB connector.

        Args:
            datasource_id: Unique identifier for the datasource
            tenant_id: Tenant identifier for multi-tenancy
            config: Configuration dictionary containing file_paths, s3_region, etc.
            credentials: Credentials dictionary (unused, AWS creds from environment)
            pool_config: Connection pool configuration

        Expected config structure:
            {
                "name": "datasource_name",
                "file_paths": ["path/to/file1.csv", "path/to/file2.csv"],
                "s3_region": "us-east-1",  # Optional, defaults to us-east-1
                "s3_endpoint": "s3.amazonaws.com",  # Optional
                "s3_use_ssl": true,  # Optional, defaults to true
                "database": ":memory:"  # Optional, defaults to :memory:
            }

        """
        super().__init__(
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            connector_key=self.CONNECTOR_KEY,
            config=config,
            credentials=credentials,
            pool_config=pool_config,
        )
        self._views: dict[str, str] = {}  # Map of view_name -> s3_path

    def _get_database_path(self) -> str:
        """Get database path from config.

        Returns:
            Database path (defaults to in-memory)

        """
        return str(self.config.get("name", ":memory:"))

    def _extract_view_name(self, file_path: str) -> str:
        """Extract view name from file path.

        Extracts the filename without extension from the full S3 path.
        Example: "ignite/.../categories.csv" -> "categories"

        Args:
            file_path: Full file path (S3 path without s3:// prefix)

        Returns:
            View name derived from filename without extension

        """
        path = Path(file_path)
        return path.stem  # Gets filename without extension

    def _build_s3_path(self, file_path: str) -> str:
        """Build full S3 URI from file path.

        Args:
            file_path: File path (may or may not have s3:// prefix)

        Returns:
            Full S3 URI

        """
        if file_path.startswith("s3://"):
            return file_path
        return f"s3://{self.config.get('s3_bucket', 'aipal-bucket-assets')}/{file_path}"

    async def _create_pool(self) -> duckdb.DuckDBPyConnection:
        """Create DuckDB connection and register CSV files as views.

        Returns:
            DuckDB connection instance

        Raises:
            ConnectionTestFailedError: If connection creation or view registration fails

        """

        def _create_connection() -> duckdb.DuckDBPyConnection:
            """Synchronous connection creation (runs in thread pool)."""
            try:
                database_path = self._get_database_path()
                conn = duckdb.connect(database=database_path, read_only=False)

                # Install and load httpfs extension for S3 support
                logger.info("Installing httpfs extension for S3 support")
                conn.execute("INSTALL httpfs;")
                conn.execute("LOAD httpfs;")

                # Configure S3 settings
                s3_region = self.config.get("s3_region", "us-west-1")
                s3_endpoint = self.config.get("s3_endpoint", "s3.amazonaws.com")
                s3_use_ssl = self.config.get("s3_use_ssl", True)

                logger.info(
                    "Configuring S3 settings",
                    extra={
                        "s3_region": s3_region,
                        "s3_endpoint": s3_endpoint,
                        "s3_use_ssl": s3_use_ssl,
                    },
                )

                # Set S3 configuration
                conn.execute(f"SET s3_region='{s3_region}';")
                conn.execute(f"SET s3_endpoint='{s3_endpoint}';")
                conn.execute(f"SET s3_use_ssl={'true' if s3_use_ssl else 'false'};")

                # Configure AWS credentials from environment
                aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
                aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
                aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

                if aws_access_key_id and aws_secret_access_key:
                    logger.info("Configuring AWS credentials from environment")
                    conn.execute(f"SET s3_access_key_id='{aws_access_key_id}';")
                    conn.execute(f"SET s3_secret_access_key='{aws_secret_access_key}';")

                    if aws_session_token:
                        conn.execute(f"SET s3_session_token='{aws_session_token}';")
                else:
                    logger.warning(
                        "AWS credentials not found in environment. "
                        "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to access S3 files."
                    )

                # Register CSV files as views
                file_paths = self.config.get("file_paths", [])
                logger.info(
                    "Registering CSV files as views",
                    extra={"file_count": len(file_paths)},
                )

                for file_path in file_paths:
                    view_name = self._extract_view_name(file_path)
                    s3_path = self._build_s3_path(file_path)
                    print(s3_path, "################")
                    # Create view using read_csv_auto for automatic schema detection
                    create_view_sql = f"""
                        CREATE VIEW {view_name} AS
                        SELECT * FROM read_csv_auto('{s3_path}', header=true)
                    """

                    logger.info(
                        "Creating view for CSV file",
                        extra={
                            "view_name": view_name,
                            "s3_path": s3_path,
                        },
                    )

                    conn.execute(create_view_sql)
                    self._views[view_name] = s3_path

                logger.info(
                    "DuckDB connection created successfully",
                    extra={
                        "database": database_path,
                        "views_created": len(self._views),
                        "view_names": list(self._views.keys()),
                    },
                )

                return conn

            except Exception as e:
                logger.error(
                    "Failed to create DuckDB connection",
                    extra={"error": str(e)},
                )
                raise ConnectionTestFailedError(
                    f"Failed to create DuckDB connection: {str(e)}",
                    connector_key=self.CONNECTOR_KEY,
                ) from e

        # Run synchronous DuckDB operations in thread pool
        return await asyncio.to_thread(_create_connection)

    async def _close_pool(self) -> None:
        """Close the DuckDB connection."""

        def _close_connection() -> None:
            """Synchronous connection close (runs in thread pool)."""
            if self._pool is not None:
                self._pool.close()
                self._views.clear()
                logger.info("DuckDB connection closed")

        await asyncio.to_thread(_close_connection)

    async def _test_connection_impl(self) -> bool:
        """Test DuckDB connection by executing a simple query.

        Returns:
            True if connection is successful

        Raises:
            ConnectionTestFailedError: If connection test fails

        """

        def _test_connection() -> bool:
            """Synchronous connection test (runs in thread pool)."""
            try:
                result = self._pool.execute("SELECT 1 as test").fetchone()
                return bool(result[0] == 1)

            except Exception as e:
                logger.error(
                    "DuckDB connection test failed",
                    extra={"error": str(e)},
                )
                raise ConnectionTestFailedError(
                    f"DuckDB connection test failed: {str(e)}",
                    connector_key=self.CONNECTOR_KEY,
                ) from e

        return await asyncio.to_thread(_test_connection)

    async def _execute_query_impl(
        self,
        query: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute DuckDB query and return complete result set.

        Args:
            query: SQL query to execute
            params: Query parameters (positional tuple or named dict)

        Returns:
            QueryResult with columns, rows, and metadata

        Raises:
            QueryExecutionError: If query execution fails

        """

        def _execute_query() -> QueryResult:
            """Synchronous query execution (runs in thread pool)."""
            try:
                # DuckDB supports both positional and named parameters
                if params:
                    result = self._pool.execute(query, params)
                else:
                    result = self._pool.execute(query)

                # Fetch all results
                rows = result.fetchall()
                columns = [desc[0] for desc in result.description] if result.description else []

                logger.debug(
                    "Query executed successfully",
                    extra={
                        "row_count": len(rows),
                        "column_count": len(columns),
                    },
                )

                return QueryResult(
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    execution_time_ms=0.0,  # Will be set by base class
                    metadata={
                        "views": list(self._views.keys()),
                    },
                )

            except duckdb.Error as e:
                logger.error(
                    "DuckDB query execution error",
                    extra={
                        "error": str(e),
                        "query_snippet": query[:200],
                    },
                )
                raise QueryExecutionError(
                    f"DuckDB query failed: {str(e)}",
                    connector_key=self.CONNECTOR_KEY,
                    query=query[:200],
                ) from e

            except Exception as e:
                logger.error(
                    "Unexpected error executing DuckDB query",
                    extra={"error": str(e)},
                )
                raise QueryExecutionError(
                    f"Query execution failed: {str(e)}",
                    connector_key=self.CONNECTOR_KEY,
                    query=query[:200],
                ) from e

        return await asyncio.to_thread(_execute_query)

    def get_metadata(self) -> ConnectorMetadata:
        """Get DuckDB connector metadata.

        Returns:
            ConnectorMetadata describing capabilities

        """
        return ConnectorMetadata(
            connector_key=self.CONNECTOR_KEY,
            version=self.VERSION,
            capabilities=[
                ConnectorCapability.ASYNC_CURSOR,
            ],
            max_pool_size=1,  # Single connection for in-memory database
            default_timeout=int(self.pool_config.get("command_timeout", 60.0)),
            supports_transactions=True,
        )

    @property
    def views(self) -> dict[str, str]:
        """Get registered views mapping.

        Returns:
            Dictionary mapping view names to S3 paths

        """
        return self._views.copy()
