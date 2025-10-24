"""Safe SQL executor with advanced validation and caching.

This module provides a safe wrapper around database connectors with:
- Pre-execution validation
- Query timeout enforcement
- Result size limits
- Result caching
- Execution logging
"""

import hashlib
import time
from typing import Any, Dict, Optional
from uuid import UUID

from opentelemetry import trace

from core.data_connectors.factory import create_connector
from core.integrations.schema import DataSourceResponse
from core.logging import get_logger
from core.services.sql.validator import get_sql_validator

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ExecutionError(Exception):
    """SQL execution error."""

    pass


class ValidationError(Exception):
    """SQL validation error."""

    pass


class SafeSQLExecutor:
    """Safe SQL executor with validation, caching, and limits."""

    def __init__(
        self,
        max_rows: int = 10000,
        timeout_seconds: float = 30.0,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ) -> None:
        """Initialize safe SQL executor.

        Args:
            max_rows: Maximum rows to return
            timeout_seconds: Query execution timeout
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds

        """
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl

    async def execute(
        self,
        sql: str,
        datasource_id: UUID,
        datasource: DataSourceResponse,
        tenant_id: UUID,
        dialect: str = "postgres",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Execute SQL query safely.

        Args:
            sql: SQL query to execute
            datasource_id: Datasource identifier
            datasource: Datasource
            tenant_id: Tenant identifier
            dialect: SQL dialect
            validate: Whether to validate SQL before execution

        Returns:
            Dict with execution results

        Raises:
            ValidationError: If SQL validation fails
            ExecutionError: If SQL execution fails

        """
        with tracer.start_as_current_span(
            "sql_executor.execute",
            attributes={
                "datasource_id": str(datasource_id),
                "tenant_id": str(tenant_id),
                "sql_length": len(sql),
            },
        ) as span:
            start_time = time.time()
            query_id = self._generate_query_id(sql, datasource_id)

            try:
                logger.info(
                    "SQL execution started",
                    extra={
                        "query_id": query_id,
                        "datasource_id": str(datasource_id),
                        "sql_length": len(sql),
                    },
                )

                # Step 1: Validation
                if validate:
                    validator = get_sql_validator(dialect=dialect)
                    validation_result = validator.validate(
                        sql=sql,
                        check_readonly=True,
                        check_complexity=True,
                    )

                    if not validation_result.is_valid:
                        span.set_attribute("validation_failed", True)
                        raise ValidationError(f"SQL validation failed: {', '.join(validation_result.errors)}")

                    if validation_result.warnings:
                        logger.warning(
                            "SQL validation warnings",
                            extra={
                                "query_id": query_id,
                                "warnings": validation_result.warnings,
                            },
                        )

                    span.set_attribute("complexity_score", validation_result.complexity.get("score", 0))

                # Step 2: Check cache
                if self.use_cache:
                    cached_result = await self._get_cached_result(query_id)
                    if cached_result:
                        execution_time = (time.time() - start_time) * 1000
                        logger.info(
                            "SQL execution completed (cached)",
                            extra={
                                "query_id": query_id,
                                "execution_time_ms": execution_time,
                                "rows_returned": len(cached_result),
                            },
                        )
                        span.set_attribute("cache_hit", True)
                        return {
                            "success": True,
                            "data": cached_result,
                            "rows_returned": len(cached_result),
                            "execution_time_ms": execution_time,
                            "cached": True,
                        }

                span.set_attribute("cache_hit", False)

                # Step 3: Get datasource connection
                connector = create_connector(datasource)
                await connector.initialize()

                # Step 4: Execute query with limits
                df = await connector.execute_query(
                    query=sql,
                    timeout=self.timeout_seconds,
                )

                await connector.close()
                # Step 5: Apply row limit
                rows_before_limit = len(df)
                if len(df) > self.max_rows:
                    logger.warning(
                        "Result set exceeds max_rows, truncating",
                        extra={
                            "query_id": query_id,
                            "rows_before": rows_before_limit,
                            "rows_after": self.max_rows,
                        },
                    )
                    df = df.head(self.max_rows)

                # Step 6: Convert to JSON-serializable format
                result_data = df.to_dict(orient="records")

                # Step 7: Cache result
                if self.use_cache:
                    await self._cache_result(query_id, result_data)

                execution_time = (time.time() - start_time) * 1000

                span.set_attribute("rows_returned", len(result_data))
                span.set_attribute("execution_time_ms", execution_time)

                logger.info(
                    "SQL execution completed",
                    extra={
                        "query_id": query_id,
                        "rows_returned": len(result_data),
                        "execution_time_ms": execution_time,
                        "truncated": rows_before_limit > self.max_rows,
                    },
                )

                return {
                    "success": True,
                    "data": result_data,
                    "rows_returned": len(result_data),
                    "execution_time_ms": execution_time,
                    "cached": False,
                    "truncated": rows_before_limit > self.max_rows,
                }

            except ValidationError:
                raise
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(
                    "SQL execution failed",
                    extra={
                        "query_id": query_id,
                        "error": str(e),
                        "execution_time_ms": execution_time,
                    },
                )
                span.set_attribute("error", True)
                raise ExecutionError(f"SQL execution failed: {str(e)}") from e

    def _generate_query_id(self, sql: str, datasource_id: UUID) -> str:
        """Generate unique query ID for caching.

        Args:
            sql: SQL query
            datasource_id: Datasource identifier

        Returns:
            Query ID (hash)

        """
        content = f"{sql}:{str(datasource_id)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_cached_result(self, _query_id: str) -> Optional[list]:
        """Get cached query result.

        Args:
            _query_id: Query identifier

        Returns:
            Cached result or None

        """
        # TODO: Implement caching using Redis
        return None

    async def _cache_result(self, _query_id: str, _data: list) -> None:
        """Cache query result.

        Args:
            _query_id: Query identifier
            _data: Result data

        """
        # TODO: Implement caching using Redis
        pass
