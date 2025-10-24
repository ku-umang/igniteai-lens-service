"""SQL service for orchestrating MAC-SQL workflows.

This service provides a high-level interface to the MAC-SQL agent system,
managing workflow execution, caching, and observability.
"""

from typing import Any, Dict
from uuid import UUID

from opentelemetry import trace

from core.agents.sql.state import MACSSQLInput, MACSSQLOutput
from core.agents.sql.workflow import MACSSQLWorkflow
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class SQLService:
    """Service for managing SQL generation and execution via MAC-SQL agents."""

    def __init__(self) -> None:
        """Initialize SQL service."""
        self.workflow = MACSSQLWorkflow()

    async def generate_sql(
        self,
        question: str,
        datasource_id: UUID,
        tenant_id: UUID,
        session_id: str | None = None,
        explain_mode: bool = False,
        use_cache: bool = True,
        timeout_seconds: float = 30.0,
        max_rows: int = 10000,
    ) -> MACSSQLOutput:
        """Generate SQL from natural language question.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context
            explain_mode: Return reasoning without executing
            use_cache: Use cached results if available
            timeout_seconds: Query execution timeout
            max_rows: Maximum rows to return

        Returns:
            MAC-SQL output with generated SQL and results

        """
        with tracer.start_as_current_span(
            "sql_service.generate_sql",
            attributes={
                "question": question,
                "datasource_id": str(datasource_id),
                "tenant_id": str(tenant_id),
                "explain_mode": explain_mode,
            },
        ):
            logger.info(
                "SQL generation requested",
                extra={
                    "question": question,
                    "datasource_id": str(datasource_id),
                    "tenant_id": str(tenant_id),
                    "explain_mode": explain_mode,
                },
            )

            # Create input
            input_data = MACSSQLInput(
                question=question,
                datasource_id=datasource_id,
                tenant_id=tenant_id,
                session_id=session_id,
                explain_mode=explain_mode,
                use_cache=use_cache,
                timeout_seconds=timeout_seconds,
                max_rows=max_rows,
            )

            # Run workflow
            output = await self.workflow.run(input_data)

            logger.info(
                "SQL generation completed",
                extra={
                    "success": output.success,
                    "sql_generated": bool(output.sql),
                    "execution_time_ms": output.execution_time_ms,
                },
            )

            return output

    async def execute_sql(
        self,
        question: str,
        datasource_id: UUID,
        tenant_id: UUID,
        session_id: str | None = None,
        use_cache: bool = True,
        timeout_seconds: float = 30.0,
        max_rows: int = 10000,
    ) -> Dict[str, Any]:
        """Generate and execute SQL from natural language question.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context
            use_cache: Use cached results if available
            timeout_seconds: Query execution timeout
            max_rows: Maximum rows to return

        Returns:
            Dict with SQL, results, and metadata

        """
        output = await self.generate_sql(
            question=question,
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            session_id=session_id,
            explain_mode=False,  # Execute, don't just explain
            use_cache=use_cache,
            timeout_seconds=timeout_seconds,
            max_rows=max_rows,
        )

        return {
            "sql": output.sql,
            "data": output.data,
            "rows_returned": output.rows_returned,
            "execution_time_ms": output.execution_time_ms,
            "cached": output.cached,
            "complexity_score": output.complexity_score,
            "success": output.success,
            "error_message": output.error_message,
        }

    async def explain_sql(
        self,
        question: str,
        datasource_id: UUID,
        tenant_id: UUID,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Generate SQL and return reasoning without executing.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context

        Returns:
            Dict with SQL and agent reasoning

        """
        output = await self.generate_sql(
            question=question,
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            session_id=session_id,
            explain_mode=True,  # Explain only
        )

        return {
            "sql": output.sql,
            "reasoning": {
                "schema_selection": output.schema_selection_reasoning,
                "query_decomposition": output.decomposition_reasoning,
                "sql_refinement": output.refinement_reasoning,
            },
            "complexity_score": output.complexity_score,
            "validation": {
                "is_valid": output.is_valid,
                "errors": output.validation_errors,
            },
            "success": output.success,
            "error_message": output.error_message,
        }
