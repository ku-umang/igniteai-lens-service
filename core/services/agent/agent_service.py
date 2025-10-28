from typing import Any, Dict, List
from uuid import UUID

from opentelemetry import trace

from core.agents.sql.state import AgentInput, AgentOutput, ChatMessage
from core.agents.workflow import AgentWorkflow
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AgentService:
    """Service for managing agent workflows."""

    def __init__(self) -> None:
        """Initialize agent service."""
        self.workflow = AgentWorkflow()

    async def run(
        self,
        question: str,
        datasource_id: UUID,
        tenant_id: UUID,
        session_id: str | None = None,
        chat_history: List[ChatMessage] | None = None,
        explain_mode: bool = False,
        use_cache: bool = True,
        timeout_seconds: float = 30.0,
        max_rows: int = 10000,
    ) -> AgentOutput:
        """Generate SQL from natural language question.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context
            chat_history: Optional conversation history for context
            explain_mode: Return reasoning without executing
            use_cache: Use cached results if available
            timeout_seconds: Query execution timeout
            max_rows: Maximum rows to return

        Returns:
            Agent output with generated SQL and results

        """
        with tracer.start_as_current_span(
            "agent_service.run",
            attributes={
                "question": question,
                "datasource_id": str(datasource_id),
                "tenant_id": str(tenant_id),
                "explain_mode": explain_mode,
            },
        ):
            logger.info(
                "Agent workflow requested",
                extra={
                    "question": question,
                    "datasource_id": str(datasource_id),
                    "tenant_id": str(tenant_id),
                    "explain_mode": explain_mode,
                },
            )

            # Create input
            input_data = AgentInput(
                question=question,
                datasource_id=datasource_id,
                tenant_id=tenant_id,
                session_id=session_id,
                chat_history=chat_history or [],
                explain_mode=explain_mode,
                use_cache=use_cache,
                timeout_seconds=timeout_seconds,
                max_rows=max_rows,
            )

            # Run workflow
            output = await self.workflow.run(input_data)

            logger.info(
                "Agent workflow completed",
                extra={
                    "success": output.success,
                    "sql_generated": bool(output.sql),
                    "execution_time_ms": output.execution_time_ms,
                },
            )

            return output

    async def chat(
        self,
        question: str,
        datasource_id: UUID,
        tenant_id: UUID,
        session_id: str | None = None,
        chat_history: List[ChatMessage] | None = None,
        use_cache: bool = True,
        timeout_seconds: float = 30.0,
        max_rows: int = 10000,
    ) -> Dict[str, Any]:
        """Chat with the Agent agent.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context
            chat_history: Optional conversation history for context
            use_cache: Use cached results if available
            timeout_seconds: Query execution timeout
            max_rows: Maximum rows to return

        Returns:
            Dict with SQL, results, and metadata

        """
        output = await self.run(
            question=question,
            datasource_id=datasource_id,
            tenant_id=tenant_id,
            session_id=session_id,
            chat_history=chat_history,
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
            "visualization_spec": output.visualization_spec,
            "success": output.success,
            "error_message": output.error_message,
        }
