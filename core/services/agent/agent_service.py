from typing import List
from uuid import UUID

from opentelemetry import trace

from core.agents.state import AgentInput, AgentOutput, ChatMessage
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
    ) -> AgentOutput:
        """Run data analyst agent to analyze data and answer questions.

        Args:
            question: User's natural language question
            datasource_id: Target datasource identifier
            tenant_id: Tenant identifier
            session_id: Optional session ID for context
            chat_history: Optional conversation history for context

        Returns:
            Agent output with analysis and results

        """
        with tracer.start_as_current_span(
            "agent_service.run",
            attributes={
                "question": question,
                "datasource_id": str(datasource_id),
                "tenant_id": str(tenant_id),
            },
        ):
            logger.info(
                "Agent workflow requested",
                extra={
                    "question": question,
                    "datasource_id": str(datasource_id),
                    "tenant_id": str(tenant_id),
                },
            )

            # Create input
            input_data = AgentInput(
                question=question,
                datasource_id=datasource_id,
                tenant_id=tenant_id,
                session_id=session_id,
                chat_history=chat_history or [],
            )

            # Run workflow
            output = await self.workflow.run(input_data)

            logger.info(
                "Agent workflow completed",
            )

            return output
