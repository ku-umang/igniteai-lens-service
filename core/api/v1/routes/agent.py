from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from opentelemetry import trace

from core.agents.state import ChatMessage
from core.api.v1.schemas.agent import (
    AgentResponse,
    AgentRunRequest,
)
from core.dependencies import (
    AgentServiceDep,
    MessageServiceDep,
    SessionServiceDep,
    verify_session_ownership_helper,
)
from core.logging import get_logger
from core.security.auth import get_current_tenant_id, get_current_user_id

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/run",
    summary="Run data analyst agent",
    description="Run data analyst agent to analyze data and answer questions",
    response_model=AgentResponse,
)
async def run(
    request: AgentRunRequest,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: SessionServiceDep,
    message_service: MessageServiceDep,
    agent_service: AgentServiceDep,
) -> AgentResponse:
    """Run data analyst agent to analyze data and answer questions.

    Args:
        request: Run data analyst agent request
        tenant_id: Tenant identifier from auth
        user_id: User identifier from auth
        session_service: Session service instance
        message_service: Message service instance
        agent_service: Agent service instance

    Returns:
        AgentResponse: Agent execution response with message ID, SQL, results, and analysis

    Raises:
        HTTPException: If run data analyst agent fails

    """
    with tracer.start_as_current_span(
        "api.agent.run_data_analyst_agent",
        attributes={
            "tenant_id": str(tenant_id),
            "session_id": str(request.session_id),
        },
    ):
        # Validate and fetch session using existing helper
        try:
            session = await verify_session_ownership_helper(
                session_id=request.session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_service=session_service,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found or access denied: {str(e)}",
            ) from e
        try:
            logger.info(
                "Run data analyst agent requested",
                extra={
                    "tenant_id": str(tenant_id),
                    "datasource_id": str(session.datasource_id),
                    "session_id": str(session.id),
                },
            )

            # Fetch conversation history
            chat_history = []
            if request.max_history_messages > 0:
                messages = await message_service.get_session_history(
                    session_id=request.session_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    limit=request.max_history_messages,
                )
                # Convert Message models to ChatMessage (reverse to get oldest first)
                chat_history = [ChatMessage(question=msg.question, sql=msg.sql) for msg in reversed(messages)]

            # Run agent workflow
            agent_output = await agent_service.run(
                question=request.question,
                datasource_id=session.datasource_id,
                tenant_id=tenant_id,
                session_id=str(request.session_id),
                chat_history=chat_history,
            )

            # Save message to database
            message = await message_service.save_chat_interaction(
                session_id=request.session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                question=request.question,
                sql=agent_output.sql,
                visualization_spec=agent_output.visualization_spec,
            )

            # Build and return response
            return AgentResponse(
                message_id=message.id,
                question=request.question,
                sql=agent_output.sql,
                data=agent_output.data,
                num_rows=agent_output.num_rows,
                answer=agent_output.answer,
                insights=agent_output.insights,
                visualization_spec=agent_output.visualization_spec,
                total_time_ms=agent_output.total_time_ms,
                llm_calls=agent_output.llm_calls,
                success=agent_output.success,
                error_message=agent_output.error_message,
            )

        except Exception as e:
            logger.error(
                "Run data analyst agent failed",
                extra={
                    "error": str(e),
                    "tenant_id": str(tenant_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Run data analyst agent failed: {str(e)}",
            ) from e
