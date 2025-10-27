"""MAC-SQL agent API endpoints.

This module provides REST API endpoints for the MAC-SQL agent system:
- Generate SQL from natural language
- Execute SQL and return results
- Explain SQL generation reasoning
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from opentelemetry import trace

from core.agents.sql.state import ChatMessage
from core.api.v1.schemas.agent import (
    ExecuteSQLRequest,
    ExecuteSQLResponse,
)
from core.dependencies.agent import AgentServiceDep
from core.dependencies.message import MessageServiceDep
from core.dependencies.session import get_session_service, verify_session_ownership_helper
from core.logging import get_logger
from core.security.auth import get_current_tenant_id, get_current_user_id
from core.services.session.session_service import SessionService

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/agent", tags=["MAC-SQL Agent"])


@router.post(
    "/chat",
    response_model=ExecuteSQLResponse,
    summary="Chat with the MAC-SQL agent",
    description="Chat with the MAC-SQL agent",
)
async def chat(
    request: ExecuteSQLRequest,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    message_service: MessageServiceDep,
    agent_service: AgentServiceDep,
) -> ExecuteSQLResponse:
    """Chat with the MAC-SQL agent.

    Args:
        request: Chat request
        tenant_id: Tenant identifier from auth
        user_id: User identifier from auth
        session_service: Session service instance
        message_service: Message service instance
        agent_service: SQL service instance

    Returns:
        Chat response

    Raises:
        HTTPException: If chat fails

    """
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

    with tracer.start_as_current_span(
        "api.agent.execute_sql",
        attributes={
            "tenant_id": str(tenant_id),
            "datasource_id": str(session.datasource_id),
            "session_id": str(session.id),
        },
    ):
        try:
            logger.info(
                "SQL execution requested",
                extra={
                    "tenant_id": str(tenant_id),
                    "datasource_id": str(session.datasource_id),
                    "session_id": str(session.id),
                    "question_length": len(request.question),
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

            # Execute SQL using datasource from session
            result = await agent_service.chat(
                question=request.question,
                datasource_id=session.datasource_id,
                tenant_id=tenant_id,
                session_id=str(request.session_id),
                chat_history=chat_history,
                use_cache=request.use_cache,
                timeout_seconds=request.timeout_seconds,
                max_rows=request.max_rows,
            )

            # Save chat interaction after successful execution
            message_id = None
            if result["success"]:
                try:
                    saved_message = await message_service.save_chat_interaction(
                        session_id=request.session_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        question=request.question,
                        sql=result["sql"] if result["sql"] else None,
                    )
                    message_id = saved_message.id
                except Exception as msg_error:
                    logger.warning(
                        "Failed to save chat message",
                        extra={"error": str(msg_error)},
                    )

            response = ExecuteSQLResponse(
                sql=result["sql"],
                data=result["data"],
                rows_returned=result["rows_returned"],
                execution_time_ms=result["execution_time_ms"],
                cached=result["cached"],
                complexity_score=result["complexity_score"],
                success=result["success"],
                error_message=result.get("error_message"),
                message_id=message_id,
            )

            logger.info(
                "SQL execution completed",
                extra={
                    "success": response.success,
                    "rows_returned": response.rows_returned,
                    "cached": response.cached,
                    "message_saved": message_id is not None,
                },
            )

            return response

        except Exception as e:
            logger.error(
                "SQL execution failed",
                extra={
                    "error": str(e),
                    "tenant_id": str(tenant_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SQL execution failed: {str(e)}",
            ) from e
