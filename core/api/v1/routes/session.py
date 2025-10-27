"""Session management API routes."""

import uuid
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from opentelemetry import trace

from core.api.metrics import api_requests_total
from core.api.v1.schemas.agent import (
    ChatHistoryResponse,
    ChatMessageResponse,
)
from core.api.v1.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionUpdate,
)
from core.dependencies.message import MessageServiceDep
from core.dependencies.session import (
    get_session_service,
    verify_session_ownership,
    verify_session_ownership_helper,
)
from core.logging import get_logger
from core.models.session import Session
from core.schemas import PaginatedResponse
from core.security.auth import get_current_tenant_id, get_current_user_id
from core.services.session.session_metrics import session_metrics
from core.services.session.session_service import SessionService
from core.utils.pagination import paginate_per_page

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/v1/sessions", tags=["Sessions"])


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new conversation session for a user with specified datasource and LLM configuration.",
)
async def create_session(
    request: Request,
    session_data: SessionCreate,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> SessionResponse:
    """Create a new session.

    Args:
        request: FastAPI request
        session_data: Session creation data
        tenant_id: Current tenant ID
        user_id: Current user ID
        session_service: Session service instance

    Returns:
        SessionResponse: Created session details

    """
    operation_id = str(uuid.uuid4())
    session_metrics.record_operation_start("create", operation_id)

    try:
        logger.info(
            "Creating new session",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            datasource_id=str(session_data.datasource_id),
            llm_config_id=str(session_data.llm_config_id),
        )

        session, session_count = await session_service.create_session(
            tenant_id=tenant_id,
            user_id=user_id,
            datasource_id=session_data.datasource_id,
            llm_config_id=session_data.llm_config_id,
            session_metadata=session_data.session_metadata,
        )

        # Record metrics with actual session count
        session_metrics.record_session_created(str(tenant_id), count=session_count, operation_id=operation_id)
        api_requests_total.labels(version="v1", resource="sessions", operation="create", status="success").inc()

        return SessionResponse.model_validate(session)

    except Exception:
        # Record failure metrics
        session_metrics.record_operation_complete("create", operation_id, success=False)
        api_requests_total.labels(version="v1", resource="sessions", operation="create", status="error").inc()
        raise


@router.get(
    "",
    response_model=PaginatedResponse[SessionResponse],
    summary="List user sessions",
    description="Get a paginated list of sessions for the current user with optional filtering and automatic next/previous URLs.",
)
async def list_sessions(
    request: Request,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    per_page: Annotated[int, Query(ge=1, le=100, description="Number of items per page")] = 20,
) -> PaginatedResponse[SessionResponse]:
    """List sessions for the current user with automatic pagination URLs.

    This endpoint uses the PagePaginator utility which automatically generates
    next_page and previous_page URLs based on the current request.

    Args:
        request: FastAPI request
        tenant_id: Current tenant ID
        user_id: Current user ID
        session_service: Session service instance
        page: Current page number (1-indexed)
        per_page: Number of items per page

    Returns:
        PaginatedResponse[SessionResponse]: Paginated list of sessions with next/prev URLs

    """
    operation_id = str(uuid.uuid4())
    session_metrics.record_operation_start("list", operation_id)

    try:
        logger.debug(
            "Listing sessions",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            page=page,
            per_page=per_page,
        )

        # Get query from service (follows repository pattern)
        query, current_context = session_service.get_user_sessions_query(
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Use pagination utility
        result = await paginate_per_page(session_service.db_session, query, page, per_page, current_context)

        # Record metrics
        session_metrics.record_session_list(result["count"], operation_id, success=True)
        api_requests_total.labels(version="v1", resource="sessions", operation="list", status="success").inc()

        return PaginatedResponse[SessionResponse](
            count=result["count"],
            next_page=result["next_page"],
            previous_page=result["previous_page"],
            items=[SessionResponse.model_validate(item) for item in result["items"]],
        )

    except Exception:
        # Record failure metrics
        session_metrics.record_session_list(0, operation_id, success=False)
        api_requests_total.labels(version="v1", resource="sessions", operation="list", status="error").inc()
        raise


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
    description="Get detailed information about a specific session.",
)
async def get_session(
    request: Request,
    session_id: UUID,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    verified_session: Annotated[Session, Depends(verify_session_ownership)],
) -> SessionResponse:
    """Get session by ID.

    Args:
        request: FastAPI request
        session_id: Session identifier
        tenant_id: Current tenant ID
        session_service: Session service instance
        verified_session: Verified session instance (from dependency)

    Returns:
        SessionResponse: Session details

    """
    operation_id = str(uuid.uuid4())
    session_metrics.record_operation_start("get", operation_id)

    try:
        logger.debug(
            "Getting session",
            session_id=str(session_id),
            tenant_id=str(tenant_id),
        )

        # Record metrics
        session_metrics.record_session_get(operation_id, success=True)
        api_requests_total.labels(version="v1", resource="sessions", operation="get", status="success").inc()

        return SessionResponse.model_validate(verified_session)

    except Exception:
        # Record failure metrics
        session_metrics.record_session_get(operation_id, success=False)
        api_requests_total.labels(version="v1", resource="sessions", operation="get", status="error").inc()
        raise


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update session",
    description="Update session status, metadata, datasource, or LLM configuration.",
)
async def update_session(
    request: Request,
    session_id: UUID,
    session_data: SessionUpdate,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    verified_session: Annotated[Session, Depends(verify_session_ownership)],
) -> SessionResponse:
    """Update session.

    Args:
        request: FastAPI request
        session_id: Session identifier
        session_data: Session update data
        tenant_id: Current tenant ID
        session_service: Session service instance
        verified_session: Verified session instance (from dependency)

    Returns:
        SessionResponse: Updated session details

    """
    operation_id = str(uuid.uuid4())
    session_metrics.record_operation_start("update", operation_id)

    try:
        logger.info(
            "Updating session",
            session_id=str(session_id),
            tenant_id=str(tenant_id),
        )

        session = await session_service.update_session(
            session_id=session_id,
            tenant_id=tenant_id,
            session_metadata=session_data.session_metadata,
            title=session_data.title,
        )

        # Record metrics
        session_metrics.record_session_updated(operation_id, success=True)
        api_requests_total.labels(version="v1", resource="sessions", operation="update", status="success").inc()

        return SessionResponse.model_validate(session)

    except Exception:
        # Record failure metrics
        session_metrics.record_session_updated(operation_id, success=False)
        api_requests_total.labels(version="v1", resource="sessions", operation="update", status="error").inc()
        raise


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete session",
    description="Delete a session permanently.",
)
async def delete_session(
    request: Request,
    session_id: UUID,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    verified_session: Annotated[Session, Depends(verify_session_ownership)],
) -> None:
    """Delete session.

    Args:
        request: FastAPI request
        session_id: Session identifier
        tenant_id: Current tenant ID
        user_id: Current user ID
        session_service: Session service instance
        verified_session: Verified session instance (from dependency)

    """
    operation_id = str(uuid.uuid4())
    session_metrics.record_operation_start("delete", operation_id)

    try:
        logger.info(
            "Deleting session",
            session_id=str(session_id),
            tenant_id=str(tenant_id),
        )

        _, session_count = await session_service.delete_session(
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # Record metrics with actual session count
        session_metrics.record_session_deleted(str(tenant_id), count=session_count, operation_id=operation_id)
        api_requests_total.labels(version="v1", resource="sessions", operation="delete", status="success").inc()

    except Exception:
        # Record failure metrics
        session_metrics.record_operation_complete("delete", operation_id, success=False)
        api_requests_total.labels(version="v1", resource="sessions", operation="delete", status="error").inc()
        raise


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Get chat history for a session",
    description="Retrieve conversation history (questions and SQL) for a session",
)
async def get_chat_history(
    session_id: UUID,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    message_service: MessageServiceDep,
    limit: int = 50,
) -> ChatHistoryResponse:
    """Get chat history for a session.

    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return
        tenant_id: Tenant identifier from auth
        user_id: User identifier from auth
        session_service: Session service instance
        message_service: Message service instance

    Returns:
        Chat history response

    Raises:
        HTTPException: If session not found or access denied

    """
    # Validate session ownership
    try:
        await verify_session_ownership_helper(
            session_id=session_id,
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
        "api.agent.get_chat_history",
        attributes={
            "tenant_id": str(tenant_id),
            "session_id": str(session_id),
        },
    ):
        try:
            # Fetch messages with pagination
            messages, total = await message_service.get_paginated_messages(
                session_id=session_id,
                tenant_id=tenant_id,
                skip=0,
                limit=limit,
            )

            # Convert to response format
            message_responses = [
                ChatMessageResponse(
                    id=msg.id,
                    session_id=msg.session_id,
                    question=msg.question,
                    sql=msg.sql,
                    created_at=msg.created_at,
                )
                for msg in messages
            ]

            return ChatHistoryResponse(
                messages=message_responses,
                total=total,
            )

        except Exception as e:
            logger.error(
                "Failed to fetch chat history",
                extra={
                    "error": str(e),
                    "session_id": str(session_id),
                    "tenant_id": str(tenant_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch chat history: {str(e)}",
            ) from e
