"""Dependencies for session management."""

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.session import get_db_session
from core.exceptions.session import SessionAccessDeniedError
from core.models.session import Session
from core.security.auth import get_current_tenant_id, get_current_user_id
from core.services.session.session_service import SessionService


async def get_session_service(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> SessionService:
    """Dependency to get session service.

    Args:
        db_session: Database session

    Returns:
        SessionService: Session service instance

    """
    return SessionService(db_session=db_session)


async def verify_session_ownership(
    session_id: UUID,
    request: Request,
    session_service: Annotated[SessionService, Depends(get_session_service)],
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
) -> Session:
    """Verify that the session belongs to the current user and tenant.

    Args:
        session_id: Session identifier
        request: FastAPI request
        session_service: Session service instance
        tenant_id: Current tenant ID
        user_id: Current user ID

    Returns:
        Session: Session instance if ownership verified

    Raises:
        SessionNotFoundError: If session not found
        SessionAccessDeniedError: If session doesn't belong to user

    """
    # Get the session
    session = await session_service.get_session(session_id, tenant_id)

    # Verify ownership
    if session.user_id != user_id:
        raise SessionAccessDeniedError(
            session_id=session_id,
            tenant_id=tenant_id,
            message="You do not have permission to access this session",
        )

    return session
