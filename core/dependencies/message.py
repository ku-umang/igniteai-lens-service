"""Dependencies for message management."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.session import get_db_session
from core.services.message.message_service import MessageService


async def get_message_service(
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
) -> MessageService:
    """Dependency to get message service.

    Args:
        db_session: Database session

    Returns:
        MessageService: Message service instance

    """
    return MessageService(db_session=db_session)


# Type alias for dependency injection
MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]
