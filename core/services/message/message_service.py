"""Message service with business logic and tenant-aware caching."""

from typing import Optional
from uuid import UUID

from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from core.cache.tenant_cache import TenantCacheManager, get_tenant_cache
from core.config import settings
from core.logging import get_logger
from core.models.message import Message
from core.repositories.message_repository import MessageRepository

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MessageService:
    """Service for message business logic with caching and validation."""

    def __init__(
        self,
        db_session: AsyncSession,
        tenant_cache: Optional[TenantCacheManager] = None,
    ) -> None:
        """Initialize the message service.

        Args:
            db_session: SQLAlchemy async database session
            tenant_cache: Optional tenant cache manager

        """
        self.db_session = db_session
        self.repository = MessageRepository(db_session)
        self.tenant_cache = tenant_cache or get_tenant_cache()

    async def save_chat_interaction(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        question: str,
        sql: Optional[str] = None,
    ) -> Message:
        """Save a chat interaction (question + SQL) to the message history.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            question: User's natural language question
            sql: Generated SQL query (optional, null if generation failed)

        Returns:
            Message: Created message instance

        """
        with tracer.start_as_current_span("message_service_save_interaction") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))

            # Create message
            message = await self.repository.create(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                question=question,
                sql=sql,
            )

            # Invalidate session history cache
            # await self._invalidate_session_history_cache(session_id, tenant_id)

            logger.info(
                "Chat interaction saved",
                message_id=str(message.id),
                session_id=str(session_id),
                tenant_id=str(tenant_id),
            )

            span.set_attribute("message_id", str(message.id))
            return message

    async def get_session_history(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        limit: int = 10,
        use_cache: bool = True,
    ) -> list[Message]:
        """Get recent chat history for a session.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation
            user_id: User identifier
            limit: Maximum number of messages to return (default 10)
            use_cache: Whether to use cache (default True)

        Returns:
            list[Message]: List of recent messages ordered by created_at DESC

        """
        with tracer.start_as_current_span("message_service_get_history") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))
            span.set_attribute("limit", limit)
            span.set_attribute("use_cache", use_cache)

            # Try cache first
            # if use_cache and self.tenant_cache:
            #     cached_history = await self._get_cached_history(
            #         session_id, tenant_id, user_id, limit
            #     )
            #     if cached_history is not None:
            #         span.set_attribute("cache_hit", True)
            #         span.set_attribute("messages_count", len(cached_history))
            #         return cached_history

            span.set_attribute("cache_hit", False)

            # Get from database
            messages = await self.repository.get_recent_messages(
                session_id=session_id,
                tenant_id=tenant_id,
                limit=limit,
            )

            # Cache the history
            # await self._cache_history(session_id, tenant_id, user_id, messages, limit)

            span.set_attribute("messages_count", len(messages))

            logger.info(
                "Session history retrieved",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                count=len(messages),
            )

            return messages

    async def get_paginated_messages(
        self,
        session_id: UUID,
        tenant_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Message], int]:
        """Get paginated messages for a session.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            tuple[list[Message], int]: List of messages and total count

        """
        with tracer.start_as_current_span("message_service_list") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("skip", skip)
            span.set_attribute("limit", limit)

            messages, total = await self.repository.get_messages_by_session(
                session_id=session_id,
                tenant_id=tenant_id,
                skip=skip,
                limit=limit,
            )

            span.set_attribute("messages_count", len(messages))
            span.set_attribute("total_count", total)

            return messages, total

    async def _cache_history(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        messages: list[Message],
        limit: int,
    ) -> None:
        """Cache session history.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            messages: Messages to cache
            limit: Limit used for the query

        """
        if not self.tenant_cache:
            return

        try:
            cache_key = f"history:limit:{limit}"
            await self.tenant_cache.set_tenant_scoped(
                base_key=cache_key,
                value=messages,
                tenant_id=tenant_id,
                user_id=user_id,
                ttl=settings.CACHE_DEFAULT_TTL,
                session_id=str(session_id),
            )
        except Exception as e:
            logger.warning(
                "Failed to cache session history",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                error=str(e),
            )

    async def _get_cached_history(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        limit: int,
    ) -> Optional[list[Message]]:
        """Get cached session history.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            limit: Limit used for the query

        Returns:
            Optional[list[Message]]: Cached messages if found

        """
        if not self.tenant_cache:
            return None

        try:
            cache_key = f"history:limit:{limit}"
            return await self.tenant_cache.get_tenant_scoped(
                base_key=cache_key,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=str(session_id),
            )
        except Exception as e:
            logger.warning(
                "Failed to get cached session history",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                error=str(e),
            )
            return None

    async def _invalidate_session_history_cache(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> None:
        """Invalidate session history cache.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier

        """
        if not self.tenant_cache:
            return

        try:
            # Invalidate all history-related caches for this session/tenant
            await self.tenant_cache.invalidate_tenant_cache(
                tenant_id=tenant_id,
                cache_types=["history"],
            )
        except Exception as e:
            logger.warning(
                "Failed to invalidate session history cache",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                error=str(e),
            )
