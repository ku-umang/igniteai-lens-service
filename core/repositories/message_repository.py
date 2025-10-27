"""Repository for message data access with multi-tenant support."""

from typing import Any, Dict, Optional
from uuid import UUID

from opentelemetry import trace
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.models.message import Message

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MessageRepository:
    """Repository for message database operations with tenant isolation."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the message repository.

        Args:
            db_session: SQLAlchemy async database session

        """
        self.db_session = db_session

    async def create(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
        question: str,
        sql: Optional[str] = None,
        visualization_spec: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create a new message with tenant scoping.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            question: User's natural language question
            sql: Generated SQL query (optional)
            visualization_spec: Plotly chart specification (optional)

        Returns:
            Message: Created message instance

        """
        with tracer.start_as_current_span("message_repository_create") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))

            message = Message(
                session_id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                question=question,
                sql=sql,
                visualization_spec=visualization_spec,
            )

            self.db_session.add(message)
            await self.db_session.flush()
            await self.db_session.refresh(message)

            logger.info(
                "Message created",
                message_id=str(message.id),
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                has_visualization=visualization_spec is not None,
            )

            span.set_attribute("message_id", str(message.id))
            return message

    async def get_recent_messages(
        self,
        session_id: UUID,
        tenant_id: UUID,
        limit: int = 10,
    ) -> list[Message]:
        """Get recent messages for a session ordered by creation time descending.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation
            limit: Maximum number of messages to return (default 10)

        Returns:
            list[Message]: List of recent messages

        """
        with tracer.start_as_current_span("message_repository_get_recent") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("limit", limit)

            stmt = (
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.tenant_id == tenant_id,
                )
                .order_by(Message.created_at.desc())
                .limit(limit)
            )

            result = await self.db_session.execute(stmt)
            messages = list(result.scalars().all())

            span.set_attribute("messages_found", len(messages))

            logger.debug(
                "Recent messages retrieved",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                count=len(messages),
            )

            return messages

    async def get_messages_by_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Message], int]:
        """List messages for a session with pagination.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            tuple[list[Message], int]: List of messages and total count

        """
        with tracer.start_as_current_span("message_repository_list") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("skip", skip)
            span.set_attribute("limit", limit)

            # Build query
            stmt = select(Message).where(
                Message.session_id == session_id,
                Message.tenant_id == tenant_id,
            )

            # Count total
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total_result = await self.db_session.execute(count_stmt)
            total = total_result.scalar_one()

            # Apply pagination and ordering
            stmt = stmt.order_by(Message.created_at.desc()).offset(skip).limit(limit)

            # Execute query
            result = await self.db_session.execute(stmt)
            messages = list(result.scalars().all())

            span.set_attribute("messages_found", len(messages))
            span.set_attribute("total_count", total)

            logger.debug(
                "Messages listed",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                count=len(messages),
                total=total,
            )

            return messages, total

    async def count_messages_by_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> int:
        """Count messages for a session.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            int: Number of messages in the session

        """
        with tracer.start_as_current_span("message_repository_count") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            stmt = select(func.count()).where(
                Message.session_id == session_id,
                Message.tenant_id == tenant_id,
            )

            result = await self.db_session.execute(stmt)
            count = result.scalar_one()

            span.set_attribute("message_count", count)
            return count

    def get_session_messages_query(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> Select:
        """Get base query for session messages.

        This method returns an unexecuted SQLAlchemy Select query that can be
        used with pagination utilities. The query includes tenant isolation
        and ordering by created_at descending.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            Select: SQLAlchemy Select query (unexecuted)

        """
        query = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.tenant_id == tenant_id,
            )
            .order_by(Message.created_at.desc())
        )

        logger.debug(
            "Built session messages query",
            session_id=str(session_id),
            tenant_id=str(tenant_id),
        )

        return query
