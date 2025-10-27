"""Message model for storing chat conversation history."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.session import Base
from core.models.session import Session


class Message(Base):
    """Message model for storing conversation messages within sessions.

    This model supports multi-tenancy and tracks user questions and generated SQL
    for providing context-aware agent responses.
    """

    __tablename__ = "messages"

    # Primary identifier
    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        nullable=False,
        doc="Unique message identifier",
    )

    # Session relationship
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Session identifier this message belongs to",
    )

    # Multi-tenant isolation
    tenant_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        doc="Tenant identifier for multi-tenant isolation",
    )

    # User association
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        doc="User identifier who created this message",
    )

    # Message content
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="User's natural language question",
    )

    sql: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Generated SQL query (null if generation failed)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Message creation timestamp",
    )

    # Relationship to Session
    session: Mapped["Session"] = relationship(  # type: ignore[name-defined]
        "Session",
        back_populates="messages",
        lazy="select",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Composite unique constraint for tenant isolation
        UniqueConstraint("tenant_id", "id", name="uq_message_tenant_id"),
        # Composite index for efficient session history queries (most recent first)
        Index("idx_messages_session_created", "session_id", "created_at"),
        # Composite index for efficient tenant + user queries
        Index("idx_messages_tenant_user", "tenant_id", "user_id"),
    )

    def __repr__(self) -> str:
        """String representation of the message."""
        return f"<Message(id={self.id}, session_id={self.session_id}, question='{self.question[:50]}...')>"
