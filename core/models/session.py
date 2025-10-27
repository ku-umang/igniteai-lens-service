"""Session model for managing conversation sessions."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database.session import Base

if TYPE_CHECKING:
    from core.models.message import Message


class Session(Base):
    """Session model for storing conversation session data.

    This model supports multi-tenancy and tracks user sessions
    with datasources and LLM configurations.
    """

    __tablename__ = "sessions"

    # Primary identifier
    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        nullable=False,
        doc="Unique session identifier",
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
        doc="User identifier who owns this session",
    )

    # Session configuration
    datasource_id: Mapped[UUID] = mapped_column(
        nullable=False,
        doc="Datasource identifier for this session",
    )

    llm_config_id: Mapped[UUID] = mapped_column(
        nullable=False,
        doc="LLM configuration identifier for this session",
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Title of the session",
    )

    # Flexible metadata storage
    session_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        doc="Additional session metadata (conversation history, preferences, etc.)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        doc="Session creation timestamp",
    )

    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        doc="Session last update timestamp",
    )

    # Relationship to Messages
    messages: Mapped[list["Message"]] = relationship(  # type: ignore[name-defined]
        "Message",
        back_populates="session",
        lazy="select",
        cascade="all, delete-orphan",
    )

    # Table constraints and indexes
    __table_args__ = (
        # Composite unique constraint for tenant isolation
        UniqueConstraint("tenant_id", "id", name="uq_session_tenant_id"),
        # Composite index for efficient tenant + user queries
        Index("idx_sessions_tenant_user", "tenant_id", "user_id"),
    )

    def __repr__(self) -> str:
        """String representation of the session."""
        return f"<Session(id={self.id}, tenant_id={self.tenant_id}, user_id={self.user_id})>"
