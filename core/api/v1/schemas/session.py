"""Pydantic schemas for session API."""

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.schemas.base import TimestampMixin


class SessionBase(BaseModel):
    """Base session schema with common fields."""

    model_config = ConfigDict(from_attributes=True)

    datasource_id: UUID = Field(..., description="Datasource identifier")
    llm_config_id: UUID = Field(..., description="LLM configuration identifier")
    session_metadata: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional session metadata",
    )


class SessionCreate(SessionBase):
    """Schema for creating a new session."""

    @field_validator("session_metadata")
    @classmethod
    def validate_session_metadata(cls, v: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Validate session_metadata is a dictionary."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("session_metadata must be a dictionary")
        return v


class SessionUpdate(BaseModel):
    """Schema for updating an existing session."""

    model_config = ConfigDict(from_attributes=True)

    session_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Updated session metadata (merges with existing)",
    )
    title: Optional[str] = Field(default=None, description="Updated title of the session")

    @field_validator("session_metadata")
    @classmethod
    def validate_session_metadata(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Validate session_metadata is a dictionary if provided."""
        if v is not None and not isinstance(v, dict):
            raise ValueError("session_metadata must be a dictionary")
        return v


class SessionResponse(SessionBase, TimestampMixin):
    """Schema for session response."""

    id: UUID = Field(..., description="Session identifier")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    user_id: UUID = Field(..., description="User identifier")
    title: Optional[str] = Field(default=None, description="Title of the session")
    model_config = ConfigDict(from_attributes=True)


class SessionFilter(BaseModel):
    """Schema for filtering sessions."""

    datasource_id: Optional[UUID] = Field(
        default=None,
        description="Filter by datasource ID",
    )
    llm_config_id: Optional[UUID] = Field(
        default=None,
        description="Filter by LLM config ID",
    )


class ChatMessageResponse(BaseModel):
    """Response schema for a chat message."""

    id: UUID = Field(..., description="Message ID")
    session_id: UUID = Field(..., description="Session ID")
    question: str = Field(..., description="User's question")
    sql: Optional[str] = Field(None, description="Generated SQL (null if generation failed)")
    created_at: datetime = Field(..., description="Message creation timestamp")


class ChatHistoryResponse(BaseModel):
    """Response schema for chat history."""

    messages: List[ChatMessageResponse] = Field(..., description="List of chat messages")
    total: int = Field(..., description="Total number of messages in session")
