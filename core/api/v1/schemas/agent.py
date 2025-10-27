"""API schemas for MAC-SQL agent endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class GenerateSQLRequest(BaseModel):
    """Request schema for SQL generation."""

    question: str = Field(..., description="Natural language question", min_length=1, max_length=1000)
    session_id: UUID = Field(..., description="Session ID (datasource will be fetched from session)")
    explain_mode: bool = Field(default=False, description="Return reasoning without executing")
    use_cache: bool = Field(default=True, description="Use cached results if available")
    timeout_seconds: float = Field(default=30.0, description="Query execution timeout", gt=0, le=300)
    max_rows: int = Field(default=10000, description="Maximum rows to return", gt=0, le=100000)
    max_history_messages: int = Field(default=10, description="Max conversation history to include", ge=0, le=50)


class ExecuteSQLRequest(BaseModel):
    """Request schema for SQL generation and execution."""

    question: str = Field(..., description="Natural language question", min_length=1, max_length=1000)
    session_id: UUID = Field(..., description="Session ID (datasource will be fetched from session)")
    use_cache: bool = Field(default=True, description="Use cached results if available")
    timeout_seconds: float = Field(default=30.0, description="Query execution timeout", gt=0, le=300)
    max_rows: int = Field(default=10000, description="Maximum rows to return", gt=0, le=100000)
    max_history_messages: int = Field(default=10, description="Max conversation history to include", ge=0, le=50)


class ExplainSQLRequest(BaseModel):
    """Request schema for SQL explanation (reasoning mode)."""

    question: str = Field(..., description="Natural language question", min_length=1, max_length=1000)
    session_id: UUID = Field(..., description="Session ID (datasource will be fetched from session)")
    max_history_messages: int = Field(default=10, description="Max conversation history to include", ge=0, le=50)


class AgentReasoningResponse(BaseModel):
    """Response schema for agent reasoning."""

    schema_selection: Optional[str] = Field(None, description="Schema selection reasoning")
    query_decomposition: Optional[str] = Field(None, description="Query decomposition reasoning")
    sql_refinement: Optional[str] = Field(None, description="SQL refinement reasoning")


class ValidationResponse(BaseModel):
    """Response schema for SQL validation."""

    is_valid: bool = Field(..., description="Whether SQL is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")


class GenerateSQLResponse(BaseModel):
    """Response schema for SQL generation."""

    sql: str = Field(..., description="Generated SQL query")
    dialect: str = Field(default="postgres", description="SQL dialect")
    complexity_score: float = Field(default=0.0, description="Query complexity score (0-1)")
    execution_time_ms: float = Field(default=0.0, description="Total execution time in milliseconds")
    validation: ValidationResponse = Field(..., description="Validation results")
    reasoning: Optional[AgentReasoningResponse] = Field(None, description="Agent reasoning (if explain_mode)")
    success: bool = Field(default=True, description="Overall success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ExecuteSQLResponse(BaseModel):
    """Response schema for SQL execution."""

    sql: str = Field(..., description="Generated SQL query")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query results")
    rows_returned: int = Field(default=0, description="Number of rows returned")
    execution_time_ms: float = Field(default=0.0, description="Total execution time in milliseconds")
    cached: bool = Field(default=False, description="Whether results came from cache")
    complexity_score: float = Field(default=0.0, description="Query complexity score (0-1)")
    visualization_spec: Optional[Dict[str, Any]] = Field(None, description="Plotly chart specification")
    success: bool = Field(default=True, description="Overall success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    message_id: Optional[UUID] = Field(None, description="Message ID of saved chat interaction")


class ExplainSQLResponse(BaseModel):
    """Response schema for SQL explanation."""

    sql: str = Field(..., description="Generated SQL query")
    reasoning: AgentReasoningResponse = Field(..., description="Agent reasoning")
    complexity_score: float = Field(default=0.0, description="Query complexity score (0-1)")
    validation: ValidationResponse = Field(..., description="Validation results")
    success: bool = Field(default=True, description="Overall success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


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
