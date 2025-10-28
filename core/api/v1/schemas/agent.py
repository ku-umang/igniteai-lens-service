"""API schemas for agent endpoints."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class AgentRunRequest(BaseModel):
    """Request schema for SQL generation and execution."""

    question: str = Field(..., description="Natural language question", min_length=1, max_length=1000)
    session_id: UUID = Field(..., description="Session ID (datasource will be fetched from session)")
    max_history_messages: int = Field(default=10, description="Max conversation history to include", ge=0, le=50)


class AgentResponse(BaseModel):
    """Response schema for agent execution."""

    message_id: UUID = Field(..., description="Message ID for this interaction")
    question: str = Field(..., description="User's question")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Query result data")
    num_rows: int = Field(default=0, description="Number of rows returned")
    answer: Optional[str] = Field(None, description="Natural language answer to the question")
    insights: Optional[List[str]] = Field(None, description="Key insights from analysis")
    visualization_spec: Optional[Dict[str, Any]] = Field(None, description="Chart visualization specification")
    total_time_ms: float = Field(..., description="Total execution time in milliseconds")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    success: bool = Field(..., description="Whether the execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
