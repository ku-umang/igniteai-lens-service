"""Base Pydantic schemas and mixins."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""

    model_config = ConfigDict(from_attributes=True)

    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    skip: int = Field(default=0, ge=0, description="Number of records to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Number of records to return")


class SearchParams(PaginationParams):
    """Standard search parameters."""

    query: str = Field(default="", max_length=100, description="Search query")


class BulkOperationResult(BaseModel):
    """Result of a bulk operation."""

    total_requested: int = Field(..., description="Total number of items requested")
    total_processed: int = Field(..., description="Total number of items processed successfully")
    total_failed: int = Field(..., description="Total number of items that failed")
    errors: list[str] = Field(default=[], description="List of error messages for failed items")


class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: Optional[str] = Field(None, description="Application version")
    environment: Optional[str] = Field(None, description="Environment name")
    services: Optional[dict[str, str]] = Field(None, description="Service health status")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response with automatic next/previous URL generation.

    This schema is designed to work with the PagePaginator utility which automatically
    generates next and previous page URLs based on the current request.

    Example:
        # In your route:
        result = await paginate_per_page(db_session, query, page=1, per_page=20)
        return PaginatedResponse[SessionResponse](
            count=result["count"],
            next_page=result["next_page"],
            previous_page=result["previous_page"],
            items=[SessionResponse.model_validate(item) for item in result["items"]]
        )

    Attributes:
        count: Total number of items across all pages
        next_page: URL for the next page (None if on last page)
        previous_page: URL for the previous page (None if on first page)
        items: List of items for the current page

    """

    count: int = Field(..., description="Total number of items across all pages")
    next_page: Optional[str] = Field(None, description="URL for the next page")
    previous_page: Optional[str] = Field(None, description="URL for the previous page")
    items: list[T] = Field(..., description="List of items for the current page")
