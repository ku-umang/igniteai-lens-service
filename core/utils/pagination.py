"""Pagination utilities for SQLAlchemy queries with automatic URL generation.

This module provides utilities for implementing offset-based pagination with
automatic next/previous URL generation. It requires the PaginationMiddleware
to be registered in your FastAPI application.
"""

from typing import Optional

from opentelemetry import context, trace
from sqlalchemy import Select, func, select
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession

from core.middlewares.pagination import request_object

tracer = trace.get_tracer(__name__)


class PagePaginator:
    """Pagination utility for SQLAlchemy queries with automatic next/previous URL generation.

    This paginator provides offset-based pagination with automatic URL generation for next and previous pages.
    It requires the PaginationMiddleware to be registered in the application.

    Example:
        query = select(Session).where(Session.tenant_id == tenant_id)
        paginator = PagePaginator(db_session, query, page=1, per_page=20)
        result = await paginator.get_response()
        # Returns: {count: int, next_page: str, previous_page: str, items: list}

    """

    def __init__(self, db_session: AsyncSession, query: Select, page: int, per_page: int):
        """Initialize the paginator.

        Args:
            db_session: SQLAlchemy async database session
            query: SQLAlchemy select query to paginate
            page: Current page number (1-indexed)
            per_page: Number of items per page

        """
        self.db_session = db_session
        self.query = query
        self.page = page
        self.per_page = per_page
        self.limit = per_page
        self.offset = (page - 1) * per_page
        self.request = request_object.get()
        # computed later
        self.number_of_pages = 0
        self.next_page = ""
        self.previous_page = ""

    def _get_next_page(self) -> str | None:
        if self.page >= self.number_of_pages:
            return

        url = self.request.url.include_query_params(page=self.page + 1)
        return str(url)

    def _get_previous_page(self) -> str | None:
        if self.page == 1 or self.page > self.number_of_pages + 1:
            return

        url = self.request.url.include_query_params(page=self.page - 1)
        return str(url)

    async def get_response(self) -> dict:
        with tracer.start_as_current_span("paginator_get_response") as span:
            span.set_attribute("page", self.page)
            span.set_attribute("per_page", self.per_page)
            span.set_attribute("offset", self.offset)

            count = await self._get_total_count()
            items = await self.get_items()

            span.set_attribute("total_count", count)
            span.set_attribute("items_count", len(items))
            span.set_attribute("number_of_pages", self.number_of_pages)

            return {
                "count": count,
                "next_page": self._get_next_page(),
                "previous_page": self._get_previous_page(),
                "items": items,
            }

    async def get_items(self) -> list:
        """Execute the paginated query and return items.

        Returns:
            list: List of items for the current page

        """
        with tracer.start_as_current_span("paginator_get_items") as span:
            span.set_attribute("limit", self.limit)
            span.set_attribute("offset", self.offset)

            result: Result = await self.db_session.execute(self.query.limit(self.limit).offset(self.offset))
            if not (instance := result.scalars().all()):
                span.set_attribute("items_returned", 0)
                return []

            items = list(instance)
            span.set_attribute("items_returned", len(items))
            return items

    def _get_number_of_pages(self, count: int) -> int:
        rest = count % self.per_page
        quotient = count // self.per_page
        return quotient if not rest else quotient + 1

    async def _get_total_count(self) -> int:
        """Get total count of items matching the query.

        Returns:
            int: Total number of items

        """
        with tracer.start_as_current_span("paginator_get_total_count") as span:
            result: Result = await self.db_session.execute(select(func.count()).select_from(self.query.subquery()))
            record = result.scalar()
            record = record if record else 0
            self.number_of_pages = self._get_number_of_pages(record)

            span.set_attribute("total_count", record)
            span.set_attribute("number_of_pages", self.number_of_pages)

            return record


async def paginate_per_page(
    db_session: AsyncSession,
    query: Select,
    page: int,
    per_page: int,
    parent_context: Optional[context.Context] = None,
) -> dict:
    """Helper function to paginate a query with optional parent trace context.

    Args:
        db_session: SQLAlchemy async database session
        query: SQLAlchemy select query to paginate
        page: Current page number (1-indexed)
        per_page: Number of items per page
        parent_context: Optional parent OpenTelemetry context for trace propagation.
                       If not provided, uses the current active context.

    Returns:
        dict: Pagination response with count, next_page, previous_page, and items

    """
    # Use provided parent context or fall back to current context
    ctx = parent_context if parent_context is not None else context.get_current()

    with tracer.start_as_current_span("paginate_per_page", context=ctx) as span:
        span.set_attribute("page", page)
        span.set_attribute("per_page", per_page)

        paginator = PagePaginator(db_session, query, page, per_page)
        response = await paginator.get_response()

        span.set_attribute("response_count", response.get("count", 0))

        return response
