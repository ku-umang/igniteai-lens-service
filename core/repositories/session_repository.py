"""Repository for session data access with multi-tenant support."""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from opentelemetry import trace
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from core.exceptions.session import SessionNotFoundError
from core.logging import get_logger
from core.models.session import Session

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class SessionRepository:
    """Repository for session database operations with tenant isolation."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the session repository.

        Args:
            db_session: SQLAlchemy async database session

        """
        self.db_session = db_session

    async def create(
        self,
        tenant_id: UUID,
        user_id: UUID,
        datasource_id: UUID,
        llm_config_id: UUID,
        session_metadata: Optional[dict] = None,
    ) -> Session:
        """Create a new session with tenant scoping.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            datasource_id: Datasource identifier
            llm_config_id: LLM configuration identifier
            session_metadata: Optional session metadata

        Returns:
            Session: Created session instance

        """
        with tracer.start_as_current_span("session_repository_create") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))
            span.set_attribute("datasource_id", str(datasource_id))
            span.set_attribute("llm_config_id", str(llm_config_id))

            session = Session(
                tenant_id=tenant_id,
                user_id=user_id,
                datasource_id=datasource_id,
                llm_config_id=llm_config_id,
                session_metadata=session_metadata or {},
            )

            self.db_session.add(session)
            await self.db_session.flush()
            await self.db_session.refresh(session)

            logger.info(
                "Session created",
                session_id=str(session.id),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                datasource_id=str(datasource_id),
                llm_config_id=str(llm_config_id),
            )

            span.set_attribute("session_id", str(session.id))
            return session

    async def get_by_id(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> Optional[Session]:
        """Get session by ID with tenant isolation.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            Optional[Session]: Session if found and belongs to tenant

        """
        with tracer.start_as_current_span("session_repository_get_by_id") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            stmt = select(Session).where(
                Session.id == session_id,
                Session.tenant_id == tenant_id,
            )

            result = await self.db_session.execute(stmt)
            session = result.scalar_one_or_none()

            span.set_attribute("found", session is not None)

            if session:
                logger.debug(
                    "Session retrieved",
                    session_id=str(session_id),
                    tenant_id=str(tenant_id),
                )
            else:
                logger.debug(
                    "Session not found",
                    session_id=str(session_id),
                    tenant_id=str(tenant_id),
                )

            return session

    async def get_by_id_or_raise(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> Session:
        """Get session by ID or raise exception.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            Session: Session instance

        Raises:
            SessionNotFoundError: If session not found or doesn't belong to tenant

        """
        session = await self.get_by_id(session_id, tenant_id)
        if not session:
            raise SessionNotFoundError(session_id=session_id, tenant_id=tenant_id)
        return session

    async def list_by_user(
        self,
        tenant_id: UUID,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Session], int]:
        """List sessions for a user with pagination and filtering.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            tuple[list[Session], int]: List of sessions and total count

        """
        with tracer.start_as_current_span("session_repository_list_by_user") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))
            span.set_attribute("skip", skip)
            span.set_attribute("limit", limit)

            # Build query
            stmt = select(Session).where(
                Session.tenant_id == tenant_id,
                Session.user_id == user_id,
            )

            # Count total
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total_result = await self.db_session.execute(count_stmt)
            total = total_result.scalar_one()

            # Apply pagination and ordering
            stmt = stmt.order_by(Session.created_at.desc()).offset(skip).limit(limit)

            # Execute query
            result = await self.db_session.execute(stmt)
            sessions = list(result.scalars().all())

            span.set_attribute("sessions_found", len(sessions))
            span.set_attribute("total_count", total)

            logger.debug(
                "Sessions listed",
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                count=len(sessions),
                total=total,
            )

            return sessions, total

    async def update(
        self,
        session_id: UUID,
        tenant_id: UUID,
        session_metadata: Optional[dict] = None,
        title: Optional[str] = None,
    ) -> Session:
        """Update session with tenant isolation.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation
            session_metadata: Optional metadata to merge
            title: Optional new title of the session

        Returns:
            Session: Updated session instance

        Raises:
            SessionNotFoundError: If session not found or doesn't belong to tenant

        """
        with tracer.start_as_current_span("session_repository_update") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            # Get existing session
            session = await self.get_by_id_or_raise(session_id, tenant_id)

            # Update fields
            if session_metadata is not None:
                # Merge metadata
                current_metadata = session.session_metadata or {}
                current_metadata.update(session_metadata)
                session.session_metadata = current_metadata
                # Mark the JSON column as modified so SQLAlchemy tracks the change
                flag_modified(session, "session_metadata")
                span.set_attribute("updated_metadata", True)

            if title:
                session.title = title
                span.set_attribute("updated_title", str(title))

            session.updated_at = datetime.now(timezone.utc)

            await self.db_session.flush()
            await self.db_session.refresh(session)

            logger.info(
                "Session updated",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
            )

            return session

    async def delete(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> bool:
        """Delete session with tenant isolation.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            bool: True if deleted successfully

        Raises:
            SessionNotFoundError: If session not found or doesn't belong to tenant

        """
        with tracer.start_as_current_span("session_repository_delete") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            # Verify session exists and belongs to tenant
            session = await self.get_by_id_or_raise(session_id, tenant_id)

            await self.db_session.delete(session)
            await self.db_session.flush()

            logger.info(
                "Session deleted",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
            )

            span.set_attribute("deleted", True)
            return True

    async def count_active_sessions(
        self,
        tenant_id: UUID,
        user_id: UUID,
    ) -> int:
        """Count active sessions for a user.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            int: Number of active sessions

        """
        with tracer.start_as_current_span("session_repository_count_active") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))

            stmt = select(func.count()).where(
                Session.tenant_id == tenant_id,
                Session.user_id == user_id,
            )

            result = await self.db_session.execute(stmt)
            count = result.scalar_one()

            span.set_attribute("active_count", count)
            return count

    async def count_sessions_by_tenant(
        self,
        tenant_id: UUID,
    ) -> int:
        """Count total sessions for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            int: Number of sessions for the tenant

        """
        with tracer.start_as_current_span("session_repository_count_by_tenant") as span:
            span.set_attribute("tenant_id", str(tenant_id))

            stmt = select(func.count()).where(
                Session.tenant_id == tenant_id,
            )

            result = await self.db_session.execute(stmt)
            count = result.scalar_one()

            span.set_attribute("tenant_session_count", count)
            return count

    def get_user_sessions_query(
        self,
        tenant_id: UUID,
        user_id: UUID,
    ) -> Select:
        """Get base query for user sessions with filtering.

        This method returns an unexecuted SQLAlchemy Select query that can be
        used with pagination utilities. The query includes tenant/user isolation,
        optional status filtering, and ordering by created_at descending.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            Select: SQLAlchemy Select query (unexecuted)

        """
        # Build base query with tenant and user filtering
        query = select(Session).where(
            Session.tenant_id == tenant_id,
            Session.user_id == user_id,
        )

        # Order by created_at descending
        query = query.order_by(Session.created_at.desc())

        logger.debug(
            "Built user sessions query",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
        )

        return query
