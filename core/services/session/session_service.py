"""Session service with business logic and tenant-aware caching."""

from typing import Optional
from uuid import UUID

from opentelemetry import context, trace
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession

from core.cache.tenant_cache import TenantCacheManager, get_tenant_cache
from core.config import settings
from core.logging import get_logger
from core.models.session import Session
from core.repositories.session_repository import SessionRepository

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class SessionService:
    """Service for session business logic with caching and validation."""

    def __init__(
        self,
        db_session: AsyncSession,
        tenant_cache: Optional[TenantCacheManager] = None,
    ) -> None:
        """Initialize the session service.

        Args:
            db_session: SQLAlchemy async database session
            tenant_cache: Optional tenant cache manager

        """
        self.db_session = db_session
        self.repository = SessionRepository(db_session)
        self.tenant_cache = tenant_cache or get_tenant_cache()

    async def create_session(
        self,
        tenant_id: UUID,
        user_id: UUID,
        datasource_id: UUID,
        llm_config_id: UUID,
        session_metadata: Optional[dict] = None,
    ) -> tuple[Session, int]:
        """Create a new session with validation and caching.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            datasource_id: Datasource identifier
            llm_config_id: LLM configuration identifier
            session_metadata: Optional session metadata

        Returns:
            tuple[Session, int]: Created session instance and current session count for tenant

        Raises:
            SessionLimitExceededError: If user exceeds session limit
            DatasourceNotFoundError: If datasource not found
            LLMConfigNotFoundError: If LLM config not found

        """
        with tracer.start_as_current_span("session_service_create") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))

            # Validate datasource and LLM config (placeholder - implement when models exist)
            await self._validate_datasource(tenant_id, datasource_id)
            await self._validate_llm_config(tenant_id, llm_config_id)

            # Create session
            session = await self.repository.create(
                tenant_id=tenant_id,
                user_id=user_id,
                datasource_id=datasource_id,
                llm_config_id=llm_config_id,
                session_metadata=session_metadata,
            )

            # Get current session count for the tenant
            session_count = await self.repository.count_sessions_by_tenant(tenant_id)

            # Cache the session
            await self._cache_session(session)

            # Invalidate user session list cache
            await self._invalidate_user_session_list_cache(tenant_id, user_id)

            logger.info(
                "Session created via service",
                session_id=str(session.id),
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                tenant_session_count=session_count,
            )

            span.set_attribute("session_id", str(session.id))
            span.set_attribute("tenant_session_count", session_count)
            return session, session_count

    async def get_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
        use_cache: bool = True,
    ) -> Session:
        """Get session by ID with caching.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            use_cache: Whether to use cache

        Returns:
            Session: Session instance

        Raises:
            SessionNotFoundError: If session not found

        """
        with tracer.start_as_current_span("session_service_get") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("use_cache", use_cache)

            # Try cache first
            if use_cache and self.tenant_cache:
                cached_session = await self._get_cached_session(session_id, tenant_id)
                if cached_session:
                    span.set_attribute("cache_hit", True)
                    return cached_session

            span.set_attribute("cache_hit", False)

            # Get from database
            session = await self.repository.get_by_id_or_raise(session_id, tenant_id)

            # Cache the session
            await self._cache_session(session)

            return session

    async def list_user_sessions(
        self,
        tenant_id: UUID,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[Session], int]:
        """List sessions for a user with caching.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            tuple[list[Session], int]: List of sessions and total count

        """
        with tracer.start_as_current_span("session_service_list") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))

            sessions, total = await self.repository.list_by_user(
                tenant_id=tenant_id,
                user_id=user_id,
                skip=skip,
                limit=limit,
            )

            span.set_attribute("sessions_count", len(sessions))
            span.set_attribute("total_count", total)

            return sessions, total

    def get_user_sessions_query(
        self,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[Select, context.Context]:
        """Get base query for user sessions with filtering.

        This method returns an unexecuted SQLAlchemy Select query that can be
        used with pagination utilities in API routes. It wraps the repository
        method and provides a service-layer interface for query building.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            Select: SQLAlchemy Select query (unexecuted)

        """
        with tracer.start_as_current_span("session_service_list") as span:
            span.set_attribute("tenant_id", str(tenant_id))
            span.set_attribute("user_id", str(user_id))
            logger.debug("Getting user sessions query via service", tenant_id=str(tenant_id), user_id=str(user_id))
            ctx = context.get_current()
            return self.repository.get_user_sessions_query(
                tenant_id=tenant_id,
                user_id=user_id,
            ), ctx

    async def update_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
        session_metadata: Optional[dict] = None,
        title: Optional[str] = None,
    ) -> Session:
        """Update session with validation and cache invalidation.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            session_metadata: Optional metadata to merge
            title: Optional new title of the session

        Returns:
            Session: Updated session instance

        Raises:
            SessionNotFoundError: If session not found
            DatasourceNotFoundError: If datasource not found
            LLMConfigNotFoundError: If LLM config not found

        """
        with tracer.start_as_current_span("session_service_update") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            # Update session
            session = await self.repository.update(
                session_id=session_id,
                tenant_id=tenant_id,
                session_metadata=session_metadata,
                title=title,
            )

            # Invalidate cache
            await self._invalidate_session_cache(session_id, tenant_id)
            await self._invalidate_user_session_list_cache(tenant_id, session.user_id)

            logger.info(
                "Session updated via service",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
            )

            return session

    async def delete_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
        user_id: UUID,
    ) -> tuple[bool, int]:
        """Delete session with cache invalidation.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier
            user_id: User identifier

        Returns:
            tuple[bool, int]: True if deleted successfully and current session count for tenant

        Raises:
            SessionNotFoundError: If session not found

        """
        with tracer.start_as_current_span("session_service_delete") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("tenant_id", str(tenant_id))

            # Delete from database
            result = await self.repository.delete(session_id, tenant_id)

            # Get current session count for the tenant (after deletion)
            session_count = await self.repository.count_sessions_by_tenant(tenant_id)

            # Invalidate cache
            await self._invalidate_session_cache(session_id, tenant_id)
            await self._invalidate_user_session_list_cache(tenant_id, user_id)

            logger.info(
                "Session deleted via service",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                tenant_session_count=session_count,
            )

            span.set_attribute("tenant_session_count", session_count)
            return result, session_count

    async def _validate_datasource(
        self,
        tenant_id: UUID,
        datasource_id: UUID,
    ) -> None:
        """Validate that datasource exists and belongs to tenant.

        Args:
            tenant_id: Tenant identifier
            datasource_id: Datasource identifier

        Raises:
            DatasourceNotFoundError: If datasource not found or doesn't belong to tenant

        """
        # TODO: Implement actual datasource validation when datasource model exists
        # For now, this is a placeholder
        logger.debug(
            "Datasource validation placeholder",
            tenant_id=str(tenant_id),
            datasource_id=str(datasource_id),
        )

    async def _validate_llm_config(
        self,
        tenant_id: UUID,
        llm_config_id: UUID,
    ) -> None:
        """Validate that LLM config exists and belongs to tenant.

        Args:
            tenant_id: Tenant identifier
            llm_config_id: LLM configuration identifier

        Raises:
            LLMConfigNotFoundError: If LLM config not found or doesn't belong to tenant

        """
        # TODO: Implement actual LLM config validation when model exists
        # For now, this is a placeholder
        logger.debug(
            "LLM config validation placeholder",
            tenant_id=str(tenant_id),
            llm_config_id=str(llm_config_id),
        )

    async def _cache_session(
        self,
        session: Session,
    ) -> None:
        """Cache a session.

        Args:
            session: Session instance to cache

        """
        if not self.tenant_cache:
            return

        try:
            await self.tenant_cache.set_tenant_scoped(
                base_key="session",
                value=session,
                tenant_id=session.tenant_id,
                user_id=session.user_id,
                ttl=settings.CACHE_DEFAULT_TTL,
                session_id=str(session.id),
            )
        except Exception as e:
            logger.warning(
                "Failed to cache session",
                session_id=str(session.id),
                tenant_id=str(session.tenant_id),
                error=str(e),
            )

    async def _get_cached_session(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> Optional[Session]:
        """Get session from cache.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier

        Returns:
            Optional[Session]: Cached session if found

        """
        if not self.tenant_cache:
            return None

        try:
            # Note: We need user_id for the cache key, but we don't have it here.
            # For now, we'll skip caching individual session gets.
            # A better approach would be to cache by session_id only.
            return None
        except Exception as e:
            logger.warning(
                "Failed to get cached session",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                error=str(e),
            )
            return None

    async def _invalidate_session_cache(
        self,
        session_id: UUID,
        tenant_id: UUID,
    ) -> None:
        """Invalidate session cache.

        Args:
            session_id: Session identifier
            tenant_id: Tenant identifier

        """
        if not self.tenant_cache:
            return

        try:
            # Invalidate all session-related caches for this tenant
            await self.tenant_cache.invalidate_tenant_cache(
                tenant_id=tenant_id,
                cache_types=["session"],
            )
        except Exception as e:
            logger.warning(
                "Failed to invalidate session cache",
                session_id=str(session_id),
                tenant_id=str(tenant_id),
                error=str(e),
            )

    async def _invalidate_user_session_list_cache(
        self,
        tenant_id: UUID,
        user_id: UUID,
    ) -> None:
        """Invalidate user session list cache.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier

        """
        if not self.tenant_cache:
            return

        try:
            await self.tenant_cache.invalidate_user_cache(
                tenant_id=tenant_id,
                user_id=user_id,
                cache_types=["session"],
            )
        except Exception as e:
            logger.warning(
                "Failed to invalidate user session list cache",
                tenant_id=str(tenant_id),
                user_id=str(user_id),
                error=str(e),
            )
