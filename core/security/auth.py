"""Authentication and authorization utilities."""

from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, status

from core.config import settings

# Default tenant ID for development
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")


async def get_current_tenant_id(request: Request) -> UUID:
    """Extract tenant ID from request.

    Priority:
    1. From request.state (set by middleware)
    2. From x-tenant-id header
    3. Default tenant (in development mode only)
    """
    # Try from request state (set by middleware)
    if hasattr(request.state, "tenant_id") and request.state.tenant_id:
        try:
            return UUID(request.state.tenant_id)
        except (ValueError, TypeError):
            pass

    # Try from header directly
    tenant_header = request.headers.get("x-tenant-id")
    if tenant_header:
        try:
            return UUID(tenant_header)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tenant ID format") from e

    # In development, use default tenant
    if settings.ENVIRONMENT == "development":
        return DEFAULT_TENANT_ID

    # In production, tenant ID is required
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Tenant ID is required. Please provide X-Tenant-ID header"
    )


async def get_optional_tenant_id(request: Request) -> Optional[UUID]:
    """Get tenant ID if available, otherwise return None."""
    try:
        return await get_current_tenant_id(request)
    except HTTPException:
        return None


async def get_current_user_id(request: Request) -> UUID:
    """Get current user ID from request.

    For now, this is a placeholder that extracts user_id from headers.
    In production, this should extract from JWT token.

    Args:
        request: FastAPI request

    Returns:
        UUID: Current user ID

    Raises:
        HTTPException: If user ID is missing or invalid

    """
    # Try to get from header (temporary solution)
    user_id_header = request.headers.get("x-user-id")

    if user_id_header:
        try:
            return UUID(user_id_header)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format",
            ) from e

    # In development, use a default user
    from core.config import Environment, settings

    if settings.ENVIRONMENT == Environment.DEVELOPMENT:
        # Default development user
        return UUID("00000000-0000-0000-0000-000000000002")

    # In production, user ID is required (should come from JWT)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="User ID is required. Please provide X-User-ID header or valid JWT token",
    )
