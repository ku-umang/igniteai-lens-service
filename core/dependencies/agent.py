"""Dependencies for MAC-SQL agent endpoints."""

from typing import Annotated

from fastapi import Depends

from core.services.sql.sql_service import SQLService

# Singleton instance
_sql_service: SQLService | None = None


def get_agent_service() -> SQLService:
    """Get or create SQL service instance.

    Returns:
        SQLService instance

    """
    global _sql_service  # noqa: PLW0603
    if _sql_service is None:
        _sql_service = SQLService()
    return _sql_service


# Type aliases for dependency injection
AgentServiceDep = Annotated[SQLService, Depends(get_agent_service)]
