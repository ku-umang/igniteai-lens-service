"""Dependencies for agent endpoints."""

from typing import Annotated

from fastapi import Depends

from core.services.agent.agent_service import AgentService

# Singleton instance
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Get or create agent service instance.

    Returns:
        AgentService instance

    """
    global _agent_service  # noqa: PLW0603
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


# Type aliases for dependency injection
AgentServiceDep = Annotated[AgentService, Depends(get_agent_service)]
