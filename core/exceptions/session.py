"""Session-specific exceptions."""

from typing import Optional
from uuid import UUID

from core.exceptions.base import IgniteLensBaseError, NotFoundError


class SessionNotFoundError(NotFoundError):
    """Raised when a session is not found."""

    def __init__(
        self,
        session_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        message: str = "Session not found",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = str(session_id)
        if tenant_id:
            details["tenant_id"] = str(tenant_id)
        super().__init__(
            message=message,
            resource_type="session",
            resource_id=str(session_id) if session_id else None,
            details=details,
            **kwargs,
        )


class SessionExpiredError(IgniteLensBaseError):
    """Raised when a session has expired.

    .. deprecated::
        Session expiration functionality has been removed. This exception is kept for
        backward compatibility but is no longer raised by the system. Sessions no longer
        have automatic expiration - they must be manually marked as expired/inactive.
    """

    def __init__(
        self,
        session_id: Optional[UUID] = None,
        message: str = "Session has expired",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = str(session_id)
        super().__init__(message=message, details=details, **kwargs)


class InvalidSessionStateError(IgniteLensBaseError):
    """Raised when a session is in an invalid state for the requested operation."""

    def __init__(
        self,
        session_id: Optional[UUID] = None,
        current_status: Optional[str] = None,
        expected_status: Optional[str] = None,
        message: str = "Invalid session state",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = str(session_id)
        if current_status:
            details["current_status"] = current_status
        if expected_status:
            details["expected_status"] = expected_status
        super().__init__(message=message, details=details, **kwargs)


class SessionAccessDeniedError(IgniteLensBaseError):
    """Raised when access to a session is denied (cross-tenant access attempt)."""

    def __init__(
        self,
        session_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        message: str = "Access to session denied",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if session_id:
            details["session_id"] = str(session_id)
        if tenant_id:
            details["tenant_id"] = str(tenant_id)
        details["reason"] = "Cross-tenant access not allowed"
        super().__init__(message=message, details=details, **kwargs)


class DatasourceNotFoundError(NotFoundError):
    """Raised when a datasource is not found."""

    def __init__(
        self,
        datasource_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        message: str = "Datasource not found",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if datasource_id:
            details["datasource_id"] = str(datasource_id)
        if tenant_id:
            details["tenant_id"] = str(tenant_id)
        super().__init__(
            message=message,
            resource_type="datasource",
            resource_id=str(datasource_id) if datasource_id else None,
            details=details,
            **kwargs,
        )


class LLMConfigNotFoundError(NotFoundError):
    """Raised when an LLM configuration is not found."""

    def __init__(
        self,
        llm_config_id: Optional[UUID] = None,
        tenant_id: Optional[UUID] = None,
        message: str = "LLM configuration not found",
        **kwargs,
    ) -> None:
        details = kwargs.pop("details", {})
        if llm_config_id:
            details["llm_config_id"] = str(llm_config_id)
        if tenant_id:
            details["tenant_id"] = str(tenant_id)
        super().__init__(
            message=message,
            resource_type="llm_config",
            resource_id=str(llm_config_id) if llm_config_id else None,
            details=details,
            **kwargs,
        )
