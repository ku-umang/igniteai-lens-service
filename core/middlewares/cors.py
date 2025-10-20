import re
from typing import List, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def configure_cors(app: FastAPI) -> None:
    """Configure CORS middleware with environment-specific settings."""
    # Get allowed origins based on environment
    allowed_origins = _get_allowed_origins()

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
        max_age=settings.CORS_MAX_AGE,
    )

    logger.info(
        "CORS middleware configured",
        environment=settings.ENVIRONMENT.value,
        allow_origins=allowed_origins if len(str(allowed_origins)) < 200 else f"{len(allowed_origins)} origins",
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
    )


def _get_allowed_origins() -> Union[List[str], List[str]]:
    """Get allowed origins based on environment and configuration."""
    if settings.ENVIRONMENT.value == "development":
        # In development, be more permissive but still secure
        origins = []

        # Add configured origins
        if settings.CORS_ALLOWED_ORIGINS:
            origins.extend(settings.CORS_ALLOWED_ORIGINS)

        # Add common development origins if not already present
        dev_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8080",
        ]

        for origin in dev_origins:
            if origin not in origins:
                origins.append(origin)

        # Allow wildcard only if explicitly configured
        if settings.CORS_ALLOW_ALL_ORIGINS:
            return ["*"]

        return origins

    elif settings.ENVIRONMENT.value == "production":
        # In production, be strict with origins
        if not settings.CORS_ALLOWED_ORIGINS:
            logger.warning("No CORS origins configured for production environment. This may block legitimate requests.")
            return []

        # Validate origins in production
        validated_origins = []
        for origin in settings.CORS_ALLOWED_ORIGINS:
            if _is_valid_origin(origin):
                validated_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin skipped: {origin}")

        return validated_origins

    else:
        # Testing or other environments
        return settings.CORS_ALLOWED_ORIGINS or ["*"]


def _is_valid_origin(origin: str) -> bool:
    """Validate origin format."""
    if not origin:
        return False

    if origin == "*":
        return True

    try:
        if not _has_valid_protocol(origin):
            return False

        host, port = _extract_host_and_port(origin)

        if not host:
            return False

        return _is_valid_host(host) and _is_valid_port(port)

    except Exception:
        return False


def _has_valid_protocol(origin: str) -> bool:
    """Check if origin has valid HTTP/HTTPS protocol."""
    return re.match(r"^https?://.+", origin) is not None


def _extract_host_and_port(origin: str) -> tuple[str, int | None]:
    """Extract host and port from origin."""
    without_protocol = origin.split("://", 1)[1]

    if ":" in without_protocol:
        host, port_str = without_protocol.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError as e:
            raise ValueError("Invalid port format") from e
        return host, port
    else:
        return without_protocol, None


def _is_valid_port(port: int | None) -> bool:
    """Validate port number."""
    return port is None or (1 <= port <= 65535)


def _is_valid_host(host: str) -> bool:
    """Validate host (localhost, IP, or domain)."""
    if _is_localhost(host):
        return True

    if _is_valid_ip_address(host):
        return True

    return _is_valid_domain(host)


def _is_localhost(host: str) -> bool:
    """Check if host is localhost."""
    return host in ["localhost", "127.0.0.1", "0.0.0.0"]


def _is_valid_ip_address(host: str) -> bool:
    """Validate IPv4 address format."""
    ip_pattern = r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    return re.match(ip_pattern, host) is not None


def _is_valid_domain(host: str) -> bool:
    """Validate domain name format."""
    if "." not in host:
        return False

    parts = host.split(".")
    return all(_is_valid_domain_part(part) for part in parts)


def _is_valid_domain_part(part: str) -> bool:
    """Validate individual domain part."""
    if not part or len(part) > 63:
        return False
    return re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?$", part) is not None
