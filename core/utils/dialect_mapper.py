"""Utility for mapping connector keys to SQL dialects.

This module provides functions to convert datasource connector keys
to SQL dialect strings compatible with sqlglot and other SQL tools.
"""

from core.logging import get_logger

logger = get_logger(__name__)

# Mapping from connector keys to sqlglot-compatible dialect names
CONNECTOR_TO_DIALECT = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "duckdb": "duckdb",
    # Add more mappings as new connectors are supported
    # "oracle": "oracle",
    # "sqlserver": "tsql",
    # "snowflake": "snowflake",
    # "bigquery": "bigquery",
}

# Default dialect to use when connector is unknown
DEFAULT_DIALECT = "postgres"


def connector_key_to_dialect(connector_key: str) -> str:
    """Convert a datasource connector key to SQL dialect string.

    Args:
        connector_key: The connector key from datasource (e.g., "postgresql", "mysql")

    Returns:
        SQL dialect string compatible with sqlglot (e.g., "postgres", "mysql")

    Examples:
        >>> connector_key_to_dialect("postgresql")
        'postgres'
        >>> connector_key_to_dialect("mysql")
        'mysql'
        >>> connector_key_to_dialect("unknown")
        'postgres'

    """
    if not connector_key:
        logger.warning(
            "Empty connector key provided, using default dialect",
            extra={"default_dialect": DEFAULT_DIALECT},
        )
        return DEFAULT_DIALECT

    # Normalize to lowercase for case-insensitive matching
    normalized_key = connector_key.lower().strip()

    # Look up dialect in mapping
    dialect = CONNECTOR_TO_DIALECT.get(normalized_key)

    if dialect:
        logger.debug(
            "Mapped connector key to dialect",
            extra={"connector_key": connector_key, "dialect": dialect},
        )
        return dialect

    # Log warning for unknown connector and return default
    logger.warning(
        "Unknown connector key, using default dialect",
        extra={
            "connector_key": connector_key,
            "default_dialect": DEFAULT_DIALECT,
            "supported_connectors": list(CONNECTOR_TO_DIALECT.keys()),
        },
    )
    return DEFAULT_DIALECT


def get_supported_dialects() -> list[str]:
    """Get list of supported SQL dialects.

    Returns:
        List of supported dialect strings

    """
    return list(set(CONNECTOR_TO_DIALECT.values()))


def get_supported_connector_keys() -> list[str]:
    """Get list of supported connector keys.

    Returns:
        List of supported connector key strings

    """
    return list(CONNECTOR_TO_DIALECT.keys())
