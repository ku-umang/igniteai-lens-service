"""LLM configuration for the application."""

from typing import Optional

from portkey_ai import Portkey

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class LLMNotConfiguredError(Exception):
    """Raised when attempting to use LLM before it's configured."""

    pass


class LLMConfig:
    """Singleton configuration manager for LLM client."""

    def __init__(self):
        self._llm: Optional[Portkey] = None
        self._is_configured: bool = False

    def configure_llm(self) -> None:
        """Configure the global LLM settings."""
        try:
            logger.info("Configuring LLM for Ignite Lens...")
            self._llm = Portkey(
                api_key=settings.PORTKEY_API_KEY,
                virtual_key=settings.PORTKEY_DEFAULT_KEY,
            )
            self._is_configured = True
            logger.info("LLM configured successfully")
        except Exception as e:
            logger.error("Failed to configure LLM", error=str(e))
            self._is_configured = False
            raise

    def get_llm(self) -> Portkey:
        """Get the configured LLM client.

        Returns:
            Portkey: The configured Portkey client instance.

        Raises:
            LLMNotConfiguredError: If LLM hasn't been configured yet.

        """
        if not self._is_configured or self._llm is None:
            raise LLMNotConfiguredError(
                "LLM has not been configured. Call configure_llm() first or check application startup logs."
            )
        return self._llm

    @property
    def is_configured(self) -> bool:
        """Check if LLM is configured and ready to use."""
        return self._is_configured

    @property
    def llm(self) -> Optional[Portkey]:
        """Direct access to LLM client (can be None).

        For safer access, use get_llm() instead which raises an error if not configured.
        """
        return self._llm


llm_config = LLMConfig()
