"""LLM configuration for the application."""

from typing import Optional

from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class LLMNotConfiguredError(Exception):
    """Raised when attempting to use LLM before it's configured."""

    pass


class LLMConfig:
    """Singleton configuration manager for LLM client using LangChain with Portkey AI."""

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._is_configured: bool = False

    def configure_llm(self) -> None:
        """Configure the global LLM settings using LangChain with Portkey AI gateway."""
        try:
            logger.info("Configuring LLM with LangChain and Portkey AI...")

            headers = createHeaders(api_key=settings.PORTKEY_API_KEY, virtual_key=settings.PORTKEY_VIRTUAL_KEY)
            self._llm = ChatOpenAI(
                api_key="X-API-KEY",  # type: ignore[arg-type]
                base_url=PORTKEY_GATEWAY_URL,
                default_headers=headers,
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
            )

            self._is_configured = True

            logger.info("LLM configured successfully with LangChain and Portkey AI")
        except Exception as e:
            logger.error("Failed to configure LLM", error=str(e))
            self._is_configured = False
            raise

    def get_llm(self) -> ChatOpenAI:
        """Get the configured LLM client.

        Returns:
            ChatOpenAI: The configured LangChain ChatOpenAI client instance with Portkey AI.

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
    def llm(self) -> Optional[ChatOpenAI]:
        """Direct access to LLM client (can be None).

        For safer access, use get_llm() instead which raises an error if not configured.
        """
        return self._llm


llm_config = LLMConfig()
