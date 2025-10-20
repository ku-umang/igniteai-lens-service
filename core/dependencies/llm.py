"""LLM dependency for FastAPI route injection."""

from portkey_ai import Portkey

from core.llm_config import llm_config


def get_llm() -> Portkey:
    """FastAPI dependency to get the configured LLM client.

    Usage in routes:
        @router.post("/generate")
        async def generate_text(llm: Annotated[Portkey, Depends(get_llm)]):
            response = llm.chat.completions.create(...)
            return response

    Returns:
        Portkey: The configured Portkey client instance.

    Raises:
        LLMNotConfiguredError: If LLM hasn't been configured.

    """
    return llm_config.get_llm()
