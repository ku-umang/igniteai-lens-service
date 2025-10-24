"""Prompts for MAC-SQL agents."""

from core.agents.prompts.decomposer import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_USER_PROMPT_TEMPLATE,
    format_decomposer_prompt,
)
from core.agents.prompts.refiner import (
    REFINER_SYSTEM_PROMPT,
    REFINER_USER_PROMPT_TEMPLATE,
    format_refiner_prompt,
)
from core.agents.prompts.selector import (
    SELECTOR_SYSTEM_PROMPT,
    SELECTOR_USER_PROMPT_TEMPLATE,
    format_selector_prompt,
)

__all__ = [
    "SELECTOR_SYSTEM_PROMPT",
    "SELECTOR_USER_PROMPT_TEMPLATE",
    "format_selector_prompt",
    "DECOMPOSER_SYSTEM_PROMPT",
    "DECOMPOSER_USER_PROMPT_TEMPLATE",
    "format_decomposer_prompt",
    "REFINER_SYSTEM_PROMPT",
    "REFINER_USER_PROMPT_TEMPLATE",
    "format_refiner_prompt",
]
