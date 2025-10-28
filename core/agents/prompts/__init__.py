"""Prompts for MAC-SQL agents."""

from core.agents.prompts.classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_PROMPT_TEMPLATE,
    format_classifier_prompt,
)
from core.agents.prompts.optimizer import (
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_PROMPT_TEMPLATE,
    format_optimizer_prompt,
)
from core.agents.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE, format_planner_prompt
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
    "CLASSIFIER_SYSTEM_PROMPT",
    "CLASSIFIER_USER_PROMPT_TEMPLATE",
    "format_classifier_prompt",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_USER_PROMPT_TEMPLATE",
    "format_planner_prompt",
    "SELECTOR_SYSTEM_PROMPT",
    "SELECTOR_USER_PROMPT_TEMPLATE",
    "format_selector_prompt",
    "OPTIMIZER_SYSTEM_PROMPT",
    "OPTIMIZER_USER_PROMPT_TEMPLATE",
    "format_optimizer_prompt",
    "REFINER_SYSTEM_PROMPT",
    "REFINER_USER_PROMPT_TEMPLATE",
    "format_refiner_prompt",
]
