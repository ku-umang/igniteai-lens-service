import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.optimizer import (
    OPTIMIZER_SYSTEM_PROMPT,
    format_optimizer_prompt,
)
from core.agents.state import AgentState
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class OptimizerAgent:
    """Optimizer agent for question optimization in workflow.

    This agent:
    1. Analyzes the user's current question
    2. Reviews recent conversation history
    3. Rewrites the question to be more explicit and context-aware
    4. Resolves ambiguous references using history
    """

    def __init__(self) -> None:
        """Initialize the Optimizer agent."""
        self.llm = llm_config.get_llm()

    async def optimize_question(self, state: AgentState) -> AgentState:
        """Optimize the user's question using conversation history.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with optimized_question populated

        """
        with tracer.start_as_current_span(
            "optimizer_agent.optimize_question",
            attributes={
                "original_question": state.user_question,
                "history_length": len(state.chat_history),
                "tenant_id": str(state.tenant_id),
            },
        ) as span:
            start_time = time.time()

            try:
                logger.info(
                    "Optimizer agent starting",
                    extra={
                        "original_question": state.user_question,
                        "history_count": len(state.chat_history),
                    },
                )

                # If no history, skip optimization
                if not state.chat_history:
                    logger.info("No chat history, skipping optimization")
                    optimization_time = (time.time() - start_time) * 1000

                    state.optimized_question = state.user_question
                    state.optimization_reasoning = "No conversation history available"
                    state.total_time_ms = state.total_time_ms + optimization_time
                    state.llm_calls = state.llm_calls
                    state.current_step = "query_optimizer"
                    return state

                # Use LLM to optimize the question
                optimization_result = await self._llm_optimize_question(
                    current_question=state.user_question,
                    chat_history=state.chat_history,
                )

                optimization_time = (time.time() - start_time) * 1000

                span.set_attribute("optimized_question", optimization_result["optimized_question"])

                logger.info(
                    "Optimizer agent completed",
                    extra={
                        "original_question": state.user_question,
                        "optimized_question": optimization_result["optimized_question"],
                        "optimization_time_ms": optimization_time,
                    },
                )

                state.optimized_question = optimization_result["optimized_question"]
                state.optimization_reasoning = optimization_result["reasoning"]
                state.total_time_ms = state.total_time_ms + optimization_time
                state.llm_calls = state.llm_calls + 1
                state.current_step = "query_optimizer"
                return state

            except Exception as e:
                logger.error(
                    "Optimizer agent failed",
                    extra={
                        "error": str(e),
                        "question": state.user_question,
                    },
                )
                # On error, proceed with original question
                state.optimized_question = state.user_question
                state.optimization_reasoning = f"Optimization failed: {str(e)}"
                state.errors = [f"Optimizer agent error: {str(e)}"]
                state.current_step = "query_optimizer"
                return state

    async def _llm_optimize_question(
        self,
        current_question: str,
        chat_history: list,
    ) -> Dict[str, Any]:
        """Use LLM to optimize the question based on conversation history.

        Args:
            current_question: User's current question
            chat_history: List of recent message objects

        Returns:
            Optimization result with optimized question and reasoning

        """
        # Format the prompt
        user_prompt = format_optimizer_prompt(
            current_question=current_question,
            chat_history=chat_history,
        )

        # Call LLM
        messages = [
            SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content if isinstance(response.content, str) else str(response.content)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Fix improperly escaped single quotes in JSON strings (\' -> ')
            content = content.replace(r"\'", "'")

            optimization = json.loads(content)
            return optimization

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(
                "Failed to parse LLM response",
                extra={"error": str(e), "response": str(response)},
            )
            # Fallback: return original question
            return {
                "optimized_question": current_question,
                "reasoning": f"Failed to parse LLM response: {str(e)}",
            }
