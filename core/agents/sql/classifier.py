import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.classifier import (
    CLASSIFIER_SYSTEM_PROMPT,
    format_classifier_prompt,
)
from core.agents.sql.state import AgentState, ClassificationResult, QuestionType
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ClassifierAgent:
    """Classifier agent for question type classification in workflow.

    This agent:
    1. Analyzes the user's question (optimized version if available)
    2. Classifies it into one of 8 analytical types
    3. Provides confidence score and reasoning
    4. Identifies key characteristics that guided classification
    """

    def __init__(self) -> None:
        """Initialize the Classifier agent."""
        self.llm = llm_config.get_llm()

    async def classify_question(self, state: AgentState) -> Dict[str, Any]:
        """Classify the user's question into an analytical type.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with classification populated

        """
        with tracer.start_as_current_span(
            "classifier_agent.classify_question",
            attributes={
                "question": state.optimized_question or state.user_question,
                "tenant_id": str(state.tenant_id),
            },
        ) as span:
            start_time = time.time()

            try:
                # Use optimized question if available, otherwise original
                question_to_classify = state.optimized_question or state.user_question

                logger.info(
                    "Classifier agent starting",
                    extra={
                        "question": question_to_classify,
                        "has_optimized": state.optimized_question is not None,
                    },
                )

                # Use LLM to classify the question
                classification_result = await self._llm_classify_question(
                    question=question_to_classify,
                )

                classification_time = (time.time() - start_time) * 1000

                # Create ClassificationResult object
                classification = ClassificationResult(
                    question_type=QuestionType(classification_result["question_type"]),
                    confidence=classification_result["confidence"],
                    reasoning=classification_result.get("reasoning", ""),
                    characteristics=classification_result.get("characteristics", []),
                )

                span.set_attribute("question_type", classification.question_type.value)
                span.set_attribute("confidence", classification.confidence)

                logger.info(
                    "Classifier agent completed",
                    extra={
                        "question": question_to_classify,
                        "question_type": classification.question_type.value,
                        "confidence": classification.confidence,
                        "reasoning": classification.reasoning,
                        "classification_time_ms": classification_time,
                    },
                )

                return {
                    "classification": classification,
                    "total_time_ms": state.total_time_ms + classification_time,
                    "llm_calls": state.llm_calls + 1,
                    "current_step": "planner",
                }

            except Exception as e:
                logger.error(
                    "Classifier agent failed",
                    extra={
                        "error": str(e),
                        "question": state.optimized_question or state.user_question,
                    },
                )
                # On error, default to descriptive with low confidence
                fallback_classification = ClassificationResult(
                    question_type=QuestionType.DESCRIPTIVE,
                    confidence=0.5,
                    reasoning=f"Classification failed: {str(e)}. Defaulting to descriptive.",
                    characteristics=["fallback"],
                )

                return {
                    "classification": fallback_classification,
                    "errors": [f"Classifier agent error: {str(e)}"],
                    "current_step": "planner",
                }

    async def _llm_classify_question(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """Use LLM to classify the question into an analytical type.

        Args:
            question: User's question (optimized if available)

        Returns:
            Classification result with question_type, confidence, reasoning, and characteristics

        """
        # Format the prompt
        user_prompt = format_classifier_prompt(question=question)

        # Call LLM
        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
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

            classification = json.loads(content)

            # Validate required fields
            if "question_type" not in classification or "confidence" not in classification:
                raise ValueError("Missing required fields in classification response")

            # Validate question_type is one of our enum values
            valid_types = [qt.value for qt in QuestionType]
            if classification["question_type"] not in valid_types:
                raise ValueError(f"Invalid question_type: {classification['question_type']}")

            # Ensure confidence is in valid range
            classification["confidence"] = max(0.0, min(1.0, float(classification["confidence"])))

            return classification

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            logger.error(
                "Failed to parse LLM classification response",
                extra={"error": str(e), "response": str(response)},
            )
            # Fallback: return descriptive with low confidence
            return {
                "question_type": "descriptive",
                "confidence": 0.5,
                "reasoning": f"Failed to parse LLM response: {str(e)}. Defaulting to descriptive type.",
                "characteristics": ["parsing_error"],
            }
