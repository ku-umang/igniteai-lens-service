import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.analyzer import (
    ANALYZER_SYSTEM_PROMPT,
    format_analyzer_prompt,
)
from core.agents.sql.state import AgentState, AnalysisResult
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AnalysisAgent:
    """Analysis agent for synthesizing insights from multiple query results.

    This agent:
    1. Receives one or more query execution results
    2. Analyzes the data based on question classification
    3. Synthesizes insights and generates a comprehensive answer
    4. Provides supporting evidence and recommendations
    5. Acknowledges limitations and confidence level
    """

    def __init__(self) -> None:
        """Initialize the Analysis agent."""
        self.llm = llm_config.get_llm()

    async def analyze_results(self, state: AgentState) -> Dict[str, Any]:
        """Analyze query results and generate insights.

        Args:
            state: Current workflow state with query results

        Returns:
            Updated state dict with analysis_result populated

        """
        with tracer.start_as_current_span(
            "analysis_agent.analyze_results",
            attributes={
                "question": state.optimized_question or state.user_question,
                "classification": state.classification.question_type.value if state.classification else "none",
                "num_results": len(state.query_results),
            },
        ) as span:
            start_time = time.time()

            try:
                question_to_use = state.optimized_question or state.user_question

                if not state.classification:
                    raise ValueError("Classification not available")

                if not state.query_results:
                    raise ValueError("No query results available for analysis")

                logger.info(
                    "Analysis agent starting",
                    extra={
                        "question": question_to_use,
                        "classification": state.classification.question_type.value,
                        "num_results": len(state.query_results),
                    },
                )

                # Prepare query results for analysis
                results_for_analysis = [
                    {
                        "success": result.success,
                        "rows_returned": result.rows_returned,
                        "error_message": result.error_message,
                        "rows": result.rows,
                        "execution_time_ms": result.execution_time_ms,
                        "cached": result.cached,
                    }
                    for result in state.query_results
                ]

                # Get execution plan strategy
                plan_strategy = ""
                if state.execution_plan:
                    plan_strategy = state.execution_plan.strategy or "Multi-step analysis"

                # Use LLM to analyze results
                analysis_dict = await self._llm_analyze_results(
                    question=question_to_use,
                    classification_type=state.classification.question_type.value,
                    plan_strategy=plan_strategy,
                    query_results=results_for_analysis,
                )

                # Create AnalysisResult object
                analysis = AnalysisResult(
                    answer=analysis_dict["answer"],
                    insights=analysis_dict.get("insights", []),
                    supporting_evidence=analysis_dict.get("supporting_evidence", []),
                    confidence=analysis_dict.get("confidence", 0.8),
                    recommendations=analysis_dict.get("recommendations", []),
                    limitations=analysis_dict.get("limitations", []),
                    reasoning=analysis_dict.get("reasoning", ""),
                )

                analysis_time = (time.time() - start_time) * 1000

                span.set_attribute("confidence", analysis.confidence)
                span.set_attribute("num_insights", len(analysis.insights))

                logger.info(
                    "Analysis agent completed",
                    extra={
                        "question": question_to_use,
                        "num_insights": len(analysis.insights),
                        "confidence": analysis.confidence,
                        "analysis_time_ms": analysis_time,
                    },
                )

                return {
                    "analysis_result": analysis,
                    "total_time_ms": state.total_time_ms + analysis_time,
                    "llm_calls": state.llm_calls + 1,
                    "current_step": "visualizer",
                }

            except Exception as e:
                logger.error(
                    "Analysis agent failed",
                    extra={
                        "error": str(e),
                        "question": state.optimized_question or state.user_question,
                    },
                )

                # On error, create basic fallback analysis
                fallback_analysis = AnalysisResult(
                    answer=f"Analysis could not be completed due to an error: {str(e)}",
                    insights=["Error occurred during analysis"],
                    supporting_evidence=[],
                    confidence=0.0,
                    recommendations=[],
                    limitations=["Analysis failed - using fallback"],
                    reasoning=f"Analysis failed: {str(e)}",
                )

                return {
                    "analysis_result": fallback_analysis,
                    "errors": [f"Analysis agent error: {str(e)}"],
                    "current_step": "visualizer",
                }

    async def _llm_analyze_results(
        self,
        question: str,
        classification_type: str,
        plan_strategy: str,
        query_results: list,
    ) -> Dict[str, Any]:
        """Use LLM to analyze query results and generate insights.

        Args:
            question: User's original question
            classification_type: Classified question type
            plan_strategy: High-level strategy from execution plan
            query_results: List of execution results

        Returns:
            Analysis dict with answer, insights, evidence, etc.

        """
        # Format the prompt
        user_prompt = format_analyzer_prompt(
            question=question,
            classification_type=classification_type,
            plan_strategy=plan_strategy,
            query_results=query_results,
        )

        # Call LLM
        messages = [
            SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
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

            # Fix improperly escaped single quotes
            content = content.replace(r"\'", "'")

            analysis = json.loads(content)

            # Validate required fields
            if "answer" not in analysis:
                raise ValueError("Missing 'answer' in analysis response")

            # Ensure confidence is in valid range
            if "confidence" in analysis:
                analysis["confidence"] = max(0.0, min(1.0, float(analysis["confidence"])))
            else:
                analysis["confidence"] = 0.8  # Default confidence

            return analysis

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            logger.error(
                "Failed to parse LLM analysis response",
                extra={"error": str(e), "response": str(response)},
            )

            # Fallback: create basic summary from data
            total_rows = sum(r.get("rows_returned", 0) for r in query_results if r.get("success"))

            return {
                "answer": f"Analysis parsing failed. Retrieved {total_rows} rows total from {len(query_results)} queries.",
                "insights": [
                    f"Query {i + 1} returned {r.get('rows_returned', 0)} rows"
                    for i, r in enumerate(query_results)
                    if r.get("success")
                ],
                "supporting_evidence": [],
                "confidence": 0.3,
                "recommendations": [],
                "limitations": [f"Failed to parse LLM analysis response: {str(e)}"],
                "reasoning": "Using fallback analysis due to parsing error",
            }
