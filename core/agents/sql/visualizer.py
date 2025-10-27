"""LLM-powered Visualization Agent for complex chart decisions.

This module uses LLM to make intelligent chart type selections
for complex or ambiguous data scenarios.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.visualizer import VISUALIZER_SYSTEM_PROMPT, format_visualizer_prompt
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class VisualizationAgent:
    """LLM-powered agent for intelligent visualization decisions."""

    def __init__(self) -> None:
        """Initialize the visualization agent."""
        self.llm = llm_config.get_llm()

    async def generate_chart_spec(
        self,
        profile: Dict[str, Any],
        sql: str,
        question: str,
    ) -> Dict[str, Any]:
        """Generate chart specification using LLM.

        Args:
            profile: Data profile from DataProfiler
            sql: Original SQL query
            question: User's natural language question

        Returns:
            Chart specification dict with chart_type, config, and reasoning

        """
        with tracer.start_as_current_span("visualization_agent.generate_chart_spec") as span:
            try:
                # Build prompt with data profile
                user_prompt = format_visualizer_prompt(profile, sql, question)

                # Call LLM
                messages = [
                    SystemMessage(content=VISUALIZER_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]

                response = await self.llm.ainvoke(messages)

                span.set_attribute("llm_response_length", len(response.content))

                # Parse response
                chart_spec = self._parse_response(str(response.content))

                logger.info(
                    "LLM chart specification generated",
                    chart_type=chart_spec.get("chart_type"),
                )

                return chart_spec

            except Exception as e:
                logger.error("LLM chart generation failed", error=str(e))
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))

                # Return fallback specification
                return self._get_fallback_spec(profile)

    def _parse_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response and extract chart specification.

        Args:
            response_content: LLM response text

        Returns:
            Parsed chart specification

        """
        try:
            # Try to extract JSON from response
            # LLM might wrap JSON in markdown code blocks
            content = response_content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]

            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

            # Parse JSON
            spec = json.loads(content)

            # Validate required fields
            if "chart_type" not in spec:
                raise ValueError("Missing chart_type in response")
            if "config" not in spec:
                raise ValueError("Missing config in response")

            return spec

        except Exception as e:
            logger.error("Failed to parse LLM response", error=str(e), response=response_content[:200])
            raise ValueError(f"Invalid LLM response format: {e}") from e

    def _get_fallback_spec(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback chart specification if LLM fails.

        Args:
            profile: Data profile

        Returns:
            Basic chart specification

        """
        columns = profile.get("columns", [])
        if not columns:
            return {
                "chart_type": "bar_vertical",
                "config": {
                    "description": "Unable to generate chart specification",
                },
                "reasoning": "Fallback: LLM generation failed",
            }

        # Simple fallback: first two columns
        col_names = [c["name"] for c in columns]

        return {
            "chart_type": "bar_vertical",
            "config": {
                "x_col": col_names[0] if len(col_names) > 0 else None,
                "y_col": col_names[1] if len(col_names) > 1 else col_names[0],
                "description": "Basic chart (fallback)",
            },
            "reasoning": "Fallback: LLM generation failed, using default bar chart",
        }
