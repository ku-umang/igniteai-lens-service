"""LLM-powered Visualization Agent for complex chart decisions.

This module uses LLM to make intelligent chart type selections
for complex or ambiguous data scenarios.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class VisualizationAgent:
    """LLM-powered agent for intelligent visualization decisions."""

    SYSTEM_PROMPT = """You are an expert data visualization consultant. Your task is to analyze data profiles
and recommend the most appropriate Plotly chart type and configuration.

Given a data profile with column types, statistics, and patterns, you should:
1. Analyze the data structure and patterns
2. Select the appropriate chart type: line, bar_vertical, bar_horizontal, grouped_bar, scatter, pie, donut, heatmap, treemap
3. Provide column mappings for the chart
4. Give a clear, concise description
5. Explain your reasoning

Return your response as valid JSON with this structure:
{
    "chart_type": "line|bar_vertical|bar_horizontal|grouped_bar|scatter|pie|donut|heatmap|treemap",
    "config": {
        "x_col": "column_name",
        "y_col": "column_name" or "y_cols": ["col1", "col2"],
        "description": "Brief description of what the chart shows",
        "additional_params": {}
    },
    "reasoning": "Why this chart type was selected"
}

Focus on clarity, simplicity, and data-driven insights. Avoid overly complex visualizations."""

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
                user_prompt = self._build_prompt(profile, sql, question)

                # Call LLM
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
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

    def _build_prompt(
        self,
        profile: Dict[str, Any],
        sql: str,
        question: str,
    ) -> str:
        """Build prompt for LLM with data profile.

        Args:
            profile: Data profile
            sql: SQL query
            question: User question

        Returns:
            Formatted prompt string

        """
        # Simplify profile for LLM (remove verbose data)
        simplified_profile = {
            "metadata": profile.get("metadata", {}),
            "columns": [
                {
                    "name": c.get("name"),
                    "type": c.get("type"),
                    "cardinality": c.get("cardinality"),
                    "distinct_count": c.get("distinct_count"),
                }
                for c in profile.get("columns", [])
            ],
            "patterns": profile.get("patterns", []),
        }

        prompt = f"""Analyze this data and recommend a visualization:

**User Question:** {question}

**SQL Query:** {sql[:300]}...

**Data Profile:**
```json
{json.dumps(simplified_profile, indent=2)}
```

**Instructions:**
- Select the chart type that best reveals insights
- Provide clear column mappings
- Explain your reasoning briefly

Return valid JSON only."""

        return prompt

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
