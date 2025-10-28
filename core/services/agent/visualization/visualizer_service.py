"""Visualization Service for automatic chart generation.

This module orchestrates data profiling, chart selection, aggregation,
and Plotly specification generation for query results.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from opentelemetry import trace

from core.agents.state import ExecutionResult
from core.agents.steps import VisualizationAgent
from core.logging import get_logger
from core.services.agent.visualization.aggregation_engine import AggregationEngine
from core.services.agent.visualization.chart_selector import ChartSelector
from core.services.agent.visualization.chart_templates import ChartTemplates
from core.services.agent.visualization.data_profiler import DataProfiler
from core.services.agent.visualization.spec_validator import SpecValidator

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class VisualizationService:
    """Main service for generating chart visualizations from query results."""

    def __init__(self) -> None:
        """Initialize the visualization service."""
        self.profiler = DataProfiler()
        self.selector = ChartSelector()
        self.aggregator = AggregationEngine()
        self.templates = ChartTemplates()
        self.validator = SpecValidator()
        self.agent = VisualizationAgent()

    async def generate_visualization(
        self,
        execution_result: ExecutionResult,
        sql: str,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate visualization specification from query execution results.

        Args:
            execution_result: Query execution result with data
            sql: Original SQL query
            question: User's natural language question

        Returns:
            Visualization specification dict or None if generation fails

        """
        with tracer.start_as_current_span("visualization_service.generate") as span:
            start_time = time.time()

            try:
                # Check if data exists and is non-empty
                if not execution_result.success or not execution_result.rows:
                    logger.info("No data to visualize", success=execution_result.success)
                    return None

                data = execution_result.rows
                span.set_attribute("row_count", len(data))
                span.set_attribute("column_count", len(data[0].keys()) if data else 0)

                # Step 1: Profile the data
                logger.info("Profiling data for visualization")
                profile = self.profiler.profile_dataframe(data, sql, question)

                # Step 2: Aggregate data if needed
                if len(data) > AggregationEngine.MAX_POINTS:
                    logger.info("Aggregating large dataset", original_size=len(data))
                    aggregated_data = self.aggregator.aggregate(data, profile)
                else:
                    aggregated_data = data

                # Step 3: Select chart type (rule-based)
                logger.info("Selecting chart type")
                chart_type, config = self.selector.select_chart_type(profile)

                span.set_attribute("chart_type", chart_type)

                # Step 4: If complex, use LLM agent
                generation_method = "rule_based"
                if chart_type == "llm_fallback":
                    logger.info("Using LLM for complex chart selection")
                    llm_spec = await self.agent.generate_chart_spec(profile, sql, question)
                    chart_type = llm_spec.get("chart_type", "bar_vertical")
                    config = llm_spec.get("config", {})
                    generation_method = "llm"
                    span.set_attribute("llm_used", True)

                # Step 5: Generate Plotly spec using templates
                logger.info("Generating Plotly specification", chart_type=chart_type)
                plotly_spec = self._generate_plotly_spec(
                    chart_type,
                    config,
                    aggregated_data,
                    question,
                )

                if not plotly_spec:
                    logger.warning("Failed to generate Plotly spec")
                    return None

                # Step 6: Validate specification
                is_valid, errors = self.validator.validate(plotly_spec)
                if not is_valid:
                    logger.warning("Invalid Plotly spec generated", errors=errors)
                    # Return anyway - frontend can handle gracefully
                    # return None

                # Step 7: Build final visualization spec
                generation_time_ms = (time.time() - start_time) * 1000
                span.set_attribute("generation_time_ms", generation_time_ms)

                viz_spec = {
                    "chart_type": chart_type,
                    "plotly_spec": plotly_spec,
                    "title": config.get("description", question[:100]),
                    "description": config.get("description"),
                    "generation_method": generation_method,
                    "generation_time_ms": generation_time_ms,
                }

                logger.info(
                    "Visualization generated successfully",
                    chart_type=chart_type,
                    generation_method=generation_method,
                    generation_time_ms=f"{generation_time_ms:.2f}",
                )

                return viz_spec

            except Exception as e:
                logger.error("Visualization generation failed", error=str(e), exc_info=True)
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                # Don't fail the workflow - return None gracefully
                return None

    def _generate_plotly_spec(
        self,
        chart_type: str,
        config: Dict[str, Any],
        data: List[Dict[str, Any]],
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate Plotly JSON specification from chart type and config.

        Args:
            chart_type: Type of chart
            config: Chart configuration
            data: Data for visualization
            question: User question for title

        Returns:
            Plotly spec dict or None

        """
        try:
            df = pd.DataFrame(data)

            # Get title from config or use question
            title = config.get("description", question[:100])

            # Generate spec based on chart type
            if chart_type == "line":
                x_col = config.get("x_col")
                y_cols = config.get("y_cols", [])
                if not x_col or not y_cols:
                    return None
                return self.templates.line_chart_template(df, x_col, y_cols, title)

            elif chart_type == "bar_vertical":
                x_col = config.get("x_col")
                y_col = config.get("y_col")
                if not x_col or not y_col:
                    return None
                return self.templates.bar_chart_template(df, x_col, y_col, title, orientation="v")

            elif chart_type == "bar_horizontal":
                x_col = config.get("x_col")
                y_col = config.get("y_col")
                if not x_col or not y_col:
                    return None

                # Handle top_n filtering
                top_n = config.get("top_n")
                if top_n and len(df) > top_n:
                    df = df.nlargest(top_n, y_col)

                return self.templates.bar_chart_template(df, x_col, y_col, title, orientation="h")

            elif chart_type == "grouped_bar":
                x_col = config.get("x_col")
                y_cols = config.get("y_cols", [])
                if not x_col or not y_cols:
                    return None
                return self.templates.grouped_bar_chart_template(df, x_col, y_cols, title)

            elif chart_type == "scatter":
                x_col = config.get("x_col")
                y_col = config.get("y_col")
                color_col = config.get("color_col")
                if not x_col or not y_col:
                    return None
                return self.templates.scatter_plot_template(df, x_col, y_col, title, color_col=color_col)

            elif chart_type in ["pie", "donut"]:
                labels_col = config.get("labels_col")
                values_col = config.get("values_col")
                hole = config.get("hole", 0.0 if chart_type == "pie" else 0.4)
                if not labels_col or not values_col:
                    return None
                return self.templates.pie_chart_template(df, labels_col, values_col, title, hole=hole)

            elif chart_type == "heatmap":
                x_col = config.get("x_col")
                y_col = config.get("y_col")
                value_col = config.get("value_col")
                if not x_col or not y_col or not value_col:
                    return None
                return self.templates.heatmap_template(df, x_col, y_col, value_col, title)

            elif chart_type == "treemap":
                labels_col = config.get("labels_col")
                parents_col = config.get("parents_col", "")
                values_col = config.get("values_col")
                if not labels_col or not values_col:
                    return None
                return self.templates.treemap_template(df, labels_col, parents_col, values_col, title)

            else:
                logger.warning(f"Unknown chart type: {chart_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate {chart_type} spec", error=str(e))
            return None
