"""Chart Selector for rule-based chart type selection.

This module implements a decision tree for selecting appropriate chart types
based on data patterns and column types.
"""

from typing import Any, Dict, Tuple

from core.logging import get_logger

logger = get_logger(__name__)


class ChartSelector:
    """Rule-based chart type selector using decision tree logic."""

    CARDINALITY_LOW = 20
    CARDINALITY_MEDIUM = 50
    PIE_CHART_MAX_CATEGORIES = 7

    @staticmethod
    def select_chart_type(profile: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Select appropriate chart type based on data profile.

        Args:
            profile: Data profile from DataProfiler

        Returns:
            Tuple of (chart_type, config_dict)
            chart_type can be: "line", "bar_vertical", "bar_horizontal",
            "scatter", "pie", "donut", "heatmap", "treemap", "llm_fallback"

        """
        columns = profile.get("columns", [])
        patterns = profile.get("patterns", [])
        row_count = profile.get("metadata", {}).get("row_count", 0)

        # Extract column types
        temporal_cols = [c for c in columns if c.get("type") == "temporal"]
        categorical_cols = [c for c in columns if c.get("type") == "categorical"]
        quantitative_cols = [c for c in columns if c.get("type") == "quantitative"]

        logger.info(
            "Selecting chart type",
            temporal_count=len(temporal_cols),
            categorical_count=len(categorical_cols),
            quantitative_count=len(quantitative_cols),
            pattern_count=len(patterns),
        )

        # Decision tree logic
        # 1. Time series: temporal + quantitative
        if temporal_cols and quantitative_cols:
            return ChartSelector._select_time_series_chart(temporal_cols, quantitative_cols, patterns)

        # 2. Categorical + quantitative
        if categorical_cols and quantitative_cols:
            return ChartSelector._select_categorical_chart(categorical_cols, quantitative_cols, row_count)

        # 3. Two quantitative columns (correlation/scatter)
        if len(quantitative_cols) >= 2:
            return ChartSelector._select_correlation_chart(quantitative_cols)

        # 4. Proportion analysis (pie chart)
        if len(categorical_cols) == 1 and len(quantitative_cols) == 1:
            cat_col = categorical_cols[0]
            if cat_col.get("cardinality") == "low":
                distinct_count = cat_col.get("distinct_count", 0)
                if distinct_count <= ChartSelector.PIE_CHART_MAX_CATEGORIES:
                    return ChartSelector._select_proportion_chart(cat_col, quantitative_cols[0])

        # 5. Single quantitative (distribution)
        if len(quantitative_cols) == 1 and not temporal_cols and not categorical_cols:
            return (
                "bar_vertical",
                {
                    "x_col": quantitative_cols[0]["name"],
                    "y_col": "count",
                    "description": "Distribution of values",
                    "note": "Consider histogram or box plot",
                },
            )

        # 6. Complex case - fall back to LLM
        logger.info("Complex chart selection, falling back to LLM")
        return (
            "llm_fallback",
            {
                "reason": "Complex data structure requiring intelligent analysis",
                "column_count": len(columns),
                "pattern_count": len(patterns),
            },
        )

    @staticmethod
    def _select_time_series_chart(
        temporal_cols: list[Dict[str, Any]],
        quantitative_cols: list[Dict[str, Any]],
        patterns: list[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Select chart for time series data.

        Args:
            temporal_cols: Temporal columns
            quantitative_cols: Quantitative columns
            patterns: Detected patterns

        Returns:
            Chart type and configuration

        """
        temporal_col = temporal_cols[0]["name"]

        if len(quantitative_cols) == 1:
            # Single metric line chart
            return (
                "line",
                {
                    "x_col": temporal_col,
                    "y_cols": [quantitative_cols[0]["name"]],
                    "description": f"Time series of {quantitative_cols[0]['name']}",
                },
            )
        elif len(quantitative_cols) <= 4:
            # Multi-metric line chart
            return (
                "line",
                {
                    "x_col": temporal_col,
                    "y_cols": [col["name"] for col in quantitative_cols],
                    "description": "Multi-metric time series",
                },
            )
        else:
            # Too many metrics - LLM should decide
            return (
                "llm_fallback",
                {
                    "reason": "Too many metrics for simple line chart",
                    "metric_count": len(quantitative_cols),
                },
            )

    @staticmethod
    def _select_categorical_chart(
        categorical_cols: list[Dict[str, Any]],
        quantitative_cols: list[Dict[str, Any]],
        row_count: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Select chart for categorical data.

        Args:
            categorical_cols: Categorical columns
            quantitative_cols: Quantitative columns
            row_count: Number of rows

        Returns:
            Chart type and configuration

        """
        # Single categorical, single quantitative
        if len(categorical_cols) == 1 and len(quantitative_cols) == 1:
            cat_col = categorical_cols[0]
            quant_col = quantitative_cols[0]
            cardinality = cat_col.get("cardinality", "low")
            distinct_count = cat_col.get("distinct_count", 0)

            if cardinality == "low" or distinct_count < ChartSelector.CARDINALITY_LOW:
                # Vertical bar chart
                return (
                    "bar_vertical",
                    {
                        "x_col": cat_col["name"],
                        "y_col": quant_col["name"],
                        "description": f"{quant_col['name']} by {cat_col['name']}",
                    },
                )
            elif cardinality == "medium" or distinct_count < ChartSelector.CARDINALITY_MEDIUM:
                # Horizontal bar chart (better for labels)
                return (
                    "bar_horizontal",
                    {
                        "x_col": cat_col["name"],
                        "y_col": quant_col["name"],
                        "description": f"{quant_col['name']} by {cat_col['name']}",
                    },
                )
            else:
                # High cardinality - top-N bar chart
                return (
                    "bar_horizontal",
                    {
                        "x_col": cat_col["name"],
                        "y_col": quant_col["name"],
                        "top_n": 20,
                        "description": f"Top 20 {cat_col['name']} by {quant_col['name']}",
                    },
                )

        # Single categorical, multiple quantitative
        elif len(categorical_cols) == 1 and len(quantitative_cols) > 1:
            cat_col = categorical_cols[0]
            cardinality = cat_col.get("cardinality", "low")

            if cardinality == "low" and len(quantitative_cols) <= 4:
                # Grouped bar chart
                return (
                    "grouped_bar",
                    {
                        "x_col": cat_col["name"],
                        "y_cols": [col["name"] for col in quantitative_cols],
                        "description": f"Multi-metric comparison by {cat_col['name']}",
                    },
                )
            else:
                # Complex - LLM fallback
                return (
                    "llm_fallback",
                    {
                        "reason": "High cardinality with multiple metrics",
                        "category_distinct": cat_col.get("distinct_count", 0),
                        "metric_count": len(quantitative_cols),
                    },
                )

        # Multiple categoricals - complex case
        else:
            return (
                "llm_fallback",
                {
                    "reason": "Multiple categorical dimensions",
                    "category_count": len(categorical_cols),
                    "metric_count": len(quantitative_cols),
                },
            )

    @staticmethod
    def _select_correlation_chart(
        quantitative_cols: list[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Select chart for correlation analysis.

        Args:
            quantitative_cols: Quantitative columns

        Returns:
            Chart type and configuration

        """
        # Scatter plot for two quantitative columns
        if len(quantitative_cols) >= 2:
            return (
                "scatter",
                {
                    "x_col": quantitative_cols[0]["name"],
                    "y_col": quantitative_cols[1]["name"],
                    "description": f"Correlation between {quantitative_cols[0]['name']} and {quantitative_cols[1]['name']}",
                    "color_col": quantitative_cols[2]["name"] if len(quantitative_cols) > 2 else None,
                },
            )

        return ("llm_fallback", {"reason": "Insufficient quantitative columns"})

    @staticmethod
    def _select_proportion_chart(
        cat_col: Dict[str, Any],
        quant_col: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Select proportion chart (pie/donut).

        Args:
            cat_col: Categorical column
            quant_col: Quantitative column

        Returns:
            Chart type and configuration

        """
        return (
            "pie",
            {
                "labels_col": cat_col["name"],
                "values_col": quant_col["name"],
                "description": f"Proportion of {quant_col['name']} by {cat_col['name']}",
                "hole": 0.0,  # 0 = pie, 0.4 = donut
            },
        )
