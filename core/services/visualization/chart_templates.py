"""Chart Templates for Plotly visualizations.

This module provides reusable Plotly chart templates with professional styling
for various chart types used in data visualization.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd


class ChartTemplates:
    """Reusable Plotly chart templates with professional styling."""

    # Professional color palette
    COLORS = [
        "#3366CC",
        "#DC3912",
        "#FF9900",
        "#109618",
        "#990099",
        "#3B3EAC",
        "#0099C6",
        "#DD4477",
        "#66AA00",
        "#B82E2E",
    ]

    LAYOUT_DEFAULTS = {
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
        "hovermode": "closest",
    }

    @staticmethod
    def _to_json_serializable(data: List[Any]) -> List[Any]:
        """Convert data to JSON-serializable format.

        Handles Decimal, datetime, and other non-JSON types.

        Args:
            data: List of values

        Returns:
            JSON-serializable list

        """
        result = []
        for item in data:
            if isinstance(item, Decimal):
                result.append(float(item))
            elif isinstance(item, (pd.Timestamp, pd.Timedelta)):
                result.append(str(item))
            elif pd.isna(item):
                result.append(None)
            else:
                result.append(item)
        return result

    @staticmethod
    def line_chart_template(
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate line chart Plotly specification.

        Args:
            data: DataFrame with chart data
            x_col: X-axis column name
            y_cols: List of Y-axis column names (supports multiple series)
            title: Chart title
            x_label: X-axis label (optional)
            y_label: Y-axis label (optional)

        Returns:
            Plotly JSON specification

        """
        traces = []
        for idx, y_col in enumerate(y_cols):
            trace = {
                "type": "scatter",
                "mode": "lines+markers",
                "name": y_col,
                "x": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                "y": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                "line": {"color": ChartTemplates.COLORS[idx % len(ChartTemplates.COLORS)], "width": 2},
                "marker": {"size": 6},
            }
            traces.append(trace)

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
            "xaxis": {"title": x_label or x_col, "showgrid": True, "gridcolor": "#E5E5E5"},
            "yaxis": {"title": y_label or (y_cols[0] if len(y_cols) == 1 else "Value"), "showgrid": True, "gridcolor": "#E5E5E5"},
            "showlegend": len(y_cols) > 1,
        }

        return {"data": traces, "layout": layout}

    @staticmethod
    def bar_chart_template(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        orientation: str = "v",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate bar chart Plotly specification.

        Args:
            data: DataFrame with chart data
            x_col: X-axis column name (categories)
            y_col: Y-axis column name (values)
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            x_label: X-axis label (optional)
            y_label: Y-axis label (optional)

        Returns:
            Plotly JSON specification

        """
        if orientation == "h":
            # Swap axes for horizontal bar
            trace = {
                "type": "bar",
                "orientation": "h",
                "y": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                "x": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                "marker": {"color": ChartTemplates.COLORS[0]},
            }
            layout = {
                **ChartTemplates.LAYOUT_DEFAULTS,
                "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
                "xaxis": {"title": y_label or y_col, "showgrid": True, "gridcolor": "#E5E5E5"},
                "yaxis": {"title": x_label or x_col},
            }
        else:
            # Vertical bar
            trace = {
                "type": "bar",
                "x": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                "y": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                "marker": {"color": ChartTemplates.COLORS[0]},
            }
            layout = {
                **ChartTemplates.LAYOUT_DEFAULTS,
                "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
                "xaxis": {"title": x_label or x_col},
                "yaxis": {"title": y_label or y_col, "showgrid": True, "gridcolor": "#E5E5E5"},
            }

        return {"data": [trace], "layout": layout}

    @staticmethod
    def grouped_bar_chart_template(
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str,
        orientation: str = "v",
    ) -> Dict[str, Any]:
        """Generate grouped bar chart Plotly specification.

        Args:
            data: DataFrame with chart data
            x_col: X-axis column name (categories)
            y_cols: List of Y-axis column names (multiple series)
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal

        Returns:
            Plotly JSON specification

        """
        traces = []
        for idx, y_col in enumerate(y_cols):
            if orientation == "h":
                trace = {
                    "type": "bar",
                    "orientation": "h",
                    "name": y_col,
                    "y": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                    "x": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                    "marker": {"color": ChartTemplates.COLORS[idx % len(ChartTemplates.COLORS)]},
                }
            else:
                trace = {
                    "type": "bar",
                    "name": y_col,
                    "x": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                    "y": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                    "marker": {"color": ChartTemplates.COLORS[idx % len(ChartTemplates.COLORS)]},
                }
            traces.append(trace)

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
            "barmode": "group",
            "showlegend": True,
        }

        if orientation == "h":
            layout["xaxis"] = {"title": "Value", "showgrid": True, "gridcolor": "#E5E5E5"}
            layout["yaxis"] = {"title": x_col}
        else:
            layout["xaxis"] = {"title": x_col}
            layout["yaxis"] = {"title": "Value", "showgrid": True, "gridcolor": "#E5E5E5"}

        return {"data": traces, "layout": layout}

    @staticmethod
    def scatter_plot_template(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        color_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate scatter plot Plotly specification.

        Args:
            data: DataFrame with chart data
            x_col: X-axis column name
            y_col: Y-axis column name
            title: Chart title
            x_label: X-axis label (optional)
            y_label: Y-axis label (optional)
            color_col: Column for color grouping (optional)

        Returns:
            Plotly JSON specification

        """
        if color_col and color_col in data.columns:
            # Grouped scatter by color
            traces = []
            for idx, group_val in enumerate(data[color_col].unique()):
                group_data = data[data[color_col] == group_val]
                trace = {
                    "type": "scatter",
                    "mode": "markers",
                    "name": str(group_val),
                    "x": ChartTemplates._to_json_serializable(group_data[x_col].tolist()),
                    "y": ChartTemplates._to_json_serializable(group_data[y_col].tolist()),
                    "marker": {"size": 8, "color": ChartTemplates.COLORS[idx % len(ChartTemplates.COLORS)]},
                }
                traces.append(trace)
            showlegend = True
        else:
            # Simple scatter
            traces = [
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": ChartTemplates._to_json_serializable(data[x_col].tolist()),
                    "y": ChartTemplates._to_json_serializable(data[y_col].tolist()),
                    "marker": {"size": 8, "color": ChartTemplates.COLORS[0]},
                }
            ]
            showlegend = False

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
            "xaxis": {"title": x_label or x_col, "showgrid": True, "gridcolor": "#E5E5E5"},
            "yaxis": {"title": y_label or y_col, "showgrid": True, "gridcolor": "#E5E5E5"},
            "showlegend": showlegend,
        }

        return {"data": traces, "layout": layout}

    @staticmethod
    def pie_chart_template(
        data: pd.DataFrame,
        labels_col: str,
        values_col: str,
        title: str,
        hole: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate pie/donut chart Plotly specification.

        Args:
            data: DataFrame with chart data
            labels_col: Column for pie slice labels
            values_col: Column for pie slice values
            title: Chart title
            hole: Hole size for donut chart (0.0 = pie, 0.4 = donut)

        Returns:
            Plotly JSON specification

        """
        trace = {
            "type": "pie",
            "labels": ChartTemplates._to_json_serializable(data[labels_col].tolist()),
            "values": ChartTemplates._to_json_serializable(data[values_col].tolist()),
            "hole": hole,
            "marker": {"colors": ChartTemplates.COLORS},
            "textinfo": "label+percent",
            "textposition": "auto",
        }

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
            "showlegend": True,
        }

        return {"data": [trace], "layout": layout}

    @staticmethod
    def heatmap_template(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str,
        title: str,
    ) -> Dict[str, Any]:
        """Generate heatmap Plotly specification.

        Args:
            data: DataFrame with chart data
            x_col: X-axis column name
            y_col: Y-axis column name
            value_col: Value column for heatmap intensity
            title: Chart title

        Returns:
            Plotly JSON specification

        """
        # Pivot data for heatmap
        pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)

        # Convert each row of the heatmap matrix
        z_data = []
        for row in pivot_data.values:
            z_data.append(ChartTemplates._to_json_serializable(row.tolist()))

        trace = {
            "type": "heatmap",
            "z": z_data,
            "x": ChartTemplates._to_json_serializable(pivot_data.columns.tolist()),
            "y": ChartTemplates._to_json_serializable(pivot_data.index.tolist()),
            "colorscale": "Blues",
            "showscale": True,
        }

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
            "xaxis": {"title": x_col},
            "yaxis": {"title": y_col},
        }

        return {"data": [trace], "layout": layout}

    @staticmethod
    def treemap_template(
        data: pd.DataFrame,
        labels_col: str,
        parents_col: str,
        values_col: str,
        title: str,
    ) -> Dict[str, Any]:
        """Generate treemap Plotly specification.

        Args:
            data: DataFrame with hierarchical data
            labels_col: Column for node labels
            parents_col: Column for parent nodes
            values_col: Column for node sizes
            title: Chart title

        Returns:
            Plotly JSON specification

        """
        trace = {
            "type": "treemap",
            "labels": ChartTemplates._to_json_serializable(data[labels_col].tolist()),
            "parents": ChartTemplates._to_json_serializable(data[parents_col].tolist()),
            "values": ChartTemplates._to_json_serializable(data[values_col].tolist()),
            "marker": {"colors": ChartTemplates.COLORS},
        }

        layout = {
            **ChartTemplates.LAYOUT_DEFAULTS,
            "title": {"text": title, "font": {"size": 16, "weight": "bold"}},
        }

        return {"data": [trace], "layout": layout}
