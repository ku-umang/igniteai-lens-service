"""Data Profiler for analyzing query results and generating statistical summaries.

This module analyzes DataFrames and generates compact statistical profiles
that include column types, distributions, and patterns for visualization decisions.
"""

from typing import Any, Dict, List

import pandas as pd
from opentelemetry import trace

from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class DataProfiler:
    """Generates statistical profiles of query results for visualization decisions."""

    CARDINALITY_LOW_THRESHOLD = 20
    CARDINALITY_MEDIUM_THRESHOLD = 50
    TOP_VALUES_LIMIT = 10

    @staticmethod
    def profile_dataframe(
        data: List[Dict[str, Any]],
        sql: str,
        question: str,
    ) -> Dict[str, Any]:
        """Generate a compact statistical profile from query results.

        Args:
            data: Query results as list of dictionaries
            sql: Original SQL query
            question: User's natural language question

        Returns:
            Compact JSON profile with metadata, column info, and patterns

        """
        with tracer.start_as_current_span("data_profiler.profile_dataframe") as span:
            if not data:
                logger.warning("Empty dataset provided for profiling")
                return {
                    "metadata": {
                        "row_count": 0,
                        "column_count": 0,
                        "query": sql,
                        "question": question,
                    },
                    "columns": [],
                    "patterns": [],
                }

            # Convert to DataFrame for analysis
            df = pd.DataFrame(data)
            span.set_attribute("row_count", len(df))
            span.set_attribute("column_count", len(df.columns))

            # Generate column profiles
            columns = []
            temporal_columns = []
            categorical_columns = []
            quantitative_columns = []

            for col in df.columns:
                column_profile = DataProfiler._profile_column(df, col)
                columns.append(column_profile)

                # Track column types for pattern detection
                if column_profile["type"] == "temporal":
                    temporal_columns.append(col)
                elif column_profile["type"] == "categorical":
                    categorical_columns.append(col)
                elif column_profile["type"] == "quantitative":
                    quantitative_columns.append(col)

            # Detect visualization patterns
            patterns = DataProfiler._detect_patterns(
                df,
                temporal_columns,
                categorical_columns,
                quantitative_columns,
            )

            profile = {
                "metadata": {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "query": sql[:500],  # Truncate long queries
                    "question": question[:200],  # Truncate long questions
                },
                "columns": columns,
                "patterns": patterns,
            }

            logger.info(
                "Data profile generated",
                row_count=len(df),
                column_count=len(df.columns),
                patterns=len(patterns),
            )

            return profile

    @staticmethod
    def _profile_column(df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Profile a single column.

        Args:
            df: DataFrame containing the column
            col: Column name

        Returns:
            Column profile dictionary

        """
        series: pd.Series = df[col]  # type: ignore[assignment]
        dtype = series.dtype
        null_count = int(series.isnull().sum())

        profile: Dict[str, Any] = {
            "name": col,
            "dtype": str(dtype),
            "null_count": null_count,
            "distinct_count": int(series.nunique()),
        }

        # Detect column type and add type-specific statistics
        if pd.api.types.is_datetime64_any_dtype(dtype):
            profile.update(DataProfiler._profile_temporal(series))
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it's really categorical despite being numeric
            if series.nunique() < DataProfiler.CARDINALITY_LOW_THRESHOLD:
                profile.update(DataProfiler._profile_categorical(series))
            else:
                profile.update(DataProfiler._profile_quantitative(series))
        elif pd.api.types.is_bool_dtype(dtype):
            profile.update(DataProfiler._profile_categorical(series))
        else:
            # String/object types
            # Try to parse as datetime
            try:
                parsed_series = pd.to_datetime(series, errors="coerce")
                if parsed_series.notna().sum() > len(series) * 0.8:  # >80% valid dates
                    profile.update(DataProfiler._profile_temporal(parsed_series))
                else:
                    profile.update(DataProfiler._profile_categorical(series))
            except Exception:
                profile.update(DataProfiler._profile_categorical(series))

        return profile

    @staticmethod
    def _profile_temporal(series: pd.Series) -> Dict[str, Any]:
        """Profile temporal column.

        Args:
            series: Pandas Series with temporal data

        Returns:
            Temporal profile

        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                "type": "temporal",
                "min": None,
                "max": None,
            }

        return {
            "type": "temporal",
            "min": str(non_null.min()),
            "max": str(non_null.max()),
        }

    @staticmethod
    def _profile_categorical(series: pd.Series) -> Dict[str, Any]:
        """Profile categorical column.

        Args:
            series: Pandas Series with categorical data

        Returns:
            Categorical profile

        """
        distinct_count = series.nunique()

        # Determine cardinality level
        if distinct_count < DataProfiler.CARDINALITY_LOW_THRESHOLD:
            cardinality = "low"
        elif distinct_count < DataProfiler.CARDINALITY_MEDIUM_THRESHOLD:
            cardinality = "medium"
        else:
            cardinality = "high"

        # Get top values (for low/medium cardinality)
        top_values = {}
        if cardinality in ["low", "medium"]:
            value_counts = series.value_counts().head(DataProfiler.TOP_VALUES_LIMIT)
            top_values = {str(k): int(v) for k, v in value_counts.items()}

        return {
            "type": "categorical",
            "cardinality": cardinality,
            "top_values": top_values,
        }

    @staticmethod
    def _profile_quantitative(series: pd.Series) -> Dict[str, Any]:
        """Profile quantitative column.

        Args:
            series: Pandas Series with numeric data

        Returns:
            Quantitative profile

        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                "type": "quantitative",
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
            }

        return {
            "type": "quantitative",
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "mean": float(non_null.mean()),
            "median": float(non_null.median()),
            "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
        }

    @staticmethod
    def _detect_patterns(
        df: pd.DataFrame,
        temporal_columns: List[str],
        categorical_columns: List[str],
        quantitative_columns: List[str],
    ) -> List[Dict[str, Any]]:
        """Detect visualization patterns in the data.

        Args:
            df: DataFrame
            temporal_columns: List of temporal column names
            categorical_columns: List of categorical column names
            quantitative_columns: List of quantitative column names

        Returns:
            List of detected patterns

        """
        patterns = []

        # Time series pattern: temporal + quantitative
        if temporal_columns and quantitative_columns:
            patterns.append(
                {
                    "type": "time_series",
                    "temporal_column": temporal_columns[0],
                    "metrics": quantitative_columns,
                }
            )

        # Categorical breakdown: categorical + quantitative
        if categorical_columns and quantitative_columns:
            patterns.append(
                {
                    "type": "categorical_breakdown",
                    "dimensions": categorical_columns[:3],  # Limit to 3
                    "metrics": quantitative_columns,
                }
            )

        # Correlation pattern: multiple quantitative columns
        if len(quantitative_columns) >= 2:
            patterns.append(
                {
                    "type": "correlation",
                    "columns": quantitative_columns[:2],  # Focus on first two
                }
            )

        # Distribution pattern: single quantitative
        if len(quantitative_columns) == 1 and not temporal_columns:
            patterns.append(
                {
                    "type": "distribution",
                    "column": quantitative_columns[0],
                }
            )

        # Proportion pattern: single categorical, single quantitative
        if len(categorical_columns) == 1 and len(quantitative_columns) == 1:
            cat_col = categorical_columns[0]
            distinct_count = df[cat_col].nunique()
            if distinct_count <= 7:  # Good for pie chart
                patterns.append(
                    {
                        "type": "proportion",
                        "category_column": cat_col,
                        "value_column": quantitative_columns[0],
                    }
                )

        return patterns
