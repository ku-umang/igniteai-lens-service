"""Aggregation Engine for reducing dataset size for visualization.

This module implements intelligent aggregation strategies to reduce large datasets
while preserving meaningful patterns for visualization.
"""

from typing import Any, Dict, List

import pandas as pd
from opentelemetry import trace

from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AggregationEngine:
    """Intelligently aggregates data for efficient visualization."""

    MAX_POINTS = 500
    SCATTER_SAMPLE_SIZE = 1000
    TOP_N_LIMIT = 20

    @staticmethod
    def aggregate(
        data: List[Dict[str, Any]],
        profile: Dict[str, Any],
        max_points: int = MAX_POINTS,
    ) -> List[Dict[str, Any]]:
        """Aggregate data based on profile and size constraints.

        Args:
            data: Original query results
            profile: Data profile from DataProfiler
            max_points: Maximum number of data points to return

        Returns:
            Aggregated data suitable for visualization

        """
        with tracer.start_as_current_span("aggregation_engine.aggregate") as span:
            row_count = profile.get("metadata", {}).get("row_count", 0)
            span.set_attribute("original_row_count", row_count)
            span.set_attribute("max_points", max_points)

            if row_count <= max_points:
                logger.info("No aggregation needed", row_count=row_count)
                return data

            # Convert to DataFrame for aggregation
            df = pd.DataFrame(data)

            # Determine aggregation strategy based on patterns
            columns = profile.get("columns", [])

            # Extract column info
            temporal_cols = [c["name"] for c in columns if c.get("type") == "temporal"]
            categorical_cols = [c["name"] for c in columns if c.get("type") == "categorical"]
            quantitative_cols = [c["name"] for c in columns if c.get("type") == "quantitative"]

            # Prioritize aggregation strategies
            aggregated_df = None

            # 1. Time-based aggregation for time series
            if temporal_cols and quantitative_cols:
                aggregated_df = AggregationEngine._aggregate_time_series(
                    df,
                    temporal_cols[0],
                    quantitative_cols,
                    max_points,
                )

            # 2. Top-N aggregation for categorical data
            elif categorical_cols and quantitative_cols:
                aggregated_df = AggregationEngine._aggregate_categorical(
                    df,
                    categorical_cols,
                    quantitative_cols,
                    max_points,
                )

            # 3. Sampling for scatter plots
            elif len(quantitative_cols) >= 2:
                aggregated_df = AggregationEngine._stratified_sample(
                    df,
                    min(max_points, AggregationEngine.SCATTER_SAMPLE_SIZE),
                )

            # 4. Default: simple sampling
            else:
                aggregated_df = df.sample(n=min(max_points, len(df)), random_state=42)

            if aggregated_df is None:
                aggregated_df = df.sample(n=min(max_points, len(df)), random_state=42)

            result = aggregated_df.to_dict(orient="records")
            span.set_attribute("aggregated_row_count", len(result))

            logger.info(
                "Data aggregated",
                original_rows=row_count,
                aggregated_rows=len(result),
                reduction_ratio=f"{len(result) / row_count:.2%}",
            )

            return result

    @staticmethod
    def _aggregate_time_series(
        df: pd.DataFrame,
        time_col: str,
        metric_cols: List[str],
        max_points: int,
    ) -> pd.DataFrame:
        """Aggregate time series data by time granularity.

        Args:
            df: DataFrame with time series data
            time_col: Temporal column name
            metric_cols: Metric column names
            max_points: Maximum points to return

        Returns:
            Aggregated DataFrame

        """
        # Convert to datetime if not already
        df[time_col] = pd.to_datetime(df[time_col])

        # Determine time range
        time_range = (df[time_col].max() - df[time_col].min()).days

        # Select appropriate granularity
        if time_range > 730:  # > 2 years
            freq = "ME"  # Monthly
            granularity = "monthly"
        elif time_range > 90:  # 3 months - 2 years
            freq = "W"  # Weekly
            granularity = "weekly"
        elif time_range > 1:  # > 1 day
            freq = "D"  # Daily
            granularity = "daily"
        else:
            freq = "h"  # Hourly
            granularity = "hourly"

        logger.info(
            "Time series aggregation",
            time_range_days=time_range,
            granularity=granularity,
        )

        # Group by time period and aggregate
        df_sorted = df.sort_values(time_col)
        df_sorted.set_index(time_col, inplace=True)

        # Resample and aggregate (mean for metrics)
        agg_dict = dict.fromkeys(metric_cols, "mean")
        aggregated = df_sorted.resample(freq).agg(agg_dict).reset_index()

        # Drop rows with all NaN values
        aggregated = aggregated.dropna(how="all", subset=metric_cols)

        # If still too many points, sample evenly
        if len(aggregated) > max_points:
            step = len(aggregated) // max_points
            aggregated = aggregated.iloc[::step]

        return aggregated

    @staticmethod
    def _aggregate_categorical(
        df: pd.DataFrame,
        cat_cols: List[str],
        metric_cols: List[str],
        max_points: int,
    ) -> pd.DataFrame:
        """Aggregate categorical data using Top-N strategy.

        Args:
            df: DataFrame with categorical data
            cat_cols: Categorical column names
            metric_cols: Metric column names
            max_points: Maximum points to return

        Returns:
            Aggregated DataFrame

        """
        # Use first categorical and first metric for Top-N selection
        primary_cat = cat_cols[0]
        primary_metric = metric_cols[0]

        # Group by category and aggregate
        grouped = df.groupby(primary_cat, as_index=False).agg(dict.fromkeys(metric_cols, "sum"))

        # Sort by primary metric descending
        grouped = grouped.sort_values(primary_metric, ascending=False)  # type: ignore[arg-type]

        # Take top N
        top_n = min(max_points, AggregationEngine.TOP_N_LIMIT, len(grouped))
        result = grouped.head(top_n)

        logger.info(
            "Categorical aggregation",
            original_categories=len(grouped),
            top_n=top_n,
        )

        return result

    @staticmethod
    def _stratified_sample(
        df: pd.DataFrame,
        sample_size: int,
    ) -> pd.DataFrame:
        """Stratified sampling for scatter plots.

        Args:
            df: DataFrame to sample
            sample_size: Number of samples to return

        Returns:
            Sampled DataFrame

        """
        if len(df) <= sample_size:
            return df

        # Simple random sampling (could be enhanced with stratification)
        sampled = df.sample(n=sample_size, random_state=42)

        logger.info(
            "Stratified sampling",
            original_size=len(df),
            sample_size=len(sampled),
        )

        return sampled
