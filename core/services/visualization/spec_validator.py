"""Spec Validator for Plotly chart specifications.

This module validates Plotly chart specifications to ensure they are
properly formatted and contain all required fields.
"""

from typing import Any, Dict, List, Tuple

from core.logging import get_logger

logger = get_logger(__name__)


class SpecValidator:
    """Validates Plotly chart specifications."""

    VALID_CHART_TYPES = {
        "scatter",
        "bar",
        "line",
        "pie",
        "heatmap",
        "treemap",
        "sunburst",
    }

    @staticmethod
    def validate(spec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Plotly chart specification.

        Args:
            spec: Plotly specification dictionary

        Returns:
            Tuple of (is_valid, error_messages)

        """
        errors: List[str] = []

        # Check top-level structure
        if not isinstance(spec, dict):
            errors.append("Specification must be a dictionary")
            return False, errors

        # Check required fields
        if "data" not in spec:
            errors.append("Missing required field: 'data'")
        elif not isinstance(spec["data"], list):
            errors.append("'data' must be a list of traces")

        if "layout" not in spec:
            errors.append("Missing required field: 'layout'")
        elif not isinstance(spec["layout"], dict):
            errors.append("'layout' must be a dictionary")

        # If basic structure is invalid, return early
        if errors:
            return False, errors

        # Validate traces
        for idx, trace in enumerate(spec.get("data", [])):
            trace_errors = SpecValidator._validate_trace(trace, idx)
            errors.extend(trace_errors)

        # Validate layout
        layout_errors = SpecValidator._validate_layout(spec.get("layout", {}))
        errors.extend(layout_errors)

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning("Spec validation failed", errors=errors)
        else:
            logger.info("Spec validation passed")

        return is_valid, errors

    @staticmethod
    def _validate_trace(trace: Any, idx: int) -> List[str]:
        """Validate a single trace in the data array.

        Args:
            trace: Trace dictionary
            idx: Trace index

        Returns:
            List of error messages

        """
        errors: List[str] = []

        if not isinstance(trace, dict):
            errors.append(f"Trace {idx}: must be a dictionary")
            return errors

        # Check trace type
        trace_type = trace.get("type")
        if not trace_type:
            errors.append(f"Trace {idx}: missing 'type' field")
        elif trace_type not in SpecValidator.VALID_CHART_TYPES:
            logger.warning(f"Trace {idx}: unknown trace type '{trace_type}'")

        # Validate data fields based on trace type
        if trace_type == "scatter" or trace_type == "line":
            if "x" not in trace:
                errors.append(f"Trace {idx}: scatter/line requires 'x' field")
            if "y" not in trace:
                errors.append(f"Trace {idx}: scatter/line requires 'y' field")

        elif trace_type == "bar":
            if "x" not in trace and "y" not in trace:
                errors.append(f"Trace {idx}: bar requires 'x' or 'y' field")

        elif trace_type == "pie":
            if "labels" not in trace:
                errors.append(f"Trace {idx}: pie requires 'labels' field")
            if "values" not in trace:
                errors.append(f"Trace {idx}: pie requires 'values' field")

        elif trace_type == "heatmap":
            if "z" not in trace:
                errors.append(f"Trace {idx}: heatmap requires 'z' field")

        return errors

    @staticmethod
    def _validate_layout(layout: Dict[str, Any]) -> List[str]:
        """Validate layout configuration.

        Args:
            layout: Layout dictionary

        Returns:
            List of error messages

        """
        errors: List[str] = []

        if not isinstance(layout, dict):
            errors.append("Layout must be a dictionary")
            return errors

        # Optional but recommended fields - just log warnings
        if "title" not in layout:
            logger.debug("Layout: missing recommended 'title' field")

        # Validate xaxis/yaxis if present
        for axis in ["xaxis", "yaxis"]:
            if axis in layout and not isinstance(layout[axis], dict):
                errors.append(f"Layout: '{axis}' must be a dictionary")

        return errors
