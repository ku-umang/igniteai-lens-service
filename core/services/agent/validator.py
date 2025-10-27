"""SQL validator with advanced validation and complexity analysis.

This module provides comprehensive SQL validation including:
- Syntax validation
- Read-only enforcement
- Complexity analysis
- Security checks
"""

from typing import Any, Dict, List

from opentelemetry import trace

from core.logging import get_logger
from core.utils.sql_parser import get_sql_parser

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ValidationResult:
    """Result of SQL validation."""

    def __init__(
        self,
        is_valid: bool,
        errors: List[str] | None = None,
        warnings: List[str] | None = None,
        complexity: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation result.

        Args:
            is_valid: Whether SQL is valid
            errors: List of validation errors
            warnings: List of warnings
            complexity: Complexity analysis

        """
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.complexity = complexity or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dict representation

        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "complexity": self.complexity,
        }


class SQLValidator:
    """SQL validator with advanced validation and complexity analysis."""

    # Complexity thresholds
    MAX_COMPLEXITY_SCORE = 8.0
    MAX_JOIN_COUNT = 10
    MAX_SUBQUERY_COUNT = 5
    MAX_NESTING_DEPTH = 5

    def __init__(self, dialect: str = "postgres") -> None:
        """Initialize SQL validator.

        Args:
            dialect: SQL dialect (postgres, mysql, sqlite, etc.)

        """
        self.dialect = dialect
        self.parser = get_sql_parser(dialect=dialect)

    def validate(
        self,
        sql: str,
        check_readonly: bool = True,
        check_complexity: bool = True,
    ) -> ValidationResult:
        """Validate SQL query.

        Args:
            sql: SQL query string
            check_readonly: Whether to enforce read-only
            check_complexity: Whether to check complexity limits

        Returns:
            ValidationResult with errors and warnings

        """
        with tracer.start_as_current_span(
            "sql_validator.validate",
            attributes={"sql_length": len(sql), "dialect": self.dialect},
        ) as span:
            errors: List[str] = []
            warnings: List[str] = []
            complexity = {}

            # Step 1: Syntax validation
            is_valid, syntax_errors = self.parser.validate_syntax(sql)
            if not is_valid:
                errors.extend(syntax_errors)
                span.set_attribute("validation_result", "syntax_error")
                return ValidationResult(is_valid=False, errors=errors)

            # Step 2: Read-only check
            if check_readonly:
                is_readonly, violations = self.parser.is_read_only(sql)
                if not is_readonly:
                    errors.extend(violations)
                    span.set_attribute("validation_result", "readonly_violation")
                    return ValidationResult(is_valid=False, errors=errors)

            # Step 3: Complexity analysis
            if check_complexity:
                complexity = self.parser.analyze_complexity(sql)

                # Check complexity limits
                if complexity.get("score", 0) > self.MAX_COMPLEXITY_SCORE:
                    warnings.append(
                        f"Query complexity score ({complexity['score']}) exceeds recommended limit ({self.MAX_COMPLEXITY_SCORE})"
                    )

                if complexity.get("join_count", 0) > self.MAX_JOIN_COUNT:
                    warnings.append(f"Join count ({complexity['join_count']}) exceeds recommended limit ({self.MAX_JOIN_COUNT})")

                if complexity.get("subquery_count", 0) > self.MAX_SUBQUERY_COUNT:
                    warnings.append(
                        f"Subquery count ({complexity['subquery_count']}) exceeds recommended limit ({self.MAX_SUBQUERY_COUNT})"
                    )

                if complexity.get("max_depth", 0) > self.MAX_NESTING_DEPTH:
                    warnings.append(
                        f"Nesting depth ({complexity['max_depth']}) exceeds recommended limit ({self.MAX_NESTING_DEPTH})"
                    )

                span.set_attribute("complexity_score", complexity.get("score", 0))

            # Step 4: Security checks
            security_warnings = self._check_security(sql)
            warnings.extend(security_warnings)

            span.set_attribute("validation_result", "valid" if not errors else "invalid")
            span.set_attribute("error_count", len(errors))
            span.set_attribute("warning_count", len(warnings))

            logger.info(
                "SQL validation completed",
                extra={
                    "is_valid": len(errors) == 0,
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "complexity_score": complexity.get("score", 0),
                },
            )

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                complexity=complexity,
            )

    def _check_security(self, sql: str) -> List[str]:
        """Check for potential security issues.

        Args:
            sql: SQL query string

        Returns:
            List of security warnings

        """
        warnings = []

        # Check for common SQL injection patterns
        sql_lower = sql.lower()

        if "--" in sql and "/*" not in sql:
            warnings.append("Query contains SQL comments (--) which may indicate injection attempts")

        if "union" in sql_lower and "select" in sql_lower:
            # This is actually OK for legitimate UNION queries, but flag as info
            logger.debug("Query contains UNION SELECT pattern", extra={"sql": sql[:100]})

        if "information_schema" in sql_lower or "pg_catalog" in sql_lower:
            warnings.append("Query accesses system catalog tables - ensure this is intentional")

        return warnings

    def estimate_rows(self, sql: str) -> int:
        """Estimate number of rows that will be returned.

        Args:
            sql: SQL query string

        Returns:
            Estimated row count (simplified heuristic)

        """
        # This is a very simplified heuristic
        # In production, you'd want to use EXPLAIN or query stats

        sql_lower = sql.lower()

        # Check for LIMIT clause
        if "limit" in sql_lower:
            try:
                # Extract limit value (simplified)
                parts = sql_lower.split("limit")
                if len(parts) > 1:
                    limit_val = parts[1].strip().split()[0]
                    return int(limit_val)
            except (ValueError, IndexError):
                pass

        # Check for aggregations
        if any(agg in sql_lower for agg in ["count(", "sum(", "avg(", "max(", "min("]):
            if "group by" not in sql_lower:
                return 1  # Single aggregate row
            else:
                return 1000  # Estimated grouped rows

        # Default estimate
        return 10000  # Conservative default


# Singleton instance
_sql_validator: Dict[str, SQLValidator] = {}


def get_sql_validator(dialect: str = "postgres") -> SQLValidator:
    """Get or create SQL validator instance.

    Args:
        dialect: SQL dialect

    Returns:
        SQLValidator instance

    """
    if dialect not in _sql_validator:
        _sql_validator[dialect] = SQLValidator(dialect=dialect)
    return _sql_validator[dialect]
