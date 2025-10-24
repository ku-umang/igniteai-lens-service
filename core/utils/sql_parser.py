"""SQL parsing utilities using sqlglot.

This module provides utilities for parsing, analyzing, and validating SQL queries.
"""

from typing import Any, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.optimizer import optimize

from core.logging import get_logger

logger = get_logger(__name__)


class SQLParser:
    """SQL parser and analyzer using sqlglot."""

    DANGEROUS_STATEMENTS = {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "REPLACE",
        "MERGE",
    }

    def __init__(self, dialect: str = "postgres") -> None:
        """Initialize SQL parser.

        Args:
            dialect: SQL dialect (postgres, mysql, sqlite, etc.)

        """
        self.dialect = dialect

    def parse(self, sql: str) -> Optional[exp.Expression]:
        """Parse SQL query into AST.

        Args:
            sql: SQL query string

        Returns:
            Parsed expression or None if parsing fails

        """
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            logger.error("Failed to parse SQL", extra={"error": str(e), "sql": sql[:200]})
            return None

    def validate_syntax(self, sql: str) -> tuple[bool, List[str]]:
        """Validate SQL syntax.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, list of error messages)

        """
        errors = []
        try:
            ast = self.parse(sql)
            if ast is None:
                errors.append("Failed to parse SQL query")
                return False, errors
            return True, []
        except Exception as e:
            errors.append(f"Syntax error: {str(e)}")
            return False, errors

    def is_read_only(self, sql: str) -> tuple[bool, List[str]]:
        """Check if SQL query is read-only.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_read_only, list of violations)

        """
        violations = []

        try:
            ast = self.parse(sql)
            if ast is None:
                violations.append("Unable to parse query for read-only check")
                return False, violations

            # Check for dangerous statement types
            for node in ast.walk():
                node_type = node.__class__.__name__.upper()
                if any(danger in node_type for danger in self.DANGEROUS_STATEMENTS):
                    violations.append(f"Forbidden statement type: {node_type}")

            return len(violations) == 0, violations

        except Exception as e:
            violations.append(f"Error checking read-only: {str(e)}")
            return False, violations

    def analyze_complexity(self, sql: str) -> Dict[str, Any]:
        """Analyze SQL query complexity.

        Args:
            sql: SQL query string

        Returns:
            Dict with complexity metrics

        """
        try:
            ast = self.parse(sql)
            if ast is None:
                return {"error": "Failed to parse SQL", "score": 10}

            # Count various complexity indicators
            join_count = sum(1 for _ in ast.find_all(exp.Join))
            subquery_count = sum(1 for _ in ast.find_all(exp.Subquery))
            cte_count = sum(1 for _ in ast.find_all(exp.CTE))
            window_count = sum(1 for _ in ast.find_all(exp.Window))
            union_count = sum(1 for _ in ast.find_all(exp.Union))
            aggregate_count = sum(1 for _ in ast.find_all(exp.AggFunc))

            # Calculate max nesting depth
            max_depth = self._calculate_depth(ast)

            # Calculate complexity score (1-10)
            score = min(
                10,
                1
                + join_count * 0.5
                + subquery_count * 1.5
                + cte_count * 1.0
                + window_count * 1.5
                + union_count * 1.0
                + aggregate_count * 0.3
                + max_depth * 0.5,
            )

            return {
                "score": round(score, 1),
                "join_count": join_count,
                "subquery_count": subquery_count,
                "cte_count": cte_count,
                "window_count": window_count,
                "union_count": union_count,
                "aggregate_count": aggregate_count,
                "max_depth": max_depth,
                "estimated_complexity": "low" if score < 3 else "medium" if score < 6 else "high",
            }

        except Exception as e:
            logger.error("Failed to analyze complexity", extra={"error": str(e), "sql": sql[:200]})
            return {"error": str(e), "score": 10}

    def extract_tables(self, sql: str) -> Set[str]:
        """Extract all table names from SQL query.

        Args:
            sql: SQL query string

        Returns:
            Set of table names

        """
        try:
            ast = self.parse(sql)
            if ast is None:
                return set()

            tables = set()
            for table_node in ast.find_all(exp.Table):
                if hasattr(table_node, "name"):
                    tables.add(table_node.name)

            return tables

        except Exception as e:
            logger.error("Failed to extract tables", extra={"error": str(e), "sql": sql[:200]})
            return set()

    def extract_columns(self, sql: str) -> Set[str]:
        """Extract all column names from SQL query.

        Args:
            sql: SQL query string

        Returns:
            Set of column names

        """
        try:
            ast = self.parse(sql)
            if ast is None:
                return set()

            columns = set()
            for col_node in ast.find_all(exp.Column):
                if hasattr(col_node, "name"):
                    columns.add(col_node.name)

            return columns

        except Exception as e:
            logger.error("Failed to extract columns", extra={"error": str(e), "sql": sql[:200]})
            return set()

    def optimize_query(self, sql: str) -> Optional[str]:
        """Optimize SQL query.

        Args:
            sql: SQL query string

        Returns:
            Optimized SQL or None if optimization fails

        """
        try:
            ast = self.parse(sql)
            if ast is None:
                return None

            optimized_ast = optimize(ast, dialect=self.dialect)
            return optimized_ast.sql(dialect=self.dialect)

        except Exception as e:
            logger.warning("Failed to optimize SQL", extra={"error": str(e), "sql": sql[:200]})
            return None

    def _calculate_depth(self, node: exp.Expression, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of SQL expression.

        Args:
            node: SQL expression node
            current_depth: Current depth level

        Returns:
            Maximum depth

        """
        if not node:
            return current_depth

        max_child_depth = current_depth
        for child in node.iter_expressions():
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth


# Singleton instance
_sql_parser: Optional[SQLParser] = None


def get_sql_parser(dialect: str = "postgres") -> SQLParser:
    """Get or create SQL parser instance.

    Args:
        dialect: SQL dialect

    Returns:
        SQLParser instance

    """
    global _sql_parser  # noqa: PLW0603
    if _sql_parser is None or _sql_parser.dialect != dialect:
        _sql_parser = SQLParser(dialect=dialect)
    return _sql_parser
