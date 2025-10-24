"""MAC-SQL agent module for text-to-SQL generation.

This module implements the MAC-SQL (Multi-Agent Collaborative SQL) architecture
with three specialized agents:
- Selector: Schema selection from retrieved metadata
- Decomposer: Query decomposition and planning
- Refiner: SQL generation and refinement
"""

from core.agents.sql.state import (
    ExecutionResult,
    GeneratedSQL,
    MACSSQLInput,
    MACSSQLOutput,
    MACSSQLState,
    QueryPlan,
    SchemaContext,
)

__all__ = [
    "MACSSQLState",
    "MACSSQLInput",
    "MACSSQLOutput",
    "SchemaContext",
    "QueryPlan",
    "GeneratedSQL",
    "ExecutionResult",
]
