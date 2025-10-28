"""Multi-agent SQL generation with classification and iterative planning.

This module implements an advanced multi-agent architecture for text-to-SQL generation:
- Optimizer: Context-aware question optimization
- Classifier: Question type classification (trend, comparison, forecasting, etc.)
- Planner: Multi-step execution planning with iterative support
- Selector: Schema selection from retrieved metadata
- Refiner: SQL generation and refinement
- Analyzer: Multi-query result synthesis
- Visualizer: Chart generation
"""

from core.agents.sql.state import (
    AgentInput,
    AgentOutput,
    AgentState,
    AnalysisResult,
    ClassificationResult,
    ExecutionPlan,
    ExecutionResult,
    GeneratedSQL,
    QueryPlan,
    QueryStep,
    QueryStepStatus,
    QuestionType,
    SchemaContext,
)

__all__ = [
    "AgentState",
    "AgentInput",
    "AgentOutput",
    "SchemaContext",
    "QueryPlan",
    "ExecutionPlan",
    "QueryStep",
    "QueryStepStatus",
    "GeneratedSQL",
    "ExecutionResult",
    "ClassificationResult",
    "QuestionType",
    "AnalysisResult",
]
