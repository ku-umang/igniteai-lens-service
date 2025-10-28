import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    format_planner_iterative_prompt,
    format_planner_prompt,
)
from core.agents.sql.state import (
    AgentState,
    ExecutionPlan,
    QueryStep,
    QueryStepStatus,
)
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class PlannerAgent:
    """Planner agent for multi-step query planning in workflow.

    This agent:
    1. Analyzes the user's question and classification
    2. Creates a multi-step execution plan with individual QuerySteps
    3. Supports iterative planning - can be called multiple times
    4. Decides when additional queries are needed based on previous results
    5. Adapts strategy based on question type (what_if, trend, correlation, etc.)
    """

    def __init__(self) -> None:
        """Initialize the Planner agent."""
        self.llm = llm_config.get_llm()

    async def create_plan(self, state: AgentState) -> Dict[str, Any]:
        """Create or update an execution plan for the user's question.

        This method handles both initial planning and iterative planning.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with execution_plan populated or updated

        """
        with tracer.start_as_current_span(
            "planner_agent.create_plan",
            attributes={
                "question": state.optimized_question or state.user_question,
                "classification": state.classification.question_type.value if state.classification else "none",
                "is_iterative": state.execution_plan is not None,
            },
        ) as span:
            start_time = time.time()

            try:
                question_to_use = state.optimized_question or state.user_question

                if not state.classification:
                    raise ValueError("Classification not available")

                # Determine if this is initial or iterative planning
                is_iterative = state.execution_plan is not None and len(state.query_results) > 0

                logger.info(
                    "Planner agent starting",
                    extra={
                        "question": question_to_use,
                        "classification": state.classification.question_type.value,
                        "confidence": state.classification.confidence,
                        "is_iterative": is_iterative,
                    },
                )

                if is_iterative:
                    # Iterative planning: update existing plan based on results
                    plan_dict = await self._llm_iterative_planning(
                        question=question_to_use,
                        classification=state.classification,
                        execution_plan=state.execution_plan,  # type: ignore
                        query_results=state.query_results,
                    )
                else:
                    # Initial planning: create new plan
                    # Extract schema info if available
                    schema_info = self._extract_schema_info(state.schema_context) if state.schema_context else None

                    plan_dict = await self._llm_create_plan(
                        question=question_to_use,
                        classification=state.classification,
                        schema_context=schema_info,
                    )

                # Build ExecutionPlan
                query_steps = []
                for step_data in plan_dict.get("steps", []):
                    query_step = QueryStep(
                        step_number=step_data["step_number"],
                        description=step_data["description"],
                        purpose=step_data["purpose"],
                        depends_on=step_data.get("depends_on", []),
                        status=QueryStepStatus.PENDING,
                        required_tables=step_data.get("required_tables", []),
                        aggregations=step_data.get("aggregations", []),
                        filters=step_data.get("filters", []),
                    )
                    query_steps.append(query_step)

                execution_plan = ExecutionPlan(
                    steps=query_steps,
                    current_step_index=0,
                    is_complete=len(query_steps) == 0,  # Empty plan means complete
                    requires_iteration=plan_dict.get("requires_iteration", False),
                    reasoning=plan_dict.get("reasoning", ""),
                    strategy=plan_dict.get("strategy", ""),
                )

                planning_time = (time.time() - start_time) * 1000

                span.set_attribute("num_steps", len(query_steps))
                span.set_attribute("requires_iteration", execution_plan.requires_iteration)

                logger.info(
                    "Planner agent completed",
                    extra={
                        "num_steps": len(query_steps),
                        "requires_iteration": execution_plan.requires_iteration,
                        "strategy": execution_plan.strategy,
                        "planning_time_ms": planning_time,
                    },
                )

                return {
                    "execution_plan": execution_plan,
                    "total_time_ms": state.total_time_ms + planning_time,
                    "llm_calls": state.llm_calls + 1,
                    "current_step": "selector",  # Next: retrieve schema for first step
                }

            except Exception as e:
                logger.error(
                    "Planner agent failed",
                    extra={
                        "error": str(e),
                        "question": state.optimized_question or state.user_question,
                    },
                )
                # On error, create a simple single-step fallback plan
                fallback_plan = ExecutionPlan(
                    steps=[
                        QueryStep(
                            step_number=1,
                            description="Execute single query to answer question",
                            purpose="Fallback plan due to planning error",
                            depends_on=[],
                            status=QueryStepStatus.PENDING,
                            required_tables=[],
                            aggregations=[],
                            filters=[],
                        )
                    ],
                    current_step_index=0,
                    is_complete=False,
                    requires_iteration=False,
                    reasoning=f"Planning failed: {str(e)}. Using fallback single-step plan.",
                    strategy="Simple single-query fallback",
                )

                return {
                    "execution_plan": fallback_plan,
                    "errors": [f"Planner agent error: {str(e)}"],
                    "current_step": "selector",
                }

    async def _llm_create_plan(
        self,
        question: str,
        classification: Any,
        schema_context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        """Use LLM to create initial execution plan.

        Args:
            question: User's question (optimized if available)
            classification: ClassificationResult
            schema_context: Optional schema context

        Returns:
            Plan dict with steps, strategy, reasoning

        """
        # Format the prompt
        user_prompt = format_planner_prompt(
            question=question,
            classification_type=classification.question_type.value,
            classification_confidence=classification.confidence,
            classification_characteristics=classification.characteristics,
            schema_context=schema_context,
        )

        # Call LLM
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content if isinstance(response.content, str) else str(response.content)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # Fix improperly escaped single quotes
            content = content.replace(r"\'", "'")

            plan = json.loads(content)

            # Validate required fields
            if "steps" not in plan:
                raise ValueError("Missing 'steps' in plan response")

            return plan

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            logger.error(
                "Failed to parse LLM plan response",
                extra={"error": str(e), "response": str(response)},
            )
            # Fallback: create simple single-step plan
            return {
                "steps": [
                    {
                        "step_number": 1,
                        "description": f"Query data to answer: {question[:100]}",
                        "purpose": "Answer the user's question",
                        "depends_on": [],
                        "required_tables": [],
                        "aggregations": [],
                        "filters": [],
                    }
                ],
                "requires_iteration": False,
                "strategy": "Simple single-query approach",
                "reasoning": f"Failed to parse LLM response. Using fallback plan. Error: {str(e)}",
            }

    async def _llm_iterative_planning(
        self,
        question: str,
        classification: Any,
        execution_plan: ExecutionPlan,
        query_results: list,
    ) -> Dict[str, Any]:
        """Use LLM to update execution plan based on previous results.

        Args:
            question: User's question
            classification: ClassificationResult
            execution_plan: Current execution plan
            query_results: Results from completed steps

        Returns:
            Updated plan dict

        """
        # Format completed steps
        completed_steps = [
            {
                "step_number": step.step_number,
                "description": step.description,
                "status": step.status.value,
            }
            for step in execution_plan.steps
            if step.status in [QueryStepStatus.COMPLETED, QueryStepStatus.FAILED]
        ]

        # Format results
        results_summary = [
            {
                "success": result.success,
                "rows_returned": result.rows_returned,
                "error_message": result.error_message,
                "rows": result.rows[:2] if result.rows else [],  # First 2 rows as sample
            }
            for result in query_results
        ]

        # Format the prompt
        user_prompt = format_planner_iterative_prompt(
            question=question,
            classification_type=classification.question_type.value,
            classification_confidence=classification.confidence,
            completed_steps=completed_steps,
            previous_results=results_summary,
        )

        # Call LLM
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await self.llm.ainvoke(messages)

        # Parse JSON response
        try:
            content = response.content if isinstance(response.content, str) else str(response.content)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            content = content.replace(r"\'", "'")
            plan = json.loads(content)

            return plan

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            logger.error(
                "Failed to parse iterative plan response",
                extra={"error": str(e), "response": str(response)},
            )
            # Fallback: mark plan as complete (no more steps)
            return {
                "steps": [],
                "requires_iteration": False,
                "strategy": "Completing with current results",
                "reasoning": f"Failed to parse iterative response. Marking plan complete. Error: {str(e)}",
            }

    def _extract_schema_info(self, schema_context: Any) -> Dict[str, Any]:
        """Extract schema info from schema context.

        Args:
            schema_context: SchemaContext from Selector

        Returns:
            Dict with tables, columns, and relationships

        """
        # Extract unique table names
        tables = [table.get("metadata", {}).get("table_name", "unknown") for table in schema_context.tables]

        # Group columns by table
        columns_by_table: Dict[str, list] = {}
        for col in schema_context.columns:
            metadata = col.get("metadata", {})
            table_qualified = metadata.get("table_qualified_name", "")
            parts = table_qualified.split(".")
            table_name = parts[1] if len(parts) > 1 else "unknown"
            col_name = metadata.get("column_name", "unknown")

            if table_name != "unknown":
                if table_name not in columns_by_table:
                    columns_by_table[table_name] = []
                columns_by_table[table_name].append(col_name)

        return {
            "tables": tables,
            "columns": columns_by_table,
            "relationships": schema_context.relationships,
        }
