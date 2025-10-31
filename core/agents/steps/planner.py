import json
import time
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.planner import (
    PLANNER_SYSTEM_PROMPT,
    format_planner_iterative_prompt,
    format_planner_prompt,
)
from core.agents.state import AgentState, ExecutionPlan, ExecutionResult, QueryStep, QueryStepStatus, SchemaContext
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class PlannerAgent:
    """Planner agent for creating natural multi-step logical execution plans.

    This agent:
    1. Analyzes the user's question to understand its type and complexity
    2. Creates a natural, logical multi-step breakdown (typically 2-5 steps)
    3. Focuses on WHAT needs to be done, not HOW it will be implemented in SQL
    4. Supports iterative planning - can be called multiple times
    5. Decides when additional queries are needed based on previous results

    IMPORTANT: The execution plan represents LOGICAL reasoning steps, NOT separate SQL queries.
    The Refiner agent will consolidate all steps into a SINGLE comprehensive SQL query using
    CTEs, subqueries, and advanced SQL features.
    """

    def __init__(self) -> None:
        """Initialize the Planner agent."""
        self.llm = llm_config.get_llm()

    async def create_plan(self, state: AgentState) -> AgentState:
        """Create or update an execution plan for the user's question.

        This method handles both initial planning and iterative planning.

        Args:
            state: Current workflow state

        Returns:
            Updated state with execution_plan populated or updated

        """
        with tracer.start_as_current_span(
            "planner_agent.create_plan",
            attributes={
                "question": state.optimized_question or state.user_question,
                "is_iterative": state.execution_plan is not None,
            },
        ) as span:
            start_time = time.time()

            try:
                question_to_use = state.optimized_question or state.user_question

                # Determine if this is initial or iterative planning
                is_iterative = state.execution_plan is not None and state.execution_plan.requires_iteration

                logger.info(
                    "Planner agent starting",
                    extra={
                        "question": question_to_use,
                        "is_iterative": is_iterative,
                    },
                )

                if is_iterative:
                    # Iterative planning: update existing plan based on results
                    plan_dict = await self._llm_iterative_planning(
                        question=question_to_use,
                        execution_plan=state.execution_plan,  # type: ignore
                        execution_result=state.execution_result,
                    )

                else:
                    # Initial planning: create new plan
                    schema_info = self._extract_schema_info(state.schema_context)  # type: ignore

                    plan_dict = await self._llm_create_plan(
                        question=question_to_use,
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
                state.execution_plan = execution_plan
                state.total_time_ms = state.total_time_ms + planning_time
                state.llm_calls = state.llm_calls + 1
                state.current_step = "execution_planner"
                return state

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
                    requires_iteration=False,
                    reasoning=f"Planning failed: {str(e)}. Using fallback single-step plan.",
                    strategy="Simple single-query fallback",
                )

                state.execution_plan = fallback_plan
                state.errors = [f"Planner agent error: {str(e)}"]
                state.current_step = "execution_planner"
                return state

    async def _llm_create_plan(
        self,
        question: str,
        schema_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to create initial execution plan.

        Args:
            question: User's question (optimized if available)
            schema_context: schema context

        Returns:
            Plan dict with steps, strategy, reasoning

        """
        # Format the prompt
        user_prompt = format_planner_prompt(
            question=question,
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

            # Soft limit: Recommend maximum 5 steps (log warning if exceeded)
            steps = plan.get("steps", [])
            if len(steps) > 5:
                logger.warning(
                    "Plan has more than 5 steps - consider breaking into sub-questions",
                    extra={
                        "question": question,
                        "num_steps": len(steps),
                    },
                )
            elif len(steps) > 3:
                logger.info(
                    "Plan has more than 3 steps - logical breakdown",
                    extra={
                        "question": question,
                        "num_steps": len(steps),
                    },
                )

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
                        "description": f"Query data to answer: {question}",
                        "purpose": "Answer the user's question",
                        "depends_on": [],
                        "required_tables": schema_context["tables"],
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
        execution_plan: ExecutionPlan,
        execution_result: Optional[ExecutionResult],
    ) -> Dict[str, Any]:
        """Use LLM to update execution plan based on previous result.

        Args:
            question: User's question
            execution_plan: Current execution plan
            execution_result: Single execution result from the generated SQL

        Returns:
            Updated plan dict

        """
        # Format execution result summary
        result_summary = None
        if execution_result:
            result_summary = {
                "success": execution_result.success,
                "rows_returned": execution_result.rows_returned,
                "error_message": execution_result.error_message,
                "rows": execution_result.rows[:2] if execution_result.rows else [],  # First 2 rows as sample
            }
        else:
            result_summary = {
                "success": False,
                "rows_returned": 0,
                "error_message": "No execution result available",
                "rows": [],
            }

        # Format plan steps for context
        plan_steps = [
            {
                "step_number": step.step_number,
                "description": step.description,
                "purpose": step.purpose,
            }
            for step in execution_plan.steps
        ]

        # Format the prompt
        user_prompt = format_planner_iterative_prompt(
            question=question,
            completed_steps=plan_steps,
            previous_results=[result_summary],
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

    def _extract_schema_info(self, schema_context: SchemaContext) -> Dict[str, Any]:
        """Extract schema info from schema context.

        Args:
            schema_context: SchemaContext from Selector

        Returns:
            Dict with tables, columns, and relationships

        """
        # Extract unique table names
        tables = [table["table_qualified_name"] for table in schema_context.tables]

        # Group columns by table
        columns_by_table: Dict[str, list] = {}
        for col in schema_context.columns:
            metadata = col.get("metadata", {})
            table_qualified_name = col["table_qualified_name"]
            col_name = metadata["column_name"]

            if table_qualified_name not in columns_by_table:
                columns_by_table[table_qualified_name] = []
            columns_by_table[table_qualified_name].append(col_name)

        return {
            "tables": tables,
            "columns": columns_by_table,
            "relationships": schema_context.relationships,
        }
