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

                if not state.classification:
                    logger.error("Classification not available", extra={"question": question_to_use})
                    raise ValueError("Classification not available")

                # Determine if this is initial or iterative planning
                is_iterative = state.execution_plan is not None and state.execution_plan.requires_iteration

                logger.info(
                    "Planner agent starting",
                    extra={
                        "question": question_to_use,
                        "classification": state.classification.question_type.value,
                        "is_iterative": is_iterative,
                    },
                )

                if is_iterative:
                    # Iterative planning: update existing plan based on results
                    plan_dict = await self._llm_iterative_planning(
                        question=question_to_use,
                        classification=state.classification,
                        execution_plan=state.execution_plan,  # type: ignore
                        execution_result=state.execution_result,
                    )

                else:
                    # Initial planning: create new plan
                    schema_info = self._extract_schema_info(state.schema_context)  # type: ignore

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
                    current_step_index=0,
                    is_complete=False,
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
        classification: Any,
        schema_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to create initial execution plan.

        Args:
            question: User's question (optimized if available)
            classification: ClassificationResult
            schema_context: schema context

        Returns:
            Plan dict with steps, strategy, reasoning

        """
        # Format the prompt
        user_prompt = format_planner_prompt(
            question=question,
            classification_type=classification.question_type.value,
            classification_confidence=classification.confidence,
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

            # Apply validation and constraints
            plan = self._validate_and_constrain_plan(
                plan=plan,
                classification=classification,
                question=question,
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
        classification: Any,
        execution_plan: ExecutionPlan,
        execution_result: Optional[ExecutionResult],
    ) -> Dict[str, Any]:
        """Use LLM to update execution plan based on previous result.

        Args:
            question: User's question
            classification: ClassificationResult
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
            classification_type=classification.question_type.value,
            classification_confidence=classification.confidence,
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

    def _validate_and_constrain_plan(
        self,
        plan: Dict[str, Any],
        classification: Any,
        question: str,
    ) -> Dict[str, Any]:
        """Validate and constrain the execution plan based on confidence and complexity.

        Args:
            plan: Raw plan dict from LLM
            classification: ClassificationResult with type and confidence
            question: User's question for logging

        Returns:
            Validated and potentially modified plan dict

        """
        steps = plan.get("steps", [])
        num_steps = len(steps)

        # Hard limit: Maximum 3 steps
        if num_steps > 3:
            logger.warning(
                "Plan exceeds maximum 3 steps, truncating",
                extra={
                    "question": question,
                    "original_steps": num_steps,
                    "classification": classification.question_type.value,
                },
            )
            plan["steps"] = steps[:3]
            plan["reasoning"] = f"{plan.get('reasoning', '')} [Note: Plan was truncated from {num_steps} to 3 steps]"
            num_steps = 3

        # Confidence-based single-step enforcement
        # If high confidence (>= 0.8) and simple question type, strongly prefer single step
        simple_types = ["descriptive", "comparison", "trend"]
        is_simple_type = classification.question_type.value in simple_types
        is_high_confidence = classification.confidence >= 0.8

        if is_high_confidence and is_simple_type and num_steps > 1:
            logger.warning(
                "High-confidence simple query with multiple steps - consider simplifying",
                extra={
                    "question": question,
                    "classification": classification.question_type.value,
                    "confidence": classification.confidence,
                    "num_steps": num_steps,
                    "steps": [step.get("description") for step in steps],
                },
            )

            # For very high confidence (>= 0.9) descriptive/comparison queries, force single step
            if classification.confidence >= 0.9 and num_steps > 1:
                logger.info(
                    "Forcing single-step plan for very high confidence simple query",
                    extra={
                        "question": question,
                        "classification": classification.question_type.value,
                        "confidence": classification.confidence,
                        "original_steps": num_steps,
                    },
                )

                # Create simplified single-step plan
                plan["steps"] = [
                    {
                        "step_number": 1,
                        "description": f"Query to answer: {question}",
                        "purpose": "Answer the user's question in a single query using SQL capabilities",
                        "depends_on": [],
                        "required_tables": list({table for step in steps for table in step.get("required_tables", [])}),
                        "aggregations": list({agg for step in steps for agg in step.get("aggregations", [])}),
                        "filters": list({filter_ for step in steps for filter_ in step.get("filters", [])}),
                    }
                ]
                plan["strategy"] = "Single-step solution using SQL's GROUP BY, conditional aggregation, and JOINs"
                plan["reasoning"] = (
                    f"High-confidence {classification.question_type.value} query simplified to single step. "
                    f"Original plan had {num_steps} steps but can be combined using modern SQL capabilities."
                )
                plan["requires_iteration"] = False

        return plan

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
