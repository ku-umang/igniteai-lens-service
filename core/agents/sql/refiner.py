import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.refiner import (
    REFINER_SYSTEM_PROMPT,
    format_refiner_prompt,
)
from core.agents.sql.state import AgentState, GeneratedSQL
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class RefinerAgent:
    """Refiner agent for SQL generation in workflow.

    This agent:
    1. Takes the logical query plan
    2. Converts it to executable SQL
    3. Optimizes for the target dialect
    4. Ensures SQL safety and correctness
    """

    def __init__(self) -> None:
        """Initialize the Refiner agent."""
        self.llm = llm_config.get_llm()

    async def refine_to_sql(self, state: AgentState) -> Dict[str, Any]:
        """Generate executable SQL from query plan or execution plan step.

        Supports both legacy single-query mode (query_plan) and new multi-query mode (execution_plan).

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with generated_sql populated

        """
        with tracer.start_as_current_span(
            "refiner_agent.refine_to_sql",
            attributes={
                "question": state.user_question,
                "has_execution_plan": state.execution_plan is not None,
                "has_query_plan": state.query_plan is not None,
            },
        ) as span:
            question_to_use = state.optimized_question or state.user_question
            try:
                if not state.schema_context:
                    raise ValueError("Schema context required")

                # Determine which planning mode we're in
                if state.execution_plan:
                    # New multi-step mode
                    return await self._refine_execution_plan_step(state, question_to_use, span)
                elif state.query_plan:
                    # Legacy single-query mode
                    return await self._refine_query_plan(state, question_to_use, span)
                else:
                    raise ValueError("Either execution_plan or query_plan required")

            except Exception as e:
                logger.error(
                    "Refiner agent failed",
                    extra={
                        "error": str(e),
                        "question": question_to_use,
                    },
                )
                return {
                    "errors": [f"Refiner agent error: {str(e)}"],
                    "current_step": "error",
                }

    async def _refine_execution_plan_step(self, state: AgentState, question: str, span: Any) -> Dict[str, Any]:
        """Generate SQL for current step in execution plan.

        Args:
            state: Current workflow state
            question: User's question
            span: OpenTelemetry span

        Returns:
            Updated state dict

        """
        execution_plan = state.execution_plan
        if not execution_plan or not execution_plan.steps:
            raise ValueError("Execution plan has no steps")

        # Get current step
        current_index = execution_plan.current_step_index
        if current_index >= len(execution_plan.steps):
            raise ValueError(f"Invalid step index: {current_index} >= {len(execution_plan.steps)}")

        current_step = execution_plan.steps[current_index]

        logger.info(
            "Refiner agent starting (execution plan mode)",
            extra={
                "question": question,
                "step_number": current_step.step_number,
                "step_description": current_step.description,
            },
        )

        # Extract selected schema
        selected_schema = self._extract_selected_schema(state)

        # Convert QueryStep to query plan dict for prompt
        query_plan_dict = {
            "steps": [current_step.description, current_step.purpose],
            "join_strategy": None,  # Will be inferred from tables
            "aggregations": current_step.aggregations,
            "filters": current_step.filters,
        }

        # Include context from previous results if this step depends on others
        previous_results_context = self._get_previous_results_context(state, current_step)

        # Get SQL dialect
        dialect = state.dialect

        logger.debug(
            "Using SQL dialect for generation",
            extra={"dialect": dialect, "step": current_step.step_number},
        )

        # Use LLM to generate SQL for this step
        sql_dict = await self._llm_generate_sql(
            question=question,
            selected_schema=selected_schema,
            query_plan=query_plan_dict,
            dialect=dialect,
            step_context=previous_results_context,
        )

        # Build GeneratedSQL
        generated_sql = GeneratedSQL(
            sql=sql_dict.get("sql", ""),
            dialect=sql_dict.get("dialect", dialect),
            is_valid=True,  # Will be validated by validator
            validation_errors=[],
            reasoning=sql_dict.get("reasoning", ""),
        )

        span.set_attribute("sql_length", len(generated_sql.sql))
        span.set_attribute("step_number", current_step.step_number)

        logger.info(
            "Refiner agent completed",
            extra={
                "sql_length": len(generated_sql.sql),
                "step_number": current_step.step_number,
            },
        )

        return {
            "generated_sql": generated_sql,
            "llm_calls": state.llm_calls + 1,
            "current_step": "validator",
        }

    async def _refine_query_plan(self, state: AgentState, question: str, span: Any) -> Dict[str, Any]:
        """Generate SQL for legacy query plan (backward compatibility).

        Args:
            state: Current workflow state
            question: User's question
            span: OpenTelemetry span

        Returns:
            Updated state dict

        """
        query_plan = state.query_plan
        if not query_plan:
            raise ValueError("Query plan required")

        logger.info(
            "Refiner agent starting (legacy query plan mode)",
            extra={"question": question},
        )

        # Extract selected schema info for the prompt
        selected_schema = self._extract_selected_schema(state)

        # Convert query plan to dict for prompt
        query_plan_dict = {
            "steps": query_plan.steps,
            "join_strategy": query_plan.join_strategy,
            "aggregations": query_plan.aggregations,
            "filters": query_plan.filters,
        }

        # Get SQL dialect from state (resolved from datasource)
        dialect = state.dialect

        logger.debug(
            "Using SQL dialect for generation",
            extra={"dialect": dialect},
        )

        # Use LLM to generate SQL
        sql_dict = await self._llm_generate_sql(
            question=question,
            selected_schema=selected_schema,
            query_plan=query_plan_dict,
            dialect=dialect,
        )

        # Build GeneratedSQL
        generated_sql = GeneratedSQL(
            sql=sql_dict.get("sql", ""),
            dialect=sql_dict.get("dialect", dialect),
            is_valid=True,  # Will be validated by validator
            validation_errors=[],
            reasoning=sql_dict.get("reasoning", ""),
        )

        span.set_attribute("sql_length", len(generated_sql.sql))

        logger.info(
            "Refiner agent completed",
            extra={"sql_length": len(generated_sql.sql)},
        )

        return {
            "generated_sql": generated_sql,
            "llm_calls": state.llm_calls + 1,
            "current_step": "validator",
        }

    def _get_previous_results_context(self, state: AgentState, current_step: Any) -> str:
        """Get context from previous query results for dependent steps.

        Args:
            state: Current workflow state
            current_step: Current QueryStep being executed

        Returns:
            Context string describing previous results

        """
        if not current_step.depends_on or not state.query_results:
            return ""

        context_lines = ["Previous query results (for context):"]

        for dep_step_num in current_step.depends_on:
            # Find result for that step (0-indexed in list, but step numbers are 1-indexed)
            result_idx = dep_step_num - 1
            if 0 <= result_idx < len(state.query_results):
                result = state.query_results[result_idx]
                if result.success:
                    context_lines.append(f"\nStep {dep_step_num} returned {result.rows_returned} rows")
                    # Include sample data structure
                    if result.rows:
                        sample_row = result.rows[0]
                        columns = list(sample_row.keys())[:5]
                        context_lines.append(f"  Columns: {', '.join(columns)}")
                else:
                    context_lines.append(f"\nStep {dep_step_num} failed: {result.error_message}")

        return "\n".join(context_lines)

    def _extract_selected_schema(self, state: AgentState) -> Dict[str, Any]:
        """Extract selected schema from state.

        Args:
            state: Current workflow state

        Returns:
            Dict with selected tables, columns, relationships, and example queries

        """
        if not state.schema_context:
            return {
                "selected_tables": [],
                "selected_columns": {},
                "relationships": [],
                "example_queries": [],
            }

        # Extract unique table names
        tables = list({table.get("metadata", {}).get("table_name") for table in state.schema_context.tables})

        # Group columns by table
        columns_by_table: Dict[str, list] = {}
        for col in state.schema_context.columns:
            table_name = col.get("metadata", {}).get("table_qualified_name", "").split(".")[1]
            col_name = col.get("metadata", {}).get("column_name")
            data_type = col.get("metadata", {}).get("data_type")

            if table_name and col_name:
                if table_name not in columns_by_table:
                    columns_by_table[table_name] = []
                columns_by_table[table_name].append(f"{col_name} ({data_type})" if data_type else col_name)

        return {
            "selected_tables": tables,
            "selected_columns": columns_by_table,
            "relationships": state.schema_context.relationships,
            "example_queries": state.schema_context.example_queries,
        }

    async def _llm_generate_sql(
        self,
        question: str,
        selected_schema: Dict[str, Any],
        query_plan: Dict[str, Any],
        dialect: str = "postgres",
        step_context: str = "",
    ) -> Dict[str, Any]:
        """Use LLM to generate SQL from query plan.

        Args:
            question: User's natural language question
            selected_schema: Selected tables and columns
            query_plan: Query plan from Decomposer or QueryStep
            dialect: Target SQL dialect
            step_context: Optional context from previous query results

        Returns:
            Generated SQL dict

        """
        # Format the prompt
        user_prompt = format_refiner_prompt(
            question=question,
            selected_schema=selected_schema,
            query_plan=query_plan,
            dialect=dialect,
            step_context=step_context,
        )

        # Call LLM
        messages = [
            SystemMessage(content=REFINER_SYSTEM_PROMPT),
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

            # Replace patterns like: "text\n" + "more" with "text\nmore"
            content = re.sub(r'"\s*\+\s*\n?\s*"', "", content)

            # Fix improperly escaped single quotes in JSON strings (\' -> ')
            # This happens when LLMs incorrectly escape single quotes in SQL code within JSON
            content = content.replace(r"\'", "'")

            sql_result = json.loads(content)
            return sql_result

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error("Failed to parse LLM response", extra={"error": str(e), "response": str(response)})

            # Fallback: try to extract SQL from response text
            content_str = response.content if isinstance(response.content, str) else str(response.content)

            # Look for SQL in code blocks
            if "```sql" in content_str:
                sql = content_str.split("```sql")[1].split("```")[0].strip()
            elif "```" in content_str:
                sql = content_str.split("```")[1].split("```")[0].strip()
            else:
                sql = "SELECT 1 -- Error: Failed to generate SQL"

            return {
                "sql": sql,
                "dialect": dialect,
                "estimated_complexity": 1,
                "optimizations_applied": [],
                "reasoning": f"Failed to parse LLM response, extracted SQL from text. Error: {str(e)}",
            }
