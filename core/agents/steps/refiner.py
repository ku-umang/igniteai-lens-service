import json
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.refiner import (
    REFINER_SYSTEM_PROMPT,
    format_refiner_prompt,
)
from core.agents.state import AgentState, GeneratedSQL
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

    async def refine_to_sql(self, state: AgentState) -> AgentState:
        """Generate a single comprehensive SQL query from execution plan.

        The execution plan is used as structured reasoning context to generate ONE SQL query
        that answers the entire question using CTEs, subqueries, and complex joins as needed.

        Args:
            state: Current workflow state

        Returns:
            Updated state with generated_sql populated

        """
        with tracer.start_as_current_span(
            "refiner_agent.refine_to_sql",
            attributes={
                "question": state.user_question,
                "has_execution_plan": state.execution_plan is not None,
            },
        ) as span:
            question_to_use = state.optimized_question or state.user_question
            try:
                if not state.schema_context:
                    raise ValueError("Schema context required")

                if not state.execution_plan:
                    raise ValueError("Execution plan required")

                return await self._refine_execution_plan(state, question_to_use, span)

            except Exception as e:
                logger.error(
                    "Refiner agent failed",
                    extra={
                        "error": str(e),
                        "question": question_to_use,
                    },
                )
                raise Exception("Failed to generate sql from query plan") from e

    async def _refine_execution_plan(self, state: AgentState, question: str, span: Any) -> AgentState:
        """Generate a single SQL query from the entire execution plan.

        The execution plan steps are used as structured reasoning to understand
        the problem complexity, but ONE comprehensive SQL query is generated.

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

        logger.info(
            "Refiner agent starting (single SQL generation mode)",
            extra={
                "num_plan_steps": len(execution_plan.steps),
                "strategy": execution_plan.strategy,
            },
        )

        # Extract selected schema
        selected_schema = self._extract_selected_schema(state)

        # Convert ExecutionPlan to dict for prompt - include ALL steps
        query_plan_dict = {
            "steps": [
                {
                    "description": step.description,
                    "purpose": step.purpose,
                    "required_tables": step.required_tables,
                    "aggregations": step.aggregations,
                    "filters": step.filters,
                }
                for step in execution_plan.steps
            ],
            "reasoning": execution_plan.reasoning,
        }

        # Get SQL dialect
        dialect = state.dialect

        logger.debug(
            "Using SQL dialect for generation",
            extra={"dialect": dialect, "num_steps": len(execution_plan.steps)},
        )

        # Use LLM to generate ONE comprehensive SQL query
        sql_dict = await self._llm_generate_sql(
            question=question,
            selected_schema=selected_schema,
            query_plan=query_plan_dict,
            dialect=dialect,
            strategy=execution_plan.strategy,
        )

        # Build GeneratedSQL
        generated_sql = GeneratedSQL(
            sql=sql_dict.get("sql", ""),
            dialect=sql_dict.get("dialect", dialect),
            reasoning=sql_dict.get("reasoning", ""),
        )

        span.set_attribute("sql_length", len(generated_sql.sql))
        span.set_attribute("num_plan_steps", len(execution_plan.steps))

        logger.info(
            "Refiner agent completed",
            extra={
                "sql_length": len(generated_sql.sql),
                "num_plan_steps": len(execution_plan.steps),
            },
        )

        # Store the single generated SQL
        state.generated_sql = generated_sql
        state.llm_calls = state.llm_calls + 1
        state.current_step = "refiner"
        return state

    def _extract_selected_schema(self, state: AgentState) -> Dict[str, Any]:
        """Extract selected schema from state.

        Args:
            state: Current workflow state

        Returns:
            Dict with selected tables, columns, relationships, and example queries

        """
        if not state.schema_context:
            raise ValueError("Schema context required")
        # Extract unique table names
        tables = [table["table_qualified_name"] for table in state.schema_context.tables]

        # Group columns by table
        columns_by_table: Dict[str, list] = {}
        for col in state.schema_context.columns:
            table_qualified_name = col["table_qualified_name"]
            col_name = col["metadata"]["column_name"]
            data_type = col["metadata"]["data_type"]
            sample_values = col["metadata"]["sample_values"]

            if table_qualified_name and col_name:
                if table_qualified_name not in columns_by_table:
                    columns_by_table[table_qualified_name] = []
                columns_by_table[table_qualified_name].append(
                    f"{col_name}: data_type: {data_type} - sample_values: {sample_values}"
                )

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
        strategy: str = "",
    ) -> Dict[str, Any]:
        """Use LLM to generate a single comprehensive SQL query from execution plan.

        Args:
            question: User's natural language question
            selected_schema: Selected tables and columns
            query_plan: Full execution plan with all steps (used as reasoning context)
            dialect: Target SQL dialect
            strategy: High-level strategy from the execution plan

        Returns:
            Generated SQL dict

        """
        # Format the prompt
        user_prompt = format_refiner_prompt(
            question=question,
            selected_schema=selected_schema,
            query_plan=query_plan,
            dialect=dialect,
            strategy=strategy,
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
                "reasoning": f"Failed to parse LLM response, extracted SQL from text. Error: {str(e)}",
            }
