"""Decomposer agent for MAC-SQL workflow.

The Decomposer agent breaks down the user's question into logical query steps
and creates a query plan that can be converted to SQL.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.decomposer import (
    DECOMPOSER_SYSTEM_PROMPT,
    format_decomposer_prompt,
)
from core.agents.sql.state import MACSSQLState, QueryPlan
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class DecomposerAgent:
    """Decomposer agent for query planning in MAC-SQL workflow.

    This agent:
    1. Analyzes the user's question and selected schema
    2. Breaks the question into logical query steps
    3. Identifies necessary operations (joins, filters, aggregations, etc.)
    4. Estimates query complexity
    """

    def __init__(self) -> None:
        """Initialize the Decomposer agent."""
        self.llm = llm_config.get_llm()

    async def decompose_query(self, state: MACSSQLState) -> Dict[str, Any]:
        """Create a logical query plan for the user's question.

        Args:
            state: Current MAC-SQL workflow state

        Returns:
            Updated state dict with query_plan populated

        """
        with tracer.start_as_current_span(
            "decomposer_agent.decompose_query",
            attributes={
                "question": state.user_question,
                "tables_selected": len(state.schema_context.tables) if state.schema_context else 0,
            },
        ) as span:
            try:
                if not state.schema_context:
                    raise ValueError("Schema context not available")

                logger.info(
                    "Decomposer agent starting",
                    extra={"question": state.user_question},
                )

                # Extract selected schema info
                selected_info = self._extract_schema_info(state.schema_context)

                # Use LLM to create query plan
                plan_dict = await self._llm_decompose_query(
                    question=state.user_question,
                    selected_tables=selected_info["tables"],
                    selected_columns=selected_info["columns"],
                    join_paths=selected_info["join_paths"],
                )

                # Build QueryPlan
                query_plan = QueryPlan(
                    steps=plan_dict.get("steps", []),
                    join_strategy=plan_dict.get("join_strategy"),
                    aggregations=plan_dict.get("aggregations", []),
                    filters=plan_dict.get("filters", []),
                    complexity_score=plan_dict.get("complexity_score", 0.5),
                    reasoning=plan_dict.get("reasoning", ""),
                )

                span.set_attribute("complexity_score", query_plan.complexity_score)
                span.set_attribute("num_steps", len(query_plan.steps))

                logger.info(
                    "Decomposer agent completed",
                    extra={
                        "num_steps": len(query_plan.steps),
                        "complexity_score": query_plan.complexity_score,
                    },
                )

                return {
                    "query_plan": query_plan,
                    "llm_calls": state.llm_calls + 1,
                    "current_step": "refiner",
                }

            except Exception as e:
                logger.error(
                    "Decomposer agent failed",
                    extra={
                        "error": str(e),
                        "question": state.user_question,
                    },
                )
                return {
                    "errors": [f"Decomposer agent error: {str(e)}"],
                    "current_step": "error",
                }

    def _extract_schema_info(self, schema_context: Any) -> Dict[str, Any]:
        """Extract schema info from schema context.

        Args:
            schema_context: SchemaContext from Selector

        Returns:
            Dict with tables, columns, and join_paths

        """
        # Extract unique table names
        tables = list({table.get("metadata", {}).get("table_name") for table in schema_context.tables})

        # Group columns by table
        columns_by_table: Dict[str, list] = {}
        for col in schema_context.columns:
            table_name = col.get("metadata", {}).get("table_qualified_name", "").split(".")[1]
            col_name = col.get("metadata", {}).get("column_name")
            if table_name and col_name:
                if table_name not in columns_by_table:
                    columns_by_table[table_name] = []
                columns_by_table[table_name].append(col_name)

        return {
            "tables": tables,
            "columns": columns_by_table,
            "join_paths": schema_context.relationships,
        }

    async def _llm_decompose_query(
        self,
        question: str,
        selected_tables: list,
        selected_columns: dict,
        join_paths: list,
    ) -> Dict[str, Any]:
        """Use LLM to create query plan.

        Args:
            question: User's natural language question
            selected_tables: List of selected table names
            selected_columns: Dict mapping table names to column lists
            join_paths: List of join path definitions

        Returns:
            Query plan dict

        """
        # Format the prompt
        user_prompt = format_decomposer_prompt(
            question=question,
            selected_tables=selected_tables,
            selected_columns=selected_columns,
            join_paths=join_paths,
        )

        # Call LLM
        messages = [
            SystemMessage(content=DECOMPOSER_SYSTEM_PROMPT),
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

            plan = json.loads(content)
            return plan

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error("Failed to parse LLM response", extra={"error": str(e), "response": str(response)})
            # Fallback: create basic plan
            return {
                "steps": ["Retrieve data from selected tables", "Apply filters", "Return results"],
                "join_strategy": None if len(selected_tables) <= 1 else "JOIN tables as needed",
                "aggregations": [],
                "filters": [],
                "complexity_score": 0.3,
                "reasoning": f"Failed to parse LLM response, using fallback plan. Error: {str(e)}",
            }
