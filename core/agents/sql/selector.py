"""Selector agent for MAC-SQL workflow.

The Selector agent analyzes the user's question and retrieves relevant schema,
then selects the minimal set of tables and columns needed to answer the question.
"""

import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.selector import (
    SELECTOR_SYSTEM_PROMPT,
    format_selector_prompt,
)
from core.agents.sql.state import MACSSQLState, SchemaContext
from core.integrations.platform_client import PlatformClient
from core.integrations.schema import RetrievalResponse
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class SelectorAgent:
    """Selector agent for schema selection in MAC-SQL workflow.

    This agent:
    1. Retrieves relevant schema from the platform service
    2. Uses an LLM to select the minimal set of tables/columns needed
    3. Identifies join paths between tables
    """

    def __init__(self) -> None:
        """Initialize the Selector agent."""
        self.llm = llm_config.get_llm()

    async def select_schema(self, state: MACSSQLState) -> Dict[str, Any]:
        """Select relevant schema for the user's question.

        Args:
            state: Current MAC-SQL workflow state

        Returns:
            Updated state dict with schema_context populated

        """
        with tracer.start_as_current_span(
            "selector_agent.select_schema",
            attributes={
                "question": state.user_question,
                "datasource_id": str(state.datasource_id),
                "tenant_id": str(state.tenant_id),
            },
        ) as span:
            start_time = time.time()

            try:
                logger.info(
                    "Selector agent starting",
                    extra={
                        "question": state.user_question,
                        "datasource_id": str(state.datasource_id),
                    },
                )

                # Step 1: Retrieve schema from platform service
                retrieval_result = await self._retrieve_schema(
                    question=state.user_question,
                    datasource_id=str(state.datasource_id),
                    tenant_id=str(state.tenant_id),
                )

                # Step 2: Use LLM to select minimal schema
                selection = await self._llm_select_schema(
                    question=state.user_question,
                    retrieval_result=retrieval_result,
                )

                # Step 3: Build SchemaContext
                schema_context = SchemaContext(
                    tables=selection.get("selected_tables_full", []),
                    columns=selection.get("selected_columns_full", []),
                    relationships=selection.get("join_paths", []),
                    example_queries=retrieval_result.get("example_queries", []),
                    reasoning=selection.get("reasoning", ""),
                )

                retrieval_time = (time.time() - start_time) * 1000

                span.set_attribute("tables_selected", len(schema_context.tables))
                span.set_attribute("columns_selected", len(schema_context.columns))

                logger.info(
                    "Selector agent completed",
                    extra={
                        "tables_selected": len(schema_context.tables),
                        "columns_selected": len(schema_context.columns),
                        "retrieval_time_ms": retrieval_time,
                    },
                )

                return {
                    "schema_context": schema_context,
                    "retrieval_time_ms": retrieval_time,
                    "llm_calls": state.llm_calls + 1,
                    "current_step": "decomposer",
                }

            except Exception as e:
                logger.error(
                    "Selector agent failed",
                    extra={
                        "error": str(e),
                        "question": state.user_question,
                    },
                )
                return {
                    "errors": [f"Selector agent error: {str(e)}"],
                    "current_step": "error",
                }

    async def _retrieve_schema(
        self,
        question: str,
        datasource_id: str,
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Retrieve schema from platform service.

        Args:
            question: User's natural language question
            datasource_id: Datasource identifier
            tenant_id: Tenant identifier

        Returns:
            Retrieval result with tables, columns, relationships

        """
        async with PlatformClient() as client:
            result = await client.retrieve_from_knowledge(
                query=question,
                datasource_id=datasource_id,
                tenant_id=tenant_id,
            )

            # Validate and parse the response
            retrieval_response = RetrievalResponse.model_validate(result)

            # Convert to dict format expected by the selector
            return {
                "tables": [
                    {"content": t.content, "qualified_name": t.metadata["qualified_name"], "metadata": t.metadata}
                    for t in retrieval_response.tables
                ],
                "columns": [
                    col
                    for table in retrieval_response.tables
                    for col in [{"content": c.content, "score": c.score, "metadata": c.metadata} for c in table.columns]
                ],
                "relationships": [
                    {
                        "source_table": rel.source_table,
                        "target_table": rel.target_table,
                        "relationship_type": rel.relationship_type,
                        "join_hint": rel.join_hint,
                        "column_mappings": rel.column_mappings,
                    }
                    for table in retrieval_response.tables
                    for rel in table.related_tables
                ],
                "example_queries": [
                    {"content": query.content, "metadata": query.metadata} for query in retrieval_response.example_queries
                ],
            }

    async def _llm_select_schema(
        self,
        question: str,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to select minimal schema from retrieved results.

        Args:
            question: User's natural language question
            retrieval_result: Schema retrieved from platform service

        Returns:
            Selected schema with reasoning

        """
        # Format the prompt
        user_prompt = format_selector_prompt(
            question=question,
            tables=retrieval_result.get("tables", []),
            columns=retrieval_result.get("columns", []),
            relationships=retrieval_result.get("relationships", []),
        )

        # Call LLM
        messages = [
            SystemMessage(content=SELECTOR_SYSTEM_PROMPT),
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

            selection = json.loads(content)

            # Enrich selection with full metadata from retrieval
            selection["selected_tables_full"] = [
                table
                for table in retrieval_result.get("tables", [])
                if table.get("metadata", {}).get("table_name") in selection.get("selected_tables", [])
            ]

            selection["selected_columns_full"] = []
            for col in retrieval_result.get("columns", []):
                for table_name, columns in selection.get("selected_columns", {}).items():
                    if (
                        col.get("metadata", {}).get("table_qualified_name", "").split(".")[1] == table_name
                        and col.get("metadata", {}).get("column_name") in columns
                    ):
                        selection["selected_columns_full"].append(col)

            return selection

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error("Failed to parse LLM response", extra={"error": str(e), "response": str(response)})
            # Fallback: return all retrieved schema
            return {
                "selected_tables": [t.get("metadata", {}).get("table_name") for t in retrieval_result.get("tables", [])],
                "selected_columns": {},
                "join_paths": retrieval_result.get("relationships", []),
                "reasoning": f"Failed to parse LLM response, using all retrieved schema. Error: {str(e)}",
                "selected_tables_full": retrieval_result.get("tables", []),
                "selected_columns_full": retrieval_result.get("columns", []),
            }
