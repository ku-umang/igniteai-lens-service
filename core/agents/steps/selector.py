import json
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace

from core.agents.prompts.selector import (
    SELECTOR_SYSTEM_PROMPT,
    format_selector_prompt,
)
from core.agents.state import AgentState, SchemaContext
from core.integrations.platform_client import PlatformClient
from core.integrations.schema import RetrievalResponse
from core.llm_config import llm_config
from core.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class SelectorAgent:
    """Selector agent for schema selection in workflow.

    This agent:
    1. Retrieves relevant schema from the platform service
    2. Uses an LLM to select the minimal set of tables/columns needed
    3. Identifies join paths between tables
    """

    def __init__(self) -> None:
        """Initialize the Selector agent."""
        self.llm = llm_config.get_llm()

    async def select_schema(self, state: AgentState) -> AgentState:
        """Select relevant schema for the user's question.

        Args:
            state: Current workflow state

        Returns:
            Updated state with schema_context populated

        """
        # Use optimized question if available, otherwise fall back to original
        question_to_use = state.optimized_question or state.user_question

        with tracer.start_as_current_span(
            "selector_agent.select_schema",
            attributes={
                "question": question_to_use,
                "datasource_id": str(state.datasource_id),
            },
        ) as span:
            start_time = time.time()

            try:
                logger.info("Selector agent starting")

                # Step 1: Retrieve schema from platform service
                retrieval_result = await self._retrieve_schema(
                    question=question_to_use,
                    datasource_id=str(state.datasource_id),
                    tenant_id=str(state.tenant_id),
                )

                # Step 2: Use LLM to select minimal schema
                selection_result = await self._llm_select_schema(
                    question=question_to_use,
                    retrieval_result=retrieval_result,
                )

                # Step 3: Build SchemaContext
                schema_context = SchemaContext(
                    tables=selection_result["selected_tables"],
                    columns=selection_result["selected_columns"],
                    relationships=selection_result["relationships"],
                    example_queries=retrieval_result["example_queries"],
                    reasoning=selection_result["reasoning"],
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

                state.schema_context = schema_context
                state.total_time_ms = state.total_time_ms + retrieval_time
                state.llm_calls = state.llm_calls + 1
                state.current_step = "schema_selector"
                return state

            except Exception as e:
                logger.error(
                    "Selector agent failed",
                    extra={
                        "error": str(e),
                        "question": state.user_question,
                    },
                )
                raise Exception(f"Selector agent error: {str(e)}") from e

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
            tables = []
            columns = []
            relationships = []

            for table in retrieval_response.tables:
                tables.append(
                    {
                        "content": table.content,
                        "table_qualified_name": table.metadata["qualified_name"],
                        "metadata": table.metadata,
                    }
                )
                for column in table.columns:
                    columns.append(
                        {
                            "content": column.content,
                            "table_qualified_name": column.metadata["table_qualified_name"],
                            "metadata": column.metadata,
                        }
                    )
                for relationship in table.related_tables:
                    relationships.append(
                        {
                            "source_table": relationship.source_table,
                            "target_table": relationship.target_table,
                            "relationship_type": relationship.relationship_type,
                            "join_hint": relationship.join_hint,
                            "column_mappings": relationship.column_mappings,
                        }
                    )

            return {
                "tables": tables,
                "columns": columns,
                "relationships": relationships,
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

            # Fix improperly escaped single quotes in JSON strings (\' -> ')
            content = content.replace(r"\'", "'")

            selection = json.loads(content)

            # Enrich selection with full metadata from retrieval
            selection["selected_tables"] = [
                table
                for table in retrieval_result.get("tables", [])
                if table["table_qualified_name"] in selection.get("selected_tables", [])
            ]

            selected_columns = []
            for col in retrieval_result.get("columns", []):
                for table_qualified_name, columns in selection.get("selected_columns", {}).items():
                    if col["table_qualified_name"] == table_qualified_name and col["metadata"]["column_name"] in columns:
                        selected_columns.append(col)

            selection["selected_columns"] = selected_columns

            return selection

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error("Failed to parse LLM response", extra={"error": str(e), "response": str(response)})
            raise Exception(f"Failed to select relevant schema: {str(e)}") from e
