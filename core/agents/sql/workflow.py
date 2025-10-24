"""LangGraph workflow for MAC-SQL agent system.

This module defines the workflow that orchestrates the three MAC-SQL agents:
Selector -> Decomposer -> Refiner -> Executor
"""

import time
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from opentelemetry import trace

from core.agents.sql.decomposer import DecomposerAgent
from core.agents.sql.refiner import RefinerAgent
from core.agents.sql.selector import SelectorAgent
from core.agents.sql.state import ExecutionResult, MACSSQLInput, MACSSQLOutput, MACSSQLState
from core.integrations.platform_client import PlatformClient
from core.logging import get_logger
from core.services.sql.executor import SafeSQLExecutor
from core.services.sql.validator import get_sql_validator
from core.utils.dialect_mapper import connector_key_to_dialect

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class MACSSQLWorkflow:
    """MAC-SQL workflow orchestrator using LangGraph."""

    def __init__(self) -> None:
        """Initialize the MAC-SQL workflow."""
        self.selector = SelectorAgent()
        self.decomposer = DecomposerAgent()
        self.refiner = RefinerAgent()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow.

        Returns:
            Compiled LangGraph workflow

        """
        # Create state graph
        graph = StateGraph(MACSSQLState)

        # Add nodes for each agent
        graph.add_node("selector", self._selector_node)
        graph.add_node("decomposer", self._decomposer_node)
        graph.add_node("refiner", self._refiner_node)
        graph.add_node("validator", self._validator_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("finalizer", self._finalizer_node)

        # Define workflow edges
        graph.set_entry_point("selector")
        graph.add_edge("selector", "decomposer")
        graph.add_edge("decomposer", "refiner")
        graph.add_edge("refiner", "validator")

        # Conditional edge from validator
        graph.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "execute": "executor",
                "explain": "finalizer",
                "retry": "refiner",
                "error": "finalizer",
            },
        )

        # Conditional edge from executor
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "retry": "refiner",
                "finalize": "finalizer",
            },
        )

        graph.add_edge("finalizer", END)

        return graph.compile()

    async def _selector_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Selector agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.selector.select_schema(state)

    async def _decomposer_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Decomposer agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.decomposer.decompose_query(state)

    async def _refiner_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Refiner agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.refiner.refine_to_sql(state)

    async def _validator_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Validator node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        try:
            if not state.generated_sql or not state.generated_sql.sql:
                return {
                    "errors": ["No SQL generated"],
                    "current_step": "error",
                }

            # Validate SQL
            validator = get_sql_validator(dialect=state.generated_sql.dialect)
            validation_result = validator.validate(
                sql=state.generated_sql.sql,
                check_readonly=True,
                check_complexity=True,
            )

            # Update generated_sql with validation results
            updated_sql = state.generated_sql.model_copy()
            updated_sql.is_valid = validation_result.is_valid
            updated_sql.validation_errors = validation_result.errors

            if validation_result.is_valid:
                logger.info(
                    "SQL validation passed",
                    extra={
                        "complexity": validation_result.complexity.get("score", 0),
                        "warnings": len(validation_result.warnings),
                    },
                )
                return {
                    "generated_sql": updated_sql,
                    "current_step": "execute" if not state.explain_mode else "explain",
                }
            else:
                logger.warning(
                    "SQL validation failed",
                    extra={"errors": validation_result.errors},
                )

                # Retry logic (max iterations)
                if state.iteration_count < state.max_iterations:
                    return {
                        "generated_sql": updated_sql,
                        "iteration_count": state.iteration_count + 1,
                        "current_step": "retry",
                        "errors": validation_result.errors,
                    }
                else:
                    return {
                        "generated_sql": updated_sql,
                        "current_step": "error",
                        "errors": validation_result.errors,
                    }

        except Exception as e:
            logger.error("Validation node failed", extra={"error": str(e)})
            return {
                "errors": [f"Validation error: {str(e)}"],
                "current_step": "error",
            }

    async def _executor_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Executor node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        try:
            if not state.generated_sql or not state.generated_sql.is_valid:
                return {
                    "errors": ["Cannot execute invalid SQL"],
                    "current_step": "error",
                }

            # Execute SQL
            executor = SafeSQLExecutor(
                max_rows=state.max_rows,
                timeout_seconds=state.timeout_seconds,
                use_cache=state.use_cache,
            )

            result = await executor.execute(
                sql=state.generated_sql.sql,
                datasource_id=state.datasource_id,
                datasource=state.datasource,
                tenant_id=state.tenant_id,
                dialect=state.generated_sql.dialect,
                validate=False,  # Already validated
            )

            execution_result = ExecutionResult(
                success=result["success"],
                rows=result["data"],
                truncated=result["truncated"],
                rows_returned=result["rows_returned"],
                execution_time_ms=result["execution_time_ms"],
                cached=result.get("cached", False),
                error_message=None,
            )

            logger.info(
                "SQL execution completed",
                extra={
                    "rows_returned": execution_result.rows_returned,
                    "execution_time_ms": execution_result.execution_time_ms,
                    "cached": execution_result.cached,
                },
            )

            return {
                "execution_result": execution_result,
                "current_step": "finalize",
            }

        except Exception as e:
            logger.error("Execution node failed", extra={"error": str(e)})
            execution_result = ExecutionResult(
                success=False,
                rows_returned=0,
                execution_time_ms=0.0,
                error_message=str(e),
            )

            # Retry logic - allow refiner to attempt fix if iterations remain
            error_msg = f"Execution error: {str(e)}"
            if state.iteration_count < state.max_iterations:
                logger.info(
                    "Execution failed, routing to refiner for retry",
                    extra={
                        "iteration": state.iteration_count + 1,
                        "max_iterations": state.max_iterations,
                        "error": str(e),
                    },
                )
                return {
                    "execution_result": execution_result,
                    "iteration_count": state.iteration_count + 1,
                    "current_step": "retry",
                    "errors": [error_msg],
                }
            else:
                logger.warning(
                    "Execution failed, max iterations reached",
                    extra={
                        "iteration": state.iteration_count,
                        "max_iterations": state.max_iterations,
                        "error": str(e),
                    },
                )
                return {
                    "execution_result": execution_result,
                    "current_step": "error",
                    "errors": [error_msg],
                }

    async def _finalizer_node(self, state: MACSSQLState) -> Dict[str, Any]:
        """Finalizer node to prepare output.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        total_time = sum(
            [
                state.retrieval_time_ms,
                state.execution_result.execution_time_ms if state.execution_result else 0,
            ]
        )

        return {
            "total_time_ms": total_time,
            "current_step": "completed",
        }

    def _route_after_validation(self, state: MACSSQLState) -> str:
        """Route workflow after validation.

        Args:
            state: Current workflow state

        Returns:
            Next node name

        """
        if state.current_step == "error":
            return "error"
        elif state.current_step == "retry":
            return "retry"
        elif state.explain_mode or state.current_step == "explain":
            return "explain"
        else:
            return "execute"

    def _route_after_execution(self, state: MACSSQLState) -> str:
        """Route workflow after execution.

        Args:
            state: Current workflow state

        Returns:
            Next node name - either "retry" (to refiner) or "finalize"

        """
        if state.current_step == "retry":
            return "retry"
        else:
            return "finalize"

    async def run(self, input_data: MACSSQLInput) -> MACSSQLOutput:
        """Run the MAC-SQL workflow.

        Args:
            input_data: Input parameters

        Returns:
            MAC-SQL output

        """
        with tracer.start_as_current_span(
            "macsql_workflow.run",
            attributes={
                "question": input_data.question,
                "datasource_id": str(input_data.datasource_id),
                "explain_mode": input_data.explain_mode,
            },
        ) as span:
            start_time = time.time()

            try:
                # Fetch datasource to get dialect information
                logger.info(
                    "Fetching datasource for dialect resolution",
                    extra={"datasource_id": str(input_data.datasource_id)},
                )

                dialect = "postgres"  # Default fallback
                try:
                    async with PlatformClient() as client:
                        datasource = await client.get_datasource(
                            datasource_id=input_data.datasource_id,
                            tenant_id=input_data.tenant_id,
                        )
                        dialect = connector_key_to_dialect(datasource.connector_key)
                        logger.info(
                            "Resolved SQL dialect from datasource",
                            extra={
                                "connector_key": datasource.connector_key,
                                "dialect": dialect,
                            },
                        )
                except Exception as e:
                    logger.error(
                        "Failed to fetch datasource, using default dialect",
                        extra={"error": str(e), "default_dialect": dialect},
                    )
                    raise e

                # Initialize state
                initial_state = MACSSQLState(
                    user_question=input_data.question,
                    datasource_id=input_data.datasource_id,
                    datasource=datasource,
                    tenant_id=input_data.tenant_id,
                    session_id=input_data.session_id,
                    explain_mode=input_data.explain_mode,
                    use_cache=input_data.use_cache,
                    timeout_seconds=input_data.timeout_seconds,
                    max_rows=input_data.max_rows,
                    dialect=dialect,
                )

                # Run workflow
                final_state_dict = await self.workflow.ainvoke(initial_state)

                # Convert dict back to MACSSQLState object
                # LangGraph returns state as dict, need to reconstruct
                final_state = MACSSQLState(**final_state_dict)

                # Build output
                output = self._build_output(final_state)

                execution_time = (time.time() - start_time) * 1000
                span.set_attribute("execution_time_ms", execution_time)
                span.set_attribute("success", output.success)

                logger.info(
                    "MAC-SQL workflow completed",
                    extra={
                        "success": output.success,
                        "execution_time_ms": execution_time,
                        "llm_calls": final_state.llm_calls,
                    },
                )

                return output

            except Exception as e:
                logger.error("MAC-SQL workflow failed", extra={"error": str(e)})
                span.set_attribute("error", True)

                # Return error output
                return MACSSQLOutput(
                    sql="",
                    dialect="postgres",
                    data=None,
                    rows_returned=0,
                    schema_selection_reasoning=None,
                    decomposition_reasoning=None,
                    refinement_reasoning=None,
                    execution_time_ms=0.0,
                    cached=False,
                    complexity_score=0.0,
                    is_valid=False,
                    validation_errors=[],
                    success=False,
                    error_message=f"Workflow failed: {str(e)}",
                )

    def _build_output(self, state: MACSSQLState) -> MACSSQLOutput:
        """Build output from final state.

        Args:
            state: Final workflow state

        Returns:
            MAC-SQL output

        """
        # Check if workflow completed successfully
        if state.current_step == "error" or state.errors:
            return MACSSQLOutput(
                sql=state.generated_sql.sql if state.generated_sql else "",
                dialect=state.generated_sql.dialect if state.generated_sql else "postgres",
                data=None,
                rows_returned=0,
                schema_selection_reasoning=None,
                decomposition_reasoning=None,
                refinement_reasoning=None,
                execution_time_ms=state.total_time_ms,
                cached=False,
                complexity_score=state.query_plan.complexity_score if state.query_plan else 0.0,
                is_valid=False,
                validation_errors=state.generated_sql.validation_errors if state.generated_sql else [],
                success=False,
                error_message="; ".join(state.errors) if state.errors else "Unknown error",
            )

        # Build successful output
        output = MACSSQLOutput(
            sql=state.generated_sql.sql if state.generated_sql else "",
            dialect=state.generated_sql.dialect if state.generated_sql else "postgres",
            data=None,  # Will be set below if execution result available
            rows_returned=0,  # Will be set below if execution result available
            schema_selection_reasoning=None,
            decomposition_reasoning=None,
            refinement_reasoning=None,
            execution_time_ms=state.total_time_ms,
            cached=False,
            complexity_score=state.query_plan.complexity_score if state.query_plan else 0.0,
            is_valid=state.generated_sql.is_valid if state.generated_sql else False,
            validation_errors=state.generated_sql.validation_errors if state.generated_sql else [],
            success=True,
            error_message=None,
        )

        # Add execution results if available
        if state.execution_result and state.execution_result.success:
            # Note: Actual data would come from executor, not stored in state
            output.rows_returned = state.execution_result.rows_returned
            output.cached = state.execution_result.cached
            output.data = state.execution_result.rows

        # Add reasoning if explain mode
        if state.explain_mode:
            output.schema_selection_reasoning = state.schema_context.reasoning if state.schema_context else None
            output.decomposition_reasoning = state.query_plan.reasoning if state.query_plan else None
            output.refinement_reasoning = state.generated_sql.reasoning if state.generated_sql else None

        return output
