import time
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from opentelemetry import trace

from core.agents.sql.analyzer import AnalysisAgent
from core.agents.sql.classifier import ClassifierAgent
from core.agents.sql.optimizer import OptimizerAgent
from core.agents.sql.planner import PlannerAgent
from core.agents.sql.refiner import RefinerAgent
from core.agents.sql.selector import SelectorAgent
from core.agents.sql.state import AgentInput, AgentOutput, AgentState, ExecutionResult, QueryStepStatus
from core.integrations.platform_client import PlatformClient
from core.logging import get_logger
from core.services.agent.executor import SafeSQLExecutor
from core.services.agent.validator import get_sql_validator
from core.utils.dialect_mapper import connector_key_to_dialect

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AgentWorkflow:
    """Agent workflow orchestrator using LangGraph with multi-query support.

    New multi-query mode uses classifier + planner + iterative execution + analysis.
    """

    def __init__(self) -> None:
        """Initialize the Agent workflow."""
        self.optimizer = OptimizerAgent()
        self.classifier = ClassifierAgent()
        self.planner = PlannerAgent()
        self.selector = SelectorAgent()
        self.refiner = RefinerAgent()
        self.analyzer = AnalysisAgent()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the LangGraph workflow with multi-query support.

        New workflow:
        Optimizer → Classifier → Planner → Selector → Refiner → Validator → Executor
                                   ↑                                           ↓
                                   └──────────── (iteration) ─────────────────┘
                                                                               ↓
                                                                           Analysis → Visualizer → Finalizer

        Returns:
            Compiled LangGraph workflow

        """
        # Create state graph
        graph = StateGraph(AgentState)

        # Add nodes for each agent
        graph.add_node("optimizer", self._optimizer_node)
        graph.add_node("classifier", self._classifier_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("selector", self._selector_node)
        graph.add_node("refiner", self._refiner_node)
        graph.add_node("validator", self._validator_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("analyzer", self._analyzer_node)
        graph.add_node("visualizer", self._visualizer_node)
        graph.add_node("finalizer", self._finalizer_node)

        # Define workflow edges
        graph.set_entry_point("optimizer")
        graph.add_edge("optimizer", "classifier")
        graph.add_edge("classifier", "planner")
        graph.add_edge("planner", "selector")

        # Main execution flow
        graph.add_edge("selector", "refiner")
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

        # Conditional edge from executor (complex routing)
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "retry": "refiner",
                "next_step": "selector",  # More steps in current plan
                "planner": "planner",  # Need iterative planning
                "analyze": "analyzer",  # All queries done, analyze results
            },
        )

        # Analysis → Visualizer → Finalizer
        graph.add_edge("analyzer", "visualizer")
        graph.add_edge("visualizer", "finalizer")

        graph.add_edge("finalizer", END)

        return graph.compile()

    async def _optimizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Optimizer agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.optimizer.optimize_question(state)

    async def _classifier_node(self, state: AgentState) -> Dict[str, Any]:
        """Classifier agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with classification

        """
        return await self.classifier.classify_question(state)

    async def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Planner agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with execution_plan

        """
        return await self.planner.create_plan(state)

    async def _analyzer_node(self, state: AgentState) -> Dict[str, Any]:
        """Analyzer agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with analysis_result

        """
        return await self.analyzer.analyze_results(state)

    async def _selector_node(self, state: AgentState) -> Dict[str, Any]:
        """Selector agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.selector.select_schema(state)

    async def _refiner_node(self, state: AgentState) -> Dict[str, Any]:
        """Refiner agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.refiner.refine_to_sql(state)

    async def _validator_node(self, state: AgentState) -> Dict[str, Any]:
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

    async def _executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Executor node - handles both single and multi-query execution.

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

            # Add to query_results list (for multi-query workflow)
            updated_results = list(state.query_results) + [execution_result]

            # Update execution plan if present (mark current step as completed)
            state_updates: Dict[str, Any] = {
                "execution_result": execution_result,
                "query_results": updated_results,
                "current_step": "finalize",
            }

            if state.execution_plan:
                # Mark current step as completed and advance to next step
                updated_plan = state.execution_plan.model_copy()
                current_idx = updated_plan.current_step_index
                if current_idx < len(updated_plan.steps):
                    updated_plan.steps[current_idx].status = QueryStepStatus.COMPLETED
                    updated_plan.steps[current_idx].execution_result = execution_result
                    updated_plan.steps[current_idx].generated_sql = state.generated_sql

                # Advance to next step
                next_idx = current_idx + 1
                if next_idx < len(updated_plan.steps):
                    updated_plan.current_step_index = next_idx
                    updated_plan.steps[next_idx].status = QueryStepStatus.IN_PROGRESS
                else:
                    # All steps complete
                    updated_plan.is_complete = True

                state_updates["execution_plan"] = updated_plan

            return state_updates

        except Exception as e:
            logger.error("Execution node failed", extra={"error": str(e)})
            execution_result = ExecutionResult(
                success=False,
                rows_returned=0,
                execution_time_ms=0.0,
                error_message=str(e),
            )

            # Add failed result to query_results
            updated_results = list(state.query_results) + [execution_result]

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
                    "query_results": updated_results,
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
                    "query_results": updated_results,
                    "current_step": "error",
                    "errors": [error_msg],
                }

    async def _visualizer_node(self, state: AgentState) -> Dict[str, Any]:
        """Visualizer node to generate chart specifications.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with visualization_spec

        """
        try:
            # Determine which result to visualize (support both single and multi-query)
            result_to_visualize = None

            # Prefer the last successful result from query_results if available
            if state.query_results:
                result_to_visualize = next((r for r in reversed(state.query_results) if r.success and r.rows), None)
            # Fall back to execution_result for backward compatibility
            elif state.execution_result and state.execution_result.success:
                result_to_visualize = state.execution_result

            if not result_to_visualize or not result_to_visualize.rows:
                logger.info("Skipping visualization - no data available")
                return {"current_step": "finalize"}

            # Import here to avoid circular dependencies
            from core.services.agent.visualization.visualizer_service import VisualizationService

            viz_service = VisualizationService()

            # Generate visualization spec
            viz_spec_dict = await viz_service.generate_visualization(
                execution_result=result_to_visualize,
                sql=state.generated_sql.sql if state.generated_sql else "",
                question=state.optimized_question or state.user_question,
            )

            if viz_spec_dict:
                # Convert dict to VisualizationSpec model
                from core.agents.sql.state import VisualizationSpec

                viz_spec = VisualizationSpec(**viz_spec_dict)
                logger.info(
                    "Visualization generated",
                    chart_type=viz_spec.chart_type,
                    generation_method=viz_spec.generation_method,
                )
                return {
                    "visualization_spec": viz_spec,
                    "current_step": "finalize",
                }
            else:
                logger.warning("Visualization generation returned None")
                return {"current_step": "finalize"}

        except Exception as e:
            logger.error("Visualizer node failed", extra={"error": str(e)})
            # Don't fail the workflow - continue without visualization
            return {"current_step": "finalize"}

    async def _finalizer_node(self, state: AgentState) -> Dict[str, Any]:
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

    def _route_after_validation(self, state: AgentState) -> str:
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

    def _route_after_execution(self, state: AgentState) -> str:
        """Route workflow after execution with multi-query support.

        Args:
            state: Current workflow state

        Returns:
            Next node name:
            - "retry": Go back to refiner for SQL fix
            - "next_step": More steps in current plan, go to selector
            - "planner": Need iterative planning based on results
            - "analyze": All queries done, proceed to analysis

        """
        # Check for retry first
        if state.current_step == "retry":
            return "retry"

        # If no execution plan, go straight to analyze (legacy mode or single query)
        if not state.execution_plan:
            return "analyze"

        execution_plan = state.execution_plan

        # Move to next step
        next_step_index = execution_plan.current_step_index + 1

        # Check if there are more steps in the current plan
        if next_step_index < len(execution_plan.steps):
            logger.info(
                "More query steps remaining",
                extra={
                    "current_step": execution_plan.current_step_index,
                    "next_step": next_step_index,
                    "total_steps": len(execution_plan.steps),
                },
            )
            # Update current_step_index and go to selector for next step
            # Note: We can't update state here, so we'll update in the calling node
            # For now, return the routing decision
            return "next_step"

        # All planned steps are complete
        # Check if plan requires iteration (may need more queries based on results)
        if execution_plan.requires_iteration:
            logger.info(
                "Execution plan requires iteration",
                extra={
                    "completed_steps": len(execution_plan.steps),
                    "query_results": len(state.query_results),
                },
            )
            return "planner"

        # All done, proceed to analysis
        logger.info(
            "All query steps completed, proceeding to analysis",
            extra={
                "total_steps": len(execution_plan.steps),
                "total_results": len(state.query_results),
            },
        )
        return "analyze"

    async def run(self, input_data: AgentInput) -> AgentOutput:
        """Run the workflow.

        Args:
            input_data: Input parameters

        Returns:
            Agent output

        """
        with tracer.start_as_current_span(
            "agent_workflow.run",
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
                initial_state = AgentState(
                    user_question=input_data.question,
                    datasource_id=input_data.datasource_id,
                    datasource=datasource,
                    tenant_id=input_data.tenant_id,
                    session_id=input_data.session_id,
                    chat_history=input_data.chat_history,
                    explain_mode=input_data.explain_mode,
                    use_cache=input_data.use_cache,
                    timeout_seconds=input_data.timeout_seconds,
                    max_rows=input_data.max_rows,
                    dialect=dialect,
                )

                # Run workflow
                final_state_dict = await self.workflow.ainvoke(initial_state)

                # LangGraph returns state as dict, need to reconstruct
                final_state = AgentState(**final_state_dict)

                print(final_state)
                # Build output
                output = self._build_output(final_state)

                execution_time = (time.time() - start_time) * 1000
                span.set_attribute("execution_time_ms", execution_time)
                span.set_attribute("success", output.success)

                logger.info(
                    "Agent workflow completed",
                    extra={
                        "success": output.success,
                        "execution_time_ms": execution_time,
                        "llm_calls": final_state.llm_calls,
                    },
                )

                return output

            except Exception as e:
                logger.error("Agent workflow failed", extra={"error": str(e)})
                span.set_attribute("error", True)

                # Return error output
                return AgentOutput(
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

    def _build_output(self, state: AgentState) -> AgentOutput:
        """Build output from final state with support for multi-query workflow.

        Args:
            state: Final workflow state

        Returns:
            Agent output

        """
        # Check if workflow completed successfully
        if state.current_step == "error" or state.errors:
            return AgentOutput(
                sql=state.generated_sql.sql if state.generated_sql else "",
                dialect=state.generated_sql.dialect if state.generated_sql else "postgres",
                question_type=state.classification.question_type.value if state.classification else None,
                classification_confidence=state.classification.confidence if state.classification else None,
                data=None,
                rows_returned=0,
                schema_selection_reasoning=None,
                planning_reasoning=None,
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

        # Log state for debugging
        logger.info(
            "Building output from state",
            extra={
                "has_classification": state.classification is not None,
                "has_execution_plan": state.execution_plan is not None,
                "num_query_results": len(state.query_results) if state.query_results else 0,
                "has_analysis_result": state.analysis_result is not None,
                "has_execution_result": state.execution_result is not None,
            },
        )

        # Determine if this was a multi-query workflow
        is_multi_query = state.execution_plan is not None and len(state.query_results) > 1

        # Collect all executed SQL queries
        all_queries = []
        if is_multi_query and state.execution_plan:
            for step in state.execution_plan.steps:
                if step.generated_sql and step.generated_sql.sql:
                    all_queries.append(step.generated_sql.sql)
        elif state.generated_sql and state.generated_sql.sql:
            all_queries.append(state.generated_sql.sql)

        # Build successful output
        output = AgentOutput(
            sql=all_queries[-1] if all_queries else "",  # Last query as primary
            all_queries=all_queries if len(all_queries) > 1 else None,
            dialect=state.generated_sql.dialect if state.generated_sql else "postgres",
            question_type=state.classification.question_type.value if state.classification else None,
            classification_confidence=state.classification.confidence if state.classification else None,
            data=None,  # Will be set below
            rows_returned=0,  # Will be set below
            num_queries_executed=len(state.query_results) if state.query_results else 1,
            schema_selection_reasoning=None,
            planning_reasoning=None,
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

        # Add execution results
        if state.query_results:
            # For multi-query, use last successful result for primary data
            last_successful = next((r for r in reversed(state.query_results) if r.success), None)
            if last_successful:
                output.rows_returned = last_successful.rows_returned
                output.cached = last_successful.cached
                output.data = last_successful.rows

            # Include all results if multiple queries
            if len(state.query_results) > 1:
                output.all_results = [
                    {
                        "success": r.success,
                        "rows_returned": r.rows_returned,
                        "data": r.rows if r.success else None,
                        "error": r.error_message,
                    }
                    for r in state.query_results
                ]
        elif state.execution_result and state.execution_result.success:
            # Legacy single query result
            output.rows_returned = state.execution_result.rows_returned
            output.cached = state.execution_result.cached
            output.data = state.execution_result.rows

        # Add analysis result if available
        if state.analysis_result:
            logger.info(
                "Adding analysis to output",
                extra={
                    "has_insights": len(state.analysis_result.insights) > 0,
                    "has_answer": bool(state.analysis_result.answer),
                },
            )
            output.analysis = {
                "answer": state.analysis_result.answer,
                "confidence": state.analysis_result.confidence,
            }
            output.insights = state.analysis_result.insights
            output.answer = state.analysis_result.answer
        else:
            logger.warning("No analysis_result in state - analysis fields will be empty")

        # Add visualization spec if available
        if state.visualization_spec:
            output.visualization_spec = state.visualization_spec.model_dump()

        # Add reasoning if explain mode
        if state.explain_mode:
            output.schema_selection_reasoning = state.schema_context.reasoning if state.schema_context else None
            output.classification_reasoning = state.classification.reasoning if state.classification else None
            output.planning_reasoning = state.execution_plan.reasoning if state.execution_plan else None
            output.decomposition_reasoning = state.query_plan.reasoning if state.query_plan else None
            output.refinement_reasoning = state.generated_sql.reasoning if state.generated_sql else None
            output.analysis_reasoning = state.analysis_result.reasoning if state.analysis_result else None

        return output
