import time
from typing import Any

from langgraph.graph import END, StateGraph
from opentelemetry import trace

from core.agents.state import AgentInput, AgentOutput, AgentState, ExecutionResult
from core.agents.steps import AnalysisAgent, OptimizerAgent, PlannerAgent, RefinerAgent, SelectorAgent
from core.integrations.platform_client import PlatformClient
from core.logging import get_logger
from core.services.agent.executor import SafeSQLExecutor
from core.services.agent.validator import get_sql_validator
from core.utils.dialect_mapper import connector_key_to_dialect

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class AgentWorkflow:
    """Agent workflow orchestrator using LangGraph with single-SQL generation.

    The workflow uses a planner to create a structured reasoning plan, then generates
    a single comprehensive SQL query that answers the entire question.
    """

    def __init__(self) -> None:
        """Initialize the Agent workflow."""
        self.optimizer = OptimizerAgent()
        self.planner = PlannerAgent()
        self.selector = SelectorAgent()
        self.refiner = RefinerAgent()
        self.analyzer = AnalysisAgent()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the Agent workflow with single-SQL generation.

        Returns:
            Compiled workflow

        """
        # Create state graph
        graph = StateGraph(AgentState)

        # Add nodes for each agent
        graph.add_node("optimizer", self._optimizer_node)
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
        graph.add_edge("optimizer", "selector")
        graph.add_edge("selector", "planner")

        # Main execution flow
        graph.add_edge("planner", "refiner")
        graph.add_edge("refiner", "validator")

        # Conditional edge from validator
        graph.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "execute": "executor",
                "retry": "refiner",
                "error": "planner",
            },
        )

        # Conditional edge from executor (simplified routing - single SQL execution)
        graph.add_conditional_edges(
            "executor",
            self._route_after_execution,
            {
                "planner": "planner",  # Need iterative planning (if execution failed)
                "analyze": "analyzer",  # Proceed to analysis
            },
        )

        # Analysis → Visualizer → Finalizer
        graph.add_edge("analyzer", "visualizer")
        graph.add_edge("visualizer", "finalizer")

        graph.add_edge("finalizer", END)

        return graph.compile()

    async def _optimizer_node(self, state: AgentState) -> AgentState:
        """Optimizer agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state with optimized query

        """
        return await self.optimizer.optimize_question(state)

    async def _selector_node(self, state: AgentState) -> AgentState:
        """Selector agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state with schema_context populated

        """
        return await self.selector.select_schema(state)

    async def _planner_node(self, state: AgentState) -> AgentState:
        """Planner agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with execution_plan

        """
        return await self.planner.create_plan(state)

    async def _analyzer_node(self, state: AgentState) -> AgentState:
        """Analyzer agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict with analysis_result

        """
        return await self.analyzer.analyze_results(state)

    async def _refiner_node(self, state: AgentState) -> AgentState:
        """Refiner agent node.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        return await self.refiner.refine_to_sql(state)

    async def _validator_node(self, state: AgentState) -> AgentState:
        """Validator node - validates the single generated SQL.

        Args:
            state: Current workflow state

        Returns:
            Updated state dict

        """
        try:
            if not state.generated_sql:
                raise Exception("No SQL generated")

            # Validate SQL
            print(state.generated_sql.sql, "!!!!!!!!!!!!!")
            validator = get_sql_validator(dialect=state.generated_sql.dialect)
            validation_result = validator.validate(
                sql=state.generated_sql.sql,
                check_readonly=True,
                check_complexity=True,
            )

            # Update generated_sql with validation results
            state.generated_sql.is_validated = validation_result.is_valid
            state.generated_sql.validation_errors = validation_result.errors

            if validation_result.is_valid:
                state.current_step = "execute"
                return state
            else:
                state.current_step = "retry"
                return state

        except Exception as e:
            logger.error("Validation node failed", extra={"error": str(e)})
            state.current_step = "error"
            return state

    async def _executor_node(self, state: AgentState) -> AgentState:
        """Executor node - executes the single generated SQL query.

        Args:
            state: Current workflow state

        Returns:
            Updated state

        """
        try:
            if not state.generated_sql or not state.generated_sql.is_validated:
                raise Exception("Cannot execute invalid SQL")

            # Execute SQL
            executor = SafeSQLExecutor(
                max_rows=state.max_rows,
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
                error_message=None,
            )

            logger.info(
                "SQL execution completed",
                extra={"rows_returned": execution_result.rows_returned, "execution_time_ms": execution_result.execution_time_ms},
            )

            # Store the single execution result
            state.execution_result = execution_result
            state.current_step = "analyze"
            return state

        except Exception as e:
            logger.error("Execution node failed", extra={"error": str(e)})
            execution_result = ExecutionResult(
                success=False,
                rows_returned=0,
                execution_time_ms=0.0,
                error_message=str(e),
            )

            # Store the failed result
            state.execution_result = execution_result
            state.execution_plan.requires_iteration = True  # type: ignore
            state.current_step = "planner"

            return state

    async def _visualizer_node(self, state: AgentState) -> AgentState:
        """Visualizer node to generate chart specifications.

        Args:
            state: Current workflow state

        Returns:
            Updated state with visualization_spec

        """
        try:
            # Check if we have execution result and data
            if not state.execution_result or not state.execution_result.rows:
                logger.info("Skipping visualization - no data available")
                state.current_step = "finalize"
                return state

            # Import here to avoid circular dependencies
            from core.services.agent.visualization.visualizer_service import VisualizationService

            viz_service = VisualizationService()

            # Generate visualization spec
            viz_spec_dict = await viz_service.generate_visualization(
                execution_result=state.execution_result,
                sql=state.generated_sql.sql if state.generated_sql else "",
                question=state.optimized_question or state.user_question,
            )

            if viz_spec_dict:
                # Convert dict to VisualizationSpec model
                from core.agents.state import VisualizationSpec

                viz_spec = VisualizationSpec(**viz_spec_dict)
                logger.info(
                    "Visualization generated",
                    chart_type=viz_spec.chart_type,
                    generation_method=viz_spec.generation_method,
                )
                state.visualization_spec = viz_spec
                state.current_step = "finalize"
                return state
            else:
                logger.warning("Visualization generation returned None")
                state.current_step = "finalize"
                return state

        except Exception as e:
            logger.error("Visualizer node failed", extra={"error": str(e)})
            # Don't fail the workflow - continue without visualization
            state.current_step = "finalize"
            return state

    async def _finalizer_node(self, state: AgentState) -> AgentState:
        """Finalizer node to prepare output.

        Args:
            state: Current workflow state

        Returns:
            Updated state

        """
        total_time = state.retrieval_time_ms
        if state.execution_result:
            total_time += state.execution_result.execution_time_ms

        state.total_time_ms = total_time
        state.current_step = "completed"
        return state

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

        else:
            return "execute"

    def _route_after_execution(self, state: AgentState) -> str:
        """Route workflow after execution (simplified for single SQL execution).

        Args:
            state: Current workflow state

        Returns:
            Next node name:
            - "planner": Need iterative planning based on results (if execution failed)
            - "analyze": Proceed to analysis (default)

        """
        if state.current_step == "planner":
            return "planner"
        else:
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
            attributes={"question": input_data.question, "datasource_id": str(input_data.datasource_id)},
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
                    max_rows=input_data.max_rows,
                    dialect=dialect,
                )

                # Run workflow
                final_state = await self.workflow.ainvoke(initial_state)

                execution_time = (time.time() - start_time) * 1000
                span.set_attribute("execution_time_ms", execution_time)

                # Extract data from final state (Pydantic models need attribute access)
                generated_sql = final_state["generated_sql"]
                execution_result = final_state["execution_result"]
                analysis_result = final_state["analysis_result"]
                visualization_spec = final_state["visualization_spec"]

                agent_output = AgentOutput(
                    sql=generated_sql.sql if generated_sql else None,
                    data=execution_result.rows if execution_result else None,
                    num_rows=execution_result.rows_returned if execution_result else 0,
                    analysis=analysis_result.model_dump() if analysis_result else None,
                    insights=analysis_result.insights if analysis_result else None,
                    answer=analysis_result.answer if analysis_result else None,
                    visualization_spec=visualization_spec.model_dump() if visualization_spec else None,
                    total_time_ms=execution_time,
                    llm_calls=final_state["llm_calls"],
                    success=True,
                    error_message=None,
                )

                logger.info(
                    "Agent workflow completed",
                    extra={
                        "execution_time_ms": execution_time,
                        "llm_calls": final_state["llm_calls"],
                    },
                )

                return agent_output

            except Exception as e:
                logger.error("Agent workflow failed", extra={"error": str(e)})
                span.set_attribute("error", True)

                # Return error output
                return AgentOutput(
                    success=False,
                    error_message=str(e),
                    total_time_ms=(time.time() - start_time) * 1000,
                )
