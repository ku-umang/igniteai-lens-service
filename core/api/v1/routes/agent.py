"""MAC-SQL agent API endpoints.

This module provides REST API endpoints for the MAC-SQL agent system:
- Generate SQL from natural language
- Execute SQL and return results
- Explain SQL generation reasoning
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from opentelemetry import trace

from core.api.v1.schemas.agent import (
    AgentReasoningResponse,
    ExecuteSQLRequest,
    ExecuteSQLResponse,
    GenerateSQLRequest,
    GenerateSQLResponse,
    ValidationResponse,
)
from core.dependencies.agent import AgentServiceDep
from core.dependencies.session import get_session_service, verify_session_ownership_helper
from core.logging import get_logger
from core.security.auth import get_current_tenant_id, get_current_user_id
from core.services.session.session_service import SessionService

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

router = APIRouter(prefix="/agent", tags=["MAC-SQL Agent"])


@router.post(
    "/generate",
    response_model=GenerateSQLResponse,
    summary="Generate SQL from natural language",
    description="Generate SQL query from a natural language question using MAC-SQL agents",
)
async def generate_sql(
    request: GenerateSQLRequest,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    agent_service: AgentServiceDep,
) -> GenerateSQLResponse:
    """Generate SQL from natural language question.

    Args:
        request: SQL generation request
        tenant_id: Tenant identifier from auth
        user_id: User identifier from auth
        session_service: Session service instance
        agent_service: SQL service instance

    Returns:
        Generated SQL with metadata

    Raises:
        HTTPException: If generation fails

    """
    with tracer.start_as_current_span(
        "agent.generate_sql",
        attributes={
            "tenant_id": str(tenant_id),
            "session_id": str(request.session_id),
            "explain_mode": request.explain_mode,
        },
    ):
        # Validate and fetch session using existing helper
        try:
            session = await verify_session_ownership_helper(
                session_id=request.session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                session_service=session_service,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found or access denied: {str(e)}",
            ) from e

        try:
            logger.info(
                "SQL generation requested",
                extra={
                    "tenant_id": str(tenant_id),
                    "datasource_id": str(session.datasource_id),
                    "session_id": str(session.id),
                    "question_length": len(request.question),
                },
            )

            # Generate SQL using datasource from session
            output = await agent_service.generate_sql(
                question=request.question,
                datasource_id=session.datasource_id,
                tenant_id=tenant_id,
                session_id=str(request.session_id),
                explain_mode=request.explain_mode,
                use_cache=request.use_cache,
                timeout_seconds=request.timeout_seconds,
                max_rows=request.max_rows,
            )

            # Build response
            reasoning = None
            if request.explain_mode:
                reasoning = AgentReasoningResponse(
                    schema_selection=output.schema_selection_reasoning,
                    query_decomposition=output.decomposition_reasoning,
                    sql_refinement=output.refinement_reasoning,
                )

            response = GenerateSQLResponse(
                sql=output.sql,
                dialect=output.dialect,
                complexity_score=output.complexity_score,
                execution_time_ms=output.execution_time_ms,
                validation=ValidationResponse(
                    is_valid=output.is_valid,
                    errors=output.validation_errors,
                ),
                reasoning=reasoning,
                success=output.success,
                error_message=output.error_message,
            )

            logger.info(
                "SQL generation completed",
                extra={
                    "success": response.success,
                    "complexity_score": response.complexity_score,
                },
            )

            return response

        except Exception as e:
            logger.error(
                "SQL generation failed",
                extra={
                    "error": str(e),
                    "tenant_id": str(tenant_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SQL generation failed: {str(e)}",
            ) from e


@router.post(
    "/execute",
    response_model=ExecuteSQLResponse,
    summary="Generate and execute SQL",
    description="Generate SQL from natural language and execute it, returning results",
)
async def execute_sql(
    request: ExecuteSQLRequest,
    tenant_id: Annotated[UUID, Depends(get_current_tenant_id)],
    user_id: Annotated[UUID, Depends(get_current_user_id)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    agent_service: AgentServiceDep,
) -> ExecuteSQLResponse:
    """Generate and execute SQL from natural language question.

    Args:
        request: SQL execution request
        tenant_id: Tenant identifier from auth
        user_id: User identifier from auth
        session_service: Session service instance
        agent_service: SQL service instance

    Returns:
        SQL query and execution results

    Raises:
        HTTPException: If execution fails

    """
    # Validate and fetch session using existing helper
    try:
        session = await verify_session_ownership_helper(
            session_id=request.session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            session_service=session_service,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found or access denied: {str(e)}",
        ) from e

    with tracer.start_as_current_span(
        "api.agent.execute_sql",
        attributes={
            "tenant_id": str(tenant_id),
            "datasource_id": str(session.datasource_id),
            "session_id": str(session.id),
        },
    ):
        try:
            logger.info(
                "SQL execution requested",
                extra={
                    "tenant_id": str(tenant_id),
                    "datasource_id": str(session.datasource_id),
                    "session_id": str(session.id),
                    "question_length": len(request.question),
                },
            )

            # Execute SQL using datasource from session
            result = await agent_service.execute_sql(
                question=request.question,
                datasource_id=session.datasource_id,
                tenant_id=tenant_id,
                session_id=str(request.session_id),
                use_cache=request.use_cache,
                timeout_seconds=request.timeout_seconds,
                max_rows=request.max_rows,
            )

            response = ExecuteSQLResponse(
                sql=result["sql"],
                data=result["data"],
                rows_returned=result["rows_returned"],
                execution_time_ms=result["execution_time_ms"],
                cached=result["cached"],
                complexity_score=result["complexity_score"],
                success=result["success"],
                error_message=result.get("error_message"),
            )

            logger.info(
                "SQL execution completed",
                extra={
                    "success": response.success,
                    "rows_returned": response.rows_returned,
                    "cached": response.cached,
                },
            )

            return response

        except Exception as e:
            logger.error(
                "SQL execution failed",
                extra={
                    "error": str(e),
                    "tenant_id": str(tenant_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SQL execution failed: {str(e)}",
            ) from e
