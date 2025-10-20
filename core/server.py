from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.openapi.utils import get_openapi
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.api.v1 import api_router
from core.cache import Cache, CustomKeyMaker, RedisBackend
from core.config import settings
from core.database.session import initialize_database
from core.exceptions import (
    IgniteLensBaseError,
    database_exception_handler,
    database_operational_exception_handler,
    generic_exception_handler,
    http_exception_handler,
    ignite_lens_exception_handler,
    starlette_http_exception_handler,
    validation_exception_handler,
)
from core.llm_config import llm_config
from core.logging import configure_logging, get_logger
from core.middlewares import (
    LoggingMiddleware,
    MetricsMiddleware,
    PaginationMiddleware,
    SecurityHeadersMiddleware,
    configure_cors,
)
from core.observability import (
    init_observability,
    instrument_app,
    shutdown_observability,
)

# Configure logging before anything else
configure_logging()
logger = get_logger(__name__)


def custom_openapi(fastapi_app: FastAPI) -> Optional[Dict[str, Any]]:
    """Custom OpenAPI schema with JWT Bearer security."""
    if fastapi_app.openapi_schema:
        return fastapi_app.openapi_schema

    openapi_schema = get_openapi(
        title=fastapi_app.title,
        version=fastapi_app.version,
        description=fastapi_app.description,
        routes=fastapi_app.routes,
    )

    # Add JWT Bearer security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter JWT token obtained from login endpoint",
        }
    }

    fastapi_app.openapi_schema = openapi_schema
    return fastapi_app.openapi_schema


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting IgniteLens Backend Service", version=settings.APP_VERSION)

    try:
        # Initialize observability
        init_observability()

        # Instrument the app
        instrument_app(fastapi_app)

        # Initialize database
        try:
            db_manager = initialize_database()
            await db_manager.initialize()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            # This is critical - we need the database to function
            raise

        # Initialize cache manager
        try:
            Cache.init(backend=RedisBackend, key_maker=CustomKeyMaker)
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize cache manager", error=str(e))
            # Continue without cache for resilience
            logger.warning("Application starting without cache functionality")

        # Configure LLM
        try:
            llm_config.configure_llm()
            logger.info("LLM configured successfully")
        except Exception as e:
            logger.error("Failed to configure LLM", error=str(e))
            # Continue without LLM for resilience
            logger.warning("Application starting without LLM functionality")

        logger.info("Application startup completed")
        yield

    except Exception as e:
        logger.error("Failed to start application", error=str(e), exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Lens Service")

        # Shutdown observability components first
        try:
            shutdown_observability()
        except Exception as e:
            logger.error("Error shutting down observability", error=str(e))

        # Close database
        try:
            from core.database.session import session_manager

            if session_manager:
                await session_manager.close()
                logger.info("Database closed successfully")
        except Exception as e:
            logger.error("Error closing database", error=str(e))

        # Close cache manager
        try:
            if Cache.backend:
                # Close Redis connection if it has a close method
                if hasattr(Cache.backend, "close"):
                    await Cache.backend.close()  # type: ignore
                logger.info("Cache manager closed successfully")
        except Exception as e:
            logger.error("Error closing cache manager", error=str(e))


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    fastapi_app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered platform for data insights and analytics",
        lifespan=lifespan,
        debug=settings.DEBUG,
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # Add CORS middleware (first for preflight handling)
    configure_cors(fastapi_app)

    # Add security headers middleware
    fastapi_app.add_middleware(SecurityHeadersMiddleware)

    # Add pagination middleware (stores request context for pagination URL generation)
    fastapi_app.add_middleware(PaginationMiddleware)

    # Add metrics middleware (tracks HTTP metrics before logging)
    fastapi_app.add_middleware(MetricsMiddleware)

    # Add logging middleware (last for complete request/response logging)
    fastapi_app.add_middleware(LoggingMiddleware)

    # Register exception handlers
    fastapi_app.add_exception_handler(IgniteLensBaseError, ignite_lens_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(PydanticValidationError, validation_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(IntegrityError, database_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(OperationalError, database_operational_exception_handler)  # type: ignore
    fastapi_app.add_exception_handler(Exception, generic_exception_handler)  # type: ignore

    # Health check endpoint
    @fastapi_app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": settings.OTEL_SERVICE_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Prometheus metrics endpoint
    @fastapi_app.get("/metrics", tags=["Monitoring"])
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Include API routers
    fastapi_app.include_router(api_router, prefix="/api")

    logger.info("FastAPI application created")
    return fastapi_app


# Create the app instance
app = create_app()
