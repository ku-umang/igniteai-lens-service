from fastapi import APIRouter

# Include all route modules
from core.api.v1.routes.session import router as session_router

api_router = APIRouter()

api_router.include_router(session_router)


__all__ = ["api_router"]
