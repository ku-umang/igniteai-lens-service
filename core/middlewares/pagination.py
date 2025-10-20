"""Pagination middleware for storing request context.

This middleware stores the current request in a context variable, making it
accessible to the PagePaginator utility for automatic URL generation.

The middleware is automatically registered in the FastAPI application
(see core/server.py) and requires no additional configuration.

How it works:
    1. The middleware intercepts each incoming request
    2. Stores the request object in a context variable
    3. The PagePaginator accesses this request to build next/previous URLs
    4. URLs preserve all query parameters from the original request

This enables automatic pagination URL generation without having to manually
pass request information through your application layers.
"""

from contextvars import ContextVar

from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

request_object: ContextVar[Request] = ContextVar("request")


class PaginationMiddleware(BaseHTTPMiddleware):
    """Middleware that stores the current request in a context variable.

    This allows the PagePaginator to access request information for
    generating next and previous page URLs automatically.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Store request in context and process the request.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            Response: The HTTP response

        """
        request_object.set(request)
        response = await call_next(request)
        return response


middleware = [Middleware(PaginationMiddleware)]
