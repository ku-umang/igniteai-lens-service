"""Datasource client for fetching connection details from external microservice."""

import asyncio
from typing import Any, Optional
from uuid import UUID

import httpx

from core.config import settings
from core.integrations.schema import DataSourceResponse, LLMConfigurationResponse
from core.logging import get_logger
from core.security.secrets_manager import secrets_manager

logger = get_logger(__name__)


class PlatformClient:
    """Client for fetching platform details from platform microservice.

    This client communicates with an external microservice that stores
    tenant platform configurations.
    """

    def __init__(self) -> None:
        """Initialize platform client."""
        self.base_url = f"{settings.PLATFORM_SERVICE_URL}"
        self.timeout = settings.PLATFORM_SERVICE_TIMEOUT
        self.retry_attempts = settings.PLATFORM_SERVICE_RETRY_ATTEMPTS
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "PlatformClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        if self._client:
            await self._client.aclose()

    async def retrieve_from_knowledge(self, query: str, datasource_id: str, tenant_id: str) -> str:
        """Retrieve information from knowledge base.

        Args:
            query: Query to ask
            datasource_id: Datasource identifier
            tenant_id: Tenant identifier

        """
        if not self._client:
            raise RuntimeError("PlatformClient must be used as async context manager")

        url = "/retrieval/retrieve"
        headers = {
            "X-Tenant-ID": tenant_id,
            "Content-Type": "application/json",
        }

        data = {
            "query": query,
            "datasource_id": datasource_id,
        }

        response = await self._client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    async def get_llm_config(self, tenant_id: UUID, llm_config_id: UUID) -> LLMConfigurationResponse:
        """Fetch LLM config details from microservice.

        Args:
            tenant_id: Tenant identifier for authorization
            llm_config_id: Unique LLM config identifier

        """
        if not self._client:
            raise RuntimeError("PlatformClient must be used as async context manager")

        url = f"/llm_configurations/{llm_config_id}"
        headers = {
            "X-Tenant-ID": str(tenant_id),
            "Content-Type": "application/json",
        }

        for attempt in range(self.retry_attempts):
            try:
                response = await self._client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                llm_config = LLMConfigurationResponse(**data)
                logger.info("Successfully fetched LLM config", extra={"llm_config_id": str(llm_config_id)})
                return llm_config
            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error fetching LLM config",
                    extra={
                        "llm_config_id": str(llm_config_id),
                        "status_code": e.response.status_code,
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except (httpx.RequestError, ValueError) as e:
                logger.error(
                    "Error fetching LLM config",
                    extra={
                        "llm_config_id": str(llm_config_id),
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff
        raise RuntimeError(f"Failed to fetch LLM config after {self.retry_attempts} attempts")

    async def get_datasource(
        self,
        datasource_id: UUID,
        tenant_id: UUID,
    ) -> DataSourceResponse:
        """Fetch datasource connection details from microservice.

        Args:
            datasource_id: Unique datasource identifier
            tenant_id: Tenant identifier for authorization

        Returns:
            DataSourceResponse with connection details

        Raises:
            httpx.HTTPStatusError: If the request fails
            ValueError: If the response is invalid

        """
        if not self._client:
            raise RuntimeError("DatasourceClient must be used as async context manager")

        url = f"/data_sources/{datasource_id}"
        headers = {
            "X-Tenant-ID": str(tenant_id),
            "Content-Type": "application/json",
        }

        # Retry logic with exponential backoff
        for attempt in range(self.retry_attempts):
            try:
                logger.info(
                    "Fetching datasource connection",
                    extra={
                        "datasource_id": str(datasource_id),
                        "tenant_id": str(tenant_id),
                        "attempt": attempt + 1,
                    },
                )

                response = await self._client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()

                # Parse response into DataSourceResponse
                connection = DataSourceResponse(**data)

                logger.info(
                    "Successfully fetched datasource connection",
                    extra={
                        "datasource_id": str(datasource_id),
                        "connector_kind": connection.kind,
                    },
                )

                if connection.has_credentials:
                    connection.credentials = {
                        "username": secrets_manager.decrypt_value(connection.credentials["username"]),  # type: ignore
                        "password": secrets_manager.decrypt_value(connection.credentials["password"]),  # type: ignore
                    }
                return connection

            except httpx.HTTPStatusError as e:
                logger.error(
                    "HTTP error fetching datasource",
                    extra={
                        "datasource_id": str(datasource_id),
                        "status_code": e.response.status_code,
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except (httpx.RequestError, ValueError) as e:
                logger.error(
                    "Error fetching datasource",
                    extra={
                        "datasource_id": str(datasource_id),
                        "error": str(e),
                        "attempt": attempt + 1,
                    },
                )
                if attempt == self.retry_attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError(f"Failed to fetch datasource after {self.retry_attempts} attempts")


# Singleton instance
_platform_client: Optional[PlatformClient] = None


async def get_platform_client() -> PlatformClient:
    """Get or create platform client instance.

    Returns:
        PlatformClient instance

    Note:
        This is a dependency injection function for FastAPI routes.
        The client should be used with async context manager.

    """
    global _platform_client
    if _platform_client is None:
        _platform_client = PlatformClient()
    return _platform_client
