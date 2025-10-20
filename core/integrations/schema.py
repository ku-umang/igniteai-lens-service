import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConnectorKind(str, enum.Enum):
    JDBC = "jdbc"
    WAREHOUSE = "warehouse"
    OBJECT_STORE = "object_store"
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    API = "api"
    FILE = "file"


class DataSourceResponse(BaseModel):
    """Data source response schema with separated config and credentials."""

    id: UUID
    tenant_id: UUID
    name: str
    slug: str
    connector_key: str
    connector_version: str
    kind: ConnectorKind
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Public configuration (non-sensitive fields only)",
    )
    has_credentials: bool = Field(default=True)
    credentials: Optional[Dict[str, Any]] = Field(
        None,
        description="Sensitive credentials (passwords, tokens, API keys, etc.)",
    )
    network_profile: Dict[str, Any]
    tags: List[str]
    connection_status: str
    created_at: datetime
    updated_at: datetime
    last_health_at: Optional[datetime] = None

    # Hidden field to ingest ORM attribute but not serialize it
    secret_uri: Optional[str] = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(from_attributes=True, json_encoders={datetime: datetime.isoformat, UUID: str})

    @model_validator(mode="after")
    def set_has_credentials(self):
        # Set has_credentials based on credentials field
        self.has_credentials = bool(self.credentials)
        return self


class LLMConfigurationResponse(BaseModel):
    """LLM configuration response schema."""

    id: UUID
    tenant_id: UUID
    config_name: str
    llm_provider: str
    llm_model: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    stop_sequences: Optional[List[str]] = None
    custom_params: Optional[Dict[str, Any]] = None
    is_default: bool
    api_key: str
    created_at: datetime
    updated_at: datetime
    created_by: UUID
    updated_by: UUID

    model_config = ConfigDict(from_attributes=True, json_encoders={datetime: datetime.isoformat, UUID: str})
