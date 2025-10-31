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
    config_json: Dict[str, Any] = Field(
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


class ColumnResultSchema(BaseModel):
    """Schema for a single column result."""

    content: str = Field(..., description="Column description text")
    score: float = Field(..., description="Relevance score (0-1)", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Column metadata")


class RelatedTableSchema(BaseModel):
    """Schema for a related table with relationship context."""

    table_name: str = Field(..., description="Table display name")
    qualified_name: str = Field(..., description="Fully qualified table name")
    relationship_type: str = Field(..., description="Type of relationship (foreign_key, reverse_foreign_key)")
    source_table: str = Field(..., description="Source table in relationship")
    target_table: str = Field(..., description="Target table in relationship")
    column_mappings: List[Dict[str, str]] = Field(default_factory=list, description="Column mappings for JOIN")
    join_hint: str = Field(..., description="SQL JOIN hint")


class TableResultSchema(BaseModel):
    """Schema for a table result with nested columns."""

    content: str = Field(..., description="Table description text")
    score: float = Field(..., description="Relevance score (0-1)", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Table metadata")
    related_tables: List[RelatedTableSchema] = Field(default_factory=list, description="Related tables via FK relationships")
    columns: List[ColumnResultSchema] = Field(default_factory=list, description="Relevant columns for this table")


class ExampleQuerySchema(BaseModel):
    """Schema for an example query result."""

    content: str = Field(..., description="Example query text")
    score: float = Field(..., description="Relevance score (0-1)", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Example query metadata")


class RetrievalResponse(BaseModel):
    """Response schema for hierarchical hybrid retrieval."""

    tables: List[TableResultSchema] = Field(..., description="Retrieved tables with nested columns")
    example_queries: List[ExampleQuerySchema] = Field(default_factory=list, description="Example queries for reference")
    query: str = Field(..., description="Original query")
    total_tables: int = Field(..., description="Total number of tables returned")
    total_columns: int = Field(..., description="Total number of columns across all tables")
    total_examples: int = Field(..., description="Total number of example queries returned")
    elapsed_time: float = Field(..., description="Total retrieval time in seconds")
