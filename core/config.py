from enum import Enum
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "IgniteLens Backend Service"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=True)

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8002)

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://platform-core-user:platform-core-password@localhost:5432/platform-core-service"
    )
    DATABASE_ECHO: bool = Field(default=False)
    DATABASE_POOL_SIZE: int = Field(default=10)
    DATABASE_MAX_OVERFLOW: int = Field(default=5)
    DATABASE_POOL_TIMEOUT: int = Field(default=30)
    DATABASE_POOL_RECYCLE: int = Field(default=1800)
    DATABASE_POOL_PRE_PING: bool = Field(default=True)

    # LLM
    PORTKEY_API_KEY: str = Field(default="")
    PORTKEY_VIRTUAL_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="gpt-4")
    LLM_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)

    # Platform Microservice
    PLATFORM_SERVICE_URL: str = Field(default="http://localhost:8000/api/v1")
    PLATFORM_SERVICE_TIMEOUT: int = Field(default=30)  # seconds
    PLATFORM_SERVICE_RETRY_ATTEMPTS: int = Field(default=3)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Cache
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_DEFAULT_TTL: int = Field(default=300)  # 5 minutes
    CACHE_MAX_CONNECTIONS: int = Field(default=50)
    CACHE_RETRY_ON_TIMEOUT: bool = Field(default=True)

    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = Field(default="json")  # json or text

    # OpenTelemetry
    OTEL_SERVICE_NAME: str = Field(default="ignitelens-backend")
    OTEL_SERVICE_VERSION: str = Field(default="0.1.0")

    # Jaeger Configuration
    JAEGER_ENABLED: bool = Field(default=True)
    JAEGER_LOGS_ENABLED: bool = Field(default=False)  # Disabled: Jaeger all-in-one doesn't support OTLP logs properly
    JAEGER_AGENT_HOST: str = Field(default="localhost")
    JAEGER_AGENT_PORT: int = Field(default=6831)
    JAEGER_COLLECTOR_ENDPOINT: Optional[str] = Field(default="http://localhost:14268/api/traces")
    JAEGER_GRPC_ENDPOINT: Optional[str] = Field(default="http://localhost:14250")
    TRACE_SAMPLING_RATE: float = Field(default=1.0, ge=0.0, le=1.0)

    # Prometheus
    METRICS_PORT: int = Field(default=8003)
    ENABLE_METRICS: bool = Field(default=True)

    # Security Headers
    SECURITY_HEADERS_ENABLED: bool = Field(default=True)
    X_FRAME_OPTIONS: str = Field(default="DENY")  # DENY, SAMEORIGIN, or ALLOW-FROM uri
    HSTS_ENABLED: bool = Field(default=True)
    HSTS_MAX_AGE: int = Field(default=31536000)  # 1 year
    HSTS_INCLUDE_SUBDOMAINS: bool = Field(default=True)
    HSTS_PRELOAD: bool = Field(default=False)
    REFERRER_POLICY: str = Field(default="strict-origin-when-cross-origin")
    CSP_ENABLED: bool = Field(default=True)
    CSP_DISABLE_IN_DEVELOPMENT: bool = Field(default=False)
    CONTENT_SECURITY_POLICY: Optional[str] = Field(
        default=(
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
            "font-src 'self' https:; connect-src 'self' https:; media-src 'self'; "
            "object-src 'none'; child-src 'none'; worker-src 'none'; "
            "frame-ancestors 'none'; form-action 'self'; base-uri 'self';"
        )
    )
    PERMISSIONS_POLICY: Optional[str] = Field(
        default="geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), speaker=()"
    )
    REMOVE_SERVER_HEADER: bool = Field(default=True)

    # CORS Settings
    CORS_ALLOWED_ORIGINS: Optional[List[str]] = Field(default=None)
    CORS_ALLOW_ALL_ORIGINS: bool = Field(default=False)
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])
    CORS_EXPOSE_HEADERS: List[str] = Field(default=["X-Correlation-ID", "X-Trace-ID"])
    CORS_MAX_AGE: int = Field(default=86400)  # 24 hours

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")


settings = Settings()
