"""Centralised, typed configuration read from the environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONTRACT_VERSION = "1.0"
CONTRACT_RELEASE = "1.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["development", "test", "production"] = Field("development", alias="MODELMESH_ENV")
    log_level: str = Field("INFO", alias="MODELMESH_LOG_LEVEL")
    default_timeout_ms: int = Field(60_000, alias="MODELMESH_DEFAULT_TIMEOUT_MS")
    max_timeout_ms: int = Field(300_000, alias="MODELMESH_MAX_TIMEOUT_MS")
    max_body_bytes: int = Field(2 * 1024 * 1024, alias="MODELMESH_MAX_BODY_BYTES")
    stream_idle_timeout_s: float = Field(30.0, alias="MODELMESH_STREAM_IDLE_TIMEOUT_S")
    stream_keepalive_s: float = Field(15.0, alias="MODELMESH_STREAM_KEEPALIVE_S")

    api_key_pepper: SecretStr = Field(SecretStr(""), alias="MODELMESH_API_KEY_PEPPER")
    admin_token: SecretStr = Field(SecretStr(""), alias="MODELMESH_ADMIN_TOKEN")

    database_url: str = Field(
        "postgresql+psycopg://modelmesh:modelmesh@localhost:55432/modelmesh", alias="MODELMESH_DATABASE_URL"
    )
    redis_url: str = Field("redis://localhost:56379/0", alias="MODELMESH_REDIS_URL")

    gemini_api_key: SecretStr | None = Field(None, alias="GEMINI_API_KEY")
    groq_api_key: SecretStr | None = Field(None, alias="GROQ_API_KEY")
    openrouter_api_key: SecretStr | None = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_app_url: str | None = Field(None, alias="OPENROUTER_APP_URL")
    openrouter_app_title: str | None = Field(None, alias="OPENROUTER_APP_TITLE")
    ollama_base_url: str | None = Field(None, alias="OLLAMA_BASE_URL")

    model_catalog_path: Path = Field(Path("config/models.yaml"), alias="MODELMESH_MODEL_CATALOG_PATH")
    pricing_path: Path = Field(Path("config/pricing.yaml"), alias="MODELMESH_PRICING_PATH")
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", alias="MODELMESH_EMBEDDING_MODEL")
    cache_default_ttl_s: int = Field(3600, alias="MODELMESH_CACHE_DEFAULT_TTL_S")
    cache_max_ttl_s: int = Field(86_400, alias="MODELMESH_CACHE_MAX_TTL_S")
    semantic_cache_enabled: bool = Field(True, alias="MODELMESH_SEMANTIC_CACHE_ENABLED")
    content_retention_ttl_days: int = Field(7, alias="MODELMESH_CONTENT_RETENTION_TTL_DAYS")
    enable_fake_provider: bool = Field(False, alias="MODELMESH_ENABLE_FAKE_PROVIDER")

    default_key_rpm: int = Field(20, alias="MODELMESH_DEFAULT_KEY_RPM")
    default_key_tpm: int = Field(50_000, alias="MODELMESH_DEFAULT_KEY_TPM")
    default_key_concurrency: int = Field(10, alias="MODELMESH_DEFAULT_KEY_CONCURRENCY")
    default_key_daily_budget_usd: float = Field(0.25, alias="MODELMESH_DEFAULT_KEY_DAILY_BUDGET_USD")

    otlp_endpoint: str | None = Field(None, alias="OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_metrics_enabled: bool = Field(False, alias="MODELMESH_OTLP_METRICS_ENABLED")
    otel_service_name: str = Field("modelmesh", alias="OTEL_SERVICE_NAME")

    @model_validator(mode="after")
    def _guard_fake_provider(self) -> Self:
        if self.enable_fake_provider and self.env == "production":
            raise ValueError("MODELMESH_ENABLE_FAKE_PROVIDER cannot be enabled when MODELMESH_ENV=production")
        return self

    def require_runtime_secrets(self) -> None:
        """Fail fast at start-up rather than running with an empty pepper or admin token."""
        missing = [
            name
            for name, value in (
                ("MODELMESH_API_KEY_PEPPER", self.api_key_pepper),
                ("MODELMESH_ADMIN_TOKEN", self.admin_token),
            )
            if not value.get_secret_value()
        ]
        if missing:
            raise RuntimeError(f"Missing required settings: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
