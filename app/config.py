from functools import lru_cache

from pydantic import Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "semantic-cache-api"
    environment: str = "development"
    redis_url: str = "redis://localhost:6379"
    cache_index_name: str = "semantic_cache_idx"
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    embedding_dimension: int = Field(default=384, ge=8)
    redis_socket_timeout_seconds: float = Field(default=2.0, gt=0)

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        RedisDsn(value)
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
