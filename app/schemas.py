from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    model: str = Field(default="demo-model", min_length=1, max_length=128)
    locale: str = Field(default="en", min_length=2, max_length=20)
    use_cache: bool = True


class GenerateResponse(BaseModel):
    answer: str
    cache_hit: bool
    similarity: float | None = None
    latency_ms: float
    model: str


class InvalidateRequest(BaseModel):
    model: str | None = Field(default=None, max_length=128)
    locale: str | None = Field(default=None, max_length=20)


class InvalidateResponse(BaseModel):
    deleted_entries: int


class HealthResponse(BaseModel):
    status: str
    redis: str


class MetricsResponse(BaseModel):
    requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    provider_calls: int
    average_latency_ms: float
