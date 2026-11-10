import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Header, HTTPException, Request, Response, status

from app.cache import RedisSemanticCache
from app.config import get_settings
from app.embeddings import DeterministicDemoEmbedder
from app.metrics import Metrics
from app.providers import DemoLLMProvider
from app.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    InvalidateRequest,
    InvalidateResponse,
    MetricsResponse,
)

settings = get_settings()
metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = redis.from_url(
        settings.redis_url,
        decode_responses=False,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
    app.state.redis = client
    app.state.cache = RedisSemanticCache(
        client,
        settings.cache_index_name,
        settings.embedding_dimension,
        settings.cache_ttl_seconds,
    )
    app.state.embedder = DeterministicDemoEmbedder(settings.embedding_dimension)
    app.state.provider = DemoLLMProvider()
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="Semantic Cache for LLM Applications",
    version="0.1.0",
    description="Redis-backed semantic response caching for LLM applications.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    try:
        await request.app.state.redis.ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc
    return HealthResponse(status="ok", redis="ok")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(
    payload: GenerateRequest,
    request: Request,
    response: Response,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    if len(x_tenant_id) > 128 or not x_tenant_id.strip():
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be non-empty and <= 128 chars")

    started = time.perf_counter()
    metrics.requests += 1
    embedding = request.app.state.embedder.embed(payload.prompt)

    if payload.use_cache:
        try:
            match = await request.app.state.cache.lookup(
                tenant=x_tenant_id,
                model=payload.model,
                locale=payload.locale,
                embedding=embedding,
                threshold=settings.similarity_threshold,
            )
        except Exception as exc:
            # Cache failures fail open: the LLM path remains available.
            match = None
            response.headers["X-Cache-Status"] = "unavailable"
        else:
            if match:
                metrics.cache_hits += 1
                elapsed = (time.perf_counter() - started) * 1000
                metrics.total_latency_ms += elapsed
                response.headers["X-Cache-Status"] = "HIT"
                return GenerateResponse(
                    answer=match.answer,
                    cache_hit=True,
                    similarity=round(match.similarity, 4),
                    latency_ms=round(elapsed, 2),
                    model=payload.model,
                )
            metrics.cache_misses += 1
            response.headers["X-Cache-Status"] = "MISS"
    else:
        response.headers["X-Cache-Status"] = "BYPASS"

    metrics.provider_calls += 1
    answer = await request.app.state.provider.generate(payload.prompt, payload.model)
    if payload.use_cache:
        try:
            await request.app.state.cache.store(
                tenant=x_tenant_id,
                model=payload.model,
                locale=payload.locale,
                prompt=payload.prompt,
                answer=answer,
                embedding=embedding,
            )
        except Exception:
            # A write failure should not discard a successful model response.
            response.headers["X-Cache-Store"] = "failed"

    elapsed = (time.perf_counter() - started) * 1000
    metrics.total_latency_ms += elapsed
    return GenerateResponse(
        answer=answer,
        cache_hit=False,
        latency_ms=round(elapsed, 2),
        model=payload.model,
    )


@app.delete("/v1/cache", response_model=InvalidateResponse)
async def invalidate(
    payload: InvalidateRequest,
    request: Request,
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID"),
):
    try:
        deleted = await request.app.state.cache.invalidate(
            tenant=x_tenant_id,
            model=payload.model,
            locale=payload.locale,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Cache invalidation failed") from exc
    return InvalidateResponse(deleted_entries=deleted)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return MetricsResponse(**metrics.snapshot())
