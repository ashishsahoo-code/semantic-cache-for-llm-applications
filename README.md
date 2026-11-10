# Semantic Cache for LLM Applications

A production-minded semantic caching layer for LLM APIs. It embeds incoming prompts, searches Redis for semantically similar requests, and reuses a cached response only when the similarity score crosses a configurable threshold and the cache scope matches.

## Why this exists

LLM applications often receive paraphrased versions of the same question. Exact-string caches miss these opportunities. Semantic caching can reduce repeat model calls, latency, and inference spend — but only when reuse is safe.

This project demonstrates:
- Embedding-based nearest-neighbor lookup with Redis.
- Configurable similarity threshold and TTL.
- Tenant, model, and locale isolation to avoid cross-context cache leakage.
- Explicit cache invalidation and cache bypass.
- Separate cache-hit and cache-miss metrics.
- A provider boundary so an LLM backend can be swapped without changing the cache.

## Architecture

```text
Client
  |
  v
FastAPI /v1/generate
  |
  +--> Embed prompt
  |
  +--> Redis vector search (scoped by tenant/model/locale)
  |       |
  |       +--> Safe match above threshold -> cached response
  |       |
  |       +--> No safe match -> LLM provider -> store response + embedding
  |
  v
JSON response + cache metadata
```

## Quick start

Requirements: Python 3.11+, Docker, Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at http://localhost:8000. Interactive API docs: http://localhost:8000/docs.

### Generate

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"prompt":"Explain semantic caching in one paragraph","model":"demo-model","locale":"en","use_cache":true}'
```

The starter uses a deterministic local demo provider and a lightweight deterministic embedding implementation so the service can run without paid API keys. These are development adapters, **not** production-quality semantic embeddings or LLM generation. Production adapters should use a real embedding model and provider.

### Invalidate a scope

```bash
curl -X DELETE http://localhost:8000/v1/cache \
  -H 'X-Tenant-ID: demo' \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo-model","locale":"en"}'
```

## Configuration

See `.env.example`. Important settings:
- `SIMILARITY_THRESHOLD`: minimum cosine similarity required for a hit.
- `CACHE_TTL_SECONDS`: response expiration.
- `REDIS_URL`: Redis connection.
- `EMBEDDING_DIMENSION`: vector dimension; must match the embedding adapter.
- `CACHE_INDEX_NAME`: Redis Search index name.

Thresholds must be calibrated against representative query pairs. A lower threshold increases hit rate but also increases the risk of returning an irrelevant answer. Cache keys are scoped by tenant, model, and locale; do not remove these boundaries casually.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service and Redis health |
| POST | `/v1/generate` | Semantic cache lookup, then provider fallback |
| DELETE | `/v1/cache` | Invalidate the current tenant's selected scope |
| GET | `/metrics` | In-memory counters and latency aggregates |

## Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
```

## Safety and correctness notes

- Similarity is not the same as equivalence. A hit threshold alone is not a complete correctness or safety policy.
- This starter isolates tenant, model, and locale, but real deployments may also need authorization scope, prompt version, region, safety policy version, and user-specific context in the cache namespace.
- Avoid caching sensitive or highly personalized prompts and responses without an explicit data-retention and privacy design.
- The demo embedding adapter is intentionally replaceable and should not be used to evaluate real semantic quality.
- Redis index creation and vector query behavior depend on Redis Stack / Redis Search being available.

## Roadmap

1. Add a production embedding adapter and offline threshold evaluation.
2. Add provider adapters (OpenAI/Azure OpenAI) with retry and timeout policies.
3. Add persistent metrics and load/latency benchmarks.
4. Add cache admission controls, content safety checks, and richer invalidation.
5. Add multi-tenant integration tests and deployment guidance.

## License

MIT
