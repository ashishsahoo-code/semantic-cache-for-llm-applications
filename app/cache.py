import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)


@dataclass
class CacheMatch:
    answer: str
    similarity: float


class RedisSemanticCache:
    """Redis Stack vector cache scoped by tenant, model, and locale."""

    def __init__(
        self,
        client: redis.Redis,
        index_name: str,
        dimension: int,
        ttl_seconds: int,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.dimension = dimension
        self.ttl_seconds = ttl_seconds
        self._index_ready = False

    async def ensure_index(self) -> None:
        if self._index_ready:
            return
        try:
            await self.client.ft(self.index_name).info()
            self._index_ready = True
            return
        except ResponseError:
            pass

        schema = (
            TagField("tenant"),
            TagField("model"),
            TagField("locale"),
            TextField("prompt"),
            TextField("answer"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dimension,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )
        definition = IndexDefinition(prefix=["semantic-cache:"], index_type=IndexType.HASH)
        try:
            await self.client.ft(self.index_name).create_index(schema, definition=definition)
        except ResponseError as exc:
            # Another worker may have created the index between INFO and CREATE.
            if "already exists" not in str(exc).lower():
                raise
        self._index_ready = True

    @staticmethod
    def _vector_bytes(vector: list[float]) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    async def lookup(
        self,
        *,
        tenant: str,
        model: str,
        locale: str,
        embedding: list[float],
        threshold: float,
    ) -> CacheMatch | None:
        await self.ensure_index()
        query_vector = self._vector_bytes(embedding)
        query = (
            f"(@tenant:{{{self._tag_escape(tenant)}}} "
            f"@model:{{{self._tag_escape(model)}}} "
            f"@locale:{{{self._tag_escape(locale)}}})"
            "=>[KNN 1 @embedding $vector AS vector_distance]"
        )
        result = await self.client.ft(self.index_name).search(
            __import__("redis.commands.search.query", fromlist=["Query"]).Query(query)
            .sort_by("vector_distance")
            .return_fields("answer", "vector_distance")
            .paging(0, 1)
            .dialect(2),
            query_params={"vector": query_vector},
        )
        if not result.docs:
            return None
        document = result.docs[0]
        distance = float(document.vector_distance)
        similarity = 1.0 - distance
        if similarity < threshold:
            return None
        return CacheMatch(answer=str(document.answer), similarity=similarity)

    async def store(
        self,
        *,
        tenant: str,
        model: str,
        locale: str,
        prompt: str,
        answer: str,
        embedding: list[float],
    ) -> str:
        await self.ensure_index()
        key = f"semantic-cache:{uuid.uuid4().hex}"
        record: dict[str, Any] = {
            "tenant": tenant,
            "model": model,
            "locale": locale,
            "prompt": prompt,
            "answer": answer,
            "embedding": self._vector_bytes(embedding),
        }
        await self.client.hset(key, mapping=record)
        await self.client.expire(key, self.ttl_seconds)
        return key

    async def invalidate(self, *, tenant: str, model: str | None, locale: str | None) -> int:
        await self.ensure_index()
        clauses = [f"@tenant:{{{self._tag_escape(tenant)}}}"]
        if model:
            clauses.append(f"@model:{{{self._tag_escape(model)}}}")
        if locale:
            clauses.append(f"@locale:{{{self._tag_escape(locale)}}}")
        query = " ".join(clauses)
        from redis.commands.search.query import Query

        result = await self.client.ft(self.index_name).search(
            Query(query).return_fields().paging(0, 10_000)
        )
        keys = [doc.id for doc in result.docs]
        if not keys:
            return 0
        return int(await self.client.delete(*keys))

    @staticmethod
    def _tag_escape(value: str) -> str:
        # Redis Search TAG syntax reserves punctuation; escape query metacharacters.
        escaped = value.replace("\\", "\\\\")
        for char in ",.<>!?@#$%^&*()-+=~":
            escaped = escaped.replace(char, f"\\{char}")
        return escaped
