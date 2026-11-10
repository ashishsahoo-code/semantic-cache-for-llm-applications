from fastapi.testclient import TestClient

from app.main import app


class FakeRedis:
    async def ping(self):
        return True

    async def aclose(self):
        return None


class FakeCache:
    def __init__(self):
        self.items = {}
        self.deleted = 0

    async def lookup(self, *, tenant, model, locale, embedding, threshold):
        key = (tenant, model, locale, tuple(embedding))
        return self.items.get(key)

    async def store(self, *, tenant, model, locale, prompt, answer, embedding):
        from app.cache import CacheMatch

        key = (tenant, model, locale, tuple(embedding))
        self.items[key] = CacheMatch(answer=answer, similarity=1.0)
        return "fake-key"

    async def invalidate(self, *, tenant, model, locale):
        count = len(self.items)
        self.items.clear()
        return count


class FakeEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]


class FakeProvider:
    async def generate(self, prompt, model):
        return f"generated:{prompt}"


def setup_app():
    app.state.redis = FakeRedis()
    app.state.cache = FakeCache()
    app.state.embedder = FakeEmbedder()
    app.state.provider = FakeProvider()
    return TestClient(app)


def test_generate_miss_then_hit():
    client = setup_app()
    try:
        headers = {"X-Tenant-ID": "tenant-a"}
        body = {"prompt": "hello world", "model": "test-model", "locale": "en"}

        first = client.post("/v1/generate", json=body, headers=headers)
        assert first.status_code == 200
        assert first.json()["cache_hit"] is False
        assert first.headers["x-cache-status"] == "MISS"

        second = client.post("/v1/generate", json=body, headers=headers)
        assert second.status_code == 200
        assert second.json()["cache_hit"] is True
        assert second.headers["x-cache-status"] == "HIT"
    finally:
        client.close()


def test_cache_can_be_bypassed():
    client = setup_app()
    try:
        response = client.post(
            "/v1/generate",
            json={"prompt": "fresh answer", "use_cache": False},
        )
        assert response.status_code == 200
        assert response.headers["x-cache-status"] == "BYPASS"
        assert response.json()["cache_hit"] is False
    finally:
        client.close()


def test_invalidate_endpoint():
    client = setup_app()
    try:
        response = client.delete(
            "/v1/cache",
            headers={"X-Tenant-ID": "tenant-a"},
            json={"model": "test-model", "locale": "en"},
        )
        assert response.status_code == 200
        assert response.json()["deleted_entries"] == 0
    finally:
        client.close()
