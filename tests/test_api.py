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

    async def lookup(self, *, tenant, model, locale, embedding, threshold):
        key = (tenant, model, locale, tuple(embedding))
        return self.items.get(key)

    async def store(self, *, tenant, model, locale, prompt, answer, embedding):
        from app.cache import CacheMatch

        key = (tenant, model, locale, tuple(embedding))
        self.items[key] = CacheMatch(answer=answer, similarity=1.0)
        return "fake-key"

    async def invalidate(self, *, tenant, model, locale):
        matching_keys = [
            key
            for key in self.items
            if key[0] == tenant
            and (model is None or key[1] == model)
            and (locale is None or key[2] == locale)
        ]
        for key in matching_keys:
            del self.items[key]
        return len(matching_keys)


class FakeEmbedder:
    def embed(self, text):
        # Keep embeddings identical so these tests isolate metadata scoping behavior.
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


def generate(client, *, tenant, model="test-model", locale="en", prompt="hello world"):
    return client.post(
        "/v1/generate",
        headers={"X-Tenant-ID": tenant},
        json={"prompt": prompt, "model": model, "locale": locale},
    )


def test_generate_miss_then_hit():
    client = setup_app()
    try:
        first = generate(client, tenant="tenant-a")
        assert first.status_code == 200
        assert first.json()["cache_hit"] is False
        assert first.headers["x-cache-status"] == "MISS"

        second = generate(client, tenant="tenant-a")
        assert second.status_code == 200
        assert second.json()["cache_hit"] is True
        assert second.headers["x-cache-status"] == "HIT"
    finally:
        client.close()


def test_cache_isolated_by_tenant_model_and_locale():
    client = setup_app()
    try:
        original = generate(client, tenant="tenant-a", model="model-v1", locale="en")
        other_tenant = generate(client, tenant="tenant-b", model="model-v1", locale="en")
        other_model = generate(client, tenant="tenant-a", model="model-v2", locale="en")
        other_locale = generate(client, tenant="tenant-a", model="model-v1", locale="fr")
        original_again = generate(client, tenant="tenant-a", model="model-v1", locale="en")

        assert [response.headers["x-cache-status"] for response in (
            original, other_tenant, other_model, other_locale, original_again
        )] == ["MISS", "MISS", "MISS", "MISS", "HIT"]
    finally:
        client.close()


def test_invalidation_only_removes_requested_tenant_scope():
    client = setup_app()
    try:
        generate(client, tenant="tenant-a", model="model-v1", locale="en")
        generate(client, tenant="tenant-a", model="model-v1", locale="fr")
        generate(client, tenant="tenant-b", model="model-v1", locale="en")

        invalidated = client.request(
            "DELETE",
            "/v1/cache",
            headers={"X-Tenant-ID": "tenant-a"},
            json={"model": "model-v1", "locale": "en"},
        )
        assert invalidated.status_code == 200
        assert invalidated.json()["deleted_entries"] == 1

        tenant_a_en = generate(client, tenant="tenant-a", model="model-v1", locale="en")
        tenant_a_fr = generate(client, tenant="tenant-a", model="model-v1", locale="fr")
        tenant_b_en = generate(client, tenant="tenant-b", model="model-v1", locale="en")

        assert tenant_a_en.headers["x-cache-status"] == "MISS"
        assert tenant_a_fr.headers["x-cache-status"] == "HIT"
        assert tenant_b_en.headers["x-cache-status"] == "HIT"
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


def test_invalid_tenant_header_is_rejected():
    client = setup_app()
    try:
        response = client.post(
            "/v1/generate",
            headers={"X-Tenant-ID": "   "},
            json={"prompt": "hello"},
        )
        assert response.status_code == 400
    finally:
        client.close()
