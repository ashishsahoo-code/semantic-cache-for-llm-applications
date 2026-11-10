from app.embeddings import DeterministicDemoEmbedder


def test_embedding_is_deterministic_and_normalized():
    embedder = DeterministicDemoEmbedder(dimension=32)
    first = embedder.embed("semantic cache test")
    second = embedder.embed("semantic cache test")

    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) ** 0.5 - 1.0) < 1e-6


def test_empty_text_returns_zero_vector():
    vector = DeterministicDemoEmbedder(dimension=16).embed("")
    assert vector == [0.0] * 16
