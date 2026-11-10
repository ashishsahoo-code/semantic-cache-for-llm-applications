import hashlib

import numpy as np


class EmbeddingProvider:
    """Embedding interface. Replace this demo adapter with a real model in production."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class DeterministicDemoEmbedder(EmbeddingProvider):
    """Stable token-hash vector for local wiring and tests; not semantic-quality embeddings."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()
