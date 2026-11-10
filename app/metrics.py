from dataclasses import dataclass, field


@dataclass
class Metrics:
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_calls: int = 0
    total_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        lookups = self.cache_hits + self.cache_misses
        return self.cache_hits / lookups if lookups else 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests else 0.0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "provider_calls": self.provider_calls,
            "average_latency_ms": round(self.average_latency_ms, 2),
        }
