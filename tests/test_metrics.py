from app.metrics import Metrics


def test_hit_rate_and_average_latency():
    metrics = Metrics(requests=2, cache_hits=1, cache_misses=1, total_latency_ms=30.0)
    assert metrics.hit_rate == 0.5
    assert metrics.average_latency_ms == 15.0


def test_empty_metrics_are_zero():
    metrics = Metrics()
    assert metrics.hit_rate == 0.0
    assert metrics.average_latency_ms == 0.0
