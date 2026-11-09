"""Telemetry data schemas and statistical aggregation for LLM serving benchmarks.

Invariants:
- All latency metrics (TTFT, TPOT, ITL, E2E) are stored in milliseconds (ms).
- Throughput metrics are stored in tokens per second (tok/s).
- Percentiles (P50, P90, P99) use linear interpolation over sorted observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import platform
import statistics
from typing import Any


@dataclass(frozen=True)
class RequestTelemetry:
    """Single-request timing record captured from an OpenAI SSE token stream."""

    request_id: int
    success: bool
    prompt_tokens_est: int
    output_tokens: int
    ttft_ms: float
    e2e_latency_ms: float
    tpot_ms: float
    itl_ms_list: list[float] = field(default_factory=list)
    error_message: str | None = None


@dataclass(frozen=True)
class DistributionSummary:
    """Percentile and central tendency summary for a latency distribution."""

    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class BenchmarkCellResult:
    """Normalized output record for one (Engine x Quantization x Concurrency) cell."""

    timestamp_utc: str
    hardware_target: str
    os_platform: str
    engine: str
    model_id: str
    quantization: str
    workload_mode: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    wall_time_s: float
    total_output_tokens: int
    system_output_throughput_tps: float
    mean_per_user_tps: float
    ttft: DistributionSummary
    tpot: DistributionSummary
    itl: DistributionSummary
    e2e: DistributionSummary
    rss_memory_mb: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute linear-interpolated percentile from a non-empty sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    weight = rank - lower_idx
    interpolated = sorted_values[lower_idx] * (1.0 - weight) + sorted_values[upper_idx] * weight
    return round(interpolated, 3)


def summarize_distribution(values: list[float]) -> DistributionSummary:
    """Summarize a list of latency measurements into Mean/P50/P90/P99."""
    if not values:
        return DistributionSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(values)
    return DistributionSummary(
        mean_ms=round(statistics.fmean(ordered), 3),
        p50_ms=compute_percentile(ordered, 50.0),
        p90_ms=compute_percentile(ordered, 90.0),
        p99_ms=compute_percentile(ordered, 99.0),
        min_ms=round(ordered[0], 3),
        max_ms=round(ordered[-1], 3),
    )


def detect_host_fingerprint() -> str:
    """Return human-readable host platform string."""
    return f"{platform.system()}-{platform.machine()}-{platform.processor() or 'unknown'}"

