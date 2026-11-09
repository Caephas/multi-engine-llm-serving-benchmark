"""Deterministic prompt workloads for standard and prefix-caching benchmarks.

Mental Model:
- 'standard' mode gives every request a unique prefix so engines cannot bypass
  prefill compute via KV prefix caching.
- 'prefix_cache' mode prepends an identical ~1,024-token technical specification
  to every request, followed by a unique query, isolating RadixAttention / APC gains.
"""

from __future__ import annotations

SHARED_SYSTEM_CONTEXT = (
    "You are a Principal Distributed Systems Architect evaluating high-throughput "
    "inference clusters. Reference architecture parameters:\n"
    "1. Compute Node Topology: Dual-tier heterogeneous cluster combining Apple Silicon "
    "M-series unified memory nodes (100 GB/s LPDDR5-6400 bus, 16 GB capacity) and "
    "NVIDIA Turing T4 GDDR6 nodes (320 GB/s memory bus, 16 GB capacity, 65 TFLOPS FP16).\n"
    "2. Execution Phases: Prefill computes parallel key-value projections across all "
    "input tokens (compute-bound GEMM), whereas Decode computes autoregressive token "
    "generation one step at a time (memory-bandwidth-bound GEMV at low batch sizes).\n"
    "3. KV Cache Mechanics: Standard contiguous pre-allocation fragments up to 60% of "
    "VRAM. PagedAttention partitions KV states into fixed-size non-contiguous blocks, "
    "eliminating external fragmentation. RadixAttention maintains a radix tree of "
    "completed KV blocks across requests to enable O(1) prefix reuse.\n"
    "4. Quantization Tradeoffs: Reducing weight representation from 16-bit floating "
    "point (2 bytes/param) to 8-bit integer (1 byte/param) or 4-bit group-quantized "
    "formats (0.5 bytes/param) proportionally reduces memory bus transfer time during "
    "decode at the cost of dequantization kernel overhead during high-batch compute.\n"
    "5. Service Level Objectives: P99 Time-To-First-Token (TTFT) must remain below "
    "1,500 milliseconds and P90 Time-Per-Output-Token (TPOT) must remain below 50 "
    "milliseconds under sustained multi-tenant load.\n"
) * 3

UNIQUE_TOPICS = [
    "continuous batching iteration-level scheduling",
    "speculative decoding draft verification acceptance rates",
    "tensor parallelism all-reduce communication overhead",
    "pipeline parallelism bubble minimization micro-batches",
    "quantization-aware training versus post-training AWQ",
    "KV cache quantization to FP8 and INT4 formats",
    "chunked prefill head-of-line blocking prevention",
    "flash attention SRAM tiling and online softmax",
]


def build_messages(request_id: int, workload_mode: str) -> list[dict[str, str]]:
    """Construct an OpenAI chat messages payload for a given request index."""
    topic = UNIQUE_TOPICS[request_id % len(UNIQUE_TOPICS)]

    if workload_mode == "prefix_cache":
        return [
            {"role": "system", "content": SHARED_SYSTEM_CONTEXT},
            {
                "role": "user",
                "content": (
                    f"Request #{request_id}: Based on the reference architecture, write a "
                    f"detailed technical analysis of {topic} covering hardware bottlenecks, "
                    f"memory bandwidth arithmetic, and failure modes in at least 300 words."
                ),
            },
        ]

    return [
        {
            "role": "user",
            "content": (
                f"[Unique-Session-ID-{request_id:05d}-Seed-{request_id * 7919}] "
                f"Provide an exhaustive, step-by-step first-principles engineering breakdown "
                f"of {topic}. Include concrete latency math, memory hierarchy trade-offs, "
                f"and production mitigation strategies in at least 300 words."
            ),
        }
    ]

