"""Unified Async OpenAI-SSE Benchmark Harness for vLLM, SGLang, llama.cpp, and MLX.

Mental Model:
- Fires `total_requests` bounded by an `asyncio.Semaphore(concurrency)`.
- Parses raw Server-Sent Events (`data: {...}`) line-by-line with nanosecond timers.
- Records exact Time-To-First-Token (TTFT) on the first non-empty content delta,
  and Inter-Token Latency (ITL) on every subsequent token chunk.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time
from typing import Any

import httpx

from harness.prompts import build_messages
from harness.schema import (
    BenchmarkCellResult,
    RequestTelemetry,
    detect_host_fingerprint,
    summarize_distribution,
)


async def measure_single_stream(
    client: httpx.AsyncClient,
    endpoint_url: str,
    model_id: str,
    request_id: int,
    workload_mode: str,
    max_tokens: int,
) -> RequestTelemetry:
    """Execute one streaming chat completion request and record token timestamps."""
    messages = build_messages(request_id=request_id, workload_mode=workload_mode)
    prompt_chars = sum(len(m["content"]) for m in messages)
    prompt_tokens_est = max(1, prompt_chars // 4)

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    token_timestamps_ns: list[int] = []
    start_ns = time.perf_counter_ns()

    try:
        async with client.stream("POST", endpoint_url, json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    token_timestamps_ns.append(time.perf_counter_ns())

    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        return RequestTelemetry(
            request_id=request_id,
            success=False,
            prompt_tokens_est=prompt_tokens_est,
            output_tokens=0,
            ttft_ms=0.0,
            e2e_latency_ms=round(elapsed_ms, 3),
            tpot_ms=0.0,
            error_message=str(exc),
        )

    end_ns = time.perf_counter_ns()
    output_tokens = len(token_timestamps_ns)
    if output_tokens == 0:
        return RequestTelemetry(
            request_id=request_id,
            success=False,
            prompt_tokens_est=prompt_tokens_est,
            output_tokens=0,
            ttft_ms=0.0,
            e2e_latency_ms=round((end_ns - start_ns) / 1_000_000.0, 3),
            tpot_ms=0.0,
            error_message="Zero non-empty content chunks received from stream.",
        )

    ttft_ms = (token_timestamps_ns[0] - start_ns) / 1_000_000.0
    e2e_ms = (end_ns - start_ns) / 1_000_000.0
    itl_ms_list = [
        round((token_timestamps_ns[i] - token_timestamps_ns[i - 1]) / 1_000_000.0, 3)
        for i in range(1, output_tokens)
    ]
    tpot_ms = (
        ((token_timestamps_ns[-1] - token_timestamps_ns[0]) / 1_000_000.0) / (output_tokens - 1)
        if output_tokens > 1
        else 0.0
    )

    return RequestTelemetry(
        request_id=request_id,
        success=True,
        prompt_tokens_est=prompt_tokens_est,
        output_tokens=output_tokens,
        ttft_ms=round(ttft_ms, 3),
        e2e_latency_ms=round(e2e_ms, 3),
        tpot_ms=round(tpot_ms, 3),
        itl_ms_list=itl_ms_list,
    )


async def run_benchmark_cell(
    base_url: str,
    engine: str,
    model_id: str,
    quantization: str,
    hardware_target: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    workload_mode: str = "standard",
    warmup_requests: int = 2,
) -> BenchmarkCellResult:
    """Run warmup followed by a bounded-concurrency benchmark sweep."""
    endpoint_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    limits = httpx.Limits(max_connections=concurrency + 16, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=300.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        for w_idx in range(warmup_requests):
            await measure_single_stream(
                client=client,
                endpoint_url=endpoint_url,
                model_id=model_id,
                request_id=9000 + w_idx,
                workload_mode=workload_mode,
                max_tokens=min(32, max_tokens),
            )

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded_worker(req_idx: int) -> RequestTelemetry:
            async with semaphore:
                return await measure_single_stream(
                    client=client,
                    endpoint_url=endpoint_url,
                    model_id=model_id,
                    request_id=req_idx,
                    workload_mode=workload_mode,
                    max_tokens=max_tokens,
                )

        wall_start_ns = time.perf_counter_ns()
        records = await asyncio.gather(*[_bounded_worker(i) for i in range(total_requests)])
        wall_time_s = (time.perf_counter_ns() - wall_start_ns) / 1_000_000_000.0

    ok_records = [r for r in records if r.success]
    all_itls = [itl for r in ok_records for itl in r.itl_ms_list]
    total_out_tokens = sum(r.output_tokens for r in ok_records)
    sys_tps = total_out_tokens / wall_time_s if wall_time_s > 0 else 0.0

    per_user_tps_list = [1000.0 / r.tpot_ms for r in ok_records if r.tpot_ms > 0]
    mean_user_tps = sum(per_user_tps_list) / len(per_user_tps_list) if per_user_tps_list else 0.0

    rusage_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rusage_rss / (1024 * 1024) if detect_host_fingerprint().startswith("Darwin") else rusage_rss / 1024.0

    return BenchmarkCellResult(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        hardware_target=hardware_target,
        os_platform=detect_host_fingerprint(),
        engine=engine,
        model_id=model_id,
        quantization=quantization,
        workload_mode=workload_mode,
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=len(ok_records),
        failed_requests=len(records) - len(ok_records),
        wall_time_s=round(wall_time_s, 3),
        total_output_tokens=total_out_tokens,
        system_output_throughput_tps=round(sys_tps, 2),
        mean_per_user_tps=round(mean_user_tps, 2),
        ttft=summarize_distribution([r.ttft_ms for r in ok_records]),
        tpot=summarize_distribution([r.tpot_ms for r in ok_records if r.tpot_ms > 0]),
        itl=summarize_distribution(all_itls),
        e2e=summarize_distribution([r.e2e_latency_ms for r in ok_records]),
        rss_memory_mb=round(rss_mb, 2),
    )


def persist_cell_result(result: BenchmarkCellResult, output_dir: Path) -> None:
    """Append benchmark cell result to both JSONL and flat CSV artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark_results.jsonl"
    csv_path = output_dir / "benchmark_summary.csv"

    with jsonl_path.open("a", encoding="utf-8") as jf:
        jf.write(json.dumps(result.to_dict()) + "\n")

    flat_row = {
        "timestamp_utc": result.timestamp_utc,
        "hardware_target": result.hardware_target,
        "engine": result.engine,
        "model_id": result.model_id,
        "quantization": result.quantization,
        "workload_mode": result.workload_mode,
        "concurrency": result.concurrency,
        "ok_requests": result.successful_requests,
        "failed_requests": result.failed_requests,
        "system_throughput_tps": result.system_output_throughput_tps,
        "mean_per_user_tps": result.mean_per_user_tps,
        "ttft_p50_ms": result.ttft.p50_ms,
        "ttft_p90_ms": result.ttft.p90_ms,
        "ttft_p99_ms": result.ttft.p99_ms,
        "tpot_p50_ms": result.tpot.p50_ms,
        "tpot_p90_ms": result.tpot.p90_ms,
        "tpot_p99_ms": result.tpot.p99_ms,
        "e2e_p50_ms": result.e2e.p50_ms,
        "e2e_p99_ms": result.e2e.p99_ms,
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(flat_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Engine OpenAI SSE Benchmark Harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Server base URL")
    parser.add_argument("--engine", required=True, help="vllm | sglang | llama.cpp | mlx")
    parser.add_argument("--model", required=True, help="Model identifier")
    parser.add_argument("--quant", required=True, help="FP16 | Q8_0 | Q4_K_M | AWQ-INT4")
    parser.add_argument("--hardware", default="Apple-M2-16GB", help="Hardware label")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent inflight requests")
    parser.add_argument("--requests", type=int, default=8, help="Total measured requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max output tokens per request")
    parser.add_argument("--workload", choices=["standard", "prefix_cache"], default="standard")
    parser.add_argument("--warmup", type=int, default=2, help="Unmeasured warmup requests")
    parser.add_argument("--output-dir", default="results", help="Output directory for JSONL/CSV")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark_cell(
            base_url=args.base_url,
            engine=args.engine,
            model_id=args.model,
            quantization=args.quant,
            hardware_target=args.hardware,
            concurrency=args.concurrency,
            total_requests=args.requests,
            max_tokens=args.max_tokens,
            workload_mode=args.workload,
            warmup_requests=args.warmup,
        )
    )
    persist_cell_result(result, Path(args.output_dir))
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

