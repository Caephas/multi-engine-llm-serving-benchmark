"""Self-contained integration test for the benchmark harness using a local mock SSE server.

Verifies:
1. SSE token stream parsing (`data: {...}` and `data: [DONE]`).
2. Accurate TTFT (simulated 40ms prefill) and ITL/TPOT (simulated 10ms/token decode).
3. JSONL and CSV persistence without touching real GPU weights.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile

from harness.benchmark import persist_cell_result, run_benchmark_cell


async def _handle_mock_sse(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Minimal asyncio HTTP server simulating an OpenAI `/v1/chat/completions` SSE stream."""
    await reader.readuntil(b"\r\n\r\n")

    headers = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n"
    )
    writer.write(headers.encode("utf-8"))
    await writer.drain()

    # Simulate 30ms Prefill (TTFT)
    await asyncio.sleep(0.030)

    # Stream 10 tokens with 10ms Inter-Token Latency (ITL / TPOT)
    for tok_idx in range(10):
        if tok_idx > 0:
            await asyncio.sleep(0.010)
        chunk = {"choices": [{"delta": {"content": f" tok{tok_idx}"}}]}
        writer.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode("utf-8"))
        await writer.drain()

    writer.write(b"data: [DONE]\r\n\r\n")
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def run_selftest() -> None:
    server = await asyncio.start_server(_handle_mock_sse, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    async with server:
        result = await run_benchmark_cell(
            base_url=base_url,
            engine="mock-engine",
            model_id="google/gemma-4-E4B-it",
            quantization="Q4_K_M",
            hardware_target="Apple-M2-16GB-Selftest",
            concurrency=4,
            total_requests=8,
            max_tokens=10,
            workload_mode="standard",
            warmup_requests=1,
        )

    assert result.successful_requests == 8, f"Expected 8 successes, got {result.successful_requests}"
    assert result.failed_requests == 0
    assert result.total_output_tokens == 80
    assert 20.0 <= result.ttft.p50_ms <= 80.0, f"Unexpected TTFT P50: {result.ttft.p50_ms}ms"
    assert 8.0 <= result.tpot.p50_ms <= 20.0, f"Unexpected TPOT P50: {result.tpot.p50_ms}ms"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        persist_cell_result(result, out_path)
        assert (out_path / "benchmark_results.jsonl").exists()
        assert (out_path / "benchmark_summary.csv").exists()

    print("SELFTEST PASSED:")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(run_selftest())

