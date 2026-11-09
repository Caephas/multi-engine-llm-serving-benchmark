"""Automated Apple Silicon (M2 16GB) Benchmark Orchestrator for llama.cpp and MLX.

Mental Model:
1. Spawns the target inference engine (`llama-server` or `mlx_lm.server`) as a
   managed subprocess with explicit parallel KV slots (`--parallel`) and Metal GPU
   offload (`-ngl 99`).
2. Polls the OpenAI `/v1/models` readiness endpoint until weights are resident in
   Unified Memory.
3. Sweeps the concurrency ladder (`1, 8, 32`) for both `standard` and `prefix_cache`
   workloads while monitoring macOS `vm.swapusage` to detect SSD swap contamination.
4. Terminates the server cleanly before advancing to the next quantization tier.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
import sys
import time
from typing import AsyncIterator

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.benchmark import persist_cell_result, run_benchmark_cell


def read_macos_swap_used_mb() -> float:
    """Query `sysctl vm.swapusage` on macOS to detect unified memory spillover to SSD."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        for token in out.split():
            if token.endswith("M") and "used =" in out[: out.find(token)]:
                return float(token.rstrip("M"))
    except Exception:
        pass
    return 0.0


def build_engine_command(
    engine: str,
    model_path_or_repo: str,
    port: int,
    max_parallel_slots: int,
    ctx_per_slot: int = 1024,
) -> list[str]:
    """Construct the exact CLI invocation for `llama-server` or `mlx_lm.server`."""
    if engine == "llama.cpp":
        total_ctx = max_parallel_slots * ctx_per_slot
        cmd = [
            "llama-server",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "-ngl",
            "99",
            "--parallel",
            str(max_parallel_slots),
            "-c",
            str(total_ctx),
            "--cont-batching",
            "--cache-reuse",
            "64",
        ]
        if Path(model_path_or_repo).expanduser().exists():
            cmd.extend(["-m", str(Path(model_path_or_repo).expanduser())])
        else:
            cmd.extend(["-hf", model_path_or_repo])
        return cmd

    if engine == "mlx":
        return [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            model_path_or_repo,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    raise ValueError(f"Unsupported local Mac engine: {engine}")


@asynccontextmanager
async def managed_engine_server(
    cmd: list[str],
    base_url: str,
    startup_timeout_s: float = 300.0,
) -> AsyncIterator[subprocess.Popen[str]]:
    """Launch an engine subprocess, await HTTP readiness, and guarantee teardown."""
    print(f"\n[Server] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + startup_timeout_s
    ready = False

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    output = proc.stdout.read() if proc.stdout else ""
                    raise RuntimeError(f"Engine exited prematurely (code {proc.returncode}):\n{output}")
                try:
                    resp = await client.get(f"{base_url}/v1/models")
                    if resp.status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)

        if not ready:
            raise TimeoutError(f"Server at {base_url} did not become ready within {startup_timeout_s}s.")

        print(f"[Server] Ready at {base_url}")
        yield proc

    finally:
        if proc.poll() is None:
            print("[Server] Shutting down engine subprocess...")
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()


async def run_engine_sweep(
    engine: str,
    model_spec: str,
    model_label: str,
    quantization: str,
    concurrencies: list[int],
    max_tokens: int,
    workload_mode: str,
    port: int,
    output_dir: Path,
) -> None:
    """Boot the local server once for a given quantization tier and sweep all concurrencies."""
    max_c = max(concurrencies)
    base_url = f"http://127.0.0.1:{port}"
    cmd = build_engine_command(
        engine=engine,
        model_path_or_repo=model_spec,
        port=port,
        max_parallel_slots=max(8, max_c),
    )

    swap_before_mb = read_macos_swap_used_mb()
    async with managed_engine_server(cmd=cmd, base_url=base_url):
        for c in concurrencies:
            total_reqs = max(4, c * 2)
            print(
                f"  -> Running cell: engine={engine} quant={quantization} "
                f"workload={workload_mode} concurrency={c} requests={total_reqs}"
            )
            result = await run_benchmark_cell(
                base_url=base_url,
                engine=engine,
                model_id=model_label,
                quantization=quantization,
                hardware_target="Apple-M2-16GB",
                concurrency=c,
                total_requests=total_reqs,
                max_tokens=max_tokens,
                workload_mode=workload_mode,
                warmup_requests=2,
            )
            persist_cell_result(result, output_dir)
            swap_after_mb = read_macos_swap_used_mb()
            swap_delta = round(swap_after_mb - swap_before_mb, 1)
            print(
                f"     [Done] Sys Throughput: {result.system_output_throughput_tps:6.2f} tok/s | "
                f"User Speed: {result.mean_per_user_tps:5.2f} tok/s | "
                f"TTFT P50: {result.ttft.p50_ms:6.1f} ms | "
                f"TPOT P50: {result.tpot.p50_ms:5.1f} ms | "
                f"Swap Delta: +{swap_delta} MB"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Apple M2 Local Engine Benchmark Runner")
    parser.add_argument("--engine", choices=["llama.cpp", "mlx"], required=True)
    parser.add_argument(
        "--model-spec",
        required=True,
        help="Path to local .gguf file, HF `-hf` repo:quant, or MLX HF repo ID",
    )
    parser.add_argument("--model-label", default="google/gemma-4-E4B-it", help="Normalized model ID")
    parser.add_argument("--quant", required=True, help="FP16 | Q8_0 | Q4_K_M | MLX-4bit")
    parser.add_argument("--concurrencies", default="1,8,32", help="Comma-separated concurrency ladder")
    parser.add_argument("--max-tokens", type=int, default=128, help="Output tokens per request")
    parser.add_argument("--workload", choices=["standard", "prefix_cache"], default="standard")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    c_ladder = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    asyncio.run(
        run_engine_sweep(
            engine=args.engine,
            model_spec=args.model_spec,
            model_label=args.model_label,
            quantization=args.quant,
            concurrencies=c_ladder,
            max_tokens=args.max_tokens,
            workload_mode=args.workload,
            port=args.port,
            output_dir= Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()

