"""Automated NVIDIA Tesla T4 (16GB GDDR6) Benchmark Orchestrator for vLLM and SGLang."""

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


def query_nvidia_vram_used_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def cleanup_gpu_processes() -> None:
    for pat in ["vllm.entrypoints", "VLLM::EngineCore", "sglang.launch_server", "sglang::"]:
        subprocess.run(["pkill", "-9", "-f", pat], stderr=subprocess.DEVNULL)
    time.sleep(3.0)


def build_cuda_engine_command(
    engine: str,
    model_repo: str,
    model_label: str,
    port: int,
) -> list[str]:
    if engine == "vllm":
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_repo,
            "--served-model-name",
            model_label,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "float16",
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.85",
            "--enforce-eager",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0,"audio":0}',
            "--enable-prefix-caching",
            "--disable-log-requests",
        ]

    if engine == "sglang":
        return [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_repo,
            "--served-model-name",
            model_label,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "float16",
            "--context-length",
            "2048",
            "--mem-fraction-static",
            "0.80",
            "--attention-backend",
            "triton",
            "--disable-cuda-graph",
        ]

    raise ValueError(f"Unsupported CUDA engine: {engine}")


@asynccontextmanager
async def managed_cuda_server(
    cmd: list[str],
    base_url: str,
    startup_timeout_s: float = 600.0,
) -> AsyncIterator[subprocess.Popen[str]]:
    cleanup_gpu_processes()
    log_path = Path("cuda_server.log")
    log_file = log_path.open("w", encoding="utf-8")
    print(f"\n[CUDA Server] Free VRAM before launch — Used: {query_nvidia_vram_used_mb():.0f} MiB")
    print(f"[CUDA Server] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    start_t = time.monotonic()
    deadline = start_t + startup_timeout_s
    ready = False
    ticks = 0

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    log_file.flush()
                    logs = log_path.read_text(encoding="utf-8", errors="replace")
                    raise RuntimeError(f"CUDA server exited (code {proc.returncode}):\n{logs[-6000:]}")
                try:
                    resp = await client.get(f"{base_url}/v1/models")
                    if resp.status_code == 200:
                        ready = True
                        break
                except httpx.HTTPError:
                    pass
                ticks += 1
                if ticks % 6 == 0:
                    log_file.flush()
                    lines = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    last_line = lines[-1][-95:] if lines else "starting..."
                    print(
                        f"  [Booting {int(time.monotonic() - start_t)}s] "
                        f"VRAM: {query_nvidia_vram_used_mb():.0f} MiB | {last_line}"
                    )
                await asyncio.sleep(2.0)

        if not ready:
            log_file.flush()
            logs = log_path.read_text(encoding="utf-8", errors="replace")
            raise TimeoutError(f"CUDA server timed out:\n{logs[-4000:]}")

        print(f"[CUDA Server] Ready | Active VRAM: {query_nvidia_vram_used_mb():.0f} MiB")
        yield proc

    finally:
        log_file.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        cleanup_gpu_processes()


async def run_t4_sweep(
    engine: str,
    model_repo: str,
    model_label: str,
    quantization: str,
    concurrencies: list[int],
    max_tokens: int,
    workload_mode: str,
    port: int,
    output_dir: Path,
) -> None:
    base_url = f"http://127.0.0.1:{port}"
    cmd = build_cuda_engine_command(engine, model_repo, model_label, port)

    async with managed_cuda_server(cmd, base_url):
        for c in concurrencies:
            total_reqs = max(4, c * 2)
            print(
                f"  -> Cell: engine={engine} quant={quantization} "
                f"workload={workload_mode} C={c} reqs={total_reqs}"
            )
            result = await run_benchmark_cell(
                base_url=base_url,
                engine=engine,
                model_id=model_label,
                quantization=quantization,
                hardware_target="NVIDIA-T4-16GB",
                concurrency=c,
                total_requests=total_reqs,
                max_tokens=max_tokens,
                workload_mode=workload_mode,
                warmup_requests=2,
            )
            persist_cell_result(result, output_dir)
            print(
                f"     [Done] Sys Throughput: {result.system_output_throughput_tps:6.2f} tok/s | "
                f"User Speed: {result.mean_per_user_tps:5.2f} tok/s | "
                f"TTFT P50: {result.ttft.p50_ms:6.1f} ms | "
                f"TPOT P50: {result.tpot.p50_ms:5.1f} ms | "
                f"VRAM: {query_nvidia_vram_used_mb():.0f} MiB"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA Tesla T4 (16GB) vLLM & SGLang Runner")
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--model-label", default="google/gemma-4-E4B-it")
    parser.add_argument("--quant", required=True)
    parser.add_argument("--concurrencies", default="1,8,32")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--workload", choices=["standard", "prefix_cache"], default="standard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    c_ladder = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]
    asyncio.run(
        run_t4_sweep(
            engine=args.engine,
            model_repo=args.model_repo,
            model_label=args.model_label,
            quantization=args.quant,
            concurrencies=c_ladder,
            max_tokens=args.max_tokens,
            workload_mode=args.workload,
            port=args.port,
            output_dir=Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()
