"""Chart generator for the Multi-Engine LLM Serving Benchmark."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as cf:
        for r in csv.DictReader(cf):
            rows.append(
                {
                    "hardware_target": r["hardware_target"],
                    "engine": r["engine"],
                    "quantization": r["quantization"],
                    "workload_mode": r["workload_mode"],
                    "concurrency": int(r["concurrency"]),
                    "system_throughput_tps": float(r["system_throughput_tps"]),
                    "mean_per_user_tps": float(r["mean_per_user_tps"]),
                    "ttft_p50_ms": float(r["ttft_p50_ms"]),
                    "tpot_p50_ms": float(r["tpot_p50_ms"]),
                }
            )
    return rows


def plot_throughput_scaling(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    series = [
        ("sglang", "Q4_K_M", "#d97706", "o-", "SGLang (T4 · AWQ-4bit)"),
        ("vllm", "Q4_K_M", "#0d9488", "s-", "vLLM (T4 · AWQ-4bit)"),
        ("llama.cpp", "Q8_0", "#64748b", "d--", "llama.cpp (M2 · Q8_0)"),
        ("llama.cpp", "Q4_K_M", "#2563eb", "^-", "llama.cpp (M2 · Q4_K_M)"),
    ]

    for eng, quant, color, style, label in series:
        pts = sorted(
            [
                r
                for r in rows
                if r["engine"] == eng
                and r["quantization"] == quant
                and r["workload_mode"] == "standard"
            ],
            key=lambda x: x["concurrency"],
        )
        if pts:
            ax.plot(
                [p["concurrency"] for p in pts],
                [p["system_throughput_tps"] for p in pts],
                style,
                color=color,
                linewidth=2.4,
                markersize=7.5,
                label=label,
            )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32])
    ax.set_xticklabels(["1", "8", "32"])
    ax.set_xlabel("Concurrent In-Flight Requests (Log2 Scale)", fontsize=11)
    ax.set_ylabel("Total System Output Throughput (tokens/sec)", fontsize=11)
    ax.set_title("Continuous Batching Scaling — Gemma 4 E4B Instruct (M2 vs. Tesla T4)", fontsize=12.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()

    dest = out_dir / "throughput_vs_concurrency.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_quantization_roofline(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.6), dpi=160)

    tiers_m2 = ["Q8_0 (8.19 GB)", "Q4_K_M (4.62 GB)"]
    quants_m2 = ["Q8_0", "Q4_K_M"]
    roofline_m2 = [100.0 / 8.19, 100.0 / 4.62]
    measured_c1 = [
        next((r["mean_per_user_tps"] for r in rows if r["engine"] == "llama.cpp" and r["quantization"] == q and r["workload_mode"] == "standard" and r["concurrency"] == 1), 0.0)
        for q in quants_m2
    ]
    measured_c32 = [
        next((r["system_throughput_tps"] for r in rows if r["engine"] == "llama.cpp" and r["quantization"] == q and r["workload_mode"] == "standard" and r["concurrency"] == 32), 0.0)
        for q in quants_m2
    ]

    x = range(len(tiers_m2))
    w = 0.28
    ax1.bar([i - w / 2 for i in x], measured_c1, width=w, color="#2563eb", label="Measured Decode @ C=1")
    ax1.bar([i + w / 2 for i in x], roofline_m2, width=w, color="#cbd5e1", edgecolor="#475569", hatch="//", label="100 GB/s Full-File Roofline")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(tiers_m2)
    ax1.set_ylabel("Single-Stream Decode Speed (tokens/sec)")
    ax1.set_title("M2 @ C=1 (Memory-Bandwidth-Bound)\nQ4_K_M is 1.58x Faster than Q8_0", fontweight="bold", fontsize=10.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8.5)

    configs_c32 = ["llama.cpp Q4_K_M\n(M2 @ C=32)", "llama.cpp Q8_0\n(M2 @ C=32)", "vLLM AWQ-4bit\n(T4 @ C=32)"]
    vals_c32 = [
        measured_c32[1],
        measured_c32[0],
        next((r["system_throughput_tps"] for r in rows if r["engine"] == "vllm" and r["quantization"] == "Q4_K_M" and r["workload_mode"] == "standard" and r["concurrency"] == 32), 0.0),
    ]
    colors_c32 = ["#2563eb", "#64748b", "#0d9488"]
    bars = ax2.bar(range(len(configs_c32)), vals_c32, width=0.45, color=colors_c32)
    for bar, v in zip(bars, vals_c32):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 4, f"{v:.1f} tok/s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(configs_c32)))
    ax2.set_xticklabels(configs_c32, fontsize=9.5)
    ax2.set_ylim(0, max(vals_c32) * 1.18 if vals_c32 else 260)
    ax2.set_ylabel("Total System Throughput (tokens/sec)")
    ax2.set_title("C=32 Compute-Bound Crossover\nQ8_0 Beats Q4_K_M on M2; vLLM Hits 231.7 tok/s", fontweight="bold", fontsize=10.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Roofline Regime Shift: C=1 Memory-Bound vs. C=32 Compute-Bound (Gemma 4 E4B)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    dest = out_dir / "quantization_roofline.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def plot_prefix_cache_ablation(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    targets = [
        ("vllm", "Q4_K_M", 8, "vLLM @ C=8\n(Paged APC)"),
        ("vllm", "Q4_K_M", 32, "vLLM @ C=32\n(Paged APC)"),
        ("llama.cpp", "Q4_K_M", 8, "llama.cpp @ C=8\n(Isolated Slots)"),
        ("llama.cpp", "Q4_K_M", 32, "llama.cpp @ C=32\n(+6.27 GB Swap)"),
    ]

    std_ttft = []
    cached_ttft = []
    labels = []

    for eng, quant, conc, label in targets:
        s_val = next(
            (r["ttft_p50_ms"] for r in rows if r["engine"] == eng and r["quantization"] == quant and r["workload_mode"] == "standard" and r["concurrency"] == conc),
            0.0,
        )
        c_val = next(
            (r["ttft_p50_ms"] for r in rows if r["engine"] == eng and r["quantization"] == quant and r["workload_mode"] == "prefix_cache" and r["concurrency"] == conc),
            0.0,
        )
        std_ttft.append(s_val)
        cached_ttft.append(c_val)
        labels.append(label)

    x = range(len(labels))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], std_ttft, width=w, color="#64748b", label="Standard Short Prompt (Uncached)")
    b2 = ax.bar([i + w / 2 for i in x], cached_ttft, width=w, color="#0d9488", label="Shared 1,024-Token System Prefix (Cached)")

    for bar, v in zip(b1, std_ttft):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 350, f"{v:.0f}ms", ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(b2, cached_ttft):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 350, f"{v:.0f}ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(max(std_ttft), max(cached_ttft)) * 1.16)
    ax.set_ylabel("Median Time-To-First-Token — TTFT P50 (ms)", fontsize=10.5)
    ax.set_title("Shared Prefix KV Caching: vLLM PagedAttention APC vs. llama.cpp Contiguous Slots", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9.5)
    fig.tight_layout()

    dest = out_dir / "prefix_cache_ttft.png"
    fig.savefig(dest)
    plt.close(fig)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark charts from CSV")
    parser.add_argument("--csv", default="results/benchmark_summary.csv", help="Input CSV path")
    parser.add_argument("--out-dir", default="results/plots", help="Output plots directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.csv))

    p1 = plot_throughput_scaling(rows, out_dir)
    p2 = plot_quantization_roofline(rows, out_dir)
    p3 = plot_prefix_cache_ablation(rows, out_dir)
    print(f"Generated plots:\n  1. {p1}\n  2. {p2}\n  3. {p3}")


if __name__ == "__main__":
    main()
