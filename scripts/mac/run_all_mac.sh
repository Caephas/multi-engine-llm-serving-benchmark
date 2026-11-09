#!/usr/bin/env bash
# Full Apple Silicon M2 (16GB) Benchmark Sweep: llama.cpp (Metal) vs MLX
# Usage: bash scripts/mac/run_all_mac.sh

set -euo pipefail

MODEL_LABEL="google/gemma-4-E4B-it"
GGUF_REPO="unsloth/gemma-4-E4B-it-GGUF"

echo "============================================================"
echo "Track 1A: llama.cpp (Apple Metal) — Q4_K_M / Q8_0 / FP16"
echo "============================================================"

for QUANT in "Q4_K_M" "Q8_0" "F16"; do
  python3 scripts/mac/run_mac_suite.py \
    --engine llama.cpp \
    --model-spec "${GGUF_REPO}:${QUANT}" \
    --model-label "${MODEL_LABEL}" \
    --quant "${QUANT}" \
    --concurrencies "1,8,32" \
    --max-tokens 128 \
    --workload standard
done

echo "============================================================"
echo "Track 1B: llama.cpp Prefix Caching Ablation (Q4_K_M)"
echo "============================================================"

python3 scripts/mac/run_mac_suite.py \
  --engine llama.cpp \
  --model-spec "${GGUF_REPO}:Q4_K_M" \
  --model-label "${MODEL_LABEL}" \
  --quant "Q4_K_M" \
  --concurrencies "1,8,32" \
  --max-tokens 128 \
  --workload prefix_cache

echo "============================================================"
echo "Track 1C: Apple MLX (mlx_lm.server) — 4bit / 8bit / FP16"
echo "============================================================"

if python3 -c "import mlx_lm" >/dev/null 2>&1; then
  for MLX_TIER in "4bit:MLX-4bit" "8bit:MLX-8bit" "bf16:FP16"; do
    SUFFIX="${MLX_TIER%%:*}"
    LABEL="${MLX_TIER##*:}"
    python3 scripts/mac/run_mac_suite.py \
      --engine mlx \
      --model-spec "mlx-community/gemma-4-E4B-it-${SUFFIX}" \
      --model-label "${MODEL_LABEL}" \
      --quant "${LABEL}" \
      --concurrencies "1,8" \
      --max-tokens 128 \
      --workload standard
  done
else
  echo "[Skip] mlx-lm not installed. Install with: pip3 install mlx-lm"
fi

