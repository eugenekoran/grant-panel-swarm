#!/usr/bin/env bash
# Runs on the GPU host. Stock Qwen3 chat template plus a <think> prefill (thinking on), localhost only; reach it via SSH tunnel.
# Env knobs: DP (data-parallel size, default 1), MAX_LEN (default 40960), GPU_UTIL (default 0.90), MAX_LORAS (default 4).
set -euo pipefail
cd ~/aw
set -a; source ~/aw.env; set +a
export PATH="$HOME/aw/.venv/bin:$PATH"  # JIT kernels need ninja from the venv
exec .venv/bin/vllm serve Qwen/Qwen3-14B \
  --served-model-name qwen3-14b-base \
  --enable-lora --max-lora-rank 128 --max-loras "${MAX_LORAS:-4}" \
  --lora-modules aw-sdf-none=adapters/aw-sdf-none aw-sdf-sft=adapters/aw-sdf-sft aw-sdf-kto=adapters/aw-sdf-kto \
  --chat-template qwen3_think_prefill.jinja --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len "${MAX_LEN:-40960}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --data-parallel-size "${DP:-1}" \
  --host 127.0.0.1 --port 8000 --api-key "$VLLM_API_KEY" \
  "$@"
