#!/usr/bin/env bash
# Runs on the GPU host. Expects ~/aw.env (HF_TOKEN, VLLM_API_KEY), copied there by the operator.
set -euo pipefail
cd ~/aw
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
set -a; source ~/aw.env; set +a
uv venv --python 3.12 .venv
# Lambda's driver is 570 (CUDA 12.8); the default vLLM wheel needs CUDA 13, so take the cu129 build.
V="${VLLM_VERSION:-0.30.0}"
uv pip install --python .venv/bin/python "vllm==$V" "huggingface_hub[hf_transfer]" ninja \
  --extra-index-url "https://wheels.vllm.ai/$V/cu129" --torch-backend=cu129 --index-strategy unsafe-best-match --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
# torchcodec/torchaudio come built for CUDA 13 and break vLLM's import; text serving doesn't need them.
uv pip uninstall --python .venv/bin/python torchcodec torchaudio || true
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/python download_adapters.py
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
