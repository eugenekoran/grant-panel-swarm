#!/usr/bin/env bash
# Run locally. Takes a fresh Lambda instance to a serving vLLM behind an SSH tunnel on localhost:8000.
#   scripts/lambda/bootstrap.sh <ip> [serve env, e.g. DP=2 or GPU_UTIL=0.75]
# Steps: write ssh_config (Host aw), copy scripts + secrets, setup.sh (~5 min), serve.sh in tmux, wait, tunnel.
set -euo pipefail
cd "$(dirname "$0")/../.."
IP="${1:?usage: bootstrap.sh <ip> [VAR=value ...]}"; shift
SERVE_ENV="$*"
CFG=scripts/lambda/ssh_config
cat > "$CFG" <<EOF
Host *
  User ubuntu
  IdentityFile ~/.ssh/lambda_key.pem
  UserKnownHostsFile ~/.ssh/known_hosts_lambda
  StrictHostKeyChecking accept-new
  ConnectTimeout 20
  ServerAliveInterval 30
Host aw
  HostName $IP
EOF
S="ssh -F $CFG aw"

echo "== copy"
$S 'mkdir -p ~/aw'
scp -q -F "$CFG" scripts/lambda/{setup.sh,serve.sh,download_adapters.py,qwen3_think_prefill.jinja} aw:aw/
# Secrets go over stdin into a mode-600 file, never onto a command line.
grep -E '^(HF_TOKEN|VLLM_API_KEY)=' .env | $S 'umask 077; cat > ~/aw.env'

echo "== setup (vLLM + weights, ~5 min)"
$S 'bash ~/aw/setup.sh > ~/aw/setup.log 2>&1 || { tail -30 ~/aw/setup.log; exit 1; }; tail -1 ~/aw/setup.log'

echo "== serve ($SERVE_ENV)"
$S "tmux kill-session -t vllm 2>/dev/null; tmux new -d -s vllm '$SERVE_ENV bash ~/aw/serve.sh > ~/aw/vllm.log 2>&1'"
$S 'for i in $(seq 90); do grep -qE "Application startup complete|Error" ~/aw/vllm.log && break; sleep 10; done
    grep -E "GPU KV cache size|startup complete|Error" ~/aw/vllm.log | tail -3'
$S 'grep -q "Application startup complete" ~/aw/vllm.log' || { echo "server failed; see: ssh -F $CFG aw tail -50 aw/vllm.log"; exit 1; }

echo "== tunnel localhost:8000 -> aw:8000"
pkill -f "ssh -F $CFG -f -N" 2>/dev/null || true
ssh -F "$CFG" -f -N -o ExitOnForwardFailure=yes -L 8000:localhost:8000 aw
echo "ready. next: uv run python scripts/lambda/smoke.py"
