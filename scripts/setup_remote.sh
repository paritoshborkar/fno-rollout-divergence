#!/usr/bin/env bash
# Bootstrap a fresh Linux GPU instance (e.g. LambdaLabs) to run this repo.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/paritoshborkar/fno-rollout-divergence/refs/heads/main/scripts/setup_remote.sh | bash
# or copy this file to the instance and run it directly.
#
# Override REPO_URL/REPO_DIR as env vars if needed, e.g.:
#   REPO_URL=https://github.com/paritoshborkar/fno-rollout-divergence.git bash setup_remote.sh
set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:paritoshborkar/fno-rollout-divergence.git}"
REPO_DIR="${REPO_DIR:-$HOME/fno-rollout-divergence}"

echo "==> Installing build essentials"
sudo apt-get update -y
sudo apt-get install -y build-essential git curl

echo "==> Installing uv"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "==> Installing Julia (juliaup)"
if ! command -v juliaup &>/dev/null; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
export PATH="$HOME/.juliaup/bin:$PATH"

echo "==> Cloning repo"
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "    $REPO_DIR already exists, skipping clone"
fi
cd "$REPO_DIR"

echo "==> Installing Python dependencies (uv sync)"
uv sync

echo "==> Installing Julia dependencies (Oceananigans project)"
julia --project=julia/oceananigans -e 'using Pkg; Pkg.instantiate()'

echo "==> Done. Repo ready at $REPO_DIR"
echo "    Start a new shell (or 'source ~/.bashrc') to pick up uv/julia on PATH permanently."
