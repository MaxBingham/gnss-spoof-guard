#!/usr/bin/env bash
# One-time setup on macOS / Linux: Python 3.12 (uv-managed) + dependencies into ~/.venvs/sf-hack-v3.
#   bash scripts/setup.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with: brew install uv   (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  exit 1
fi
uv python install 3.12
uv sync --project "$SFHACK_ROOT"
uv run --project "$SFHACK_ROOT" python -c "import mujoco, cv2, numpy; print('mujoco', mujoco.__version__, '| opencv', cv2.__version__, '| numpy', numpy.__version__)"
echo "venv: $UV_PROJECT_ENVIRONMENT"
echo "data: $SFHACK_DATA"
