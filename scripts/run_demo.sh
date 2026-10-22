#!/usr/bin/env bash
# Run one demo by number or name, e.g.:  bash scripts/run_demo.sh 01 --headless
set -euo pipefail
. "$(dirname "$0")/env.sh"
demo="${1:?usage: run_demo.sh <number|name> [args...]}"; shift
script="$(ls "$SFHACK_ROOT/demos/${demo}"*.py 2>/dev/null | head -n 1 || true)"
[ -n "$script" ] || { echo "No demo matching '$demo' in demos/"; exit 1; }
# The MuJoCo viewer on macOS needs mjpython (ships with the mujoco wheel); headless runs use python.
runner=python
if [[ "$(uname)" == "Darwin" && " $* " != *" --headless "* && "$(basename "$script")" != demo_live.py ]]; then
  runner=mjpython  # demo_live.py draws its own OpenCV window and needs plain python
fi
exec uv run --project "$SFHACK_ROOT" "$runner" "$script" "$@"
