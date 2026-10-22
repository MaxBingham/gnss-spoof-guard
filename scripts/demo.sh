#!/usr/bin/env bash
# One-click live demo (macOS / Linux):  bash scripts/demo.sh [extra demo_live args]
# Offline: the Berkeley map package ships in the repo (sfhack/world/sample_berkeley), uv never touches the network.
# Fallback if the viewer fails: play $SFHACK_DATA/demo/v2_demo.mp4 (see DEMO.md).
set -euo pipefail
. "$(dirname "$0")/env.sh"
export UV_OFFLINE=1
# Plain python on every OS: the viewer is an OpenCV window over an offscreen MuJoCo render (no mjpython;
# its Cocoa window has to stay on the main thread).
exec uv run --frozen --project "$SFHACK_ROOT" python "$SFHACK_ROOT/demos/demo_live.py" "$@"
