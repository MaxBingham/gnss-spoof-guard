#!/usr/bin/env bash
# V3 one-click demo (macOS / Linux):  bash scripts/demo_v3.sh [scene] [extra v3_live args]
# Scenes (see DEMO_V3.md):
#   s1         S1 San Francisco, INS + TRN + IBN, live viewer (default)
#   s1-ins     S1 dead reckoning only: the estimate drifts, the drone misses B
#   s1-outage  S1 all sources, TRN + IBN switched off for 40 s mid-cruise, then back on
#   s2         S2 Carpathian foothills, 8.4 km, INS + TRN + landing-patch IBN, 8x
#   videos     record the backup videos (S1 + S2) to $SFHACK_DATA/v3/demo/
# Needs the world packages: uv run python -m sfhack.world.build_v3 s1 s2 (once, online).
set -euo pipefail
. "$(dirname "$0")/env.sh"
export UV_OFFLINE=1
scene="${1:-s1}"; shift || true
live="$SFHACK_ROOT/demos/v3_live.py"
# The MuJoCo viewer on macOS needs mjpython (ships with the mujoco wheel).
runner=python
if [[ "$(uname)" == "Darwin" && "$scene" != "videos" && " $* " != *" --headless "* && " $* " != *" --record"* ]]; then
  runner=mjpython
fi
run() { uv run --frozen --project "$SFHACK_ROOT" "$runner" "$live" "$@"; }

case "$scene" in
  s1)        run --world s1 --config ins_trn_ibn "$@" ;;
  s1-ins)    run --world s1 --config ins "$@" ;;
  s1-outage) run --world s1 --config ins_trn_ibn --events "60:T,60:I,100:T,100:I" "$@" ;;
  s2)        run --world s2 --config ins_trn_ibn --speed 8 "$@" ;;
  videos)
    run --world s1 --config ins_trn_ibn --speed 2 --record "$@"
    run --world s1 --config ins_trn_ibn --speed 2 --events "60:T,60:I,100:T,100:I" \
        --record "$SFHACK_DATA/v3/demo/v3_s1_outage_seed0.mp4" "$@"
    run --world s2 --config ins_trn_ibn --speed 8 --record "$@" ;;
  *) echo "unknown scene '$scene' (s1 | s1-ins | s1-outage | s2 | videos)"; exit 1 ;;
esac
