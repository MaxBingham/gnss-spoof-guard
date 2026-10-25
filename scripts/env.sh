# Project environment for macOS / Linux. Source it:  . "$(dirname "$0")/env.sh"
# The virtualenv and all generated data live outside the repo (and outside any synced folder).

# Own venv for the V5 mission-planner branch (other branches run in parallel with their own venvs).
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/sf-hack-v5"
export SFHACK_DATA="${SFHACK_DATA:-$HOME/sf-hack-data}"
export UV_LINK_MODE=copy
export PYTHONUTF8=1

# No GPU device (headless CI, Cloud Agents): GLFW/EGL have nothing to bind.
# A preset MUJOCO_GL always wins, so a machine with a real context can override this.
if [[ -z "${MUJOCO_GL:-}" && "$(uname)" == "Linux" && ! -e /dev/nvidia0 && ! -d /dev/dri ]]; then
  export MUJOCO_GL=osmesa
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
fi

SFHACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export SFHACK_ROOT
mkdir -p "$SFHACK_DATA"
