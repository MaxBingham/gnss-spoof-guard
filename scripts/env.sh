# Project environment for macOS / Linux. Source it:  . "$(dirname "$0")/env.sh"
# The virtualenv and all generated data live outside the repo (and outside any synced folder).

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/sf-hack}"
export SFHACK_DATA="${SFHACK_DATA:-$HOME/sf-hack-data}"
export UV_LINK_MODE=copy
export PYTHONUTF8=1

SFHACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export SFHACK_ROOT
mkdir -p "$SFHACK_DATA"
