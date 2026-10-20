#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/env.sh"
exec uv run --project "$SFHACK_ROOT" python -m pytest "$SFHACK_ROOT/tests" "$@"
