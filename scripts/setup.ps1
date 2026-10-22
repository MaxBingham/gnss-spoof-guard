# One-time setup on Windows: installs Python 3.12 (uv-managed) and all dependencies
# into $HOME\.venvs\sf-hack-v3 (outside OneDrive).
#   powershell -ExecutionPolicy Bypass -File "scripts\setup.ps1"
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install it with: winget install --id astral-sh.uv -e"
    exit 1
}
uv python install 3.12
uv sync --project "$SFHACK_ROOT"
uv run --project "$SFHACK_ROOT" python -c "import mujoco, cv2, numpy; print('mujoco', mujoco.__version__, '| opencv', cv2.__version__, '| numpy', numpy.__version__)"
Write-Host "venv: $env:UV_PROJECT_ENVIRONMENT"
Write-Host "data: $env:SFHACK_DATA"
