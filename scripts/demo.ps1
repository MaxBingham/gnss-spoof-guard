# One-click live demo (Windows):  powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 [extra demo_live args]
# Offline: the Berkeley map package ships in the repo (sfhack/world/sample_berkeley), uv never touches the network.
# Fallback if the viewer fails: play $env:SFHACK_DATA\demo\v2_demo.mp4 (see DEMO.md).
param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$env:UV_OFFLINE = "1"
uv run --frozen --project "$SFHACK_ROOT" python "$SFHACK_ROOT\demos\demo_live.py" @Rest
exit $LASTEXITCODE
