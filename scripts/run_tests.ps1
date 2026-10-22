# Run the test suite:  powershell -ExecutionPolicy Bypass -File "scripts\run_tests.ps1"
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
uv run --project "$SFHACK_ROOT" python -m pytest "$SFHACK_ROOT\tests" @args
exit $LASTEXITCODE
