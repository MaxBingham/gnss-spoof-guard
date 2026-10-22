# Run one demo by number or name, e.g.:  powershell -ExecutionPolicy Bypass -File "scripts\run_demo.ps1" 01 --headless
param([Parameter(Mandatory = $true)][string]$Demo, [Parameter(ValueFromRemainingArguments = $true)]$Rest)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$script = Get-ChildItem -Path (Join-Path $SFHACK_ROOT "demos") -Filter "$Demo*.py" | Select-Object -First 1
if (-not $script) { Write-Host "No demo matching '$Demo' in demos\"; exit 1 }
uv run --project "$SFHACK_ROOT" python "$($script.FullName)" @Rest
exit $LASTEXITCODE
