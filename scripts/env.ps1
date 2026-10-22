# Project environment for Windows PowerShell. Dot-source it:  . "$PSScriptRoot\env.ps1"
# The repo lives inside OneDrive, so the virtualenv and all generated data live OUTSIDE it.

# Always the V3 venv (a shell that sourced V2's env.ps1 before would otherwise keep V2's venv).
$env:UV_PROJECT_ENVIRONMENT = Join-Path $HOME ".venvs\sf-hack-v3"
if (-not $env:SFHACK_DATA) { $env:SFHACK_DATA = Join-Path $HOME "sf-hack-data" }
# Hardlinks from the uv cache into the venv are fine (same drive); copy mode avoids OneDrive quirks.
$env:UV_LINK_MODE = "copy"
$env:PYTHONUTF8 = "1"

$SFHACK_ROOT = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path $env:SFHACK_DATA | Out-Null
