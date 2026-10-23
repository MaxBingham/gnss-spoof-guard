# V3 one-click demo (Windows):  powershell -ExecutionPolicy Bypass -File scripts\demo_v3.ps1 [scene] [extra v3_live args]
# Scenes (see DEMO_V3.md):
#   s1         S1 San Francisco, INS + TRN + IBN, live viewer (default)
#   s1-ins     S1 dead reckoning only: the estimate drifts, the drone misses B
#   s1-outage  S1 all sources, TRN + IBN switched off for 40 s mid-cruise, then back on
#   s2         S2 Carpathian foothills, 8.4 km, INS + TRN + landing-patch IBN, 8x
#   videos     record the backup videos (S1 + S2) to $env:SFHACK_DATA\v3\demo\
# Needs the world packages: uv run python -m sfhack.world.build_v3 s1 s2 (once, online).
param([string]$Scene = "s1", [Parameter(ValueFromRemainingArguments = $true)]$Rest)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"
$env:UV_OFFLINE = "1"
$live = Join-Path $SFHACK_ROOT "demos\v3_live.py"

function Live([string[]]$a) {
    uv run --frozen --project "$SFHACK_ROOT" python $live @a @Rest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Scene) {
    "s1"        { Live @("--world", "s1", "--config", "ins_trn_ibn") }
    "s1-ins"    { Live @("--world", "s1", "--config", "ins") }
    "s1-outage" { Live @("--world", "s1", "--config", "ins_trn_ibn", "--events", "60:T,60:I,100:T,100:I") }
    "s2"        { Live @("--world", "s2", "--config", "ins_trn_ibn", "--speed", "8") }
    "videos"    {
        Live @("--world", "s1", "--config", "ins_trn_ibn", "--speed", "2", "--record")
        Live @("--world", "s1", "--config", "ins_trn_ibn", "--speed", "2", "--events", "60:T,60:I,100:T,100:I",
               "--record", (Join-Path $env:SFHACK_DATA "v3\demo\v3_s1_outage_seed0.mp4"))
        Live @("--world", "s2", "--config", "ins_trn_ibn", "--speed", "8", "--record")
    }
    default     { Write-Host "unknown scene '$Scene' (s1 | s1-ins | s1-outage | s2 | videos)"; exit 1 }
}
