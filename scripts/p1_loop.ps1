# Keeps the P1 full benchmark (decision D2: free tiers only) collecting until every cell is done.
#
# Free-tier quotas reset daily, and the runner stops a lane once it hits its quota or daily cap. This loop relaunches it.
# Run two loops, one per process, so each process has its own run directory:
#   .\scripts\p1_loop.ps1 -Group cloud   # flash-lite and both Groq models -> full-2026-11-16; OpenRouter -> pilot set
#   .\scripts\p1_loop.ps1 -Group local   # both Ollama models -> full-2026-11-16-local
# Ollama has its own run directory because the runner appends to one observations file per run, and
# two processes must never append to the same file. Merge the two directories when analysing
# (dedupe on item_id + model, keeping the first). The pilot-era full run also holds 69 early qwen3 rows.
#
# Each cycle waits $WaitMinutes after the runner exits. The loop ends after $MaxCycles cycles, or after
# $IdleLimit cycles in a row with no new observations. The cloud default (15 cycles, about 30 hours) outlasts
# a day on which every free quota is already spent. The local default (2) stops soon after Ollama finishes.
param(
    [Parameter(Mandatory = $true)][ValidateSet("cloud", "local")][string]$Group,
    [int]$WaitMinutes = 120,
    [int]$MaxCycles = 250,
    [int]$IdleLimit = 0,
    # Cloud lanes for the full run. Drop one (e.g. -CloudLanes gemini) while its provider key is being replaced.
    [string]$CloudLanes = "gemini,groq"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "X:/modelmesh-cache/hf" }

function Count-Lines([string]$path) {
    if (Test-Path $path) { return (Get-Content $path | Measure-Object -Line).Lines } else { return 0 }
}

if ($Group -eq "cloud") {
    $runDir = "data/benchmark/full/full-2026-11-16"
} else {
    $runDir = "data/benchmark/full/full-2026-11-16-local"
}
New-Item -ItemType Directory -Force $runDir | Out-Null
if (-not (Test-Path "$runDir/key_overrides.json")) {
    Copy-Item "data/benchmark/full/full-2026-11-16/key_overrides.json" "$runDir/key_overrides.json"
}
$observations = "$runDir/observations.jsonl"
if ($IdleLimit -le 0) { $IdleLimit = if ($Group -eq "cloud") { 15 } else { 2 } }
$idle = 0
for ($cycle = 1; $cycle -le $MaxCycles; $cycle++) {
    $before = Count-Lines $observations
    $stamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content "$runDir/loop.log" "$stamp cycle $cycle start, observations $before"
    if ($Group -eq "cloud") {
        & .\scripts\pilot.ps1 -Profile full -RunId full-2026-11-16 -Lanes $CloudLanes -Exclude gemini/gemini-3.6-flash *>> "$runDir/run.console.log"
        # OpenRouter's two :free models stay on the pilot set (D2); its daily cap is 50 requests.
        & .\scripts\pilot.ps1 -Lanes openrouter *>> "data/benchmark/pilot/pilot-2026-11-11/run.openrouter.console.log"
    } else {
        & .\scripts\pilot.ps1 -Profile full -RunId full-2026-11-16-local -Lanes ollama *>> "$runDir/run.console.log"
    }
    $after = Count-Lines $observations
    Add-Content "$runDir/loop.log" "$((Get-Date).ToUniversalTime().ToString('o')) cycle $cycle end, observations $after"
    if ($after -eq $before) { $idle++ } else { $idle = 0 }
    if ($idle -ge $IdleLimit) {
        Add-Content "$runDir/loop.log" "no new observations in $IdleLimit cycles: stopping"
        break
    }
    Start-Sleep -Seconds ($WaitMinutes * 60)
}
