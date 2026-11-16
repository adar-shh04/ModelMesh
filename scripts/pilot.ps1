# Runs (or resumes) the M5 pilot against the local gateway, within free-tier limits.
# The benchmark key is read from .cache/bench_key (gitignored) and is never printed.
#
# -Lanes picks the providers to run, comma-separated, for example:
#   .\scripts\pilot.ps1 -Lanes gemini,openrouter,ollama
# Leave a lane out when its free quota is needed elsewhere (Groq's daily quota is shared with Mezvuat).
#
# The full benchmark (P1, decision D2: free tiers only) uses -Profile full and leaves out the models
# that stay on the pilot set:
#   .\scripts\pilot.ps1 -Profile full -RunId full-2026-11-15 -Lanes gemini,groq,ollama -Exclude gemini/gemini-3.6-flash
param(
    [ValidateSet("pilot", "full")][string]$Profile = "pilot",
    [string]$RunId = "pilot-2026-11-11",
    [string]$Lanes = "gemini,groq,openrouter,ollama",
    [string]$Exclude = ""
)
$ErrorActionPreference = "Stop"
$known = @("gemini", "groq", "openrouter", "ollama")
# A comma-separated string or a PowerShell array both arrive here; normalise to a list of lane names.
$selected = @($Lanes -split "[,\s]+" | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
$unknown = @($selected | Where-Object { $known -notcontains $_ })
if ($unknown.Count -gt 0) { throw "Unknown lane(s): $($unknown -join ', '). Known lanes: $($known -join ', ')." }
if ($selected.Count -eq 0) { throw "Select at least one lane." }

$env:MODELMESH_BENCH_BASE_URL = "http://127.0.0.1:8100"
$env:MODELMESH_BENCH_API_KEY = (Get-Content .cache/bench_key -Raw).Trim()
$allModels = @(
    "gemini/gemini-3.1-flash-lite", "gemini/gemini-3.6-flash",
    "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b",
    "openrouter/nvidia/nemotron-3.5-super-vl:free", "openrouter/liquid/lfm-3-1.7b:free",
    "ollama/qwen3:4b", "ollama/llama3.2:3b"
)
$excluded = @($Exclude -split "[,\s]+" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$unknownModels = @($excluded | Where-Object { $allModels -notcontains $_ })
if ($unknownModels.Count -gt 0) { throw "Unknown model(s) in -Exclude: $($unknownModels -join ', ')." }
$models = @($allModels | Where-Object {
    ($selected -contains ($_ -split "/", 2)[0]) -and ($excluded -notcontains $_)
}) -join ","
if (-not $models) { throw "No models left to run after -Lanes and -Exclude." }
# Groq is shared with another client: 8 RPM across both models and at most 400 requests per invocation.
# OpenRouter free models share 50 requests per day on an account without credits.
$paceByLane = @{
    "groq" = "groq=8:400:120000"; "gemini" = "gemini=8::120000"
    "openrouter" = "openrouter=10:50:120000"; "ollama" = "ollama=::180000"
}
$runArgs = @("-m", "modelmesh.ml.benchmark", "run", "--profile", $Profile, "--run-id", $RunId, "--models", $models)
foreach ($lane in $selected) { $runArgs += @("--pace", $paceByLane[$lane]) }
if ($selected -contains "ollama") { $runArgs += @("--warmup", "ollama") }
Write-Host "Profile: $Profile  Run: $RunId  Models: $models"
& .venv\Scripts\python.exe @runArgs
