# Runs (or resumes) the M5 pilot against the local gateway, within free-tier limits.
# The benchmark key is read from .cache/bench_key (gitignored) and is never printed.
param([string]$RunId = "pilot-2026-11-11")
$ErrorActionPreference = "Stop"
$env:MODELMESH_BENCH_BASE_URL = "http://127.0.0.1:8100"
$env:MODELMESH_BENCH_API_KEY = (Get-Content .cache/bench_key -Raw).Trim()
$models = @(
    "gemini/gemini-3.1-flash-lite", "gemini/gemini-3.6-flash",
    "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b",
    "openrouter/nvidia/nemotron-3.5-super-vl:free", "openrouter/liquid/lfm-3-1.7b:free",
    "ollama/qwen3:4b", "ollama/llama3.2:3b"
) -join ","
# Groq is shared with another client: 8 RPM across both models and at most 400 requests per invocation.
# OpenRouter free models share 50 requests per day on an account without credits.
.venv\Scripts\python.exe -m modelmesh.ml.benchmark run --profile pilot --run-id $RunId --models $models `
    --pace "groq=8:400:120000" --pace "gemini=8::120000" --pace "openrouter=10:50:120000" --pace "ollama=::180000" `
    --warmup ollama
