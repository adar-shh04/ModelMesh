# Runs the same gates as CI. Postgres and Redis must be up: docker compose up -d postgres redis
$ErrorActionPreference = "Stop"
foreach ($step in @(
    @("ruff", "check", "src", "tests", "migrations"),
    @("ruff", "format", "--check", "src", "tests", "migrations"),
    @("mypy"),
    @("lint-imports"),
    @("pytest", "tests")
)) {
    uv run @step
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
