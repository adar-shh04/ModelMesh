# Runs the same gates as CI. Postgres and the test Redis must be up: docker compose --profile test up -d postgres redis-test
$ErrorActionPreference = "Stop"
foreach ($step in @(
    @("ruff", "check", "src", "tests", "migrations", "loadtests"),
    @("ruff", "format", "--check", "src", "tests", "migrations", "loadtests"),
    @("mypy"),
    @("lint-imports"),
    @("pytest", "tests")
)) {
    uv run @step
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
