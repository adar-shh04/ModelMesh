"""Module boundaries from ADR-0001 that cannot be expressed as import-linter contracts."""

from __future__ import annotations

import subprocess
import sys


def test_runtime_import_does_not_load_offline_ml_libraries() -> None:
    script = (
        "import sys; import modelmesh.api.app; "
        "print([m for m in ('sklearn', 'mlflow', 'datasets', 'lightgbm', 'scipy', 'modelmesh.ml.benchmark', "
        "'modelmesh.ml.training', 'modelmesh.ml.evaluation') if m in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
    assert result.stdout.strip() == "[]"
