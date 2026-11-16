"""Creates a load-test tenant and API key on the target gateway and writes the key to a local file.

    uv run python loadtests/setup_key.py --base-url http://127.0.0.1:8200 --profile fake

The admin token is read from ``MODELMESH_ADMIN_TOKEN`` (the target instance's, not the benchmark gateway's).
The key is written to ``.cache/load_key`` (gitignored) and never printed. The key's limits are explicit
overrides of the D1 defaults, as DECISIONS.md requires for load tests; they are printed so the run can record
them. ``--profile real`` keeps D1-sized limits, so a real-provider run cannot outspend them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
PROTECTED = {REPO / ".cache" / "bench_key"}

PROFILES = {
    # Fake provider: limits far above the target load, so the gateway's own limiter never shapes the result.
    "fake": {"rpm": 1_000_000, "tpm": 1_000_000_000, "concurrency": 2_000, "daily_budget_usd": 10_000.0},
    # Real providers: the D1 defaults, explicitly.
    "real": {"rpm": 20, "tpm": 50_000, "concurrency": 10, "daily_budget_usd": 0.25},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="the target gateway, e.g. http://127.0.0.1:8200")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fake")
    parser.add_argument("--tenant", default="load-test")
    parser.add_argument("--out", type=Path, default=REPO / ".cache" / "load_key")
    args = parser.parse_args()
    out = args.out.resolve()
    if out in PROTECTED:
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    token = os.environ.get("MODELMESH_ADMIN_TOKEN")
    if not token:
        print("set MODELMESH_ADMIN_TOKEN to the target instance's admin token", file=sys.stderr)
        return 2
    admin = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=admin, timeout=10) as client:
        tenant = client.post("/admin/v1/tenants", json={"name": args.tenant})
        tenant.raise_for_status()
        limits = PROFILES[args.profile]
        key = client.post(
            f"/admin/v1/tenants/{tenant.json()['id']}/keys",
            json={"name": f"{args.tenant}-{args.profile}", "env": "test", "limits": limits},
        )
        key.raise_for_status()
        created = key.json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(created["key"], encoding="utf-8")
    with contextlib.suppress(OSError):  # best effort; Windows ignores most mode bits
        out.chmod(0o600)
    print(json.dumps({"key_id": created["id"], "key_file": str(out), "limits": created["limits"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
