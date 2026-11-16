"""Runs the gateway in-process for load tests (option B in loadtests/README.md).

    uv run python loadtests/serve.py --port 8200

Configuration comes from the environment, as for the container. On Windows, psycopg's async driver cannot run on
the Proactor event loop that ``uvicorn`` would pick, so this starts uvicorn on a selector loop instead.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = uvicorn.Config(
        "modelmesh.api.app:app_factory", factory=True, host=args.host, port=args.port, log_level="warning"
    )
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
