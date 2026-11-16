"""P1 command line. Offline and read-only towards the run directories; only ``judge --execute`` sends requests
(to the gateway's public API, like the benchmark runner).

    python -m modelmesh.ml.p1 freeze-split [--write]   # build the pre-registered split; --write only if unfrozen
    python -m modelmesh.ml.p1 verify-split             # check the committed manifests against the pinned hashes
    python -m modelmesh.ml.p1 merge                    # merged snapshot -> data/benchmark/full/p1-merged/
    python -m modelmesh.ml.p1 status [--json]          # coverage, failures, deferrals, yield, projected finish
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m modelmesh.ml.p1")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-split", help="build the pre-registered P1 split from item contents")
    freeze.add_argument("--write", action="store_true", help="write the manifests (refused once frozen)")
    commands.add_parser("verify-split", help="check the committed split manifests against the frozen hashes")
    commands.add_parser("merge", help="merge the P1 run directories into one snapshot (outside them)")
    status = commands.add_parser("status", help="read-only P1 progress: coverage, failures, yield, projection")
    status.add_argument("--json", action="store_true", help="print the full JSON instead of tables")
    args = parser.parse_args(argv)

    if args.command == "freeze-split":
        from modelmesh.ml.p1.split import freeze as freeze_split

        print(json.dumps(freeze_split(write=args.write), indent=1))
    elif args.command == "verify-split":
        from modelmesh.ml.p1.split import verify_frozen

        split = verify_frozen()
        print(json.dumps({"ok": True, "split_manifest_sha256": split["split_manifest_sha256"], **split["counts"]}))
    elif args.command == "merge":
        from modelmesh.ml.p1.data import merge, write_merged
        from modelmesh.ml.p1.split import load_items

        rows, report = merge(load_items())
        target = write_merged(rows, report)
        print(json.dumps({"written": str(target), **report.as_dict()}, indent=1))
    elif args.command == "status":
        from modelmesh.ml.p1.data import render_status
        from modelmesh.ml.p1.data import status as p1_status
        from modelmesh.ml.p1.split import load_items

        doc = p1_status(load_items())
        print(json.dumps(doc, indent=1) if args.json else render_status(doc))
    else:  # pragma: no cover - argparse rejects unknown commands
        sys.exit(2)


if __name__ == "__main__":
    main()
