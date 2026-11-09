"""No runtime dependency may be AGPL-licensed (DECISIONS D5), and every one must be compatible with MIT (D6)."""

from __future__ import annotations

import re
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

DISALLOWED = re.compile(r"\b(AGPL|Affero|SSPL|RSAL|BUSL|Commons Clause)\b", re.IGNORECASE)
# Strong copyleft that would constrain distribution of ModelMesh; LGPL libraries used as-is remain compatible.
COPYLEFT = re.compile(r"(?<![AL])\bGPL\b|GNU General Public", re.IGNORECASE)
PERMISSIVE_OR_WEAK = re.compile(
    r"MIT|BSD|Apache|ISC|Python Software Foundation|PSF|MPL|Mozilla|LGPL|Lesser|Unlicense|Zlib|CC0|HPND|0BSD|Boost",
    re.IGNORECASE,
)


# Packages whose metadata omits a licence, checked by hand against the upstream repository.
VERIFIED_UPSTREAM = {"py-rust-stemmers": "MIT (github.com/qdrant/py-rust-stemmers, checked 2026-11-09)"}


def runtime_closure() -> set[str]:
    seen: set[str] = set()
    pending = ["modelmesh"]
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen.add(name)
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return seen - {"modelmesh"}


def licence_text(name: str) -> str:
    meta = metadata.metadata(name)
    classifiers = " ".join(value for value in meta.get_all("Classifier", []) if value.startswith("License"))
    return " ".join(filter(None, [meta.get("License-Expression"), (meta.get("License") or "")[:300], classifiers]))


def test_runtime_dependencies_have_acceptable_licences() -> None:
    packages = runtime_closure()
    assert len(packages) > 40
    problems = {}
    for package in sorted(packages):
        text = licence_text(package)
        if DISALLOWED.search(text) or COPYLEFT.search(text):
            problems[package] = text
        elif not PERMISSIVE_OR_WEAK.search(text) and package not in VERIFIED_UPSTREAM:
            problems[package] = f"unrecognised: {text!r}"
    assert not problems, problems
