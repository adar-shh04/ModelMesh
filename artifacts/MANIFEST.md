# Router artifacts

Artifact files are gitignored; only this manifest is committed. Each artifact is one JSON document read by
`modelmesh.ml.artifacts.RouterArtifact`. Register one with `POST /admin/v1/router-models`; the gateway refuses
a file whose sha256 no longer matches the registered value.

| File | Version | Algorithm | sha256 | Source | Status |
|---|---|---|---|---|---|
| `router-p0-primary-logreg.json` | `p0-primary-logreg` | logistic regression per model | `960b804165ac2c2f9179f92acff6d79e1f03bfd6c9822765b35bf69cf03b9c0d` | `python -m modelmesh.ml.evaluation.p0` on P0 pilot data | Preliminary. Fails the ADR-0004 criterion and the ECE gate; must not be promoted |
| `router-p0-with_f1_only_synthesis-logreg.json` | `p0-with_f1_only_synthesis-logreg` | logistic regression per model | `bba7979172d2fdb22cfc3ff513a79e6578d8482b1ce2ecfed1f061d25220daec` | sensitivity variant | Preliminary; analysis only |
