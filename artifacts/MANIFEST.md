# Router artifacts

Artifact files are gitignored; only this manifest is committed. Each artifact is one JSON document read by
`modelmesh.ml.artifacts.RouterArtifact`. Register one with `POST /admin/v1/router-models`; the gateway refuses
a file whose sha256 no longer matches the registered value.

| File | Version | Algorithm | sha256 | Source | Status |
|---|---|---|---|---|---|
| `router-p0-primary-knn.json` | `p0-primary-knn` | kNN | `da1a0816c701f631c356a092bdb520e456ee40ded03445502e8063553e24e5cc` | `python -m modelmesh.ml.evaluation.p0` on P0 pilot data, feature version 2 (2026-11-16) | Preliminary. Fails the ADR-0004 criterion and the ECE gate; must not be promoted |
| `router-p0-with_f1_only_synthesis-knn.json` | `p0-with_f1_only_synthesis-knn` | kNN | `8137a714995015d1073e284dd8e75f94eedf263f78f5687f4c4a8a913d2f18ee` | sensitivity variant, feature version 2 | Preliminary; analysis only |

Superseded (feature version 1, 2026-11-15; the gateway refuses to load them): `router-p0-primary-logreg.feature-v1.json` (`960b8041…`) and `router-p0-with_f1_only_synthesis-logreg.feature-v1.json` (`bba79791…`).
