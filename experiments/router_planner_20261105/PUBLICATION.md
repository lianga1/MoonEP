# Publication and provenance

This directory preserves the original GPU-executed CUDA/Python modules and raw
timing, correctness, and failure evidence. Historical absolute paths in raw
evidence identify the measurement environment; they are not required runtime
locations. No credential material, model weights, original token text, compiled
extensions, Git object databases, or complete router-dump archives are included.
The NPZ fixtures contain expert IDs, execution-rank plans, and numerical audits.

`PACKAGE_MANIFEST.json` records original and published hashes. Portability-only
edits are listed separately; they do not rewrite the hashes recorded at GPU test
time or claim that a different kernel was measured. `inputs/MoonEP` is the tested
source snapshot (with its license), not a replacement for the parent fork API.
The optional DeepEP snapshots retain their licenses and recorded commits.

Only the documented entry points are part of this delivery. Legacy provenance
archives and logs retain historical paths and failed attempts. Source/archive
manifests from the original workspace remain historical evidence; the package
manifest is the current distribution inventory.
