# Frozen MPS paper-suite baseline

This directory is the compact, auditable snapshot of the completed 65-stage overnight suite. The full checkpoint tree is preserved at the `raw_path` recorded in `SNAPSHOT.json`. Verify either tree with `shasum -a 256 -c MANIFEST.sha256`. New flow experiments must use `models/runs/neurreps_flow_v1` and must not write into this snapshot.
