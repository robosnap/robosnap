# Third-party Layout

RoboSnap keeps modified third-party code under `third_party/` instead of expecting users to clone upstream repositories by hand.

Current GUI release layout:

```text
third_party/
  sam3/                         # modified SAM3 source snapshot
  sam-3d-objects/               # modified SAM3D source snapshot
  Hunyuan3D-Part/               # contains P3-SAM and XPart/partgen
```

Local staged snapshots for later releases may also exist here, but they are git-ignored for the GUI-first release: DiffusionLight, FoundationPose, vggt, lyra, vipe, and Grounded-SAM-2.

Policy for release:

- Keep the upstream license files inside each vendored third-party directory.
- Record upstream URL, commit/tag, local source path, and local modifications in `manifest.yaml`.
- Do not commit checkpoints or model weights. Put them under `checkpoints/` and document download commands.
- For heavily modified repos such as SAM3 and SAM3D, release a modified source snapshot or a fork/submodule pinned to the exact commit. Do not ask users to clone the original repo and manually rediscover local changes.

## Release entrypoints

Use the RoboSnap wrappers under `robosnap/` for pipeline execution. Some vendored demos, notebooks, and tests are kept as source context and are not release entrypoints. Do not prune those source folders without confirming first.

## License audit

Before publishing, run:

```bash
python3 scripts/gui/python/audit_third_party_licenses.py
python3 scripts/gui/python/audit_third_party_licenses.py --fail-on-missing
python3 scripts/gui/python/audit_third_party_licenses.py --all-components
```

The first command reports the GUI release state without blocking validation. The second command is the final GUI-release mode. The `--all-components` command is for staged dependencies that are not part of the first public GUI release. If it reports a vendored component without a visible `LICENSE`, `COPYING`, `NOTICE`, or `AUTHORS` file, resolve that manually before public release.
