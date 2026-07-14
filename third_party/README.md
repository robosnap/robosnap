# Third-party Layout

RoboSnap keeps adapted GUI dependencies under `third_party/`. The automatic-pipeline installer fetches its larger pinned sources.

Release layout:

```text
third_party/
  sam3/                         # modified SAM3 source snapshot
  sam-3d-objects/               # modified SAM3D source snapshot
  Hunyuan3D-Part/               # contains P3-SAM and XPart/partgen
  vggt/                         # fetched by scripts/setup_auto_sources.sh
  lyra/                         # fetched and patched by scripts/setup_auto_sources.sh
```

VGGT and Lyra are git-ignored source checkouts. Their URLs and commits are fixed in `scripts/setup_auto_sources.sh`; model weights stay under `checkpoints/`.
Upstream demo media, evaluation data, notebooks, and duplicate Kaolin/mip-splatting source trees are omitted from the release snapshot. Runtime dependencies are installed by the environment scripts.

Policy for release:

- Keep the upstream license files inside each vendored third-party directory.
- Record upstream URL, commit/tag, local source path, and local modifications in `manifest.yaml`.
- Do not commit checkpoints or model weights. Put them under `checkpoints/` and document download commands.
- Keep local compatibility changes as tracked patches or in the adapted source snapshot.

## Release entrypoints

Use the RoboSnap wrappers under `robosnap/` for pipeline execution. Vendored demo media, notebooks, evaluation data, and duplicate dependency sources are intentionally excluded.

## License audit

Before publishing, run:

```bash
python3 scripts/gui/python/audit_third_party_licenses.py
python3 scripts/gui/python/audit_third_party_licenses.py --fail-on-missing
python3 scripts/gui/python/audit_third_party_licenses.py --all-components
```

Run `--all-components` after `scripts/setup_auto_sources.sh` has fetched VGGT and Lyra.
