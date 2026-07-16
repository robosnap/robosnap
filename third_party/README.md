# Third-party Sources

Initialize the pinned source revisions with:

```bash
git submodule update --init --recursive
```

The Conda installers call `scripts/setup_auto_sources.sh` automatically. The same script fetches the exact revisions when Git metadata is unavailable inside a Docker build.

| Path | Source | Revision |
| --- | --- | --- |
| `sam3` | [`robosnap/sam3`](https://github.com/robosnap/sam3) | `16fff334254b7de76c2ae2fe8968fd85afc7d815` |
| `sam-3d-objects` | [`robosnap/sam-3d-objects`](https://github.com/robosnap/sam-3d-objects) | `79dbb1f59adb7d4c4e16b1fe55ee38f52a1d12f0` |
| `lyra` | [`robosnap/lyra`](https://github.com/robosnap/lyra) | `812d586ac7978b41c6dee560f99b07b1007e26fa` |
| `vggt` | [`facebookresearch/vggt`](https://github.com/facebookresearch/vggt) | `44b3afbd1869d8bde4894dd8ea1e293112dd5eba` |
| `Hunyuan3D-Part` | [`robosnap/Hunyuan3D-Part`](https://github.com/robosnap/Hunyuan3D-Part) | `b58568a328202bde2921e7d7e01368c7f558ecb3` |

SAM3, SAM-3D-Objects, Hunyuan3D-Part, and Lyra use RoboSnap forks because the release depends on source changes in those repositories. VGGT is unmodified and pinned to the official revision matching the validated pipeline. Model weights remain under `checkpoints/` and are not committed.

`DiffusionLight`, `FoundationPose`, `Grounded-SAM-2`, and a separate top-level VIPE checkout are not dependencies of the released GUI or automatic pipeline.
