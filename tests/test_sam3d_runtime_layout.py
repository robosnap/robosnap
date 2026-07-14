from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM3D_PACKAGE = ROOT / "third_party" / "sam-3d-objects" / "sam3d_objects"


def test_sam3d_runtime_modules_are_packaged():
    assert (SAM3D_PACKAGE / "inference.py").is_file()
    assert (SAM3D_PACKAGE / "load_images_and_masks.py").is_file()
    assert "sam3d_objects.init" not in (SAM3D_PACKAGE / "__init__.py").read_text()


def test_sam3d_entrypoints_do_not_depend_on_notebook_tree():
    for name in (
        "image2glb.py",
        "image2glb_metric.py",
        "run_inference.py",
        "run_inference_weighted.py",
    ):
        source = (SAM3D_PACKAGE / name).read_text()
        assert '"notebook"' not in source
        assert "from inference import" not in source
        assert "from load_images_and_masks import" not in source
