import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_render_wrapper_help():
    env = os.environ.copy()
    env["PY_RENDER"] = sys.executable
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "render_gravity_aligned_scene.sh"), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "render_layered_scene.py" in result.stdout


def test_auto_pipeline_launcher_uses_only_auto_config(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nprintf 'root=%s\\n' \"${ROBOSNAP_ROOT}\"\nprintf 'arg=%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    auto_env = tmp_path / "auto.env"
    auto_env.write_text(
        "INPUT_IMAGE=${ROBOSNAP_ROOT}/examples/test1.png\n"
        "OUTPUT_DIR=${ROBOSNAP_ROOT}/outputs/automatic\n"
        "DEVICE=cpu\n",
        encoding="utf-8",
    )
    gui_env = tmp_path / "gui.env"
    gui_env.write_text("PY_AUTO=/definitely/missing/python\n", encoding="utf-8")

    env = os.environ.copy()
    env.pop("ROBOSNAP_ROOT", None)
    env.update(
        {
            "PY_AUTO": str(fake_python),
            "ROBOSNAP_AUTO_ENV_FILE": str(auto_env),
            "ROBOSNAP_ENV_FILE": str(gui_env),
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_auto_pipeline.sh"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"root={ROOT}" in result.stdout
    assert "arg=robosnap.pipeline.auto_layered_scene" in result.stdout
    assert "/definitely/missing/python" not in result.stdout
