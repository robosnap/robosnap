"""3D generation process bridge for the RoboSnap GUI."""

from __future__ import annotations

import os
from pathlib import Path


def _default_subprocess_env_for_python(python_executable, *, conda_env_name=None, pythonpath=None):
    env = os.environ.copy()
    if pythonpath:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{existing}" if existing else str(pythonpath)
    return env


def _first_top_mask(obj_dir: Path) -> Path | None:
    for mask_dir in sorted(p for p in obj_dir.glob("top*_mask") if p.is_dir()):
        masks = sorted(mask_dir.glob("*.png"))
        if masks:
            return masks[0]
    return None


def generate_3d_meshes_for_all_objects(
    out_dir: Path,
    config_path: str = None,
    max_frames: int = 20,
    progress_callback=None,
    *,
    py_asset: str = "python",
    sam3d_dir: Path | str | None = None,
    default_config: str | None = None,
    source_video: Path | str | None = None,
    _subprocess_env_for_python=_default_subprocess_env_for_python,
):
    """
    Generate 3D meshes for all objects in the output directory.
    This runs after user clicks End button in segmentation phase.
    
    Workflow:
    1. Check if multi_mask/object_name/object_name.glb exists - if yes, skip
    2. Run run_inference.py --compose_scene on multi_mask/
    3. Run image2glb.sh to get scale (saves to single_mask/)
    
    Args:
        out_dir: Base output directory (e.g., case1/multi_mask)
        config_path: Path to the 3D generation config yaml
        max_frames: Number of top frames to use for 3D reconstruction
        progress_callback: Optional callback function(step, total, message) for progress updates
        source_video: Original input video/image used to extract single_mask/image.png when needed
    
    Returns:
        dict with keys: success, message, generated_objects, logs
    """
    import subprocess
    import os

    PY_ASSET = py_asset
    SAM3D_DIR = Path(sam3d_dir).expanduser().resolve() if sam3d_dir is not None else Path(".").resolve()
    SAM3D_CONFIG = default_config
    
    if config_path is None:
        config_path = SAM3D_CONFIG
    
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)
        if progress_callback:
            progress_callback(0, 0, msg)

    def prepare_reference_image(single_mask_dir: Path):
        target_image_path = single_mask_dir / "image.png"
        if target_image_path.exists():
            return

        candidates = [out_dir / "video.mp4"]
        if source_video is not None:
            candidates.append(Path(source_video).expanduser())

        seen = set()
        for candidate in candidates:
            candidate = Path(candidate)
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.exists():
                continue

            single_mask_dir.mkdir(parents=True, exist_ok=True)
            log(f"[3D Gen] Extracting first frame from {candidate} to {target_image_path}")
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", str(candidate), "-vframes", "1", "-q:v", "2", str(target_image_path)],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and target_image_path.exists():
                return

            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            if detail:
                log(f"[3D Gen] ffmpeg warning: {detail[-1]}")

        log(f"[3D Gen] Warning: could not prepare reference image at {target_image_path}")
    
    # Step 1: Check if all mask objects already have meshes in multi_mask/object_name/.
    # A partial set of meshes is not enough to skip run_inference.py.
    object_dirs = []
    existing_objects = []
    missing_objects = []
    if out_dir.exists():
        for obj_dir in sorted(out_dir.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name.startswith(".") or obj_dir.name == "background":
                continue
            has_mask_dir = any(
                entry.is_dir() and entry.name.startswith("top") and entry.name.endswith("_mask")
                for entry in obj_dir.iterdir()
            )
            if not has_mask_dir:
                continue

            object_dirs.append(obj_dir)
            expected_glb = obj_dir / f"{obj_dir.name}.glb"
            if expected_glb.exists():
                existing_objects.append(obj_dir.name)
            else:
                missing_objects.append(obj_dir.name)

    if object_dirs and not missing_objects:
        log(f"[3D Gen] Found meshes for all {len(existing_objects)} mask objects in {out_dir}")
        log(f"[3D Gen] Objects: {existing_objects}")
        log(f"[3D Gen] Skipping run_inference.py - all meshes already exist!")
        
        # Still need to run image2glb.py for scale if scene_composed.glb doesn't exist
        single_mask_dir = out_dir.parent / "single_mask"
        scene_composed_glb = single_mask_dir / "scene_composed.glb"

        if not scene_composed_glb.exists():
            prepare_reference_image(single_mask_dir)

            # 把 multi_mask 各 object 子目录的 GLB 和 top*_mask 第一帧补齐到 single_mask/
            import shutil as _shutil
            single_mask_dir.mkdir(parents=True, exist_ok=True)
            for idx, obj_dir in enumerate(object_dirs):
                glb_src = obj_dir / f"{obj_dir.name}.glb"
                glb_dst = single_mask_dir / f"{idx}.glb"
                if glb_src.exists() and not glb_dst.exists():
                    _shutil.copy2(str(glb_src), str(glb_dst))
                    log(f"[3D Gen] Copied {glb_src.name} -> single_mask/{idx}.glb")
                mask_dst = single_mask_dir / f"{idx}.png"
                first_mask = _first_top_mask(obj_dir)
                if first_mask is not None and not mask_dst.exists():
                    _shutil.copy2(str(first_mask), str(mask_dst))
                    log(f"[3D Gen] Copied mask {first_mask.name} -> single_mask/{idx}.png")

            # Run image2glb.py for scale extraction and scene composition
            log(f"[3D Gen] Running image2glb.py for scale extraction...")
            image2glb = SAM3D_DIR / "sam3d_objects" / "image2glb.py"
            num_masks = len([f for f in single_mask_dir.glob("*.glb") if f.name != "scene_composed.glb"]) if single_mask_dir.exists() else 0
            try:
                proc = subprocess.Popen(
                    [PY_ASSET, str(image2glb),
                     "--config", config_path,
                     "--image", str(single_mask_dir / "image.png"),
                     "--mask_dir", str(single_mask_dir),
                     "--num_masks", str(num_masks),
                     "--seed", "43",
                     "--compose_scene",
                     "--scale_only"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=_subprocess_env_for_python(PY_ASSET, conda_env_name="PY_ASSET_CONDA_PREFIX", pythonpath=str(SAM3D_DIR))
                )
                for line in proc.stdout:
                    line = line.rstrip('\n')
                    if line:
                        log(line)
                proc.wait()
                if proc.returncode == 0:
                    log(f"[3D Gen] image2glb.py completed successfully!")
                else:
                    log(f"[3D Gen] image2glb.py warning: exited with code {proc.returncode}")
            except Exception as e:
                log(f"[3D Gen] image2glb.py error: {str(e)}")
        else:
            log(f"[3D Gen] scene_composed.glb already exists, skipping image2glb.py")
        
        return {
            "success": True,
            "message": f"Using {len(existing_objects)} existing object meshes",
            "generated_objects": existing_objects,
            "logs": logs
        }

    if object_dirs:
        log(f"[3D Gen] Found {len(existing_objects)}/{len(object_dirs)} mask objects with meshes in {out_dir}")
        if existing_objects:
            log(f"[3D Gen] Existing meshes: {existing_objects}")
        log(f"[3D Gen] Missing meshes: {missing_objects}")
        log(f"[3D Gen] Running run_inference.py to generate missing meshes")
    
    # Step 2: Run run_inference.py --compose_scene
    if not object_dirs:
        log(f"[3D Gen] No mask object folders found. Starting run_inference.py with --compose_scene")
    
    run_inference = SAM3D_DIR / "sam3d_objects" / "run_inference.py"
    
    cmd = [
        PY_ASSET,
        str(run_inference),
        "--input_path", str(out_dir),
        "--compose_scene",
        "--config", config_path,
    ]

    log(f"[3D Gen] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SAM3D_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_subprocess_env_for_python(PY_ASSET, conda_env_name="PY_ASSET_CONDA_PREFIX", pythonpath=str(SAM3D_DIR))
        )
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line:
                log(line)
        proc.wait()

        if proc.returncode != 0:
            log(f"[3D Gen] run_inference.py exited with code {proc.returncode}")
            return {"success": False, "message": f"run_inference.py failed (code {proc.returncode})", "generated_objects": [], "logs": logs}

        log(f"[3D Gen] run_inference.py completed successfully!")

    except Exception as e:
        log(f"[3D Gen] Exception: {str(e)}")
        return {"success": False, "message": str(e), "generated_objects": [], "logs": logs}

    # Step 3: Extract first frame + call image2glb.py directly for scale
    single_mask_dir = out_dir.parent / "single_mask"
    prepare_reference_image(single_mask_dir)

    # 把 multi_mask 各 object 子目录的第一帧 mask 和 .glb 复制到 single_mask/
    # 命名为 0.png/0.glb, 1.png/1.glb, ... 供 image2glb.py --scale_only 使用
    single_mask_dir.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    obj_dirs_sorted = sorted([
        d for d in out_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "background"
    ])
    for idx, obj_dir in enumerate(obj_dirs_sorted):
        # 复制 GLB
        glb_src = obj_dir / f"{obj_dir.name}.glb"
        glb_dst = single_mask_dir / f"{idx}.glb"
        if glb_src.exists() and not glb_dst.exists():
            _shutil.copy2(str(glb_src), str(glb_dst))
            log(f"[3D Gen] Copied {glb_src.name} -> single_mask/{idx}.glb")
        # Copy the first selected top-mask frame as mask PNG.
        mask_dst = single_mask_dir / f"{idx}.png"
        first_mask = _first_top_mask(obj_dir)
        if first_mask is not None and not mask_dst.exists():
            _shutil.copy2(str(first_mask), str(mask_dst))
            log(f"[3D Gen] Copied mask {first_mask.name} -> single_mask/{idx}.png")

    # Count .glb files in single_mask (excluding scene_composed.glb)
    num_masks = len([f for f in single_mask_dir.glob("*.glb") if f.name != "scene_composed.glb"]) if single_mask_dir.exists() else 0
    log(f"[3D Gen] Found {num_masks} GLB files in single_mask, running image2glb.py for scale...")

    image2glb = SAM3D_DIR / "sam3d_objects" / "image2glb.py"
    cmd2 = [
        PY_ASSET, str(image2glb),
        "--config", config_path,
        "--image", str(single_mask_dir / "image.png"),
        "--mask_dir", str(single_mask_dir),
        "--num_masks", str(num_masks),
        "--seed", "43",
        "--scale_only",  # scale_only 模式：只跑推理保存 pose json，不生成 GLB
        "--compose_scene",  # 用已有 GLB 合成场景（GLB 已从 multi_mask 复制过来）
    ]

    try:
        proc2 = subprocess.Popen(
            cmd2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_subprocess_env_for_python(PY_ASSET, conda_env_name="PY_ASSET_CONDA_PREFIX", pythonpath=str(SAM3D_DIR))
        )
        for line in proc2.stdout:
            line = line.rstrip('\n')
            if line:
                log(line)
        proc2.wait()

        if proc2.returncode == 0:
            log(f"[3D Gen] image2glb.py completed successfully!")
        else:
            log(f"[3D Gen] image2glb.py warning: exited with code {proc2.returncode}")

    except Exception as e:
        log(f"[3D Gen] image2glb.py error: {str(e)}")
    
    # Re-scan for generated objects
    existing_objects = []
    if out_dir.exists():
        for obj_dir in out_dir.iterdir():
            if obj_dir.is_dir() and not obj_dir.name.startswith('.'):
                glb_files = list(obj_dir.glob("*.glb"))
                if glb_files:
                    existing_objects.append(obj_dir.name)
    
    return {
        "success": True,
        "message": f"Generated {len(existing_objects)} object meshes with scale",
        "generated_objects": existing_objects,
        "logs": logs
    }

