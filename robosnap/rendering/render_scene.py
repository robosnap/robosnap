"""
Render 3D scene from multiple viewpoints
Uses pyrender or matplotlib for visualization
"""

import argparse
import os
import subprocess
import numpy as np
from pathlib import Path
import trimesh
from PIL import Image


def resolve_blender_executable(config_path: str | None = None) -> str | None:
    env_candidates = [
        os.environ.get("BLENDER_EXECUTABLE"),
        os.environ.get("BLENDER_PATH"),
    ]
    for candidate in env_candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            candidate = cfg.get("blender_executable")
            if candidate:
                if not os.path.isabs(candidate):
                    candidate = os.path.join(os.path.dirname(config_path), candidate)
                if os.path.exists(candidate):
                    return candidate
        except Exception:
            return None

    return None


def render_with_blender(scene_path, output_dir, views, config_path: str | None):
    blender_exec = resolve_blender_executable(config_path)
    if not blender_exec:
        print("[WARN] Blender executable not found. Set BLENDER_EXECUTABLE or pass --config.")
        return False

    script_path = Path(__file__).parent / "blender_render_glb.py"
    if not script_path.exists():
        print(f"[ERROR] Blender render script not found: {script_path}")
        return False

    cmd = [
        blender_exec,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--input_glb",
        str(scene_path),
        "--output_dir",
        str(output_dir),
        "--views",
        ",".join(views),
    ]
    print(f"[INFO] Running Blender: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("[ERROR] Blender rendering failed.")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False
    expected = [Path(output_dir) / f"{v}.png" for v in views]
    missing = [p for p in expected if not p.exists() or p.stat().st_size == 0]
    if missing:
        print("[ERROR] Blender finished but outputs are missing:")
        for p in missing:
            print(f"  - {p}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return False
    return True


def render_scene_views(scene_path, output_dir, views=['front', 'top', 'perspective'], config_path: str | None = None):
    """
    Render scene from multiple viewpoints

    Args:
        scene_path: Path to scene GLB file
        output_dir: Output directory for rendered images
        views: List of view names to render

    Returns:
        bool: Success
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading scene: {scene_path}")

    # Load scene
    try:
        scene = trimesh.load(str(scene_path))
    except Exception as e:
        print(f"[ERROR] Failed to load scene: {e}")
        return False

    # Try to use pyrender for high-quality rendering
    try:
        import pyrender
        use_pyrender = True
        print("[INFO] Using pyrender for rendering")
    except ImportError:
        use_pyrender = False
        print("[INFO] Using trimesh built-in rendering (install pyrender for better quality)")

    if use_pyrender:
        try:
            success = render_with_pyrender(scene, output_dir, views)
        except Exception as e:
            print(f"[WARN] pyrender OffscreenRenderer failed: {e}")
            print("[INFO] Falling back to Blender rendering (headless)")
            success = render_with_blender(scene_path, output_dir, views, config_path)
    else:
        success = render_with_trimesh(scene, output_dir, views)

    return success


def render_with_pyrender(scene, output_dir, views):
    """Render using pyrender (high quality)"""
    import pyrender

    # Convert trimesh scene to pyrender scene
    pr_scene = pyrender.Scene.from_trimesh_scene(scene)

    # Setup camera and renderer
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    renderer = pyrender.OffscreenRenderer(1024, 1024)

    # Get scene bounds for camera positioning
    if isinstance(scene, trimesh.Scene):
        bounds = scene.bounds
    else:
        bounds = scene.bounding_box.bounds

    center = (bounds[0] + bounds[1]) / 2
    extent = np.linalg.norm(bounds[1] - bounds[0])

    # Define view configurations
    view_configs = {
        'front': {
            'eye': center + np.array([0, -extent * 1.5, extent * 0.3]),
            'target': center,
            'up': np.array([0, 0, 1])
        },
        'top': {
            'eye': center + np.array([0, 0, extent * 2.0]),
            'target': center,
            'up': np.array([0, 1, 0])
        },
        'perspective': {
            'eye': center + np.array([extent * 1.0, -extent * 1.0, extent * 1.0]),
            'target': center,
            'up': np.array([0, 0, 1])
        }
    }

    # Render each view
    for view_name in views:
        if view_name not in view_configs:
            continue

        config = view_configs[view_name]

        # Compute camera pose
        z = config['eye'] - config['target']
        z = z / np.linalg.norm(z)
        x = np.cross(config['up'], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = x
        camera_pose[:3, 1] = y
        camera_pose[:3, 2] = z
        camera_pose[:3, 3] = config['eye']

        # Add camera to scene
        cam_node = pr_scene.add(camera, pose=camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        pr_scene.add(light, pose=camera_pose)

        # Render
        color, depth = renderer.render(pr_scene)

        # Save image
        output_path = output_dir / f"{view_name}.png"
        Image.fromarray(color).save(output_path)
        print(f"  - Rendered: {output_path.name}")

        # Remove camera and light for next view
        pr_scene.remove_node(cam_node)

    renderer.delete()
    return True


def render_with_trimesh(scene, output_dir, views):
    """Render using trimesh built-in (fallback)"""

    # Get scene bounds
    if isinstance(scene, trimesh.Scene):
        bounds = scene.bounds
    else:
        bounds = scene.bounding_box.bounds

    center = (bounds[0] + bounds[1]) / 2
    extent = np.linalg.norm(bounds[1] - bounds[0])

    # Define view angles (azimuth, elevation)
    view_configs = {
        'front': {'resolution': (1024, 1024), 'fov': (60, 60)},
        'top': {'resolution': (1024, 1024), 'fov': (60, 60)},
        'perspective': {'resolution': (1024, 1024), 'fov': (60, 60)}
    }

    for view_name in views:
        if view_name not in view_configs:
            continue

        config = view_configs[view_name]

        # Use trimesh's built-in save_image (requires pyglet)
        try:
            png = scene.save_image(resolution=config['resolution'])
            output_path = output_dir / f"{view_name}.png"

            with open(output_path, 'wb') as f:
                f.write(png)

            print(f"  - Rendered: {output_path.name}")
        except Exception as e:
            print(f"  - Failed to render {view_name}: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Render 3D scene")
    parser.add_argument("--scene", type=str, required=True,
                       help="Path to scene GLB file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for rendered images")
    parser.add_argument("--views", type=str, nargs='+',
                       default=['front', 'top', 'perspective'],
                       help="Views to render")
    parser.add_argument("--config", type=str, default=None,
                       help="Optional config.yaml with blender_executable")

    args = parser.parse_args()

    print("="*60)
    print("Rendering Scene")
    print("="*60)
    print(f"Scene: {args.scene}")
    print(f"Output: {args.output_dir}")
    print(f"Views: {', '.join(args.views)}")
    print()

    success = render_scene_views(args.scene, args.output_dir, args.views, args.config)

    if success:
        print()
        print("[SUCCESS] Rendering complete")
        return 0
    else:
        print()
        print("[ERROR] Rendering failed")
        return 1


if __name__ == "__main__":
    exit(main())
