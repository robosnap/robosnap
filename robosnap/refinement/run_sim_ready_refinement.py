#!/usr/bin/env python3
# Adapter for optional sim-ready refinement.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value in (None, "") else Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a projection-stabilized, physically refined foreground scene.")
    parser.add_argument("--input-foreground", type=Path, required=True)
    parser.add_argument("--output-foreground", type=Path, required=True)
    parser.add_argument("--refinement-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, help="RoboSnap scene directory used to export SF-Real2Sim inputs.")
    parser.add_argument("--icp-report", type=Path, help="Default: <scene-dir>/depth/object_point_clouds/icp_report.json")
    parser.add_argument("--gravity-transform-json", type=Path, help="Default: <scene-dir>/gravity_alignment.json")
    parser.add_argument("--support-mask", type=Path, help="Default: <scene-dir>/support_mask.png")
    parser.add_argument("--scene-graph-input", type=Path, default=env_path("SF_SCENE_GRAPH_PATH"))
    parser.add_argument("--object-file", type=Path, help="Default: <scene-dir>/object.txt")
    parser.add_argument("--sf-python", default=sys.executable)
    parser.add_argument("--sf-extra-pythonpath", default=os.environ.get("SF_REAL2SIM_EXTRA_PYTHONPATH", ""))
    parser.add_argument("--sf-collision-method", choices=("vhacd", "coacd"), default=os.environ.get("SF_REAL2SIM_COLLISION_METHOD", "vhacd"))
    parser.add_argument("--sf-use-cached-collisions", action="store_true", default=env_flag("SF_REAL2SIM_USE_CACHED_COLLISIONS", False))
    parser.add_argument("--sf-disable-collision-split", action="store_true", default=env_flag("SF_REAL2SIM_DISABLE_COLLISION_SPLIT", True))
    parser.add_argument("--sf-num-rounds", type=int, default=env_int("SF_REAL2SIM_NUM_ROUNDS", 30))
    parser.add_argument("--sf-sdf-steps-per-round", type=int, default=env_int("SF_REAL2SIM_SDF_STEPS_PER_ROUND", 15))
    parser.add_argument("--sf-sim-steps-per-round", type=int, default=env_int("SF_REAL2SIM_SIM_STEPS_PER_ROUND", 8))
    parser.add_argument("--sf-sim-damping-steps", type=int, default=env_int("SF_REAL2SIM_SIM_DAMPING_STEPS", 6))
    parser.add_argument("--sf-convergence-threshold", type=float, default=env_float("SF_REAL2SIM_CONVERGENCE_THRESHOLD", 1e-3))
    parser.add_argument("--sf-sdf-resolution", type=int, default=env_int("SF_REAL2SIM_SDF_RESOLUTION", 128))
    parser.add_argument("--sf-num-surface-points", type=int, default=env_int("SF_REAL2SIM_NUM_SURFACE_POINTS", 1024))
    parser.add_argument("--sf-no-make-sdf-mesh-watertight", action="store_true", default=env_flag("SF_REAL2SIM_NO_MAKE_SDF_MESH_WATERTIGHT", False))
    parser.add_argument("--sf-no-require-watertight-sdf-mesh", action="store_true", default=env_flag("SF_REAL2SIM_NO_REQUIRE_WATERTIGHT_SDF_MESH", False))
    parser.add_argument("--sf-sdf-watertight-method", default=os.environ.get("SF_REAL2SIM_SDF_WATERTIGHT_METHOD", "voxel"), choices=("voxel", "pymeshfix", "trimesh"))
    parser.add_argument("--sf-disable-initial-pose-regularization", action="store_true", default=env_flag("SF_REAL2SIM_DISABLE_INITIAL_POSE_REGULARIZATION", False))
    parser.add_argument("--sf-sdf-watertight-voxel-resolution", type=int, default=env_int("SF_REAL2SIM_SDF_WATERTIGHT_VOXEL_RESOLUTION", 96))
    parser.add_argument("--projection-min-scale", type=float, default=env_float("SF_PROJECTION_MIN_SCALE", 0.35))
    parser.add_argument("--projection-max-scale", type=float, default=env_float("SF_PROJECTION_MAX_SCALE", 1.5))
    parser.add_argument("--projection-scale-samples", type=int, default=env_int("SF_PROJECTION_SCALE_SAMPLES", 81))
    parser.add_argument("--projection-min-iou", type=float, default=env_float("SF_PROJECTION_MIN_IOU", 0.55))
    parser.add_argument("--projection-raw-pose-keep-ratio", type=float, default=env_float("SF_PROJECTION_RAW_POSE_KEEP_RATIO", 0.9))
    parser.add_argument("--sf-second-pass-num-rounds", type=int, default=env_int("SF_SECOND_PASS_NUM_ROUNDS", 30))
    parser.add_argument("--sf-second-pass-sdf-steps-per-round", type=int, default=env_int("SF_SECOND_PASS_SDF_STEPS_PER_ROUND", 15))
    parser.add_argument("--sf-second-pass-sim-steps-per-round", type=int, default=env_int("SF_SECOND_PASS_SIM_STEPS_PER_ROUND", 8))
    parser.add_argument("--sf-second-pass-sim-damping-steps", type=int, default=env_int("SF_SECOND_PASS_SIM_DAMPING_STEPS", 6))
    parser.add_argument("--sf-second-pass-regularization-weight", type=float, default=env_float("SF_SECOND_PASS_REGULARIZATION_WEIGHT", 1.0))
    parser.add_argument("--sf-second-pass-min-reprojection-ratio", type=float, default=env_float("SF_SECOND_PASS_MIN_REPROJECTION_RATIO", 0.98))
    parser.add_argument("--sf-seed", type=int, default=env_int("SF_REAL2SIM_SEED", 0))
    return parser.parse_args()


def read_object_names(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_gravity_transform(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.eye(4, dtype=np.float64)
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data.get("transform_gravity_from_camera", np.eye(4)), dtype=np.float64)


def copy_mesh(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(str(source.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def git_revision(path: Path) -> dict:
    result = {"path": str(path), "commit": None, "dirty": None}
    if not path.exists():
        return result
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode == 0:
        result["commit"] = commit.stdout.strip()
    if dirty.returncode == 0:
        result["dirty"] = bool(dirty.stdout.strip())
    return result


def sf_pythonpath(args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    entries = [str(repo_root)]
    entries.extend(
        part
        for part in str(args.sf_extra_pythonpath).split(os.pathsep)
        if part
    )
    existing = os.environ.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def python_environment(python: str, pythonpath: str | None = None) -> dict:
    code = """
import importlib.metadata as metadata
import json
import platform
import sys

packages = {}
for name in ("numpy", "torch", "trimesh", "sapien", "scipy", "open3d", "opencv-python", "opencv-python-headless", "tqdm"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
}))
"""
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    proc = subprocess.run(
        [python, "-c", code],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {"python_executable": python, "error": proc.stderr.strip()}
    payload = json.loads(proc.stdout)
    payload["python_executable"] = python
    return payload


def reproducibility_record(args: argparse.Namespace) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    refinement_root = repo_root / "robosnap" / "refinement"
    sf_core = refinement_root / "sf_real2sim"
    optimizer_entrypoint = sf_core / "optimize_scene.py"
    stabilizer = refinement_root / "stabilize_scene_projection.py"
    return {
        "pipeline": [
            "support-group 3D camera projection stabilization",
            "RoboSnap SF-Real2Sim pass 1",
            "per-object 3D camera projection stabilization",
            "RoboSnap SF-Real2Sim pass 2",
            "reprojection and support quality gate",
        ],
        "seed": int(args.sf_seed),
        "robosnap": git_revision(repo_root),
        "sf_real2sim": {
            "ownership": "first_party",
            "path": str(sf_core),
            "source_tree_sha256": sha256_tree(sf_core),
        },
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "optimizer_sha256": sha256_file(optimizer_entrypoint),
        "projection_stabilizer_sha256": sha256_file(stabilizer),
        "sf_environment": python_environment(args.sf_python, sf_pythonpath(args)),
        "sf_pythonpath": sf_pythonpath(args),
        "pass1": {
            "num_rounds": int(args.sf_num_rounds),
            "sdf_steps_per_round": int(args.sf_sdf_steps_per_round),
            "sim_steps_per_round": int(args.sf_sim_steps_per_round),
            "sim_damping_steps": int(args.sf_sim_damping_steps),
        },
        "pass2": {
            "num_rounds": int(args.sf_second_pass_num_rounds),
            "sdf_steps_per_round": int(args.sf_second_pass_sdf_steps_per_round),
            "sim_steps_per_round": int(args.sf_second_pass_sim_steps_per_round),
            "sim_damping_steps": int(args.sf_second_pass_sim_damping_steps),
            "regularization_weight": float(args.sf_second_pass_regularization_weight),
            "min_reprojection_ratio": float(args.sf_second_pass_min_reprojection_ratio),
        },
        "collision_method": args.sf_collision_method,
        "root_policy": "all scene-graph nodes without an incoming Support edge are fixed",
        "determinism": {
            "level": "configuration-reproducible, not bitwise deterministic",
            "reason": "VHACD and SAPIEN can produce small numeric variation across runs",
            "stabilizer": "fixed reprojection and support quality gate",
        },
    }


def write_scene_graph(path: Path, objects: list[dict], edges: list[dict]) -> None:
    graph_nodes = [
        {
            "id": int(obj["id"]),
            "name": obj["name"],
            "description": obj["name"],
            "is_table": bool(obj.get("is_table", False)),
        }
        for obj in objects
    ]
    payload = {
        "objects": graph_nodes,
        "graph": {
            "nodes": graph_nodes,
            "edges": edges,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_binary_mask(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        return np.asarray(image.getchannel("A")) > 0
    return np.asarray(image.convert("L")) > 0


def semantic_support_surface(name: str) -> bool:
    text = name.lower()
    if re.search(r"\b(tabletop|desktop|countertop|workbench)\b", text):
        return True
    contextual = re.search(
        r"\b(on|under|near|behind|beside|next to|in front of|at the edge of)\b.*\b(table|desk)\b",
        text,
    )
    return contextual is None and re.search(r"\b(table|desk|counter)\b", text) is not None


def structural_root(name: str) -> bool:
    return re.search(
        r"\b(divider|partition|wall|floor|ceiling|window|door|room structure)\b",
        name.lower(),
    ) is not None


def lower_side_border_truncated(obj: dict, tolerance: int = 1) -> bool:
    obj_dir = obj.get("obj_dir")
    if not obj_dir:
        return False
    mask_path = Path(obj_dir) / "mask.png"
    if not mask_path.exists():
        return False
    mask = load_binary_mask(mask_path)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return False
    height, width = mask.shape
    touches_bottom = int(rows.max()) >= height - 1 - tolerance
    touches_side = bool(
        int(cols.min()) <= tolerance
        or int(cols.max()) >= width - 1 - tolerance
    )
    return bool(touches_bottom and touches_side)


def mark_support_objects(objects: list[dict], support_mask_path: Path | None) -> list[int]:
    support_mask = None
    if support_mask_path is not None and support_mask_path.exists():
        support_mask = load_binary_mask(support_mask_path)

    support_ids: list[int] = []
    for obj in objects:
        score = 0.0
        mask_path = Path(obj["obj_dir"]) / "mask.png"
        if support_mask is not None and mask_path.exists():
            object_mask = load_binary_mask(mask_path)
            if object_mask.shape == support_mask.shape and object_mask.any():
                score = float(np.logical_and(object_mask, support_mask).sum() / object_mask.sum())
        obj["support_mask_coverage"] = score
        obj["is_table"] = score >= 0.5
        if obj["is_table"]:
            support_ids.append(int(obj["id"]))

    if not support_ids:
        for obj in objects:
            obj["is_table"] = semantic_support_surface(obj["name"])
            if obj["is_table"]:
                support_ids.append(int(obj["id"]))
    return support_ids


def object_world_bounds(obj: dict) -> np.ndarray | None:
    try:
        import trimesh

        loaded = trimesh.load(obj["mesh"], force="scene")
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        mesh.apply_transform(np.loadtxt(obj["pose_gravity"]).astype(np.float64))
        return np.asarray(mesh.bounds, dtype=np.float64)
    except Exception:
        return None


def infer_scene_graph_edges(objects: list[dict], support_ids: list[int]) -> list[dict]:
    by_id = {int(obj["id"]): obj for obj in objects}
    support_bounds = {obj_id: object_world_bounds(by_id[obj_id]) for obj_id in support_ids}
    edges: list[dict] = []
    explicit_pattern = re.compile(
        r"\b(on|atop|upon|sitting on|standing on|placed on|resting on|lying on)\b.*"
        r"\b(table|desk|tabletop|desktop|counter)\b"
    )

    for obj in objects:
        object_id = int(obj["id"])
        if object_id in support_ids or structural_root(obj["name"]):
            continue
        bounds = object_world_bounds(obj)
        explicit_support = explicit_pattern.search(obj["name"].lower()) is not None
        border_truncated = lower_side_border_truncated(obj)
        obj["border_truncated"] = border_truncated
        if border_truncated and not explicit_support:
            continue
        candidates: list[tuple[float, int]] = []
        for support_id, parent_bounds in support_bounds.items():
            if bounds is None or parent_bounds is None:
                if explicit_support:
                    candidates.append((0.0, support_id))
                continue
            gap_x = max(parent_bounds[0, 0] - bounds[1, 0], bounds[0, 0] - parent_bounds[1, 0], 0.0)
            gap_y = max(parent_bounds[0, 1] - bounds[1, 1], bounds[0, 1] - parent_bounds[1, 1], 0.0)
            xy_gap = float(np.hypot(gap_x, gap_y))
            support_span = float(np.max(parent_bounds[1, :2] - parent_bounds[0, :2]))
            near_xy = xy_gap <= max(0.05, 0.08 * support_span)
            support_top = float(parent_bounds[1, 2])
            near_z = bounds[0, 2] <= support_top + 0.25 and bounds[1, 2] >= support_top - 0.35
            if explicit_support or (near_xy and near_z):
                candidates.append((xy_gap + abs(float(bounds[0, 2]) - support_top), support_id))
        if not candidates:
            continue
        _, support_id = min(candidates)
        edges.append(
            {
                "source_id": object_id,
                "target_id": support_id,
                "relation": "Support",
                "fine_relation": "Stack",
                "inference": "caption" if explicit_support else "geometry",
            }
        )
    return edges


def prepare_sf_real2sim_inputs(args: argparse.Namespace) -> dict:
    if args.scene_dir is None:
        raise ValueError("--scene-dir is required to prepare SF-Real2Sim inputs")
    scene_dir = args.scene_dir
    icp_report = args.icp_report or scene_dir / "depth" / "object_point_clouds" / "icp_report.json"
    gravity_json = args.gravity_transform_json or scene_dir / "gravity_alignment.json"
    object_file = args.object_file or scene_dir / "object.txt"
    support_mask = args.support_mask or scene_dir / "support_mask.png"
    if not icp_report.exists():
        raise FileNotFoundError(f"Missing ICP report: {icp_report}")

    report = json.loads(icp_report.read_text(encoding="utf-8"))
    object_names = read_object_names(object_file)
    T_gravity_from_camera = load_gravity_transform(gravity_json)

    sf_root = args.refinement_dir / "sf_real2sim"
    results_dir = sf_root / "results"
    scene_graph_path = sf_root / "scene_graph" / "scene_graph.json"
    results_dir.mkdir(parents=True, exist_ok=True)

    objects_meta: list[dict] = []
    for obj in report.get("objects", []):
        object_id = int(obj["object_id"])
        obj_dir = results_dir / f"obj_{object_id}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        mesh_src = Path(obj["mesh"])
        mesh_dst = obj_dir / "mesh_scaled.glb"
        copy_mesh(mesh_src, mesh_dst)

        pose_camera = np.loadtxt(obj["pose_icp"]).astype(np.float64)
        pose_gravity = T_gravity_from_camera @ pose_camera
        optimized_pose = obj_dir / "pose_optimized.txt"
        if optimized_pose.exists():
            optimized_pose.unlink()
        np.savetxt(obj_dir / "pose_camera.txt", pose_camera, fmt="%.10g")
        np.savetxt(obj_dir / "pose_gravity.txt", pose_gravity, fmt="%.10g")
        np.savetxt(obj_dir / "pose_rotated.txt", pose_gravity, fmt="%.10g")
        np.savetxt(obj_dir / "final_scale.txt", np.asarray([1.0]), fmt="%.10g")

        mask_src = Path(obj.get("mask", ""))
        if mask_src.exists():
            shutil.copy2(mask_src, obj_dir / "mask.png")

        name = object_names[object_id] if object_id < len(object_names) else f"object_{object_id}"
        objects_meta.append(
            {
                "id": object_id,
                "name": name,
                "obj_dir": str(obj_dir),
                "mesh": str(mesh_dst),
                "pose_camera": str(obj_dir / "pose_camera.txt"),
                "pose_gravity": str(obj_dir / "pose_gravity.txt"),
                "is_table": False,
            }
        )

    support_ids = mark_support_objects(objects_meta, support_mask)
    if args.scene_graph_input is not None:
        input_graph = json.loads(args.scene_graph_input.expanduser().read_text(encoding="utf-8"))
        graph = input_graph.get("graph", input_graph)
        edges = list(graph.get("edges", []))
        scene_graph_source = str(args.scene_graph_input.expanduser().resolve())
    else:
        edges = infer_scene_graph_edges(objects_meta, support_ids)
        scene_graph_source = "support_mask+caption+geometry"
    supported_object_ids = {
        int(edge["source_id"])
        for edge in edges
        if edge.get("relation", "Support") == "Support"
    }
    root_object_ids = sorted(int(obj["id"]) for obj in objects_meta if int(obj["id"]) not in supported_object_ids)
    write_scene_graph(scene_graph_path, objects_meta, edges)
    manifest = {
        "status": "prepared",
        "results_dir": str(results_dir),
        "scene_graph_path": str(scene_graph_path),
        "input_pose_name": "pose_gravity",
        "output_pose_name": "pose_optimized",
        "scene_graph_source": scene_graph_source,
        "support_object_ids": support_ids,
        "scene_graph_edges": len(edges),
        "root_object_ids": root_object_ids,
        "objects": objects_meta,
    }
    (sf_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compose_from_results(
    results_dir: Path,
    pose_name: str,
    output_path: Path,
    object_ids: list[int] | None = None,
) -> int:
    import trimesh

    scene = trimesh.Scene()
    count = 0
    if object_ids is None:
        object_dirs = sorted(results_dir.glob("obj_*"))
    else:
        object_dirs = [results_dir / f"obj_{object_id}" for object_id in object_ids]
    for obj_dir in object_dirs:
        mesh_path = obj_dir / "mesh_scaled.glb"
        pose_path = obj_dir / f"{pose_name}.txt"
        if not mesh_path.exists() or not pose_path.exists():
            continue
        pose = np.loadtxt(pose_path).astype(np.float64)
        loaded = trimesh.load(str(mesh_path), force="scene")
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded
        mesh.apply_transform(pose)
        scene.add_geometry(mesh, node_name=obj_dir.name)
        count += 1
    if count == 0:
        raise RuntimeError(f"No optimized object poses found in {results_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))
    return count


def prepare_second_pass_inputs(manifest: dict) -> dict:
    first_results = Path(manifest["results_dir"])
    pass2_root = first_results.parent / "pass2"
    pass2_results = pass2_root / "results"
    pass2_results.mkdir(parents=True, exist_ok=True)
    objects = []

    for obj in manifest["objects"]:
        object_id = int(obj["id"])
        source_dir = first_results / f"obj_{object_id}"
        target_dir = pass2_results / f"obj_{object_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        source_mesh = source_dir / "mesh_projection_stabilized.glb"
        source_pose = source_dir / "pose_projection_stabilized.txt"
        if not source_mesh.exists() or not source_pose.exists():
            raise FileNotFoundError(
                f"Missing projection-stabilized pass-2 input for object {object_id}"
            )

        mesh_hash = sha256_file(source_mesh)
        mesh_hash_path = target_dir / "mesh_scaled.sha256"
        previous_hash = (
            mesh_hash_path.read_text(encoding="utf-8").strip()
            if mesh_hash_path.exists()
            else None
        )
        if previous_hash != mesh_hash:
            collision_dir = target_dir / "collision"
            if collision_dir.exists():
                shutil.rmtree(collision_dir)
        copy_mesh(source_mesh, target_dir / "mesh_scaled.glb")
        mesh_hash_path.write_text(f"{mesh_hash}\n", encoding="utf-8")
        shutil.copy2(source_pose, target_dir / "pose_projection_stabilized.txt")
        np.savetxt(
            target_dir / "final_scale.txt",
            np.asarray([1.0]),
            fmt="%.10g",
        )
        source_mask = source_dir / "mask.png"
        if source_mask.exists():
            shutil.copy2(source_mask, target_dir / "mask.png")
        optimized_pose = target_dir / "pose_optimized.txt"
        if optimized_pose.exists():
            optimized_pose.unlink()

        item = dict(obj)
        item.update(
            {
                "obj_dir": str(target_dir),
                "mesh": str(target_dir / "mesh_scaled.glb"),
                "pose_projection_stabilized": str(
                    target_dir / "pose_projection_stabilized.txt"
                ),
                "mesh_sha256": mesh_hash,
            }
        )
        objects.append(item)

    second = {
        **manifest,
        "status": "prepared_second_pass",
        "results_dir": str(pass2_results),
        "input_pose_name": "pose_projection_stabilized",
        "output_pose_name": "pose_optimized",
        "objects": objects,
        "source_results_dir": str(first_results),
    }
    (pass2_root / "manifest.json").write_text(
        json.dumps(second, indent=2),
        encoding="utf-8",
    )
    return second


def run_sf_optimizer(
    args: argparse.Namespace,
    manifest: dict,
    *,
    label: str = "pass1",
    output_foreground: Path | None = None,
    num_rounds: int | None = None,
    sdf_steps_per_round: int | None = None,
    sim_steps_per_round: int | None = None,
    sim_damping_steps: int | None = None,
    w_regularization: float = 1.0,
    disable_initial_pose_regularization: bool | None = None,
    use_cached_collisions: bool | None = None,
) -> dict:
    output_foreground = output_foreground or args.output_foreground
    num_rounds = args.sf_num_rounds if num_rounds is None else num_rounds
    sdf_steps_per_round = (
        args.sf_sdf_steps_per_round
        if sdf_steps_per_round is None
        else sdf_steps_per_round
    )
    sim_steps_per_round = (
        args.sf_sim_steps_per_round
        if sim_steps_per_round is None
        else sim_steps_per_round
    )
    sim_damping_steps = (
        args.sf_sim_damping_steps
        if sim_damping_steps is None
        else sim_damping_steps
    )
    disable_initial_pose_regularization = (
        args.sf_disable_initial_pose_regularization
        if disable_initial_pose_regularization is None
        else disable_initial_pose_regularization
    )
    use_cached_collisions = (
        args.sf_use_cached_collisions
        if use_cached_collisions is None
        else use_cached_collisions
    )

    cmd = [
        args.sf_python,
        "-m",
        "robosnap.refinement.sf_real2sim.optimize_scene",
        "--results-dir",
        manifest["results_dir"],
        "--scene-graph-path",
        manifest["scene_graph_path"],
        "--input-pose-name",
        manifest["input_pose_name"],
        "--output-pose-name",
        manifest["output_pose_name"],
        "--collision-method",
        args.sf_collision_method,
        "--num-rounds",
        str(num_rounds),
        "--sdf-steps-per-round",
        str(sdf_steps_per_round),
        "--sim-steps-per-round",
        str(sim_steps_per_round),
        "--sim-damping-steps",
        str(sim_damping_steps),
        "--convergence-threshold",
        str(args.sf_convergence_threshold),
        "--sdf-resolution",
        str(args.sf_sdf_resolution),
        "--num-surface-points",
        str(args.sf_num_surface_points),
        "--w-regularization",
        str(w_regularization),
        "--sdf-watertight-method",
        args.sf_sdf_watertight_method,
        "--sdf-watertight-voxel-resolution",
        str(args.sf_sdf_watertight_voxel_resolution),
        "--seed",
        str(args.sf_seed),
    ]
    if disable_initial_pose_regularization:
        cmd.append("--disable-initial-pose-regularization")
    if args.sf_no_make_sdf_mesh_watertight:
        cmd.append("--no-make-sdf-mesh-watertight")
    if args.sf_no_require_watertight_sdf_mesh:
        cmd.append("--no-require-watertight-sdf-mesh")
    if use_cached_collisions:
        cmd.append("--use-cached-collisions")
    if args.sf_disable_collision_split:
        cmd.append("--disable-collision-split")

    log_name = "sf_real2sim_optimizer.log" if label == "pass1" else f"sf_real2sim_optimizer_{label}.log"
    log_path = args.refinement_dir / log_name
    env = os.environ.copy()
    extra_paths = [part for part in str(args.sf_extra_pythonpath).split(os.pathsep) if part]
    env["PYTHONPATH"] = sf_pythonpath(args)
    extra_pythonpath = os.pathsep.join(extra_paths)
    repo_root = Path(__file__).resolve().parents[2]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        last_error = lines[-1] if lines else ""
        reason = f"RoboSnap SF-Real2Sim failed with code {proc.returncode}."
        if last_error:
            reason = f"{reason} Last log line: {last_error}"
        return {
            "status": "sf_failed",
            "reason": reason,
            "cmd": cmd,
            "log": str(log_path),
            "runner": "robosnap_core",
            "extra_pythonpath": extra_pythonpath,
            "collision_method": args.sf_collision_method,
            "use_cached_collisions": bool(use_cached_collisions),
            "last_error": last_error,
        }

    edge_count = int(manifest.get("scene_graph_edges", 0))
    if len(manifest["objects"]) > 1 and edge_count == 0:
        return {
            "status": "sf_invalid",
            "reason": "The scene graph has no physical edges; all objects would be fixed roots.",
            "cmd": cmd,
            "log": str(log_path),
            "runner": "robosnap_core",
            "scene_graph_edges": edge_count,
        }

    count = compose_from_results(
        Path(manifest["results_dir"]),
        manifest["output_pose_name"],
        output_foreground,
        [int(obj["id"]) for obj in manifest["objects"]],
    )
    expected_count = len(manifest["objects"])
    if count != expected_count:
        raise RuntimeError(f"SF-Real2Sim composed {count} objects, expected {expected_count}")

    translation_changes = []
    translation_changes_by_id = {}
    rotation_changes_deg = []
    for obj in manifest["objects"]:
        object_id = int(obj["id"])
        obj_dir = Path(obj["obj_dir"])
        before = np.loadtxt(obj_dir / f"{manifest['input_pose_name']}.txt").astype(np.float64)
        after = np.loadtxt(obj_dir / f"{manifest['output_pose_name']}.txt").astype(np.float64)
        translation_change = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
        translation_changes.append(translation_change)
        translation_changes_by_id[object_id] = translation_change
        delta_rotation = after[:3, :3] @ before[:3, :3].T
        cosine = float(np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_changes_deg.append(float(np.degrees(np.arccos(cosine))))

    root_changes = [
        translation_changes_by_id.get(int(object_id), 0.0)
        for object_id in manifest.get("root_object_ids", [])
    ]
    root_max_translation = max(root_changes, default=0.0)
    status = "sf_ok" if root_max_translation <= 1e-5 else "sf_invalid"
    reason = (
        "RoboSnap SF-Real2Sim completed and optimized poses were composed."
        if status == "sf_ok"
        else "RoboSnap SF-Real2Sim moved a fixed root object; the scene was rejected."
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "status": status,
        "label": label,
        "reason": reason,
        "cmd": cmd,
        "log": str(log_path),
        "runner": "robosnap_core",
        "extra_pythonpath": extra_pythonpath,
        "collision_method": args.sf_collision_method,
        "use_cached_collisions": bool(use_cached_collisions),
        "optimized_objects": int(count),
        "scene_graph_edges": edge_count,
        "support_object_ids": manifest.get("support_object_ids", []),
        "changed_objects": int(sum(change > 1e-5 for change in translation_changes)),
        "root_object_ids": manifest.get("root_object_ids", []),
        "max_translation_change_m": max(translation_changes, default=0.0),
        "root_max_translation_change_m": root_max_translation,
        "max_rotation_change_deg": max(rotation_changes_deg, default=0.0),
        "collision_load_failures": log_text.count("Failed to cook mesh"),
        "num_rounds": int(num_rounds),
        "w_regularization": float(w_regularization),
        "seed": int(args.sf_seed),
        "optimizer": "RoboSnap SF-Real2Sim AlternatingOptimizer (SDF + SAPIEN)",
    }


def main() -> int:
    args = parse_args()
    args.refinement_dir.mkdir(parents=True, exist_ok=True)
    args.output_foreground.parent.mkdir(parents=True, exist_ok=True)
    args.output_foreground.unlink(missing_ok=True)

    status = {
        "status": "preparing",
        "reason": "Preparing fixed two-pass physical refinement.",
        "input_foreground": str(args.input_foreground),
        "output_foreground": str(args.output_foreground),
        "sf_implementation": "robosnap.refinement.sf_real2sim",
    }

    manifest = None
    try:
        if args.scene_dir is None:
            raise ValueError("--scene-dir is required for physical refinement")
        manifest = prepare_sf_real2sim_inputs(args)
        status["sf_real2sim_inputs"] = manifest
    except Exception as exc:
        status["status"] = "prepare_failed"
        status["reason"] = f"Failed to prepare SF inputs: {exc}"

    if manifest is not None:
        try:
            from robosnap.refinement.stabilize_scene_projection import (
                stabilize_scene_projection,
                stabilize_support_group_projection,
                validate_second_pass_projection,
            )

            pre_sf_output = Path(manifest["results_dir"]).parent / "pre_sf_projection.glb"
            pre_projection = stabilize_support_group_projection(
                args.scene_dir,
                Path(manifest["results_dir"]),
                manifest,
                pre_sf_output,
                input_pose_name=manifest["input_pose_name"],
            )
            status["pre_sf_projection"] = pre_projection
            manifest["input_pose_name_before_projection"] = manifest["input_pose_name"]
            manifest["input_pose_name"] = pre_projection["output_pose_name"]
            for obj in manifest["objects"]:
                object_id = int(obj["id"])
                obj["pose_pre_sf_projected"] = str(
                    Path(manifest["results_dir"])
                    / f"obj_{object_id}"
                    / f"{manifest['input_pose_name']}.txt"
                )
            (Path(manifest["results_dir"]).parent / "manifest.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            status["sf_real2sim_inputs"] = manifest

            pass1 = run_sf_optimizer(args, manifest)
            status.update(pass1)
            status["sf_pass1"] = pass1
            if pass1["status"] == "sf_ok":
                projection = stabilize_scene_projection(
                    args.scene_dir,
                    Path(manifest["results_dir"]),
                    manifest,
                    args.output_foreground,
                    input_pose_name=manifest["input_pose_name"],
                    optimized_pose_name=manifest["output_pose_name"],
                    reference_pose_name=manifest.get("input_pose_name_before_projection"),
                    min_scale=args.projection_min_scale,
                    max_scale=args.projection_max_scale,
                    scale_samples=args.projection_scale_samples,
                    min_reprojection_iou=args.projection_min_iou,
                    raw_pose_keep_ratio=args.projection_raw_pose_keep_ratio,
                )
                status["projection_stabilization"] = projection

                second_manifest = prepare_second_pass_inputs(manifest)
                status["sf_second_pass_inputs"] = second_manifest
                second_raw = Path(second_manifest["results_dir"]).parent / "second_pass_raw.glb"
                pass2 = run_sf_optimizer(
                    args,
                    second_manifest,
                    label="pass2",
                    output_foreground=second_raw,
                    num_rounds=args.sf_second_pass_num_rounds,
                    sdf_steps_per_round=args.sf_second_pass_sdf_steps_per_round,
                    sim_steps_per_round=args.sf_second_pass_sim_steps_per_round,
                    sim_damping_steps=args.sf_second_pass_sim_damping_steps,
                    w_regularization=args.sf_second_pass_regularization_weight,
                    disable_initial_pose_regularization=False,
                    use_cached_collisions=True,
                )
                status["sf_pass2"] = pass2
                if pass2["status"] != "sf_ok":
                    status["status"] = pass2["status"]
                    status["reason"] = pass2["reason"]
                else:
                    validation = validate_second_pass_projection(
                        args.scene_dir,
                        Path(second_manifest["results_dir"]),
                        second_manifest,
                        args.output_foreground,
                        input_pose_name=second_manifest["input_pose_name"],
                        optimized_pose_name=second_manifest["output_pose_name"],
                        min_reprojection_iou=args.projection_min_iou,
                        min_reprojection_ratio=args.sf_second_pass_min_reprojection_ratio,
                    )
                    status["second_pass_validation"] = validation
                    status["status"] = "sf_ok"
                    status["reason"] = (
                        "Fixed projection -> SF -> projection -> SF pipeline completed "
                        "and passed reprojection/support validation."
                    )
        except Exception as exc:
            status["status"] = "sf_invalid"
            status["reason"] = f"Fixed two-pass physical refinement failed: {exc}"

    reproducibility = reproducibility_record(args)
    status["reproducibility"] = reproducibility
    (args.refinement_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2),
        encoding="utf-8",
    )
    (args.refinement_dir / "status.json").write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )
    if status["status"] == "sf_ok":
        print(f"[sim-ready] wrote {args.output_foreground}")
    print(f"[sim-ready] wrote {args.refinement_dir / 'status.json'}")
    return 0 if status["status"] == "sf_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
