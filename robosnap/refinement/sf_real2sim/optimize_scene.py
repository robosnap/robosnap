#!/usr/bin/env python3
"""Run RoboSnap's physical layout optimization core."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np


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


def find_root_node_ids(nodes: list[dict], edges) -> list[int]:
    supported_ids = {
        int(edge.source_id if hasattr(edge, "source_id") else edge["source_id"])
        for edge in edges
        if (edge.relation if hasattr(edge, "relation") else edge.get("relation", "Support")) == "Support"
    }
    return sorted(int(node["id"]) for node in nodes if int(node["id"]) not in supported_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboSnap physical layout optimization.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--scene-graph-path", required=True)
    parser.add_argument("--input-pose-name", default="pose_gravity")
    parser.add_argument("--output-pose-name", default="pose_optimized")
    parser.add_argument("--disable-initial-pose-regularization", action="store_true")
    parser.add_argument("--regularized-pose-name", default="pose_initial_regularized")
    parser.add_argument("--save-pose-trajectory", action="store_true")
    parser.add_argument("--pose-trajectory-name", default="pose_trajectory.json")
    parser.add_argument("--collision-method", default=os.environ.get("SF_REAL2SIM_COLLISION_METHOD", "vhacd"), choices=("vhacd", "coacd"))
    parser.add_argument("--disable-collision-split", action="store_true", default=env_flag("SF_REAL2SIM_DISABLE_COLLISION_SPLIT", True))
    parser.add_argument("--use-cached-collisions", action="store_true", default=env_flag("SF_REAL2SIM_USE_CACHED_COLLISIONS", False))
    parser.add_argument("--num-rounds", type=int, default=env_int("SF_REAL2SIM_NUM_ROUNDS", 30))
    parser.add_argument("--sdf-steps-per-round", type=int, default=env_int("SF_REAL2SIM_SDF_STEPS_PER_ROUND", 15))
    parser.add_argument("--sim-steps-per-round", type=int, default=env_int("SF_REAL2SIM_SIM_STEPS_PER_ROUND", 8))
    parser.add_argument("--sim-damping-steps", type=int, default=env_int("SF_REAL2SIM_SIM_DAMPING_STEPS", 6))
    parser.add_argument("--convergence-threshold", type=float, default=env_float("SF_REAL2SIM_CONVERGENCE_THRESHOLD", 1e-3))
    parser.add_argument("--sdf-resolution", type=int, default=env_int("SF_REAL2SIM_SDF_RESOLUTION", 128))
    parser.add_argument("--num-surface-points", type=int, default=env_int("SF_REAL2SIM_NUM_SURFACE_POINTS", 1024))
    parser.add_argument("--w-regularization", type=float, default=env_float("SF_REAL2SIM_W_REGULARIZATION", 1.0))
    parser.add_argument("--w-support", type=float, default=env_float("SF_REAL2SIM_W_SUPPORT", 5.0))
    parser.add_argument("--w-contact", type=float, default=env_float("SF_REAL2SIM_W_CONTACT", 5.0))
    parser.add_argument("--w-global-penetration", type=float, default=env_float("SF_REAL2SIM_W_GLOBAL_PENETRATION", 16.0))
    parser.add_argument("--no-make-sdf-mesh-watertight", action="store_true", default=env_flag("SF_REAL2SIM_NO_MAKE_SDF_MESH_WATERTIGHT", False))
    parser.add_argument("--no-require-watertight-sdf-mesh", action="store_true", default=env_flag("SF_REAL2SIM_NO_REQUIRE_WATERTIGHT_SDF_MESH", False))
    parser.add_argument("--sdf-watertight-method", default=os.environ.get("SF_REAL2SIM_SDF_WATERTIGHT_METHOD", "voxel"), choices=("voxel", "pymeshfix", "trimesh"))
    parser.add_argument("--sdf-watertight-voxel-resolution", type=int, default=env_int("SF_REAL2SIM_SDF_WATERTIGHT_VOXEL_RESOLUTION", 96))
    parser.add_argument("--seed", type=int, default=env_int("SF_REAL2SIM_SEED", 0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    from robosnap.refinement.sf_real2sim.sdf.optimizer import OptimConfig, SDFSceneOptimizer
    import robosnap.refinement.sf_real2sim.physics.alternating_optimizer as alternating_module
    from robosnap.refinement.sf_real2sim.physics.alternating_optimizer import AlternatingConfig
    from robosnap.refinement.sf_real2sim.physics.initial_pose_regularizer import (
        InitialPoseRegularizerConfig,
        regularize_initial_poses,
    )
    from robosnap.refinement.sf_real2sim.physics.simulator import SimConfig
    from robosnap.refinement.sf_real2sim.scene_io import load_scene_data, save_optimized_poses

    class RootAwareSDFSceneOptimizer(SDFSceneOptimizer):
        def _find_roots(self):
            return find_root_node_ids(self.scene_graph["nodes"], self.edges)

    alternating_module.SDFSceneOptimizer = RootAwareSDFSceneOptimizer
    AlternatingOptimizer = alternating_module.AlternatingOptimizer

    results_dir = args.results_dir
    scene_graph_path = args.scene_graph_path
    meshes, mesh_paths, poses = load_scene_data(results_dir, args.input_pose_name)
    if len(meshes) < 2:
        print("Need at least 2 objects to optimise. Exiting.")
        return 0

    if not args.disable_initial_pose_regularization:
        poses = regularize_initial_poses(
            meshes=meshes,
            initial_poses=poses,
            scene_graph_path=scene_graph_path,
            config=InitialPoseRegularizerConfig(),
        )
        if args.regularized_pose_name:
            print(f"\nSaving regularized initial poses as {args.regularized_pose_name}.txt")
            save_optimized_poses(results_dir, poses, args.regularized_pose_name)

    sdf_cfg = OptimConfig()
    sdf_cfg.sdf_debug_dir = os.path.join(results_dir, "sdf_debug")
    sdf_cfg.sdf_resolution = args.sdf_resolution
    sdf_cfg.num_surface_points = args.num_surface_points
    sdf_cfg.w_regularization = args.w_regularization
    sdf_cfg.w_support = args.w_support
    sdf_cfg.w_contact = args.w_contact
    sdf_cfg.w_global_pen = args.w_global_penetration
    sdf_cfg.make_sdf_mesh_watertight = not args.no_make_sdf_mesh_watertight
    sdf_cfg.require_watertight_sdf_mesh = not args.no_require_watertight_sdf_mesh
    sdf_cfg.sdf_watertight_method = args.sdf_watertight_method
    sdf_cfg.sdf_watertight_voxel_resolution = args.sdf_watertight_voxel_resolution
    config = AlternatingConfig(
        sdf_config=sdf_cfg,
        sim_config=SimConfig(),
        num_rounds=args.num_rounds,
        sdf_steps_per_round=args.sdf_steps_per_round,
        sim_steps_per_round=args.sim_steps_per_round,
        sim_damping_steps=args.sim_damping_steps,
        convergence_threshold=args.convergence_threshold,
        collision_method=args.collision_method,
        split_collision_mesh=not args.disable_collision_split,
        use_cached_collisions=args.use_cached_collisions,
        save_pose_trajectory=args.save_pose_trajectory,
        pose_trajectory_path=os.path.join(results_dir, args.pose_trajectory_name) if args.save_pose_trajectory else None,
        input_pose_name=args.input_pose_name,
        output_pose_name=args.output_pose_name,
    )

    optimizer = AlternatingOptimizer(
        meshes=meshes,
        mesh_paths=mesh_paths,
        initial_poses=poses,
        scene_graph_path=scene_graph_path,
        results_dir=results_dir,
        config=config,
    )
    final_poses = optimizer.optimize()
    save_optimized_poses(results_dir, final_poses, args.output_pose_name)
    print("\nAlternating optimisation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
