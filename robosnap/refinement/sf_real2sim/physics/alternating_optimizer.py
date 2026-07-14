"""
Alternating SDF + SAPIEN optimizer.

Alternates between gradient-based SDF/scene-graph optimization and
SAPIEN physics simulation to produce robust, sim-ready object layouts.

Key efficiency guarantees:
  - SDF grids and surface samples are computed once (object-local, pose-invariant).
  - V-HACD collision meshes are computed once and cached to disk.
  - The SAPIEN scene is built once; between rounds only actor poses are updated.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ..sdf.optimizer import SDFSceneOptimizer, OptimConfig
from .collision import prepare_all_collisions
from .simulator import SapienSimulator, SimConfig


@dataclass
class AlternatingConfig:
    """Hyper-parameters for the alternating optimisation loop."""

    # Alternating loop
    num_rounds: int = 30
    sdf_steps_per_round: int = 15
    sim_steps_per_round: int = 8
    sim_damping_steps: int = 6
    convergence_threshold: float = 1e-3

    # SDF optimizer config (passed to the SDF optimizer)
    sdf_config: OptimConfig = field(default_factory=OptimConfig)

    # SAPIEN physics config
    sim_config: SimConfig = field(default_factory=SimConfig)

    # Collision mesh extraction
    collision_method: str = "vhacd"
    split_collision_mesh: bool = False
    use_cached_collisions: bool = False

    # Pose trajectory export (round-level only)
    save_pose_trajectory: bool = False
    pose_trajectory_path: Optional[str] = None
    pose_trajectory_fps: int = 24
    pose_trajectory_coordinate_system: str = "rotated_z_up_world"
    input_pose_name: Optional[str] = None
    output_pose_name: Optional[str] = None


class AlternatingOptimizer:
    """Coordinate SDF optimisation and SAPIEN simulation in alternating rounds.

    Usage::

        opt = AlternatingOptimizer(meshes, poses, scene_graph_path, config)
        final_poses = opt.optimize()
    """

    def __init__(
        self,
        meshes: dict,
        mesh_paths: dict,
        initial_poses: dict,
        scene_graph_path: str,
        results_dir: str,
        config: Optional[AlternatingConfig] = None,
    ):
        """
        Args:
            meshes: {obj_id: trimesh.Trimesh}
            mesh_paths: {obj_id: path_to_mesh_scaled.glb}
            initial_poses: {obj_id: 4x4 np.ndarray}
            scene_graph_path: Path to scene_graph.json.
            results_dir: Root results dir (for collision cache).
            config: Alternating optimisation parameters.
        """
        self.config = config or AlternatingConfig()
        self.results_dir = results_dir
        self.pose_trajectory_frames: List[dict] = []

        # --- Phase 0: one-time setup ---
        t0 = time.time()

        # 0b. V-HACD collision meshes (one-time, cached to disk)
        print("\n[AlternatingOptimizer] Preparing collision meshes ...")
        self.collision_paths = prepare_all_collisions(
            meshes,
            results_dir,
            method=self.config.collision_method,
            split_mesh=self.config.split_collision_mesh,
            use_cached_collisions=self.config.use_cached_collisions,
        )

        # 0a. SDF optimizer (one-time SDF grid + surface point precomputation)
        print("\n[AlternatingOptimizer] Creating SDF optimizer ...")
        self.sdf_optimizer = SDFSceneOptimizer(
            meshes=meshes,
            initial_poses=initial_poses,
            scene_graph_path=scene_graph_path,
            config=self.config.sdf_config,
        )

        # Collect root obj_ids from the SDF optimizer's analysis
        self.root_obj_ids: Set[str] = {
            od.obj_id for od in self.sdf_optimizer.objects.values() if od.is_root
        }



        # 0c. SAPIEN simulator (one-time scene build)
        print("\n[AlternatingOptimizer] Building SAPIEN scene ...")
        self.simulator = SapienSimulator(
            mesh_paths=mesh_paths,
            collision_paths=self.collision_paths,
            initial_poses=initial_poses,
            root_ids=self.root_obj_ids,
            config=self.config.sim_config,
        )

        print(f"\n[AlternatingOptimizer] Initialisation done "
              f"in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # Main alternating loop
    # ------------------------------------------------------------------

    def optimize(self) -> Dict[str, np.ndarray]:
        """Run the alternating SDF + SAPIEN loop and return final poses."""
        cfg = self.config
        current_poses = {
            oid: od.initial_pose.copy()
            for oid, od in self.sdf_optimizer.objects.items()
        }
        self._record_pose_frame(
            phase="initial",
            round_idx=0,
            poses=current_poses,
        )

        # # 纯优化 begin
        # for round_idx in range(cfg.num_rounds):
        #     print(f"\n{'='*60}")
        #     print(f"  Round {round_idx + 1} / {cfg.num_rounds}")
        #     print(f"{'='*60}")

        #     # --- Phase 1: SDF + scene-graph optimisation ---
        #     print(f"\n[Round {round_idx+1}] SDF optimisation "
        #           f"({cfg.sdf_steps_per_round} steps) ...")
        #     self.sdf_optimizer.config.num_iterations = cfg.sdf_steps_per_round
        #     sdf_poses = self.sdf_optimizer.optimize() #starting_poses=current_poses)

        #     # 重置物体在optimizer中的 initial pose
        #     settled_poses = sdf_poses
        #     for obj_id, pose in settled_poses.items():
        #         if obj_id in self.sdf_optimizer.objects:
        #             self.sdf_optimizer.objects[obj_id].initial_pose = pose

        # current_poses = sdf_poses
        # # 纯优化 end


        # 优化+仿真 begin
        last_completed_round = 0
        for round_idx in range(cfg.num_rounds):
            print(f"\n{'='*60}")
            print(f"  Round {round_idx + 1} / {cfg.num_rounds}")
            print(f"{'='*60}")

            # --- Phase 1: SDF + scene-graph optimisation ---
            print(f"\n[Round {round_idx+1}] SDF optimisation "
                  f"({cfg.sdf_steps_per_round} steps) ...")
            self.sdf_optimizer.config.num_iterations = cfg.sdf_steps_per_round
            sdf_poses = self.sdf_optimizer.optimize() #starting_poses=current_poses)
            self._record_pose_frame(
                phase="sdf_end",
                round_idx=round_idx + 1,
                poses=sdf_poses,
            )

            # --- Phase 2: SAPIEN physics simulation ---
            print(f"\n[Round {round_idx+1}] SAPIEN simulation "
                  f"({cfg.sim_steps_per_round} steps, "
                  f"damping {cfg.sim_damping_steps}) ...")
            self.simulator.update_poses(sdf_poses)

            # 如果是最后一个轮次，物理仿真的时间加长
            if round_idx == cfg.num_rounds - 1:
                print("Last round, increasing simulation time to 5x")
                settled_poses = self.simulator.simulate(
                    num_steps=cfg.sim_steps_per_round * 5,
                    damping_steps=cfg.sim_damping_steps * 5,
                )
            else:
                settled_poses = self.simulator.simulate(
                    num_steps=cfg.sim_steps_per_round,
                    damping_steps=cfg.sim_damping_steps,
                )
            self._record_pose_frame(
                phase="sim_end",
                round_idx=round_idx + 1,
                poses=settled_poses,
            )
            last_completed_round = round_idx + 1

            # --- Convergence check ---
            max_disp = self._max_displacement(current_poses, settled_poses)
            print(f"\n[Round {round_idx+1}] Max displacement: {max_disp:.6f}")

            current_poses = settled_poses

            # Update SDF optimizer's initial poses for the next round
            # *****important*****
            for obj_id, pose in settled_poses.items():
                if obj_id in self.sdf_optimizer.objects:
                    self.sdf_optimizer.objects[obj_id].initial_pose = pose

            if max_disp < cfg.convergence_threshold:
                print(f"[AlternatingOptimizer] Converged at round {round_idx+1}")
                break
         # 优化+仿真 end

        #  # 纯仿真 begin
        # for round_idx in range(cfg.num_rounds):
        #     print(f"\n{'='*60}")
        #     print(f"  Round {round_idx + 1} / {cfg.num_rounds}")
        #     print(f"{'='*60}")
        #     # 如果是最后一个轮次，物理仿真的时间加长
        #     if round_idx == cfg.num_rounds - 1:
        #         print("Last round, increasing simulation time to 5x")
        #         settled_poses = self.simulator.simulate(
        #             num_steps=cfg.sim_steps_per_round * 10,
        #             damping_steps=cfg.sim_damping_steps * 5,
        #         )
        #     else:
        #         settled_poses = self.simulator.simulate(
        #             num_steps=cfg.sim_steps_per_round,
        #             damping_steps=cfg.sim_damping_steps,
        #         )
        #     current_poses = settled_poses
        #  # 纯仿真 end

        self._record_pose_frame(
            phase="final",
            round_idx=last_completed_round,
            poses=current_poses,
        )
        self._save_pose_trajectory()

        return current_poses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_displacement(
        poses_a: Dict[str, np.ndarray],
        poses_b: Dict[str, np.ndarray],
    ) -> float:
        """Compute the maximum translation displacement between two pose sets."""
        max_d = 0.0
        for obj_id in poses_a:
            if obj_id not in poses_b:
                continue
            d = np.linalg.norm(poses_a[obj_id][:3, 3] - poses_b[obj_id][:3, 3])
            max_d = max(max_d, d)
        return max_d

    def _record_pose_frame(
        self,
        phase: str,
        round_idx: int,
        poses: Dict[str, np.ndarray],
    ) -> None:
        """Append one round-level pose snapshot for Blender animation export."""
        if not self.config.save_pose_trajectory:
            return

        self.pose_trajectory_frames.append(
            {
                "frame": len(self.pose_trajectory_frames),
                "round": round_idx,
                "phase": phase,
                "poses": self._poses_to_json(poses),
            }
        )

    @staticmethod
    def _poses_to_json(poses: Dict[str, np.ndarray]) -> Dict[str, list]:
        return {
            obj_id: np.asarray(pose, dtype=float).tolist()
            for obj_id, pose in sorted(poses.items())
        }

    def _save_pose_trajectory(self) -> None:
        if not self.config.save_pose_trajectory:
            return
        if not self.config.pose_trajectory_path:
            raise ValueError(
                "pose_trajectory_path must be set when save_pose_trajectory=True"
            )

        out_path = self.config.pose_trajectory_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "coordinate_system": self.config.pose_trajectory_coordinate_system,
            "input_pose_name": self.config.input_pose_name,
            "output_pose_name": self.config.output_pose_name,
            "fps": self.config.pose_trajectory_fps,
            "frames": self.pose_trajectory_frames,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[AlternatingOptimizer] Saved pose trajectory: {out_path}")
