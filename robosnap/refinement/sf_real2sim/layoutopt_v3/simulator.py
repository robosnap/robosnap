"""
SAPIEN physics simulator wrapper.

Creates a SAPIEN scene once with all collision/visual meshes loaded,
then supports efficient pose-reset + re-simulation without scene rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pose conversion helpers
# ---------------------------------------------------------------------------

def pose_matrix_to_pq(mat: np.ndarray):
    """Decompose a 4x4 rigid transform into position and quaternion [w,x,y,z]."""
    M = mat[:3, :3]
    scale = np.array([
        np.linalg.norm(M[:, 0]),
        np.linalg.norm(M[:, 1]),
        np.linalg.norm(M[:, 2]),
    ], dtype=np.float32)
    R_mat = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        R_mat[:, i] = M[:, i] / scale[i] if scale[i] > 1e-12 else M[:, i]

    quat_xyzw = R.from_matrix(R_mat).as_quat()  # scipy: [x, y, z, w]
    q_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    p = mat[:3, 3].astype(np.float32)
    return p, q_wxyz, scale


def pq_to_pose_matrix(p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Convert position + quaternion [w,x,y,z] to a 4x4 rigid transform."""
    quat_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    mat[:3, 3] = p
    return mat


# ---------------------------------------------------------------------------
# Simulator config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    timestep: float = 1 / 100.0
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
    restitution: float = 0.0
    ground_static_friction: float = 5.0
    ground_dynamic_friction: float = 5.0
    density: float = 3000.0
    ground_altitude: float = 0.0


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class SapienSimulator:
    """Persistent SAPIEN scene that can be re-used across alternating rounds.

    Actors are built once; between rounds only their poses are updated
    via ``update_poses``, avoiding expensive scene reconstruction.
    """

    def __init__(
        self,
        mesh_paths: Dict[str, str],
        collision_paths: Dict[str, List[str]],
        initial_poses: Dict[str, np.ndarray],
        root_ids: Set[str],
        config: Optional[SimConfig] = None,
    ):
        """
        Args:
            mesh_paths: {obj_id: path_to_visual_mesh.glb}
            collision_paths: {obj_id: [collision_part_0.glb, ...]}
            initial_poses: {obj_id: 4x4 rigid transform}
            root_ids: set of obj_ids that should be kinematic (fixed).
            config: Physics parameters.
        """
        self.config = config or SimConfig()
        self.root_ids = root_ids

        # Create engine & scene (once)
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(self.config.timestep)

        # Materials
        self.physical_material = self.scene.create_physical_material(
            static_friction=self.config.static_friction,
            dynamic_friction=self.config.dynamic_friction,
            restitution=self.config.restitution,
        )
        ground_material = self.scene.create_physical_material(
            static_friction=self.config.ground_static_friction,
            dynamic_friction=self.config.ground_dynamic_friction,
            restitution=self.config.restitution,
        )

        # Ground
        ground_rb = self.scene.add_ground(
            altitude=self.config.ground_altitude, material=ground_material,
        )
        ground_cs = ground_rb.get_collision_shapes()[0]
        ground_cs.set_collision_groups(group0=1, group1=3, group2=0, group3=0)

        # Build actors
        self.actors: Dict[str, sapien.ActorBase] = {}
        self._dynamic_actor_ids: List[str] = []
        self._build_actors(mesh_paths, collision_paths, initial_poses)

    # ------------------------------------------------------------------
    # Actor construction (one-time)
    # ------------------------------------------------------------------

    def _build_actors(
        self,
        mesh_paths: Dict[str, str],
        collision_paths: Dict[str, List[str]],
        initial_poses: Dict[str, np.ndarray],
    ) -> None:
        for obj_id in sorted(initial_poses.keys()):
            if obj_id not in collision_paths:
                print(f"  [SapienSimulator] skip {obj_id}: no collision meshes")
                continue

            builder = self.scene.create_actor_builder()
            pose_mat = initial_poses[obj_id]
            p, q, scale = pose_matrix_to_pq(pose_mat)
            actor_pose = sapien.Pose(p=p, q=q)

            # Visual mesh (at identity relative to actor; actor pose positions it)
            if obj_id in mesh_paths:
                builder.add_visual_from_file(
                    filename=mesh_paths[obj_id],
                    scale=scale,
                )

            # Collision parts
            for cpath in collision_paths[obj_id]:
                builder.add_collision_from_file(
                    filename=cpath,
                    scale=scale,
                    material=self.physical_material,
                    density=self.config.density,
                )

            builder.set_collision_groups(2, 2, 0, 0)

            is_root = obj_id in self.root_ids
            if is_root:
                actor = builder.build_kinematic(name=obj_id)
            else:
                actor = builder.build(name=obj_id)
                self._dynamic_actor_ids.append(obj_id)

            actor.set_pose(actor_pose)
            self.actors[obj_id] = actor

        print(f"  [SapienSimulator] built {len(self.actors)} actors "
              f"({len(self._dynamic_actor_ids)} dynamic, "
              f"{len(self.actors) - len(self._dynamic_actor_ids)} kinematic)")

    # ------------------------------------------------------------------
    # Pose update (between rounds)
    # ------------------------------------------------------------------

    def update_poses(self, poses: Dict[str, np.ndarray]) -> None:
        """Teleport all actors to new poses and reset velocities."""
        for obj_id, pose_mat in poses.items():
            actor = self.actors.get(obj_id)
            if actor is None:
                continue
            p, q, _ = pose_matrix_to_pq(pose_mat)
            actor.set_pose(sapien.Pose(p=p, q=q))
            if isinstance(actor, sapien.Actor):  # dynamic only
                actor.set_velocity(np.zeros(3))
                actor.set_angular_velocity(np.zeros(3))

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        num_steps: int = 200,
        damping_steps: int = 100,
    ) -> Dict[str, np.ndarray]:
        """Run physics simulation with velocity damping.

        During the first ``damping_steps`` steps, XY velocities are
        suppressed to near-zero while allowing slow Z settling.  This
        prevents objects from being launched by collision forces.

        Args:
            num_steps: Total simulation steps.
            damping_steps: Number of initial steps with velocity damping.

        Returns:
            Final poses as {obj_id: 4x4 matrix}.
        """
        for step in range(num_steps):
            if step < damping_steps:
                for obj_id in self._dynamic_actor_ids:
                    actor = self.actors[obj_id]
                    vel = actor.get_velocity()
                    vel_norm = np.linalg.norm(vel)
                    if vel_norm > 1e-10:
                        damped = vel / vel_norm * np.array(
                            [1e-7, 1e-7, 0.01], dtype=np.float32,
                        )
                        actor.set_velocity(damped)
            self.scene.step()

        return self.get_current_poses()

    # ------------------------------------------------------------------
    # Pose readout
    # ------------------------------------------------------------------

    def get_current_poses(self) -> Dict[str, np.ndarray]:
        """Extract current world poses of all actors as 4x4 matrices."""
        poses = {}
        for obj_id, actor in self.actors.items():
            sp = actor.get_pose()
            poses[obj_id] = pq_to_pose_matrix(
                np.asarray(sp.p), np.asarray(sp.q),
            )
        return poses
