"""
CAST-style physics-aware loss functions for SDF-based layout optimization.

Implements Equations (8)-(11) from the CAST paper:
  - Penetration loss: penalise overlapping geometry for all object pairs
  - Support loss (Eq.10): supporter is fixed, attract supported object to contact
  - Contact loss (Eq.9): bidirectional – prevent penetration AND separation
  - Regularization: keep optimized poses close to the initial estimates
"""

import torch

from .sdf_utils import query_sdf


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply a 4x4 rigid transform to a set of 3-D points.

    Args:
        points: (N, 3)
        T: (4, 4) rigid transform

    Returns:
        Transformed points (N, 3).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def _transform_points_to_local(
    world_points: torch.Tensor,
    T_world: torch.Tensor,
) -> torch.Tensor:
    """
    Transform world-frame points into an object's local frame.

    Args:
        world_points: (N, 3) in world frame.
        T_world: (4, 4) object-to-world transform.

    Returns:
        (N, 3) points in the object's local frame.
    """
    R = T_world[:3, :3]
    t = T_world[:3, 3]
    return (world_points - t) @ R


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def penetration_loss(
    sdf_grid_i: torch.Tensor,
    bounds_i: torch.Tensor,
    T_i: torch.Tensor,
    surface_pts_j: torch.Tensor,
    T_j: torch.Tensor,
) -> torch.Tensor:
    """
    Non-penetration loss for object pair (i, j).

    Penalises surface points of j that are *inside* object i (SDF < 0).

    Args:
        sdf_grid_i: (1,1,D,H,W) precomputed SDF of object i.
        bounds_i: (2,3) grid bounds for object i.
        T_i: (4,4) world-from-object transform for i.
        surface_pts_j: (N,3) surface samples on j in j's local frame.
        T_j: (4,4) world-from-object transform for j.

    Returns:
        Scalar loss.
    """
    world_pts = _transform_points(surface_pts_j, T_j)
    local_pts_i = _transform_points_to_local(world_pts, T_i)
    sdf_vals = query_sdf(sdf_grid_i, bounds_i, local_pts_i)

    # Penalise negative SDF (penetration): loss = mean(max(-sdf, 0))
    pen = torch.clamp(-sdf_vals, min=0.0)
    return pen.mean()


def support_loss(
    sdf_grid_supporter: torch.Tensor,
    bounds_supporter: torch.Tensor,
    T_supporter: torch.Tensor,
    surface_pts_supported: torch.Tensor,
    T_supported: torch.Tensor,
) -> torch.Tensor:
    """
    Support loss (CAST Eq.10).

    Encourages the *closest* surface point of the supported object to lie
    exactly on the supporter's surface (SDF = 0).  Only the supported
    object's transform carries gradients.

    Args:
        sdf_grid_supporter: (1,1,D,H,W) SDF of the supporter.
        bounds_supporter: (2,3) grid bounds.
        T_supporter: (4,4) supporter world transform (**detached**).
        surface_pts_supported: (N,3) surface samples on the supported object.
        T_supported: (4,4) supported-object world transform (has grad).

    Returns:
        Scalar loss.
    """
    T_sup_detached = T_supporter.detach()

    world_pts = _transform_points(surface_pts_supported, T_supported)
    local_pts = _transform_points_to_local(world_pts, T_sup_detached)
    sdf_vals = query_sdf(sdf_grid_supporter, bounds_supporter, local_pts)

    # Attract: the minimum SDF should be 0 (touching the surface)
    min_sdf = sdf_vals.min()
    attract = torch.abs(min_sdf)

    #如果min_sdf离得比较远，加入一个coarse的惩罚用于拉近距离，不然离太远的话clip之后的sdf函数产生不了梯度
    # 获取外部点偏离 bounding box 的距离惩罚以提供全局梯度
    grid_min, grid_max = bounds_supporter[0], bounds_supporter[1]
    # 如果点在界外，out_of_bounds_dist > 0，在界内则为 0
    out_of_bounds_dist = torch.clamp(local_pts - grid_max, min=0) + torch.clamp(grid_min - local_pts, min=0)
    # 对界外点施加一个 L2/L1 惩罚把它拉回网格内
    oob_loss = torch.norm(out_of_bounds_dist, dim=-1).mean()

    attract += 0.05 * oob_loss  # 加一个小权重的 OOB 引导 loss

    # Pull the nearest 25% of exterior points closer to the surface,
    # encouraging a broader contact region instead of a single point.
    '''
    筛选出 sdf_vals > 0 的点（在支撑物外部、尚未接触的表面点）
    对这些正值进行升序排序
    取最小的 25%（即距离支撑物表面最近的那批外部点）
    计算它们的均值，累加到 attract 项中
    '''
    positive_mask = sdf_vals > 0
    positive_vals = sdf_vals[positive_mask]
    if positive_vals.numel() > 0:
        sorted_pos, _ = torch.sort(positive_vals)
        k = max(1, positive_vals.numel() // 5)
        attract = attract + sorted_pos[:k].mean()

    # Also penalise penetration for the rest of the points
    pen = torch.clamp(-sdf_vals, min=0.0).mean()

    return attract + pen


def contact_loss(
    sdf_grid_i: torch.Tensor,
    bounds_i: torch.Tensor,
    T_i: torch.Tensor,
    surface_pts_i: torch.Tensor,
    sdf_grid_j: torch.Tensor,
    bounds_j: torch.Tensor,
    T_j: torch.Tensor,
    surface_pts_j: torch.Tensor,
) -> torch.Tensor:
    """
    Contact loss (CAST Eq.9) – bidirectional.

    Both prevents penetration *and* prevents complete separation.

    Returns:
        Scalar loss.
    """
    # Direction i -> j: surface of j queries SDF of i
    world_pts_j = _transform_points(surface_pts_j, T_j)
    local_pts_j_in_i = _transform_points_to_local(world_pts_j, T_i)
    sdf_j_in_i = query_sdf(sdf_grid_i, bounds_i, local_pts_j_in_i)

    pen_ij = torch.clamp(-sdf_j_in_i, min=0.0).mean()
    attract_ij = torch.clamp(sdf_j_in_i.min(), min=0.0)

    # Direction j -> i: surface of i queries SDF of j
    world_pts_i = _transform_points(surface_pts_i, T_i)
    local_pts_i_in_j = _transform_points_to_local(world_pts_i, T_j)
    sdf_i_in_j = query_sdf(sdf_grid_j, bounds_j, local_pts_i_in_j)

    pen_ji = torch.clamp(-sdf_i_in_j, min=0.0).mean()
    attract_ji = torch.clamp(sdf_i_in_j.min(), min=0.0)

    return pen_ij + pen_ji + attract_ij + attract_ji


def regularization_loss(
    delta_translations: torch.Tensor,
    delta_rotations: torch.Tensor,
    rot_weight: float = 5.0#1.0,
) -> torch.Tensor:
    """
    Regularization loss that penalises deviation from the initial pose.

    Args:
        delta_translations: (K, 3) translation offsets for K optimisable objects.
        delta_rotations: (K, 3) axis-angle rotation offsets.
        rot_weight: relative weight for the rotation term.

    Returns:
        Scalar loss.
    """
    trans_loss = (delta_translations ** 2).sum()
    rot_loss = (delta_rotations ** 2).sum()
    return trans_loss + rot_weight * rot_loss
