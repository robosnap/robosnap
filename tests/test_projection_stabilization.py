from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from robosnap.refinement.stabilize_scene_projection import (
    bbox_iou,
    fit_support_front_edge,
    planar_group_transform,
    project_bbox_xyxy,
    stabilize_scene_projection,
    support_contact_metrics,
    support_group_ids,
    support_relation_valid,
    support_relation_preserved,
)


class ProjectionStabilizationTest(unittest.TestCase):
    def test_bbox_iou(self):
        left = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(bbox_iou(left, left), 1.0)
        self.assertEqual(bbox_iou(left, np.array([40.0, 50.0, 60.0, 70.0])), 0.0)

    def test_supported_object_recovers_initial_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp)
            results_dir = scene_dir / "refinement" / "sf_real2sim" / "results"
            obj_dir = results_dir / "obj_0"
            camera_dir = scene_dir / "sam3d+fpose" / "vggt_single_image"
            obj_dir.mkdir(parents=True)
            camera_dir.mkdir(parents=True)

            intrinsic = np.array(
                [[120.0, 0.0, 50.0], [0.0, 120.0, 50.0], [0.0, 0.0, 1.0]]
            )
            camera = {
                "intrinsic_original_pixels": intrinsic.tolist(),
                "original_size_wh": [100, 100],
            }
            (camera_dir / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
            gravity = {
                "transform_gravity_from_camera": np.eye(4).tolist(),
                "camera_alignment": {"foreground_w2c": np.eye(4).tolist()},
            }
            (scene_dir / "gravity_alignment.json").write_text(
                json.dumps(gravity),
                encoding="utf-8",
            )

            mesh = trimesh.creation.box(extents=[0.8, 0.6, 0.4])
            mesh.export(obj_dir / "mesh_scaled.glb")
            initial = np.eye(4)
            initial[:3, 3] = [0.0, 0.0, 4.0]
            optimized = np.eye(4)
            optimized[:3, 3] = [0.7, 0.0, 2.5]
            np.savetxt(obj_dir / "pose_gravity.txt", initial)
            np.savetxt(obj_dir / "pose_v3_optimized.txt", optimized)

            target_points = trimesh.sample.sample_surface(mesh, 1000, seed=7)[0]
            target_bbox = project_bbox_xyxy(
                target_points,
                initial,
                np.eye(4),
                intrinsic,
            )
            x0, y0 = np.floor(target_bbox[:2]).astype(int)
            x1, y1 = np.ceil(target_bbox[2:]).astype(int)
            rgba = np.zeros((100, 100, 4), dtype=np.uint8)
            rgba[max(y0, 0) : min(y1, 100), max(x0, 0) : min(x1, 100), 3] = 255
            Image.fromarray(rgba).save(obj_dir / "mask.png")

            graph_path = scene_dir / "scene_graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "graph": {
                            "edges": [
                                {
                                    "source_id": 0,
                                    "target_id": 1,
                                    "relation": "Support",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "results_dir": str(results_dir),
                "scene_graph_path": str(graph_path),
                "root_object_ids": [],
                "objects": [{"id": 0}],
            }
            output = scene_dir / "fully_refined_foreground.glb"
            report = stabilize_scene_projection(
                scene_dir,
                results_dir,
                manifest,
                output,
                input_pose_name="pose_gravity",
                optimized_pose_name="pose_v3_optimized",
                scale_samples=31,
            )

            record = report["objects"][0]
            self.assertEqual(record["selected"], "projection_stabilized")
            self.assertGreater(record["selected_reprojection_iou"], 0.75)
            self.assertGreater(
                record["selected_reprojection_iou"],
                record["sf_reprojection_iou"],
            )
            self.assertGreaterEqual(
                record["selected_reprojection_iou"],
                0.95 * record["initial_reprojection_iou"],
            )
            self.assertTrue(output.exists())


    def test_support_gate_rejects_floating_or_off_surface_pose(self):
        parent = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
        child = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        parent_pose = np.eye(4)
        parent_pose[2, 3] = -0.05
        child_pose = np.eye(4)
        child_pose[2, 3] = 0.1

        before = support_contact_metrics(
            child,
            child_pose,
            parent,
            parent_pose,
        )
        self.assertTrue(support_relation_preserved(before, before))

        floating_pose = child_pose.copy()
        floating_pose[:3, 3] = [2.5, 0.0, 0.8]
        floating = support_contact_metrics(
            child,
            floating_pose,
            parent,
            parent_pose,
        )
        self.assertFalse(support_relation_preserved(before, floating))

    def test_absolute_support_gate_rejects_existing_penetration(self):
        penetrating = {
            "xy_overlap_ratio": 0.56,
            "vertical_gap_m": -0.22,
        }
        corrected = {
            "xy_overlap_ratio": 0.73,
            "vertical_gap_m": 0.012,
        }
        self.assertFalse(support_relation_valid(penetrating))
        self.assertFalse(
            support_relation_preserved(penetrating, penetrating)
        )
        self.assertTrue(support_relation_valid(corrected))
        self.assertTrue(
            support_relation_preserved(penetrating, corrected)
        )

    def test_support_front_edge_ignores_narrow_legs(self):
        mask = np.zeros((100, 120), dtype=bool)
        for x in range(10, 111):
            front_y = int(round(45.0 + 0.08 * x))
            mask[20 : front_y + 1, x] = True
        mask[50:90, 25:30] = True
        mask[52:90, 88:93] = True

        edge = fit_support_front_edge(mask)
        self.assertAlmostEqual(edge["slope"], 0.08, delta=0.015)
        self.assertGreater(edge["sample_count"], 70)

    def test_planar_group_transform_preserves_relative_pose(self):
        parent = np.eye(4)
        parent[:3, 3] = [0.4, -0.2, 0.0]
        child = np.eye(4)
        child[:3, 3] = [0.7, 0.1, 0.3]
        transform = planar_group_transform(
            np.array([0.2, 0.3]),
            -4.0,
            np.array([0.03, -0.08]),
        )
        before = np.linalg.inv(parent) @ child
        after = np.linalg.inv(transform @ parent) @ (transform @ child)
        np.testing.assert_allclose(after, before, atol=1e-10)

    def test_support_group_collects_nested_descendants(self):
        edges = [
            {"source_id": 1, "target_id": 6, "relation": "Support"},
            {"source_id": 2, "target_id": 1, "relation": "Support"},
            {"source_id": 7, "target_id": 8, "relation": "Support"},
        ]
        self.assertEqual(support_group_ids(6, edges), [1, 2, 6])

if __name__ == "__main__":
    unittest.main()
