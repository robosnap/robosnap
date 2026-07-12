from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from robosnap.refinement.run_sim_ready_refinement import (
    env_path,
    infer_scene_graph_edges,
    mark_support_objects,
    semantic_support_surface,
)
from robosnap.refinement.sf_real2sim_layoutopt import find_root_node_ids


class SimReadySceneGraphTest(unittest.TestCase):
    def test_contextual_desk_mentions_are_not_support_surfaces(self):
        self.assertTrue(semantic_support_surface("long light wood desk tabletop"))
        self.assertFalse(semantic_support_surface("silver laptop computer on the desk"))
        self.assertFalse(semantic_support_surface("gray divider panel behind the desk"))

    def test_support_mask_selects_only_the_table_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            support = np.zeros((8, 8), dtype=np.uint8)
            support[2:7, 1:7] = 255
            support_path = root / "support.png"
            Image.fromarray(support).save(support_path)

            other = np.zeros((8, 8), dtype=np.uint8)
            other[:2, :1] = 255
            objects = []
            for object_id, mask in ((6, support), (1, other)):
                obj_dir = root / f"obj_{object_id}"
                obj_dir.mkdir()
                Image.fromarray(mask).save(obj_dir / "mask.png")
                objects.append(
                    {
                        "id": object_id,
                        "name": "desk tabletop" if object_id == 6 else "laptop on the desk",
                        "obj_dir": str(obj_dir),
                    }
                )

            self.assertEqual(mark_support_objects(objects, support_path), [6])
            self.assertTrue(objects[0]["is_table"])
            self.assertFalse(objects[1]["is_table"])

    def test_fallback_graph_keeps_structural_nodes_isolated(self):
        objects = [
            {"id": 0, "name": "monitor on the desk"},
            {"id": 2, "name": "charger near the monitor"},
            {"id": 6, "name": "desk tabletop"},
            {"id": 7, "name": "gray cubicle divider behind the desk"},
        ]
        bounds = {
            0: np.array([[-0.4, -0.2, -0.02], [0.0, 0.3, 0.4]]),
            2: np.array([[-0.2, 0.2, -0.08], [-0.1, 0.3, -0.03]]),
            6: np.array([[-1.0, -1.0, -0.8], [1.0, 1.0, 0.03]]),
            7: np.array([[-0.3, 0.1, -0.1], [0.5, 0.4, 0.4]]),
        }
        with patch(
            "robosnap.refinement.run_sim_ready_refinement.object_world_bounds",
            side_effect=lambda obj: bounds[int(obj["id"])],
        ):
            edges = infer_scene_graph_edges(objects, [6])

        self.assertEqual({edge["source_id"] for edge in edges}, {0, 2})
        self.assertEqual({edge["target_id"] for edge in edges}, {6})
        self.assertNotIn(7, {edge["source_id"] for edge in edges})

    def test_truncated_border_object_requires_explicit_table_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[5:, 10:] = 255
            objects = []
            for object_id, name in (
                (5, "backpack at the right front edge"),
                (1, "laptop on the desk"),
                (6, "desk tabletop"),
            ):
                obj_dir = root / f"obj_{object_id}"
                obj_dir.mkdir()
                if object_id != 6:
                    Image.fromarray(mask).save(obj_dir / "mask.png")
                objects.append(
                    {
                        "id": object_id,
                        "name": name,
                        "obj_dir": str(obj_dir),
                    }
                )

            bounds = {
                1: np.array([[-0.2, -0.2, 0.0], [0.2, 0.2, 0.3]]),
                5: np.array([[0.6, -0.2, -0.2], [1.1, 0.3, 0.4]]),
                6: np.array([[-1.0, -1.0, -0.8], [1.0, 1.0, 0.03]]),
            }
            with patch(
                "robosnap.refinement.run_sim_ready_refinement.object_world_bounds",
                side_effect=lambda obj: bounds[int(obj["id"])],
            ):
                edges = infer_scene_graph_edges(objects, [6])

            self.assertEqual({edge["source_id"] for edge in edges}, {1})
            self.assertTrue(objects[0]["border_truncated"])
            self.assertTrue(objects[1]["border_truncated"])

    def test_empty_scene_graph_environment_is_unset(self):
        with patch.dict(os.environ, {"SF_SCENE_GRAPH_PATH": ""}):
            self.assertIsNone(env_path("SF_SCENE_GRAPH_PATH"))

        expected = Path("/tmp/scene_graph.json")
        with patch.dict(os.environ, {"SF_SCENE_GRAPH_PATH": str(expected)}):
            self.assertEqual(env_path("SF_SCENE_GRAPH_PATH"), expected)

    def test_isolated_structural_node_and_table_are_roots(self):
        nodes = [{"id": 0}, {"id": 6}, {"id": 7}]
        edges = [SimpleNamespace(source_id=0, target_id=6, relation="Support")]
        self.assertEqual(find_root_node_ids(nodes, edges), [6, 7])


if __name__ == "__main__":
    unittest.main()
