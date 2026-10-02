from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from audit_result import audit_result
from mesh_preflight import inspect_mesh


class ValidationTests(unittest.TestCase):
    def _source(self, folder: Path) -> tuple[Path, trimesh.Trimesh]:
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        path = folder / "source.glb"
        mesh.export(path)
        return path, mesh

    def test_preflight_applies_scene_transform_and_records_unverified_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
            transform = np.eye(4)
            transform[:3, 3] = (4.0, 5.0, 6.0)
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="moved", geom_name="moved", transform=transform)
            source = folder / "transformed.glb"
            scene.export(source)

            report = inspect_mesh(source, unit_status="normalized_only")

            self.assertEqual("PASS", report["status"])
            self.assertGreaterEqual(report["scene"]["non_identity_transform_count"], 1)
            self.assertFalse(report["units"]["mm_metrics_allowed"])
            self.assertAlmostEqual(4.0, report["working_mesh"]["centroid"][0], places=6)
            self.assertAlmostEqual(5.0, report["working_mesh"]["centroid"][1], places=6)
            self.assertAlmostEqual(6.0, report["working_mesh"]["centroid"][2], places=6)

    def test_audit_accepts_unchanged_observed_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, mesh = self._source(folder)
            result = folder / "result.glb"
            scene = trimesh.Scene()
            scene.add_geometry(mesh.copy(), node_name="region_observed", geom_name="region_observed")
            scene.add_geometry(
                trimesh.creation.icosphere(radius=0.1),
                node_name="region_ai",
                geom_name="region_ai",
            )
            scene.export(result)

            report = audit_result(source, result)

            self.assertEqual("PASS", report["status"])
            self.assertTrue(report["observed_geometry"]["unchanged"])
            self.assertEqual(
                "NOT_EVALUATED_NO_INDEPENDENT_GT", report["ground_truth"]["status"]
            )

    def test_audit_rejects_changed_observed_vertex(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, mesh = self._source(folder)
            changed = mesh.copy()
            changed.vertices[0, 0] += 0.01
            result = folder / "changed.glb"
            scene = trimesh.Scene()
            scene.add_geometry(
                changed, node_name="region_observed", geom_name="region_observed"
            )
            scene.export(result)

            report = audit_result(source, result)

            self.assertEqual("FAIL", report["status"])
            self.assertFalse(report["observed_geometry"]["vertices_equal"])


if __name__ == "__main__":
    unittest.main()
