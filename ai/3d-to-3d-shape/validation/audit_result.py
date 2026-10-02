"""Audit final 2-node restoration output without replacing existing evaluation.

The check is intentionally narrow: it verifies that the observed/carried node
still has the same coordinates and topology as the input mesh, records hashes,
and labels whether independent ground truth exists.  Chamfer, coverage, seam,
material, and silhouette comparisons remain in the existing dev evaluators.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mesh_preflight import load_world_mesh, sha256_file


def _hashed_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def audit_result(
    source: str | Path,
    result: str | Path,
    observed_node: str = "region_observed",
    source_space: str = "GLTF_WORLD_Y_UP",
    working_space: str = "GLTF_WORLD_Y_UP",
    atol: float = 0.0,
    config_paths: list[str | Path] | None = None,
    independent_gt: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    result_path = Path(result).resolve()
    source_mesh = load_world_mesh(source_path, source_space, working_space)
    try:
        carried_mesh = load_world_mesh(
            result_path, source_space, working_space, geometry_name=observed_node
        )
    except ValueError:
        if observed_node != "region_observed":
            raise
        observed_node = "region_carried"
        carried_mesh = load_world_mesh(
            result_path, source_space, working_space, geometry_name=observed_node
        )

    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    carried_vertices = np.asarray(carried_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    carried_faces = np.asarray(carried_mesh.faces, dtype=np.int64)

    vertex_shape_equal = source_vertices.shape == carried_vertices.shape
    face_shape_equal = source_faces.shape == carried_faces.shape
    if vertex_shape_equal and len(source_vertices):
        vertex_delta = np.linalg.norm(carried_vertices - source_vertices, axis=1)
        max_vertex_delta = float(vertex_delta.max())
        vertices_equal = bool(np.all(vertex_delta <= atol))
    else:
        max_vertex_delta = None
        vertices_equal = vertex_shape_equal
    faces_equal = bool(face_shape_equal and np.array_equal(source_faces, carried_faces))
    unchanged = vertices_equal and faces_equal

    configs = [_hashed_file(Path(path)) for path in (config_paths or [])]
    gt_record: dict[str, Any]
    if independent_gt is None:
        gt_record = {"status": "NOT_EVALUATED_NO_INDEPENDENT_GT"}
    else:
        gt_record = {"status": "AVAILABLE_NOT_EVALUATED_BY_AUDIT", **_hashed_file(Path(independent_gt))}

    errors: list[str] = []
    if not vertex_shape_equal:
        errors.append(
            f"observed vertex shape changed: {source_vertices.shape} -> {carried_vertices.shape}"
        )
    elif not vertices_equal:
        errors.append(f"observed vertices changed: max_delta={max_vertex_delta}")
    if not face_shape_equal:
        errors.append(f"observed face shape changed: {source_faces.shape} -> {carried_faces.shape}")
    elif not faces_equal:
        errors.append("observed face topology or ordering changed")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if unchanged else "FAIL",
        "scope": "AUDIT_ONLY_EXISTING_EVALUATORS_REMAIN_AUTHORITATIVE",
        "source": _hashed_file(source_path),
        "result": _hashed_file(result_path),
        "configs": configs,
        "coordinates": {
            "source_space": source_space,
            "working_space": working_space,
        },
        "observed_geometry": {
            "node": observed_node,
            "source_vertices": int(len(source_vertices)),
            "result_vertices": int(len(carried_vertices)),
            "source_faces": int(len(source_faces)),
            "result_faces": int(len(carried_faces)),
            "vertex_tolerance": atol,
            "max_vertex_delta": max_vertex_delta,
            "vertices_equal": vertices_equal,
            "faces_equal": faces_equal,
            "unchanged": unchanged,
        },
        "ground_truth": gt_record,
        "errors": errors,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observed-node", default="region_observed")
    parser.add_argument("--source-space", default="GLTF_WORLD_Y_UP")
    parser.add_argument("--working-space", default="GLTF_WORLD_Y_UP")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--independent-gt", type=Path)
    args = parser.parse_args()
    payload = audit_result(
        source=args.source,
        result=args.result,
        observed_node=args.observed_node,
        source_space=args.source_space,
        working_space=args.working_space,
        atol=args.atol,
        config_paths=args.config,
        independent_gt=args.independent_gt,
    )
    write_json(args.out, payload)
    print(f"{payload['status']}: {args.out}")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
