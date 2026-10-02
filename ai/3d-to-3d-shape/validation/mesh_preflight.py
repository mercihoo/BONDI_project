"""GLB coordinate, scene-transform, and unit preflight.

This module deliberately does not guess physical scale or the up axis.  It
applies glTF scene-node transforms, records the declared working coordinate
space, and marks metric claims as unavailable unless the caller explicitly
provides verified scale information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


COORDINATE_SPACES = ("GLTF_WORLD_Y_UP", "BLENDER_WORLD_Z_UP")
UNIT_STATUSES = ("verified_mm", "metadata_derived", "normalized_only", "unknown")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene(path: str | Path) -> trimesh.Scene:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    scene = trimesh.Scene()
    scene.add_geometry(loaded)
    return scene


def coordinate_matrix(source: str, working: str) -> np.ndarray:
    """Return the explicit coordinate conversion used by this repository.

    glTF Y-up -> Blender Z-up is ``(x, y, z) -> (x, -z, y)``.  The inverse
    conversion is used for Blender Z-up -> glTF Y-up.
    """
    if source not in COORDINATE_SPACES or working not in COORDINATE_SPACES:
        raise ValueError(f"Unsupported coordinate space: {source} -> {working}")
    if source == working:
        return np.eye(4, dtype=np.float64)
    y_to_z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return y_to_z if source == "GLTF_WORLD_Y_UP" else np.linalg.inv(y_to_z)


def _matching_world_mesh(scene: trimesh.Scene, geometry_name: str | None) -> trimesh.Trimesh:
    if geometry_name is None:
        mesh = scene.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected triangle geometry, got {type(mesh).__name__}")
        return mesh

    parts: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, name = scene.graph[node_name]
        if node_name != geometry_name and name != geometry_name:
            continue
        part = scene.geometry[name].copy()
        part.apply_transform(transform)
        parts.append(part)
    if not parts:
        available = sorted(
            {str(name) for name in scene.geometry}
            | {str(name) for name in scene.graph.nodes_geometry}
        )
        raise ValueError(
            f"Geometry/node '{geometry_name}' not found; available={available}"
        )
    return parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)


def load_world_mesh(
    path: str | Path,
    source_space: str = "GLTF_WORLD_Y_UP",
    working_space: str = "GLTF_WORLD_Y_UP",
    geometry_name: str | None = None,
) -> trimesh.Trimesh:
    mesh = _matching_world_mesh(load_scene(path), geometry_name).copy()
    mesh.apply_transform(coordinate_matrix(source_space, working_space))
    return mesh


def _node_records(scene: trimesh.Scene) -> tuple[list[dict[str, Any]], float, list[str]]:
    records: list[dict[str, Any]] = []
    max_roundtrip = 0.0
    errors: list[str] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        matrix = np.asarray(transform, dtype=np.float64)
        geometry = scene.geometry[geometry_name]
        determinant = float(np.linalg.det(matrix[:3, :3]))
        record: dict[str, Any] = {
            "node": str(node_name),
            "geometry": str(geometry_name),
            "vertices": int(len(geometry.vertices)),
            "faces": int(len(geometry.faces)),
            "transform": matrix.round(12).tolist(),
            "transform_is_identity": bool(np.allclose(matrix, np.eye(4), atol=1e-12)),
            "linear_determinant": determinant,
        }
        try:
            inverse = np.linalg.inv(matrix)
            vertices = np.asarray(geometry.vertices, dtype=np.float64)
            if len(vertices):
                sample = vertices[: min(4096, len(vertices))]
                world = trimesh.transformations.transform_points(sample, matrix)
                back = trimesh.transformations.transform_points(world, inverse)
                error = float(np.linalg.norm(back - sample, axis=1).max())
            else:
                error = 0.0
            record["local_world_local_max_error"] = error
            max_roundtrip = max(max_roundtrip, error)
        except np.linalg.LinAlgError:
            record["local_world_local_max_error"] = None
            errors.append(f"non-invertible node transform: {node_name}")
        records.append(record)
    return records, max_roundtrip, errors


def inspect_mesh(
    source: str | Path,
    source_space: str = "GLTF_WORLD_Y_UP",
    working_space: str = "GLTF_WORLD_Y_UP",
    unit_status: str = "unknown",
    units_per_mm: float | None = None,
    roundtrip_tolerance: float = 1e-9,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    if unit_status not in UNIT_STATUSES:
        raise ValueError(f"Unsupported unit status: {unit_status}")
    if units_per_mm is not None and units_per_mm <= 0:
        raise ValueError("units_per_mm must be positive")
    if unit_status == "verified_mm" and units_per_mm is None:
        raise ValueError("verified_mm requires --units-per-mm")

    scene = load_scene(source_path)
    nodes, node_roundtrip, errors = _node_records(scene)
    mesh = load_world_mesh(source_path, source_space, working_space)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    conversion = coordinate_matrix(source_space, working_space)
    if len(vertices) == 0 or len(faces) == 0:
        errors.append("mesh has no triangle geometry")
    if not np.isfinite(vertices).all():
        errors.append("mesh contains non-finite vertices")
    if node_roundtrip > roundtrip_tolerance:
        errors.append(
            f"node transform roundtrip {node_roundtrip:.3e} exceeds "
            f"tolerance {roundtrip_tolerance:.3e}"
        )

    if len(vertices):
        bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        centroid = vertices.mean(axis=0)
        extents = bounds[1] - bounds[0]
    else:
        bounds = np.zeros((2, 3), dtype=np.float64)
        centroid = np.zeros(3, dtype=np.float64)
        extents = np.zeros(3, dtype=np.float64)

    metric_scale_verified = unit_status == "verified_mm"
    warnings: list[str] = []
    if not metric_scale_verified:
        warnings.append("physical scale is not verified; do not report mm accuracy")
    if source_space != working_space:
        warnings.append("coordinate conversion is active; downstream code must not apply it again")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        },
        "coordinates": {
            "source_space": source_space,
            "working_space": working_space,
            "conversion_matrix": conversion.round(12).tolist(),
            "conversion_applied_exactly_once_by_loader": source_space != working_space,
        },
        "units": {
            "status": unit_status,
            "units_per_mm": units_per_mm,
            "metric_scale_verified": metric_scale_verified,
            "mm_metrics_allowed": metric_scale_verified,
        },
        "scene": {
            "geometry_count": int(len(scene.geometry)),
            "node_count": int(len(nodes)),
            "non_identity_transform_count": int(
                sum(not item["transform_is_identity"] for item in nodes)
            ),
            "nodes": nodes,
            "max_local_world_local_error": node_roundtrip,
            "roundtrip_tolerance": roundtrip_tolerance,
        },
        "working_mesh": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds": bounds.round(12).tolist(),
            "extents": extents.round(12).tolist(),
            "centroid": centroid.round(12).tolist(),
        },
        "warnings": warnings,
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
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-space", choices=COORDINATE_SPACES, default=COORDINATE_SPACES[0])
    parser.add_argument("--working-space", choices=COORDINATE_SPACES, default=COORDINATE_SPACES[0])
    parser.add_argument("--unit-status", choices=UNIT_STATUSES, default="unknown")
    parser.add_argument("--units-per-mm", type=float)
    parser.add_argument("--roundtrip-tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    payload = inspect_mesh(
        args.source,
        source_space=args.source_space,
        working_space=args.working_space,
        unit_status=args.unit_status,
        units_per_mm=args.units_per_mm,
        roundtrip_tolerance=args.roundtrip_tolerance,
    )
    write_json(args.out, payload)
    print(f"{payload['status']}: {args.out}")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
