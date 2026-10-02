from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SOURCE_DIR = Path(r"C:\Users\<USER>\Downloads\빗살무늬토기\프린트_STL")
OUTPUT_DIR = Path(__file__).with_name("pottery_restoration_output")

N_THETA = 360
N_Y = 240
RANDOM_SEED = 71353


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    face_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + face_count * 50:
        raise ValueError("Only binary STL is supported")
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = np.frombuffer(raw, dtype=dtype, offset=84, count=face_count)
    triangles = records["vertices"].astype(np.float64)
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, label: str) -> None:
    triangles = vertices[faces].astype(np.float32)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0
    header = label.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    record_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = np.zeros(len(faces), dtype=record_dtype)
    records["normal"] = normals
    records["vertices"] = triangles
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(faces)))
        handle.write(records.tobytes())


def fit_axis(vertices: np.ndarray) -> tuple[float, float]:
    bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    initial = np.array([(bounds[0, 0] + bounds[1, 0]) / 2, (bounds[0, 2] + bounds[1, 2]) / 2])
    y_low, y_high = np.quantile(vertices[:, 1], [0.12, 0.94])
    centers: list[np.ndarray] = []
    for low, high in zip(np.linspace(y_low, y_high, 45)[:-1], np.linspace(y_low, y_high, 45)[1:]):
        points = vertices[(vertices[:, 1] >= low) & (vertices[:, 1] < high)][:, [0, 2]]
        if len(points) < 100:
            continue
        radial = np.linalg.norm(points - initial, axis=1)
        outer = points[radial >= np.quantile(radial, 0.68)]
        a = np.column_stack((2 * outer[:, 0], 2 * outer[:, 1], np.ones(len(outer))))
        b = np.sum(outer * outer, axis=1)
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        center = solution[:2]
        if np.linalg.norm(center - initial) < 35:
            centers.append(center)
    if not centers:
        return float(initial[0]), float(initial[1])
    center = np.median(np.vstack(centers), axis=0)
    return float(center[0]), float(center[1])


def classify_surfaces(
    vertices: np.ndarray, faces: np.ndarray, center_x: float, center_z: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    centers = triangles.mean(axis=1)
    radial = np.column_stack((centers[:, 0] - center_x, np.zeros(len(centers)), centers[:, 2] - center_z))
    dots = np.sum(normals * radial, axis=1)
    radii = np.linalg.norm(radial[:, [0, 2]], axis=1)
    reference = dots[radii >= np.quantile(radii, 0.72)]
    orientation = 1.0 if np.median(reference) >= 0 else -1.0
    outer_faces = dots * orientation > 0

    outer_votes = np.zeros(len(vertices), dtype=np.int32)
    inner_votes = np.zeros(len(vertices), dtype=np.int32)
    np.add.at(outer_votes, faces[outer_faces].ravel(), 1)
    np.add.at(inner_votes, faces[~outer_faces].ravel(), 1)
    outer_vertices = outer_votes > inner_votes
    inner_vertices = inner_votes > outer_votes
    return outer_vertices, inner_vertices, {
        "outer_faces": int(np.count_nonzero(outer_faces)),
        "inner_faces": int(np.count_nonzero(~outer_faces)),
        "outer_vertices": int(np.count_nonzero(outer_vertices)),
        "inner_vertices": int(np.count_nonzero(inner_vertices)),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def fill_profile(profile: np.ndarray) -> np.ndarray:
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Insufficient profile samples")
    positions = np.arange(len(profile))
    filled = np.interp(positions, positions[valid], profile[valid])
    return moving_average(filled, 7)


def radial_grid(
    vertices: np.ndarray,
    selected: np.ndarray,
    center_x: float,
    center_z: float,
    y_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = vertices[selected]
    theta = np.mod(np.arctan2(points[:, 2] - center_z, points[:, 0] - center_x), 2 * np.pi)
    radius = np.hypot(points[:, 0] - center_x, points[:, 2] - center_z)
    theta_index = np.minimum((theta / (2 * np.pi) * N_THETA).astype(np.int32), N_THETA - 1)
    y_index = np.clip(np.searchsorted(y_edges, points[:, 1], side="right") - 1, 0, N_Y - 1)
    flat = y_index * N_THETA + theta_index
    sums = np.bincount(flat, weights=radius, minlength=N_Y * N_THETA).reshape(N_Y, N_THETA)
    counts = np.bincount(flat, minlength=N_Y * N_THETA).reshape(N_Y, N_THETA)
    grid = np.full((N_Y, N_THETA), np.nan, dtype=np.float64)
    valid = counts > 0
    grid[valid] = sums[valid] / counts[valid]
    return grid, counts


def fill_grid(grid: np.ndarray, fallback_profile: np.ndarray) -> np.ndarray:
    filled = grid.copy()
    for _ in range(80):
        missing = ~np.isfinite(filled)
        if not np.any(missing):
            break
        sums = np.zeros_like(filled)
        counts = np.zeros_like(filled, dtype=np.int16)
        for shifted in (np.roll(filled, 1, axis=1), np.roll(filled, -1, axis=1)):
            valid = np.isfinite(shifted)
            sums[valid] += shifted[valid]
            counts[valid] += 1
        for direction in (-1, 1):
            shifted = np.full_like(filled, np.nan)
            if direction == -1:
                shifted[:-1] = filled[1:]
            else:
                shifted[1:] = filled[:-1]
            valid = np.isfinite(shifted)
            sums[valid] += shifted[valid]
            counts[valid] += 1
        can_fill = missing & (counts > 0)
        if not np.any(can_fill):
            break
        filled[can_fill] = sums[can_fill] / counts[can_fill]
    missing = ~np.isfinite(filled)
    if np.any(missing):
        filled[missing] = np.broadcast_to(fallback_profile[:, None], filled.shape)[missing]
    return filled


def box_blur_wrap(values: np.ndarray, theta_window: int, y_window: int) -> np.ndarray:
    theta_pad = theta_window // 2
    theta_sum = np.zeros_like(values)
    for offset in range(-theta_pad, theta_pad + 1):
        theta_sum += np.roll(values, offset, axis=1)
    theta_blurred = theta_sum / (2 * theta_pad + 1)
    y_pad = y_window // 2
    padded = np.pad(theta_blurred, ((y_pad, y_pad), (0, 0)), mode="edge")
    cumulative = np.vstack((np.zeros((1, padded.shape[1])), np.cumsum(padded, axis=0)))
    return (cumulative[y_window:] - cumulative[:-y_window]) / y_window


def estimate_periods(residual: np.ndarray, band_count: int = 10) -> tuple[np.ndarray, list[int]]:
    prediction = np.zeros_like(residual)
    periods: list[int] = []
    edges = np.linspace(0, N_Y, band_count + 1, dtype=int)
    for start, end in zip(edges[:-1], edges[1:]):
        band = residual[start:end]
        centered = band - band.mean(axis=1, keepdims=True)
        scores = []
        candidates = range(14, min(121, N_THETA // 2))
        denominator = float(np.mean(centered * centered) + 1e-9)
        for shift in candidates:
            score = float(np.mean(centered * np.roll(centered, shift, axis=1)) / denominator)
            scores.append(score)
        period = list(candidates)[int(np.argmax(scores))]
        periods.append(period)
        shifted = np.stack([
            np.roll(band, period, axis=1),
            np.roll(band, -period, axis=1),
            np.roll(band, 2 * period, axis=1),
            np.roll(band, -2 * period, axis=1),
        ])
        prediction[start:end] = np.median(shifted, axis=0)
    return prediction, periods


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        expanded = result | np.roll(result, 1, axis=1) | np.roll(result, -1, axis=1)
        expanded[1:] |= result[:-1]
        expanded[:-1] |= result[1:]
        result = expanded
    return result


def learn_restoration(
    residual: np.ndarray,
    anomaly_quantile: float = 0.985,
    minimum_anomaly_mm: float = 0.35,
    delta_limit_mm: float = 3.0,
    dilation_steps: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[str, float]]:
    periodic_prediction, periods = estimate_periods(residual)
    smooth_residual = box_blur_wrap(residual, 5, 5)
    smooth_prediction = box_blur_wrap(periodic_prediction, 5, 5)
    anomaly = np.abs(smooth_residual - smooth_prediction)

    thresholds = np.zeros(N_Y)
    band_edges = np.linspace(0, N_Y, len(periods) + 1, dtype=int)
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        thresholds[start:end] = np.quantile(anomaly[start:end], anomaly_quantile)
    core = anomaly > np.maximum(thresholds[:, None], minimum_anomaly_mm)

    ring_one = dilate(core, 1)
    ring_two = dilate(core, dilation_steps)
    weights = np.zeros_like(residual)
    weights[ring_two] = 0.28
    weights[ring_one] = 0.62
    weights[core] = 1.0

    delta = np.clip(periodic_prediction - residual, -delta_limit_mm, delta_limit_mm)
    restored = residual + weights * delta

    rng = np.random.default_rng(RANDOM_SEED)
    intact = np.flatnonzero((~ring_two).ravel())
    sample = rng.choice(intact, size=min(12000, len(intact)), replace=False)
    differences = (periodic_prediction - residual).ravel()[sample]
    metrics = {
        "self_supervised_holdout_mae_mm": float(np.mean(np.abs(differences))),
        "self_supervised_holdout_rmse_mm": float(np.sqrt(np.mean(differences * differences))),
        "detected_damage_fraction": float(np.mean(core)),
        "blended_patch_fraction": float(np.mean(weights > 0)),
        "maximum_radial_change_mm": float(np.max(np.abs(weights * delta))),
        "anomaly_quantile": anomaly_quantile,
        "minimum_anomaly_mm": minimum_anomaly_mm,
    }
    return restored, weights, periodic_prediction, periods, metrics


def generate_shell(
    center_x: float,
    center_z: float,
    y_centers: np.ndarray,
    outer_radius: np.ndarray,
    inner_profile: np.ndarray,
    inner_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.arange(N_THETA) / N_THETA * 2 * np.pi
    cosines = np.cos(angles)
    sines = np.sin(angles)

    outer_start = int(np.argmax(np.median(outer_radius, axis=1) > 2.5))
    outer_ring_y = y_centers[outer_start:]
    outer_ring_r = outer_radius[outer_start:]
    outer_vertices = np.column_stack((
        center_x + (outer_ring_r * cosines[None, :]).ravel(),
        np.repeat(outer_ring_y, N_THETA),
        center_z + (outer_ring_r * sines[None, :]).ravel(),
    ))

    inner_candidates = np.flatnonzero(inner_valid & (inner_profile > 2.5))
    if len(inner_candidates) == 0:
        raise ValueError("Could not locate inner bowl surface")
    inner_start = int(inner_candidates[0])
    inner_ring_y = y_centers[inner_start:]
    enforced_inner = np.minimum(inner_profile[inner_start:], np.median(outer_radius[inner_start:], axis=1) - 4.0)
    enforced_inner = np.maximum(enforced_inner, 0.8)
    inner_vertices = np.column_stack((
        center_x + np.outer(enforced_inner, cosines).ravel(),
        np.repeat(inner_ring_y, N_THETA),
        center_z + np.outer(enforced_inner, sines).ravel(),
    ))

    vertices = np.vstack((outer_vertices, inner_vertices))
    outer_rows = len(outer_ring_y)
    inner_rows = len(inner_ring_y)
    outer_offset = 0
    inner_offset = len(outer_vertices)
    faces: list[tuple[int, int, int]] = []
    patch_flags: list[bool] = []

    for row in range(outer_rows - 1):
        for column in range(N_THETA):
            next_column = (column + 1) % N_THETA
            a = outer_offset + row * N_THETA + column
            b = outer_offset + row * N_THETA + next_column
            c = outer_offset + (row + 1) * N_THETA + column
            d = outer_offset + (row + 1) * N_THETA + next_column
            faces.extend(((a, c, b), (b, c, d)))
            patch_flags.extend((False, False))

    for row in range(inner_rows - 1):
        for column in range(N_THETA):
            next_column = (column + 1) % N_THETA
            a = inner_offset + row * N_THETA + column
            b = inner_offset + row * N_THETA + next_column
            c = inner_offset + (row + 1) * N_THETA + column
            d = inner_offset + (row + 1) * N_THETA + next_column
            faces.extend(((a, b, c), (b, d, c)))
            patch_flags.extend((False, False))

    outer_bottom = len(vertices)
    inner_bottom = outer_bottom + 1
    vertices = np.vstack((
        vertices,
        [center_x, y_centers[outer_start] - 0.75, center_z],
        [center_x, y_centers[inner_start] - 0.75, center_z],
    ))
    for column in range(N_THETA):
        next_column = (column + 1) % N_THETA
        faces.append((outer_bottom, outer_offset + next_column, outer_offset + column))
        patch_flags.append(False)
        faces.append((inner_bottom, inner_offset + column, inner_offset + next_column))
        patch_flags.append(False)

    outer_top = outer_offset + (outer_rows - 1) * N_THETA
    inner_top = inner_offset + (inner_rows - 1) * N_THETA
    for column in range(N_THETA):
        next_column = (column + 1) % N_THETA
        faces.extend((
            (outer_top + column, outer_top + next_column, inner_top + column),
            (outer_top + next_column, inner_top + next_column, inner_top + column),
        ))
        patch_flags.extend((False, False))

    return vertices, np.asarray(faces, dtype=np.int32), np.asarray(patch_flags, dtype=bool)


def edge_validation(faces: np.ndarray) -> dict[str, int | bool]:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = int(np.count_nonzero(counts == 1))
    nonmanifold = int(np.count_nonzero(counts > 2))
    return {
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "watertight_by_edges": boundary == 0 and nonmanifold == 0,
    }


def map_to_image(values: np.ndarray, mask: np.ndarray | None = None) -> Image.Image:
    low, high = np.quantile(values, [0.02, 0.98])
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
    red = (normalized * 255).astype(np.uint8)
    blue = ((1 - normalized) * 255).astype(np.uint8)
    green = (70 + 120 * (1 - np.abs(normalized - 0.5) * 2)).astype(np.uint8)
    rgb = np.dstack((red, green, blue))
    if mask is not None:
        rgb[mask] = np.array([255, 45, 35], dtype=np.uint8)
    return Image.fromarray(rgb).resize((N_THETA * 2, N_Y * 2), Image.Resampling.NEAREST)


def render_side(vertices: np.ndarray, faces: np.ndarray, size: tuple[int, int] = (560, 680)) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, (8, 12, 20))
    draw = ImageDraw.Draw(canvas)
    points = vertices[:, [0, 1]]
    low = points.min(axis=0)
    high = points.max(axis=0)
    scale = min((width - 30) / (high[0] - low[0]), (height - 30) / (high[1] - low[1]))
    projected = (points - (low + high) / 2) * scale
    projected[:, 0] += width / 2
    projected[:, 1] = height / 2 - projected[:, 1]
    triangles = vertices[faces]
    order = np.argsort(triangles.mean(axis=1)[:, 2])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1
    shade = 85 + 160 * np.abs(normals[:, 2] / lengths)
    for index in order:
        polygon = [tuple(projected[value]) for value in faces[index]]
        value = int(np.clip(shade[index], 0, 255))
        draw.polygon(polygon, fill=(value, int(value * 0.78), int(value * 0.42)))
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 이 경로는 빗살무늬토기 자기지도 실행(이 파일의 단독 진입점) 전용이다.
    # 최상위에 두면 build_mesh_direct.cyl_map 의 load_ssv 임포트 때 같이 평가돼
    # 원본 폴더가 없는 PC 에서 StopIteration 이 난다. 쓰는 자리에서만 찾는다.
    source_stl = next(SOURCE_DIR.glob("*.stl"))
    vertices, faces = read_binary_stl(source_stl)
    center_x, center_z = fit_axis(vertices)
    outer_mask, inner_mask, surface_stats = classify_surfaces(vertices, faces, center_x, center_z)

    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    y_edges = np.linspace(y_min, y_max, N_Y + 1)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    outer_raw, outer_counts = radial_grid(vertices, outer_mask, center_x, center_z, y_edges)
    raw_outer_profile = np.nanmedian(outer_raw, axis=1)
    outer_profile = fill_profile(raw_outer_profile)
    outer_grid = fill_grid(outer_raw, outer_profile)
    outer_profile = moving_average(np.median(outer_grid, axis=1), 7)
    residual = outer_grid - outer_profile[:, None]

    restored_residual, weights, periodic_prediction, periods, learning_metrics = learn_restoration(residual)
    ideal_residual, ideal_weights, _, ideal_periods, ideal_learning_metrics = learn_restoration(
        residual,
        anomaly_quantile=0.95,
        minimum_anomaly_mm=0.20,
        delta_limit_mm=4.0,
        dilation_steps=5,
    )
    restored_outer = outer_profile[:, None] + restored_residual
    ideal_outer = outer_profile[:, None] + ideal_residual

    inner_raw, inner_counts = radial_grid(vertices, inner_mask, center_x, center_z, y_edges)
    raw_inner_profile = np.nanmedian(inner_raw, axis=1)
    inner_valid = np.isfinite(raw_inner_profile)
    inner_profile = fill_profile(raw_inner_profile)
    inner_profile = np.minimum(inner_profile, outer_profile - 4.0)

    restored_vertices, restored_faces, _ = generate_shell(
        center_x, center_z, y_centers, restored_outer, inner_profile, inner_valid
    )
    restored_path = OUTPUT_DIR / "02_self_supervised_restored.stl"
    write_binary_stl(restored_path, restored_vertices, restored_faces, "SELF-SUPERVISED POTTERY RESTORATION")

    outer_start = int(np.argmax(np.median(restored_outer, axis=1) > 2.5))
    patch_cell = (weights[outer_start:-1] > 0.02) | (weights[outer_start + 1:] > 0.02)
    outer_face_count = (len(y_centers[outer_start:]) - 1) * N_THETA * 2
    selected_face_indices: list[int] = []
    for row in range(patch_cell.shape[0]):
        for column in range(N_THETA):
            if patch_cell[row, column] or patch_cell[row, (column + 1) % N_THETA]:
                base = (row * N_THETA + column) * 2
                selected_face_indices.extend((base, base + 1))
    patch_faces = restored_faces[np.asarray(selected_face_indices, dtype=np.int64)]
    patch_path = OUTPUT_DIR / "03_restoration_patch_surface.stl"
    write_binary_stl(patch_path, restored_vertices, patch_faces, "RESTORATION PATCH SURFACE - VR ONLY")

    ideal_vertices, ideal_faces, _ = generate_shell(
        center_x, center_z, y_centers, ideal_outer, inner_profile, inner_valid
    )
    ideal_path = OUTPUT_DIR / "04_idealized_self_supervised_restored.stl"
    write_binary_stl(ideal_path, ideal_vertices, ideal_faces, "IDEALIZED SELF-SUPERVISED POTTERY RESTORATION")

    ideal_patch_cell = (ideal_weights[outer_start:-1] > 0.02) | (ideal_weights[outer_start + 1:] > 0.02)
    ideal_selected_face_indices: list[int] = []
    for row in range(ideal_patch_cell.shape[0]):
        for column in range(N_THETA):
            if ideal_patch_cell[row, column] or ideal_patch_cell[row, (column + 1) % N_THETA]:
                base = (row * N_THETA + column) * 2
                ideal_selected_face_indices.extend((base, base + 1))
    ideal_patch_faces = ideal_faces[np.asarray(ideal_selected_face_indices, dtype=np.int64)]
    ideal_patch_path = OUTPUT_DIR / "05_idealized_restoration_patch_surface.stl"
    write_binary_stl(
        ideal_patch_path,
        ideal_vertices,
        ideal_patch_faces,
        "IDEALIZED RESTORATION PATCH SURFACE - VR ONLY",
    )

    original_copy = OUTPUT_DIR / "01_original_reference.stl"
    original_copy.write_bytes(source_stl.read_bytes())

    validation = edge_validation(restored_faces)
    report = {
        "source": str(source_stl),
        "method": "single-object self-supervised periodic exemplar completion",
        "axis_center_xz": [center_x, center_z],
        "grid": [N_Y, N_THETA],
        "learned_periods_by_height_band": periods,
        "source_surface_classification": surface_stats,
        "source_outer_grid_coverage": float(np.mean(outer_counts > 0)),
        "source_inner_grid_coverage": float(np.mean(inner_counts > 0)),
        "conservative_learning_metrics": learning_metrics,
        "idealized_learning_metrics": ideal_learning_metrics,
        "restored_vertices": int(len(restored_vertices)),
        "restored_faces": int(len(restored_faces)),
        "patch_surface_faces": int(len(patch_faces)),
        "idealized_patch_surface_faces": int(len(ideal_patch_faces)),
        **validation,
        "caveat": "The result is a computational hypothesis, not an authenticated archaeological reconstruction.",
    }
    (OUTPUT_DIR / "restoration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    original_map = map_to_image(residual)
    detected_map = map_to_image(residual, weights > 0.02)
    restored_map_image = map_to_image(restored_residual)
    diagnostics = Image.new("RGB", (original_map.width * 3, original_map.height + 34), (245, 245, 245))
    diagnostics.paste(original_map, (0, 34))
    diagnostics.paste(detected_map, (original_map.width, 34))
    diagnostics.paste(restored_map_image, (original_map.width * 2, 34))
    draw = ImageDraw.Draw(diagnostics)
    draw.text((8, 10), "unwrapped original residual", fill=(20, 20, 20))
    draw.text((original_map.width + 8, 10), "auto-detected restoration mask", fill=(20, 20, 20))
    draw.text((original_map.width * 2 + 8, 10), "self-supervised restored residual", fill=(20, 20, 20))
    diagnostics.save(OUTPUT_DIR / "unwrapped_restoration_diagnostics.png")

    ideal_detected_map = map_to_image(residual, ideal_weights > 0.02)
    ideal_map_image = map_to_image(ideal_residual)
    ideal_diagnostics = Image.new("RGB", (original_map.width * 3, original_map.height + 34), (245, 245, 245))
    ideal_diagnostics.paste(original_map, (0, 34))
    ideal_diagnostics.paste(ideal_detected_map, (original_map.width, 34))
    ideal_diagnostics.paste(ideal_map_image, (original_map.width * 2, 34))
    draw = ImageDraw.Draw(ideal_diagnostics)
    draw.text((8, 10), "unwrapped original residual", fill=(20, 20, 20))
    draw.text((original_map.width + 8, 10), "idealized restoration mask", fill=(20, 20, 20))
    draw.text((original_map.width * 2 + 8, 10), "idealized restored residual", fill=(20, 20, 20))
    ideal_diagnostics.save(OUTPUT_DIR / "unwrapped_idealized_diagnostics.png")

    original_preview = render_side(vertices, faces)
    restored_preview = render_side(restored_vertices, restored_faces)
    ideal_preview = render_side(ideal_vertices, ideal_faces)
    comparison = Image.new("RGB", (original_preview.width * 3, original_preview.height + 34), (245, 245, 245))
    comparison.paste(original_preview, (0, 34))
    comparison.paste(restored_preview, (original_preview.width, 34))
    comparison.paste(ideal_preview, (original_preview.width * 2, 34))
    draw = ImageDraw.Draw(comparison)
    draw.text((8, 10), "original", fill=(20, 20, 20))
    draw.text((original_preview.width + 8, 10), "conservative restoration", fill=(20, 20, 20))
    draw.text((original_preview.width * 2 + 8, 10), "idealized restoration", fill=(20, 20, 20))
    comparison.save(OUTPUT_DIR / "original_vs_restored.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
