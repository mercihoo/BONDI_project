#!/usr/bin/env python3
"""합성 — 관측 영역은 원본 메시(질감 그대로), 결손 영역만 LaS-Comp 메시로 채운다. 2 노드 GLB. GPU 불필요.

    python tools/composite.py --source artifacts/gupdari71489/source_model.glb \
        --result results/<라벨>/output_mesh.glb --out results/<라벨>/composite.glb \
        [--tau-voxels 1.5] [--min-area 0.003] [--ymin 0.0]

왜: LaS-Comp 디코더는 관측 표면도 다시 그린다 (굽다리 2차: 관측점 33% 가 1 복셀 이상 이동). 그 표면을 그대로
     내보내면 실측을 바꾼 것이라 NFR-ETH 위반이다. 그래서 LaS-Comp 메시는 **원본 표면에서 τ 이상 떨어진 면**만 남기고,
     남은 조각 중 작은 섬(관측부 재합성 잔재)은 버린다. 남는 큰 조각이 결손 채움이다.
프레임: 결과 GLB 는 LaS-Comp 내부 프레임(yz 교환, 부호 미정)이다 — 후보를 입력 표면 거리로 재어 가장 맞는 것을 고르고 기록한다.
--ymin: 바닥에서 이 비율(bbox 높이 기준) 아래의 채움은 버린다 — 굽다리 투창은 LaS-Comp 이 메워 버리므로(64³) 몸통만 채우고
        굽다리는 기하 방식에 맡길 때 쓴다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

VOXEL = 1.0 / 64
DENSE_N = 2_000_000

# 후보 프레임: (축 순열, 부호). 결과가 입력에 yz 교환을 걸었다는 것은 알지만 부호(거울/회전)는 모른다.
CANDS = {
    "same":              ([0, 1, 2], [1, 1, 1]),
    "swap_yz":           ([0, 2, 1], [1, 1, 1]),
    "swap_yz_negz":      ([0, 2, 1], [1, 1, -1]),   # = x 축 -90° 회전
    "swap_yz_negy":      ([0, 2, 1], [1, -1, 1]),   # = x 축 +90° 회전
    "swap_yz_negyz":     ([0, 2, 1], [1, -1, -1]),
}


def apply_frame(p: np.ndarray, cand) -> np.ndarray:
    perm, sign = cand
    return p[:, perm] * np.asarray(sign, dtype=p.dtype)


def frame_is_improper(cand) -> bool:
    perm, sign = cand
    par = 1
    for i in range(3):
        for j in range(i + 1, 3):
            if perm[i] > perm[j]:
                par = -par
    return par * int(np.prod(sign)) < 0


def load_world_mesh(path: Path):
    """GLB 를 장면으로 읽어 노드 변환을 펼친 월드 좌표 메시(기하만)와 원래 장면을 돌려준다."""
    sc = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if isinstance(g, trimesh.Trimesh) and len(g.faces):
            parts.append(trimesh.Trimesh(g.vertices.copy(), g.faces.copy(), process=False).apply_transform(T))
    if not parts:
        sys.exit("메시 없음: %s" % path)
    return sc, trimesh.util.concatenate(parts)


def junction_fraction(las_t: trimesh.Trimesh, src: trimesh.Trimesh, bins: int = 64) -> float:
    """몸통·굽다리 경계 = 세로축 둘레 반지름이 가장 좁아지는 높이 (bbox 높이 비율). 완성된 LaS 메시로 잰다 — 원본은 조각이라 빈 구간이 있다.
    위·아래 20% 는 제외 (테두리·받침 밑은 원래 좁아질 수 있다)."""
    v = np.asarray(las_t.vertices, dtype=np.float64)
    c = (src.bounds[0] + src.bounds[1]) / 2
    r = np.hypot(v[:, 0] - c[0], v[:, 2] - c[2])
    y0, y1 = src.bounds[0][1], src.bounds[1][1]
    idx = np.clip(((v[:, 1] - y0) / (y1 - y0) * bins).astype(int), 0, bins - 1)
    rmax = np.full(bins, np.nan)
    for b in range(bins):
        m = idx == b
        if m.any():
            rmax[b] = np.percentile(r[m], 98)
    lo, hi = int(bins * 0.2), int(bins * 0.8)
    band = rmax[lo:hi]
    j = lo + int(np.nanargmin(band))
    return (j + 0.5) / bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="손상 원본 GLB (질감 포함, 월드 프레임 기준)")
    ap.add_argument("--result", required=True, help="LaS-Comp output_mesh.glb")
    ap.add_argument("--out", required=True)
    # τ 0.65 복셀 = 사용자 확정값(2026-09-17). 1.5 → 1.0 → 0.5 → 0.75 를 눈으로 비교한 뒤 0.65 로 정착.
    # (겹침은 τ 0.5 에서 24%, 0.75 에서 12% — 0.65 는 그 사이. 파단면 윤곽과 이음새 틈의 절충점)
    ap.add_argument("--tau-voxels", type=float, default=0.65, help="원본 표면에서 이 거리(복셀 단위) 안의 LaS 면은 버린다")
    ap.add_argument("--min-area", type=float, default=0.003, help="LaS 전체 면적 대비 이보다 작은 조각(섬)은 버린다")
    ap.add_argument("--ymin", default="0", help="bbox 높이 비율 — 이 아래 채움은 버린다. 0 = 안 씀, auto = 반지름이 가장 좁아지는 높이(몸통·굽다리 경계)")
    ap.add_argument("--color", default="200,188,168", help="채움 재질 baseColor (0-255)")
    a = ap.parse_args()

    src_scene, src = load_world_mesh(Path(a.source))
    _, las = load_world_mesh(Path(a.result))
    scale = float((src.bounds[1] - src.bounds[0]).max())
    tau = a.tau_voxels * VOXEL * scale
    print("원본: %d 면 · 최대 변 %.4f · 복셀 %.4f · τ = %.4f (%.1f 복셀)" % (len(src.faces), scale, VOXEL * scale, tau, a.tau_voxels))
    print("LaS : %d 면" % len(las.faces))

    # ── 프레임 고르기: 원본 표면 표본 → 후보 프레임의 LaS 표면 거리 중앙값이 가장 작은 것
    src_pts, _ = trimesh.sample.sample_surface(src, 200_000)
    las_pts, _ = trimesh.sample.sample_surface(las, 200_000)
    src_pts = np.asarray(src_pts, dtype=np.float64); las_pts = np.asarray(las_pts, dtype=np.float64)
    meds = {}
    for name, cand in CANDS.items():
        d, _ = cKDTree(apply_frame(las_pts, cand)).query(src_pts)
        meds[name] = float(np.median(d))
    frame = min(meds, key=meds.get)
    print("프레임 후보(원본→LaS 거리 중앙값): " + " · ".join("%s %.4f" % kv for kv in meds.items()))
    print("→ 채택: %s" % frame)
    cand = CANDS[frame]
    faces = las.faces[:, [0, 2, 1]] if frame_is_improper(cand) else las.faces      # 거울 변환이면 감김을 뒤집어 법선을 지킨다
    las_t = trimesh.Trimesh(apply_frame(np.asarray(las.vertices, dtype=np.float64), cand), faces, process=False)

    # ── 결손 판정: LaS 면 중심 → 원본 표면(200 만 점) 거리
    dense, _ = trimesh.sample.sample_surface(src, DENSE_N)
    tree = cKDTree(np.asarray(dense, dtype=np.float64))
    cen = las_t.triangles_center
    d, _ = tree.query(cen)
    keep = d > tau
    n_tau = int(keep.sum())
    ymin = a.ymin
    if ymin == "auto":
        ymin = junction_fraction(las_t, src)
        print("몸통·굽다리 경계(auto): bbox 높이의 %.3f" % ymin)
    ymin = float(ymin)
    if ymin > 0:
        y0 = src.bounds[0][1] + ymin * (src.bounds[1][1] - src.bounds[0][1])
        keep &= cen[:, 1] > y0
    sub = las_t.submesh([np.where(keep)[0]], append=True) if keep.any() else None
    if sub is None or len(sub.faces) == 0:
        sys.exit("남는 면이 없다 — τ 를 줄여라")

    # ── 섬 제거: 관측부를 다시 그린 잔재는 작은 조각으로 남는다
    comps = sub.split(only_watertight=False)
    areas = np.array([c.area for c in comps])
    big = [c for c, ar in zip(comps, areas) if ar >= a.min_area * las_t.area]
    if not big:
        sys.exit("섬 제거 후 남는 조각이 없다 — --min-area 를 줄여라")
    fill = trimesh.util.concatenate(big)
    print("LaS 면 %d → τ 밖 %d → 높이 필터 뒤 %d → 섬 제거 뒤 %d 면 (%d 조각 유지 / %d 조각 중)"
          % (len(las_t.faces), n_tau, int(keep.sum()), len(fill.faces), len(big), len(comps)))
    print("채움 면적 = LaS 전체의 %.1f%% · 원본 면적의 %.1f%%" % (100 * fill.area / las_t.area, 100 * fill.area / src.area))

    # 복원본에 모델 예측 색이 있으면(output_mesh_color.glb) 채움에 그 색을 쓴다. 없으면 지정 단색.
    res_colored = trimesh.load(a.result, force="mesh", process=False)      # load_world_mesh 는 visual 을 버린다
    if getattr(res_colored.visual, "kind", None) == "vertex" and len(res_colored.vertices) == len(las.vertices):
        vc = np.asarray(res_colored.visual.vertex_colors)
        idx = cKDTree(np.asarray(las_t.vertices)).query(np.asarray(fill.vertices))[1]
        fill.visual = trimesh.visual.ColorVisuals(mesh=fill, vertex_colors=vc[idx])
        print("채움 색: 모델 예측 정점 색 (평균 RGB %s)" % vc[idx][:, :3].mean(0).round(1))
    else:
        rgb = [int(v) for v in a.color.split(",")]
        fill.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name="lascomp_gap_fill", baseColorFactor=rgb + [255], metallicFactor=0.0, roughnessFactor=0.85))
        print("채움 색: 지정 단색", rgb)
    fill.fix_normals()

    out_scene = src_scene.copy()
    out_scene.add_geometry(fill, node_name="lascomp_gap_fill", geom_name="lascomp_gap_fill")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out_scene.export(str(out))
    rep = {
        "source": a.source, "result": a.result, "out": str(out),
        "frame": frame, "frame_medians": {k: round(v, 5) for k, v in meds.items()},
        "tau_voxels": a.tau_voxels, "tau_abs": round(tau, 5), "min_area_frac": a.min_area, "ymin_frac": round(ymin, 4), "ymin_mode": a.ymin,
        "las_faces": int(len(las_t.faces)), "faces_outside_tau": n_tau, "faces_after_height": int(keep.sum()),
        "fill_faces": int(len(fill.faces)), "components_total": int(len(comps)), "components_kept": len(big),
        "fill_area_frac_of_las": round(float(fill.area / las_t.area), 4),
        "fill_area_frac_of_source": round(float(fill.area / src.area), 4),
        "source_faces": int(len(src.faces)),
    }
    out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out, "·", out.with_suffix(".json"))


if __name__ == "__main__":
    main()
