#!/usr/bin/env python3
"""완성 결과를 수치로 잰다 — Q3(양자화 전후 품질) · Q4(관측부 보존). GPU 불필요.

    python tools/eval_completion.py --result results/<라벨> --gt samples/CompC_datasets/plyobj/gtdata/bimba.ply \
        --partial samples/CompC_datasets/plyobj/indata/bimba.ply --label <라벨>

재는 것
  chamfer_l2      : 결과 ↔ 정답. 양쪽을 각자 단위 상자로 정규화한 뒤 잰다 (프레임 차이 흡수).
                    LaS-Comp 이 입력에 yz 교환을 걸 수 있어, yz 를 바꾼 경우도 같이 재고 **작은 쪽**을 쓴다.
                    어느 쪽이 채택됐는지 함께 남겨 프레임 문제를 숨기지 않는다.
  observed_keep   : Q4. 입력 부분 점군의 각 점 → 결과 표면 최근접 거리. ERS 가 관측부를 그대로 넣었으면
                    이 값은 복셀 한 칸(1/64 ≈ 0.0156) 안에 들어야 한다. 그 밖으로 나간 점의 비율도 낸다.
  n_verts/n_faces : 메시 크기 (있을 때)

결과 폴더에서 찾는 파일: *.obj / *.ply 중 'input_' 으로 시작하지 않는 것. 메시가 있으면 표면에서 점을 뽑고,
점군만 있으면 그대로 쓴다.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

try:
    import trimesh
    from scipy.spatial import cKDTree
except ImportError as e:
    sys.exit("필요: pip install trimesh scipy  (%s)" % e)

VOXEL = 1.0 / 64          # LaS-Comp 내부 해상도 — 관측부 보존 판정 기준


def load_geometry(path: Path):
    """메시면 Trimesh 를, 점군이면 None 을 돌려준다 (점군은 load_points 로)."""
    obj = trimesh.load(str(path), force=None, process=False)
    if isinstance(obj, trimesh.Scene):
        geos = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        # glb 는 노드 변환(Y-up 등)을 지니므로 월드 좌표로 펼쳐서 합친다 — 그래야 GT·입력과 같은 프레임 비교가 된다
        obj = trimesh.util.concatenate(
            [g.copy().apply_transform(obj.graph.get(name)[0]) for name, g in obj.geometry.items()
             if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]) if geos else None
    if isinstance(obj, trimesh.Trimesh) and len(obj.faces) > 0:
        return obj
    return None


def load_points(path: Path, n: int = 50000) -> tuple[np.ndarray, dict]:
    mesh = load_geometry(path)
    if mesh is not None:
        meta = {"n_verts": int(len(mesh.vertices)), "n_faces": int(len(mesh.faces))}
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts, dtype=np.float64), meta
    obj = trimesh.load(str(path), force=None, process=False)
    pts = np.asarray(getattr(obj, "vertices", obj), dtype=np.float64)
    return pts, {}


DENSE_N = 2_000_000


def build_reference(result_path: Path, result_pts: np.ndarray) -> tuple["cKDTree", str]:
    """입력 점 → 결과 거리를 잴 기준(KD 트리)을 **한 번만** 만든다.

    결과가 메시면 표면을 200 만 점으로 뽑아 쓴다. 표본 간격 ≈ sqrt(면적/2e6) ≈ 0.0006 으로 복셀(0.0149)의
    1/25 — 판정에 영향 없는 오차다. 5 만 점(간격 ≈0.008)으로 재던 이전 방식은 점군 기준 0% 를 메시 기준 9.8% 로
    부풀렸다. `trimesh.nearest.on_surface`(정확 표면 거리)는 88 만 면 메시에서 후보 삼각형 배열이 터져
    RAM OOM 으로 커널에 죽었다 — 쓰지 않는다. 점군이면 그 점들이 기준이다."""
    mesh = load_geometry(result_path)
    if mesh is not None:
        dense, _ = trimesh.sample.sample_surface(mesh, DENSE_N)
        return cKDTree(np.asarray(dense, dtype=np.float64)), "dense_sample_2M"
    return cKDTree(result_pts), "nearest_sample"


def normalize(p: np.ndarray) -> np.ndarray:
    lo, hi = p.min(0), p.max(0)
    c = (lo + hi) / 2
    s = (hi - lo).max()
    return (p - c) / (s if s > 0 else 1.0)


def chamfer_l2(a: np.ndarray, b: np.ndarray) -> float:
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(np.mean(da ** 2) + np.mean(db ** 2))


def swap_yz(p: np.ndarray) -> np.ndarray:
    q = p.copy(); q[:, [1, 2]] = q[:, [2, 1]]; return q


def pick_result_file(result_dir: Path) -> Path:
    # LaS-Comp 은 output_mesh.glb · output_points.ply · sparse_structure.ply 를 낸다. 메시(.glb/.obj)를 먼저,
    # 없으면 output_points, 마지막이 sparse_structure(16³ 희소 구조 — 평가용으로는 너무 거칠다).
    cands = [p for p in result_dir.iterdir()
             if p.suffix.lower() in (".glb", ".obj", ".ply") and not p.name.startswith("input_")]
    if not cands:
        sys.exit("결과 폴더에 메시/점군이 없다: %s" % result_dir)
    def rank(p):
        n = p.name.lower()
        return (0 if p.suffix.lower() in (".glb", ".obj") else 1,
                0 if "mesh" in n else (1 if "output_points" in n else 2), n)
    cands.sort(key=rank)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="결과 폴더 (results/<라벨>)")
    ap.add_argument("--gt", required=True, help="정답 ply")
    ap.add_argument("--partial", required=True, help="입력 부분 점군 ply (관측부 보존 판정용)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="eval_reports")
    ap.add_argument("--n", type=int, default=50000, help="표면 표본 수")
    a = ap.parse_args()

    result_dir = Path(a.result)
    rf = pick_result_file(result_dir)
    if "output_points" in rf.name.lower():
        # 저자 스크립트는 output_points 에 **입력 점군을 이어붙인다** (pts_all = vstack([pts_mesh, xyz_original])).
        # 그 파일로 관측부 보존을 재면 구조상 항상 통과한다 — 무효. 메시(.glb/.obj)가 있어야 Q4 다.
        print("  경고: 결과가 output_points.ply 다 — 입력 점군이 섞여 있어 관측부 보존(Q4)이 무효다. "
              "메시 파일로 재라.", file=sys.stderr)
    res_pts, meta = load_points(rf, a.n)
    gt_pts, _ = load_points(Path(a.gt), a.n)
    part_pts, _ = load_points(Path(a.partial), a.n)

    # ── Q3: Chamfer (정규화 후, 프레임 두 가지 중 작은 쪽) ──
    rn, gn = normalize(res_pts), normalize(gt_pts)
    cd_same = chamfer_l2(rn, gn)
    cd_swap = chamfer_l2(rn, normalize(swap_yz(gt_pts)))
    frame = "same" if cd_same <= cd_swap else "yz_swapped"
    cd = min(cd_same, cd_swap)

    # ── Q4: 관측부 보존 — 결과 좌표계는 저장 시 입력 원좌표로 되돌려져 있다(스크립트가 xyz_original 기준으로 저장)
    #      그래도 프레임이 어긋날 수 있어 두 프레임 중 작은 쪽을 쓰고 어느 쪽인지 남긴다.
    ref, how = build_reference(rf, res_pts)          # 기준은 한 번만 — 두 프레임을 같은 트리에 질의
    d_same, _ = ref.query(part_pts)
    d_swap, _ = ref.query(swap_yz(part_pts))
    d = d_same if np.median(d_same) <= np.median(d_swap) else d_swap
    keep_frame = "same" if d is d_same else "yz_swapped"
    # 입력 프레임 스케일로 복셀 크기를 맞춘다 (입력 점군의 최대 변 기준)
    part_scale = (part_pts.max(0) - part_pts.min(0)).max()
    voxel_abs = VOXEL * part_scale

    rep = {
        "label": a.label or result_dir.name,
        "result_file": rf.name,
        **meta,
        "chamfer_l2_normalized": round(cd, 6),
        "chamfer_frame": frame,
        "chamfer_l2_same": round(cd_same, 6),
        "chamfer_l2_yz_swapped": round(cd_swap, 6),
        "observed_keep_median": round(float(np.median(d)), 6),
        "observed_keep_p95": round(float(np.percentile(d, 95)), 6),
        "observed_keep_max": round(float(d.max()), 6),
        "observed_frame": keep_frame,
        "observed_distance_method": how,
        "voxel_size_in_input_units": round(float(voxel_abs), 6),
        "observed_outside_one_voxel_ratio": round(float((d > voxel_abs).mean()), 4),
    }
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / ("%s.json" % rep["label"])).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("  평가 —", rep["label"], "(", rf.name, ")")
    print("=" * 60)
    if meta:
        print("  메시            : %d 정점 · %d 면" % (meta["n_verts"], meta["n_faces"]))
    print("  Chamfer(정규화) : %.6f   [프레임: %s]" % (cd, frame))
    print("  관측부 보존     : 중앙값 %.5f · p95 %.5f · 최대 %.5f  [프레임: %s · 거리: %s]"
          % (rep["observed_keep_median"], rep["observed_keep_p95"], rep["observed_keep_max"], keep_frame, how))
    print("  복셀 1칸 밖 비율: %.2f%%   (복셀 크기 %.5f)" % (100 * rep["observed_outside_one_voxel_ratio"], voxel_abs))
    print("  저장            :", out / ("%s.json" % rep["label"]))
    print("=" * 60)


if __name__ == "__main__":
    main()
