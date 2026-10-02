#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**격자를 안 쓰고** AI 점군에서 바로 면을 만든다 — 푸아송 · 볼피벗.

왜 이 스크립트가 있나
---------------------
`make_mesh_2node.py` 는 `(h,θ)` 격자를 깐다. 그게 **회전체라는 가정**을 하나 넣는
것이라, *"가정 없이 AI 가 예측한 대로만 만들면 어떻게 되나"* 를 물을 수 있다.

이 스크립트가 그 답이다. **비교군을 만들기 위한 것이지 현행 경로가 아니다.**
결과는 v23a~v23d 로 등록돼 있고, 요약은 `결과.md` §5.12 에 있다.

무엇이 달라지나
---------------
격자가 하던 일 넷이 전부 없어진다.

  ① 결손 판정      "점 없는 칸 = 결손" 이라는 규칙이 사라진다
  ② 투창 보호      n-fold 판정을 걸 자리가 없다
  ③ 출처 분리      관측과 AI 가 **한 표면으로 녹는다** (NFR-ETH-003 위반)
  ④ 원본 텍스처    정점을 전부 새로 만드니 UV 가 날아간다

그래서 출력은 2노드가 아니라 **한 덩어리**(`region_fused`)다. 이름이 곧 한계다.

핵심 관찰 (71489, v3 TTA8)
--------------------------
푸아송은 **밀도 절단** 세기로 구멍을 열고 닫는데, **투창과 결손을 못 가른다** —
점군에서 둘 다 *"점이 없는 곳"* 이라 구별할 정보가 점 자체에 없다.

    밀도 절단   투창 열림   결손 메움
      없음         5.3%      99.9%     ← 투창까지 메운다
      15%         68.8%      13.3%
      30%         82.9%       5.9%     ← 결손도 안 메운다
    v22 (격자)    89.8%      99.7%     ← 둘 다 된다

쓰는 법
-------
    $PY make_mesh_direct.py --src work/pred_71489_A_normal_v3tta8.npz \
        --method poisson --depth 8 --trim 0.15 --name _tmp_v23b
    $PY versions.py --build v23b        # 이쪽이 편하다
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RESTORE_ROOT", HERE.parent))
WORK = Path(os.environ.get("ADAPOINTR_WORK", HERE / "work"))
OUT = WORK / "restored_adapointr"
SRC_GLB = ROOT / "gupdari-71489" / "in" / "A_normal.glb"

sys.path.insert(0, str(HERE))
import make_mesh_71489 as M                     # noqa: E402
from make_mesh_2node import load_original, norm_params    # noqa: E402


def outward_normals(pcd, P, up, c2, knn=30):
    """법선을 **축에서 바깥으로** 통일한다.

    푸아송은 법선 방향으로 안팎을 정한다. `estimate_normals` 는 부호를 안 맞춰 주므로
    그대로 넣으면 표면이 뒤집혀 누더기가 된다. 이 유물은 회전체라
    **축에서 멀어지는 쪽이 바깥**이라는 확실한 기준이 있다.
    """
    import open3d as o3d
    ax = [i for i in range(3) if i != up]
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    n = np.asarray(pcd.normals).copy()
    r = np.zeros_like(P)
    r[:, ax[0]] = P[:, ax[0]] - c2[0]
    r[:, ax[1]] = P[:, ax[1]] - c2[1]
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    rn[rn < 1e-9] = 1.0
    flip = (n * (r / rn)).sum(1) < 0
    n[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)
    return int(flip.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(WORK / "pred_71489_A_normal_v3tta8.npz"))
    ap.add_argument("--orig", type=Path, default=SRC_GLB)
    ap.add_argument("--method", choices=("poisson", "bpa"), default="poisson")
    ap.add_argument("--depth", type=int, default=8,
                    help="푸아송 8분트리 깊이. 크면 세밀하나 잡음도 같이 산다")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="""푸아송 **밀도 하위 몇 할을 버리나** (0~1).

    푸아송은 빈 곳도 매끄럽게 덮어 버린다(watertight 를 지향한다). 밀도가 낮은
    정점을 버려야 구멍이 열린다. **그런데 투창과 결손을 같이 연다** — 이 스크립트의
    핵심 관찰이다.""")
    ap.add_argument("--radii", default="2,4,6",
                    help="볼피벗 공 반지름. 평균 최근접거리의 배수로 준다")
    ap.add_argument("--knn", type=int, default=30, help="법선 추정 이웃 수")
    ap.add_argument("--name", default="_tmp_direct")
    args = ap.parse_args()

    import open3d as o3d

    d = np.load(args.src, allow_pickle=True)
    P = np.asarray(d["pred"], np.float64)
    new = np.asarray(d["new"], bool)
    up = int(d["up"])
    print("예측 %d점 (그중 새 점 %d · %.1f%%)" % (len(P), new.sum(), 100 * new.mean()))

    orig = load_original(args.orig)
    cen, scl = norm_params(orig)
    mg = M.load_measure_glb()
    _, c2 = mg.fit_axis(P, up)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    nflip = outward_normals(pcd, P, up, c2, args.knn)
    print("  법선 바깥으로 정렬: %d점 뒤집음" % nflip)

    # 밀도 진단 — 관측 재현부와 AI 새 점부가 얼마나 다른가.
    # 푸아송·볼피벗은 **균일 밀도를 전제**하므로 이 비율이 곧 품질 한계다.
    kd = o3d.geometry.KDTreeFlann(pcd)
    rng = np.random.RandomState(0)

    def spacing(idx):
        idx = rng.choice(idx, min(2000, len(idx)), replace=False)
        return float(np.median([np.sqrt(kd.search_knn_vector_3d(pcd.points[int(i)], 2)[2][1])
                                for i in idx]))

    s_obs, s_ai = spacing(np.nonzero(~new)[0]), spacing(np.nonzero(new)[0])
    print("  점 간격 중앙: 관측재현부 %.4f · AI 새 점부 %.4f  (**%.1f배**)"
          % (s_obs, s_ai, s_ai / max(s_obs, 1e-9)))

    if args.method == "poisson":
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.depth)
        dens = np.asarray(dens)
        n0 = len(mesh.vertices)
        if args.trim > 0:
            mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.trim))
            mesh.remove_unreferenced_vertices()
        print("  푸아송 depth=%d · 밀도 하위 %.0f%% 버림: 정점 %d → %d"
              % (args.depth, 100 * args.trim, n0, len(mesh.vertices)))
    else:
        avg = float(np.mean(pcd.compute_nearest_neighbor_distance()))
        radii = [avg * float(x) for x in args.radii.split(",")]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        print("  볼피벗 반지름 %s (평균 최근접 %.4f 의 배수)" % (args.radii, avg))

    V = np.asarray(mesh.vertices) * scl + cen        # 원본 자로 되돌린다
    F = np.asarray(mesh.triangles)
    if len(F) == 0:
        print("[!] 면이 하나도 안 나왔다.")
        return 1
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
    pieces = len(tm.split(only_watertight=False))
    ang = tm.face_adjacency_angles
    dih = float(np.degrees(np.median(ang[np.isfinite(ang)])))
    print("  면 %d · 조각 %d · 이면각 중앙 %.2f°  (관측 3.04° · v22 2.25°)"
          % (len(F), pieces, dih))

    # **한 덩어리로 낸다.** 관측과 AI 가 녹아 붙어 나눌 수가 없다 —
    # 이름이 곧 이 방법의 한계다 (`region_observed` / `region_ai` 가 아니다).
    tm.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="region_fused", baseColorFactor=[0.72, 0.69, 0.64, 1.0],
            metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True))
    OUT.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(tm, geom_name="region_fused")
    glb = OUT / (args.name + ".glb")
    scene.export(str(glb))
    print("\n  region_fused  V=%d F=%d  — **출처 분리 없음.** 원본 텍스처도 잃는다"
          % (len(V), len(F)))
    print("저장 → %s" % glb)

    side = OUT / (args.name + ".json")
    side.write_text(json.dumps(
        {"method": args.method, "depth": args.depth, "trim": args.trim,
         "radii": args.radii, "면": len(F), "조각": pieces, "이면각": round(dih, 2),
         "점간격_관측재현부": round(s_obs, 5), "점간격_AI새점부": round(s_ai, 5)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
