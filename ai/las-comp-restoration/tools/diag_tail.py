#!/usr/bin/env python3
"""입력 → output_points(1.6만) 와 입력 → 메시 밀집 표본(200만) 의 거리를 **같은 배열**로 나란히 재서
꼬리가 왜 다른지 가른다. 메시 기준으로 복셀 밖인 입력 점들이 output_points 기준으로는 얼마나 가까운지,
그 점들이 단위 상자 밖(LaS-Comp 이 버리는 점)인지, output_points 중 메시에서 떨어진 점은 몇 개인지.

    python tools/diag_tail.py results/<라벨> samples/CompC_datasets/plyobj/indata/bimba.ply
"""
import sys
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_completion import load_geometry, load_points, DENSE_N

res_dir, partial = Path(sys.argv[1]), Path(sys.argv[2])
inp, _ = load_points(partial, 50000)
inp_int = inp.copy(); inp_int[:, [1, 2]] = inp_int[:, [2, 1]]          # plyobj: 내부 프레임은 (x, z, y)
op, _ = load_points(res_dir / "output_points.ply", 50000)
mesh = load_geometry(res_dir / "output_mesh.glb")
dense, _ = trimesh.sample.sample_surface(mesh, DENSE_N)
dense = np.asarray(dense)

VOX = 1.0 / 64 * (inp_int.max(0) - inp_int.min(0)).max()
d_op, _ = cKDTree(op).query(inp_int)
d_me, _ = cKDTree(dense).query(inp_int)
d_op_me, _ = cKDTree(dense).query(op)

def q(d): return "중앙 %.5f  p95 %.5f  p99 %.5f  최대 %.5f  복셀밖 %.2f%%" % (
    np.median(d), np.percentile(d, 95), np.percentile(d, 99), d.max(), 100 * (d > VOX).mean())

print("입력 점 %d · output_points %d · 메시 표본 %d · 복셀 %.5f" % (len(inp_int), len(op), len(dense), VOX))
print("입력 → output_points :", q(d_op))
print("입력 → 메시(200만)   :", q(d_me))
print("output_points → 메시 :", q(d_op_me))

outside_cube = (np.abs(inp_int) > 0.5).any(1)
print("단위 상자 밖 입력 점: %d (%.2f%%)" % (outside_cube.sum(), 100 * outside_cube.mean()))

bad = d_me > VOX
print("\n메시 기준 복셀 밖 입력 점 %d개 — 그 점들의 output_points 거리:" % bad.sum(), q(d_op[bad]) if bad.any() else "-")
print("  그중 단위 상자 밖: %d" % (bad & outside_cube).sum())
# 공간 분포: 어느 축 방향에 몰려 있나 (얇은 부위·바닥 등 추정용)
if bad.any():
    b = inp_int[bad]
    print("  bbox 전체  min %s max %s" % (np.round(inp_int.min(0), 3), np.round(inp_int.max(0), 3)))
    print("  밖 점 bbox min %s max %s" % (np.round(b.min(0), 3), np.round(b.max(0), 3)))
    print("  밖 점 평균 위치 %s (전체 평균 %s)" % (np.round(b.mean(0), 3), np.round(inp_int.mean(0), 3)))

far_op = d_op_me > VOX
print("\noutput_points 중 메시에서 복셀 밖: %d개 (%.2f%%)" % (far_op.sum(), 100 * far_op.mean()))
