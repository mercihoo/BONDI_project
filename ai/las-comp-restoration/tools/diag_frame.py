#!/usr/bin/env python3
"""output_mesh.glb 와 입력·output_points 의 좌표 프레임이 맞는지 가른다.

증상: 같은 메시에서 뽑았을 output_points.ply 는 입력 점의 100% 가 복셀 안인데, .glb 메시는 9~14% 가 밖이다.
가설: .glb 내보내기(glTF 는 Y-up) 나 yz 교환·축 부호가 입력 프레임과 어긋나 있다. 좌우 대칭에 가까운 bimba 는
거울상이어도 Chamfer·중앙값이 멀쩡해 보이므로, 축 부호 8 가지 × yz 교환 2 가지 = 16 프레임을 전부 시험한다.

    python tools/diag_frame.py results/<라벨> samples/CompC_datasets/plyobj/indata/bimba.ply
"""
import sys, itertools
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_completion import load_geometry, load_points, DENSE_N

res_dir, partial = Path(sys.argv[1]), Path(sys.argv[2])
mesh = load_geometry(res_dir / "output_mesh.glb")
dense, _ = trimesh.sample.sample_surface(mesh, DENSE_N)
tree = cKDTree(np.asarray(dense))
VOX = 1.0 / 64

def frames(p):
    for swap in (False, True):
        q = p.copy()
        if swap: q[:, [1, 2]] = q[:, [2, 1]]
        for s in itertools.product((1, -1), repeat=3):
            yield ("yz" if swap else "id") + "".join("+" if v > 0 else "-" for v in s), q * np.array(s, dtype=float)

def report(name, pts):
    scale = (pts.max(0) - pts.min(0)).max()
    rows = []
    for tag, q in frames(pts):
        d, _ = tree.query(q)
        rows.append((float((d > VOX * scale).mean()), float(np.median(d)), float(np.percentile(d, 95)), tag))
    rows.sort()
    print("── %s → 메시 (프레임 상위 4 / 16) ──" % name)
    for out, med, p95, tag in rows[:4]:
        print("   %-6s 복셀 밖 %5.1f%%  중앙값 %.5f  p95 %.5f" % (tag, 100 * out, med, p95))
    return rows[0]

part, _ = load_points(partial, 50000)
outp, _ = load_points(res_dir / "output_points.ply", 50000)
best_in = report("입력 부분 점군", part)
best_out = report("output_points.ply", outp)
print()
print("판정: 입력 최적 프레임 %s (복셀 밖 %.1f%%) · output_points 최적 프레임 %s (복셀 밖 %.1f%%)"
      % (best_in[3], 100 * best_in[0], best_out[3], 100 * best_out[0]))
if best_out[0] < 0.01 and best_out[3] not in ("id+++", "yz+++"):
    print("→ .glb 가 output_points 와 다른 프레임에 있다: %s 변환을 적용해야 같은 좌표계다" % best_out[3])
elif best_out[0] >= 0.01:
    print("→ 어느 프레임으로도 output_points 가 메시 위에 없다 — output_points 가 이 메시에서 뽑힌 것이 아니거나, 스케일·평행이동이 다르다")
