#!/usr/bin/env python3
"""모델이 예측한 정점 색을 결과 GLB 에 붙인다 — 저자 스크립트가 버리는 색을 살린다.

    python tools/apply_colors.py results/<라벨>

읽는 것: `output_mesh.glb`(저자 저장, 색 없음) + `output_mesh_colors.npz`(우리 색 훅이 남긴 vertex_attrs).
쓰는 것: `output_mesh_color.glb`(정점 색), `output_mesh_color.obj`+`.mtl`(OBJ 로도 — 정점 색 포함).

vertex_attrs 레이아웃은 cube2mesh.py 의 LAYOUTS['color'] = (8, 6) 에 따라 **색 3 + 법선 3** 이다.
앞 3 채널을 색으로 쓰고, 값 범위가 [-1,1] 로 보이면 (x+1)/2 로 옮긴다(디코더 출력이 tanh 계열일 때).
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

R = Path(sys.argv[1])
gz = R / "output_mesh_colors.npz"
gl = R / "output_mesh.glb"
if not gz.exists():
    sys.exit("색 파일이 없다: %s  (LASCOMP_COLOR_OUT 로 실행해야 생긴다)" % gz)

d = np.load(gz)
attrs = d["vertex_attrs"]
mesh = trimesh.load(str(gl), force="mesh", process=False)
print("메시 정점 %d · vertex_attrs %s · 값 %.3f~%.3f" % (len(mesh.vertices), attrs.shape, attrs.min(), attrs.max()))
if len(attrs) != len(mesh.vertices):
    sys.exit("정점 수가 다르다 (%d vs %d) — 같은 실행의 파일인지 확인" % (len(attrs), len(mesh.vertices)))

col = attrs[:, :3].astype(np.float64)
if col.min() < -0.05:                      # [-1,1] 로 보이면 [0,1] 로
    col = (col + 1.0) / 2.0
    print("  범위 [-1,1] → [0,1] 로 옮김")
lo, hi = np.percentile(col, [1, 99])
print("  색 1~99 백분위: %.3f ~ %.3f" % (lo, hi))
col = np.clip(col, 0, 1)
rgba = np.concatenate([(col * 255).round().astype(np.uint8),
                       np.full((len(col), 1), 255, np.uint8)], axis=1)
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)
out = R / "output_mesh_color.glb"
mesh.export(str(out))
print("저장:", out, "(정점 색)")
obj = R / "output_mesh_color.obj"
mesh.export(str(obj))                      # trimesh 는 정점 색을 OBJ 확장 문법(v x y z r g b)으로 쓴다
print("저장:", obj, "(+ .mtl)")
print("평균 색 RGB:", (col.mean(0) * 255).round().astype(int))
