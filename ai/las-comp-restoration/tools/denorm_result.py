#!/usr/bin/env python3
"""텍스트 조건 실행 결과(정규화 프레임)를 원좌표로 되돌린다 — 이미지 경로의 결과와 같은 프레임(원좌표·y/z 교환)이 되게.

    python tools/denorm_result.py results/<라벨> samples/artifacts/<이름>/partial_norm.json

output_mesh.glb · sparse_structure.ply · output_points.ply 를 제자리에서 바꾸고, 원본은 *_norm.* 으로 남긴다.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

R, meta = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
c = np.asarray(meta["center"], dtype=np.float64)
c_swapped = c[[0, 2, 1]]                      # 결과는 y/z 가 교환된 프레임 — 중심도 교환해서 더한다
k = meta["scale"] / meta["factor"]

def back(p):
    return p * k + c_swapped

m_path = R / "output_mesh.glb"
if m_path.exists() and not (R / "output_mesh_norm.glb").exists():
    shutil.copy(m_path, R / "output_mesh_norm.glb")
    m = trimesh.load(str(m_path), force="mesh", process=False)
    m.vertices = back(np.asarray(m.vertices, dtype=np.float64))
    m.export(str(m_path))
    print("output_mesh.glb 원좌표로 (bbox %s)" % np.round(m.extents, 3))
for name in ("sparse_structure.ply", "output_points.ply"):
    p = R / name
    if p.exists() and not (R / (p.stem + "_norm.ply")).exists():
        shutil.copy(p, R / (p.stem + "_norm.ply"))
        pc = trimesh.load(str(p), force=None, process=False)
        pts = back(np.asarray(getattr(pc, "vertices", pc), dtype=np.float64))
        trimesh.PointCloud(pts).export(str(p))
        print("%s 원좌표로" % name)
