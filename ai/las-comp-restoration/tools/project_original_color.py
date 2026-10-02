#!/usr/bin/env python3
"""복원본 **기하는 그대로 두고**, 원본이 관측된 자리의 색만 원본에서 가져온다 (텍스처 투영).

    python tools/project_original_color.py --source <원본.glb> --result <복원본.glb> --out <출력.glb> [--tau-voxels 1.5]

메시를 자르거나 정점을 옮기지 않는다 — 복원본의 정점·면은 100% 그대로고 **정점 색만** 바뀐다:
  · 원본 표면에서 τ 안에 있는 정점 → 원본의 사진 텍스처 색 (= 실측 외관이 보인다)
  · τ 밖(모델이 새로 만든 곳)    → 모델이 예측한 색, 없으면 지정한 채움색
그래서 "어디가 실측이고 어디가 복원인지" 가 색으로 구분된다 — 유물 복원에서 요구되는 구분 표시이기도 하다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import CANDS, VOXEL, apply_frame, frame_is_improper  # noqa: E402


def load_mesh_keep_color(path: Path) -> trimesh.Trimesh:
    """노드 변환을 적용하되 **색(텍스처/정점색)을 유지**해 읽는다.

    composite.py 의 load_world_mesh 는 Trimesh(vertices, faces) 로 다시 만들어 visual 을 버린다 —
    그걸 썼다가 원본 텍스처가 균일 회색(102)으로, 모델 예측 색이 통째로 사라졌다.
    trimesh 의 force="mesh" 는 장면을 변환까지 적용해 합치면서 visual 을 살린다.
    """
    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
        sys.exit("메시 없음: %s" % path)
    return m

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--result", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tau-voxels", type=float, default=1.5)
ap.add_argument("--fill-color", default="200,188,168", help="모델 색이 없을 때 결손부에 쓸 색")
ap.add_argument("--keep-model-color", action="store_true", help="결손부에 모델 예측 색을 쓴다 (기본: 있으면 쓴다)")
a = ap.parse_args()

src = load_mesh_keep_color(Path(a.source))
res = load_mesh_keep_color(Path(a.result))
scale = float((src.bounds[1] - src.bounds[0]).max())
tau = a.tau_voxels * VOXEL * scale

# 프레임 맞추기
sp = np.asarray(trimesh.sample.sample_surface(src, 200_000)[0], dtype=np.float64)
rp = np.asarray(trimesh.sample.sample_surface(res, 200_000)[0], dtype=np.float64)
meds = {k: float(np.median(cKDTree(apply_frame(rp, c)).query(sp)[0])) for k, c in CANDS.items()}
frame = min(meds, key=meds.get)
cand = CANDS[frame]
faces = res.faces[:, [0, 2, 1]] if frame_is_improper(cand) else res.faces
V = apply_frame(np.asarray(res.vertices, dtype=np.float64), cand)
out_mesh = trimesh.Trimesh(V, faces, process=False)
print("프레임 %s (%.4f) · τ %.4f (%.1f 복셀) · 복원본 정점 %d" % (frame, meds[frame], tau, a.tau_voxels, len(V)))

# 원본의 텍스처를 정점 색으로 바꾼다 (UV·이미지 → 정점 RGBA)
src_col = src.visual.to_color().vertex_colors if getattr(src.visual, "kind", None) == "texture" else src.visual.vertex_colors
src_col = np.asarray(src_col, dtype=np.uint8)
print("원본 정점 색: %d 개 · 평균 RGB %s" % (len(src_col), src_col[:, :3].mean(0).round(1)))

# 복원본 정점마다 가장 가까운 원본 정점
d, idx = cKDTree(np.asarray(src.vertices, dtype=np.float64)).query(V)
observed = d <= tau
print("관측부로 판정된 정점: %d / %d = %.1f%%" % (observed.sum(), len(V), 100 * observed.mean()))

# 결손부 색: 복원본에 정점색(모델 예측)이 있으면 그것, 없으면 지정색
if getattr(res.visual, "kind", None) == "vertex":
    fill = np.asarray(res.visual.vertex_colors, dtype=np.uint8).copy()
    print("결손부 색: 모델 예측 색 사용")
else:
    rgb = [int(x) for x in a.fill_color.split(",")]
    fill = np.tile(np.array(rgb + [255], np.uint8), (len(V), 1))
    print("결손부 색: 지정색", rgb)

colors = fill.copy()
colors[observed] = src_col[idx[observed]]
out_mesh.visual = trimesh.visual.ColorVisuals(mesh=out_mesh, vertex_colors=colors)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out_mesh.export(str(out))
out_mesh.export(str(out.with_suffix(".obj")))
rep = {"source": a.source, "result": a.result, "out": str(out), "frame": frame,
       "tau_voxels": a.tau_voxels, "tau_abs": round(tau, 5),
       "n_verts": int(len(V)), "observed_verts": int(observed.sum()),
       "observed_frac": round(float(observed.mean()), 4),
       "geometry_changed": False}
out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", out, "·", out.with_suffix(".obj"), "(기하 변경 없음)")
