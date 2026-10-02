#!/usr/bin/env python3
"""복원본을 원본 표면과의 거리로 **두 조각으로 갈라** 각각 따로 저장한다. 합치지 않는다(합성 아님).

    python tools/split_by_original.py --source <원본.glb> --result <복원본.glb> --out-dir <폴더> [--tau-voxels 1.5]

산출 (둘 다 복원본의 면을 그대로 쓴다 — 정점을 옮기거나 원본을 끼워 넣지 않는다):
  part_observed.glb : 원본 표면 **가까이**(τ 안) 있는 면 = 모델이 관측부를 다시 그린 부분
  part_filled.glb   : 원본 표면에서 **먼**(τ 밖) 면 = 모델이 새로 만든 부분(결손 채움)
  split.json        : 면 수·면적 비율·프레임

"복원된 부분을 지우면 어떻게 되나" 를 눈으로 보려는 도구다. 색이 있는 복원본(output_mesh_color.glb)을 넣으면 색도 따라간다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import CANDS, VOXEL, apply_frame, frame_is_improper, load_world_mesh  # noqa: E402

DENSE_N = 2_000_000

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--result", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--tau-voxels", type=float, default=1.5)
ap.add_argument("--min-area", type=float, default=0.0, help="이 비율보다 작은 조각(섬)은 버린다. 0 = 그대로 둔다")
a = ap.parse_args()

_, src = load_world_mesh(Path(a.source))
_, res = load_world_mesh(Path(a.result))
scale = float((src.bounds[1] - src.bounds[0]).max())
tau = a.tau_voxels * VOXEL * scale

# 프레임 맞추기 (복원본은 LaS-Comp 내부 프레임)
sp = np.asarray(trimesh.sample.sample_surface(src, 200_000)[0], dtype=np.float64)
rp = np.asarray(trimesh.sample.sample_surface(res, 200_000)[0], dtype=np.float64)
meds = {k: float(np.median(cKDTree(apply_frame(rp, c)).query(sp)[0])) for k, c in CANDS.items()}
frame = min(meds, key=meds.get)
cand = CANDS[frame]
faces = res.faces[:, [0, 2, 1]] if frame_is_improper(cand) else res.faces
res_t = trimesh.Trimesh(apply_frame(np.asarray(res.vertices, dtype=np.float64), cand), faces, process=False)
res_t.visual = res.visual                      # 정점색이 있으면 그대로 물려준다
print("프레임 %s (%.4f) · 원본 최대 변 %.4f · τ %.4f (%.1f 복셀)" % (frame, meds[frame], scale, tau, a.tau_voxels))

dense = np.asarray(trimesh.sample.sample_surface(src, DENSE_N)[0], dtype=np.float64)
d, _ = cKDTree(dense).query(res_t.triangles_center)
far = d > tau

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
rep = {"source": a.source, "result": a.result, "frame": frame, "tau_voxels": a.tau_voxels,
       "tau_abs": round(tau, 5), "result_faces": int(len(res_t.faces)),
       "frame_medians": {k: round(v, 5) for k, v in meds.items()}}

for tag, mask, desc in (("observed", ~far, "모델이 관측부를 다시 그린 부분 (원본 표면 τ 안)"),
                        ("filled", far, "모델이 새로 만든 부분 (원본 표면 τ 밖)")):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print("  %s: 면이 없다" % tag)
        continue
    m = res_t.submesh([idx], append=True)
    if a.min_area > 0:
        comps = m.split(only_watertight=False)
        big = [c for c in comps if c.area >= a.min_area * res_t.area]
        if big:
            m = trimesh.util.concatenate(big)
            print("  %s: 섬 제거 %d → %d 조각" % (tag, len(comps), len(big)))
    p = out / ("part_%s.glb" % tag)
    m.export(str(p))
    rep["%s_faces" % tag] = int(len(m.faces))
    rep["%s_area_frac" % tag] = round(float(m.area / res_t.area), 4)
    print("  part_%s.glb : %d 면 (%.1f%%) — %s" % (tag, len(m.faces), 100 * m.area / res_t.area, desc))

(out / "split.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", out / "split.json")
