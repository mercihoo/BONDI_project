#!/usr/bin/env python3
"""τ·섬 임계를 고르기 위한 측정 — τ 를 바꿔 가며 (1) 채움 면적 (2) 조각 크기 분포 (3) 원본과 겹치는 정도를 잰다.

    python tools/tune_tau.py --source <원본.glb> --result <복원본.glb> [--taus 0.25,0.5,0.75,1,1.5,2,3]

읽는 법:
  겹침(overlap) : 채움으로 분류된 면 중 원본 표면에서 **1 복셀 안**에 있는 것의 비율.
                  τ 가 작을수록 커지고, 이게 크면 복원분이 원본 위를 덮어 이중 표면이 된다.
  큰 조각/부스러기: 조각을 면적 순으로 정렬해 누적 90% 를 채우는 조각을 '큰 조각'으로 본다.
                  나머지가 부스러기이고, 섬 임계는 그 둘을 가르는 지점에 두면 된다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import CANDS, VOXEL, apply_frame, frame_is_improper, load_world_mesh  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--result", required=True)
ap.add_argument("--taus", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0")
a = ap.parse_args()

_, src = load_world_mesh(Path(a.source))
_, res = load_world_mesh(Path(a.result))
scale = float((src.bounds[1] - src.bounds[0]).max())
vox = VOXEL * scale

sp = np.asarray(trimesh.sample.sample_surface(src, 200_000)[0], dtype=np.float64)
rp = np.asarray(trimesh.sample.sample_surface(res, 200_000)[0], dtype=np.float64)
meds = {k: float(np.median(cKDTree(apply_frame(rp, c)).query(sp)[0])) for k, c in CANDS.items()}
frame = min(meds, key=meds.get)
cand = CANDS[frame]
faces = res.faces[:, [0, 2, 1]] if frame_is_improper(cand) else res.faces
res_t = trimesh.Trimesh(apply_frame(np.asarray(res.vertices, dtype=np.float64), cand), faces, process=False)

dense = np.asarray(trimesh.sample.sample_surface(src, 2_000_000)[0], dtype=np.float64)
tree = cKDTree(dense)
d, _ = tree.query(res_t.triangles_center)
print("프레임 %s · 복셀 %.5f · 복원본 %d 면" % (frame, vox, len(res_t.faces)))
print("복원본 면→원본 거리(복셀): 중앙 %.2f · p75 %.2f · p95 %.2f"
      % (np.median(d) / vox, np.percentile(d, 75) / vox, np.percentile(d, 95) / vox))
print()
print("  τ(복셀)  채움면적%  조각수  큰조각  누적90%%까지  최소큰조각%   겹침%")
for t in [float(x) for x in a.taus.split(",")]:
    keep = d > t * vox
    if keep.sum() == 0:
        print("  %5.2f     (없음)" % t); continue
    sub = res_t.submesh([np.where(keep)[0]], append=True)
    comps = sub.split(only_watertight=False)
    ar = np.array(sorted((c.area for c in comps), reverse=True))
    tot = ar.sum()
    cum = np.cumsum(ar) / tot
    n90 = int(np.searchsorted(cum, 0.90) + 1)                 # 면적 90% 를 채우는 조각 수
    smallest_big = 100 * ar[n90 - 1] / res_t.area             # 그중 가장 작은 조각의 전체 대비 %
    # 겹침: 채움 면 중 원본에서 1 복셀 안에 있는 비율 (τ<1 일 때만 0 이 아니다)
    ov = float((d[keep] <= vox).mean()) * 100
    print("  %5.2f    %7.1f  %6d  %6d  %11.1f%%  %10.3f%%  %6.1f%%"
          % (t, 100 * sub.area / src.area, len(comps), n90, 100 * cum[n90 - 1], smallest_big, ov))
