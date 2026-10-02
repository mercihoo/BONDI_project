#!/usr/bin/env python3
"""결과 GLB 의 프레임(내부 yz 교환·거울 여부)을 원본 표면 거리로 고른다. 이름만 찍는다.

    python tools/pick_frame.py <원본.glb> <결과.glb>     → 예: swap_yz

composite.py 의 후보·판정을 그대로 쓴다 (합성을 하지 않아도 프레임은 알아야 렌더가 맞는다).
"""
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composite import CANDS, apply_frame, load_world_mesh  # noqa: E402

_, src = load_world_mesh(sys.argv[1])
_, res = load_world_mesh(sys.argv[2])
sp = np.asarray(trimesh.sample.sample_surface(src, 100_000)[0], dtype=np.float64)
rp = np.asarray(trimesh.sample.sample_surface(res, 100_000)[0], dtype=np.float64)
best, meds = None, {}
for name, cand in CANDS.items():
    meds[name] = float(np.median(cKDTree(apply_frame(rp, cand)).query(sp)[0]))
best = min(meds, key=meds.get)
print(best)
if os.environ.get("PICK_FRAME_VERBOSE"):
    print(" · ".join("%s %.4f" % kv for kv in sorted(meds.items(), key=lambda kv: kv[1])), file=sys.stderr)
