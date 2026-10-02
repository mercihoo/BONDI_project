#!/usr/bin/env python3
"""부분 점군을 단위 정육면체 안(반폭 0.49)으로 정규화해 저장하고, 되돌릴 center·scale 을 JSON 으로 남긴다.

    python tools/make_partial_norm.py samples/artifacts/<이름>/partial.ply samples/artifacts/<이름>/partial_norm.ply

왜: 텍스트 조건 스크립트(run_lascomp_text_condition_single.py)에는 --normalize-partial 이 없다. 점이 [-0.5, 0.5] 밖이면
voxelize_unit_cube 가 버린다. 이미지 스크립트의 정규화(중심·최대 변)와 같게 하되, 경계에 정확히 걸린 점이 잘리지 않게 0.98 을 곱한다.
축 교환(--yz-flip)은 스크립트 안에서 뒤에 일어나는데, 축별 중심·균일 스케일이라 순서와 무관하다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

FACTOR = 0.98

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
pc = trimesh.load(str(src), force=None, process=False)
p = np.asarray(getattr(pc, "vertices", pc), dtype=np.float64)
lo, hi = p.min(0), p.max(0)
center = (lo + hi) / 2
scale = float((hi - lo).max())
q = (p - center) / scale * FACTOR
trimesh.PointCloud(q).export(str(dst))
meta = {"center": center.tolist(), "scale": scale, "factor": FACTOR, "n": int(len(q)),
        "note": "norm = (p - center) / scale * factor ; 되돌리기 = q / factor * scale + center (결과는 y/z 교환 프레임이므로 center 도 교환해서 더한다)"}
dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("저장: %s (%d 점, 반폭 %.3f) · %s" % (dst, len(q), np.abs(q).max(), dst.with_suffix(".json")))
