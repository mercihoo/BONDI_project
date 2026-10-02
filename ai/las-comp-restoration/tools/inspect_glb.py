#!/usr/bin/env python3
"""GLB 를 읽어 LaS-Comp 입력으로 쓸 때 알아야 할 것을 뽑는다 — 정점·면 수, bbox, 위 축 추정, 조각 수, 수밀 여부.
README 의 A_normal.glb 규격(정점 94,877 · 면 149,150 · bbox [0.904, 1.002, 0.843]) 과 대조한다.

    python tools/inspect_glb.py artifacts/gupdari71489/source_model.glb
"""
import sys
from pathlib import Path
import numpy as np, trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_completion import load_geometry

p = Path(sys.argv[1])
m = load_geometry(p)
if m is None:
    sys.exit("메시가 아니다: %s" % p)

ext = m.bounds[1] - m.bounds[0]
up = int(np.argmax(ext))               # 굽다리바리는 높이가 가장 크다 — 가장 긴 축을 위 축으로 본다
pieces = m.split(only_watertight=False)
print("파일        :", p.name, "(%.1f MB)" % (p.stat().st_size / 1e6))
print("정점 / 면   : %d / %d" % (len(m.vertices), len(m.faces)))
print("bbox 크기   :", np.round(ext, 3), " 중심", np.round(m.bounds.mean(0), 3))
print("가장 긴 축  :", "XYZ"[up], "→ 위 축 추정 (%s)" % ("Y-up → --yz-flip" if up == 1 else "Z-up → --no-yz-flip" if up == 2 else "X-up?? 확인 필요"))
print("조각 수     :", len(pieces), "| 수밀:", m.is_watertight, "| 면 방향 일관:", m.is_winding_consistent)
print("단위 상자 밖: 반폭 %.3f — %s" % (np.abs(m.vertices).max(), "--normalize-partial 필요" if np.abs(m.vertices).max() > 0.5 else "그대로 들어감"))
print("README 규격 : 정점 94,877 · 면 149,150 · bbox [0.904 1.002 0.843] →",
      "일치" if (abs(len(m.vertices) - 94877) < 50 and abs(len(m.faces) - 149150) < 50) else "다른 파일 (다른 소스일 수 있음 — 스캔/생성 여부 확인)")
print("색/텍스처   :", type(m.visual).__name__, "| 정점색" if hasattr(m.visual, "vertex_colors") and len(getattr(m.visual, "vertex_colors", [])) else "")
