#!/usr/bin/env python3
"""두 실행의 채움을 **높이로 갈라 합친다** — 위(몸통·테두리)는 A, 아래(굽다리)는 B.

    python tools/hybrid_fill.py --source <원본.glb> --above <A_composite.glb> --below <B_composite.glb> \
        --out <출력.glb> [--split auto|<비율>]

왜 — 채움이 많은 실행이 늘 더 좋지 않다. 굽다리바리에서 guide8 은 테두리를 더 잘 닫았지만
**굽다리 투창까지 메웠다** (투창 띠 채운 칸 100.0% · 남은 창 0 개. guide7 은 93.9% · 창 3 개).
투창은 64³ 격자에서 2~3 칸 폭이라 벽이 조금만 두꺼워지면 표면 추출이 이어 버린다.
LaS-Comp 의 ERS 는 `pred_voxel[voxel_mask] = 1.0` 한 줄, 즉 **관측 복셀을 1 로만 만든다** —
"여기는 비어 있어야 한다" 를 강제하는 장치가 없어서, 어떤 파라미터로도 투창을 지키게 할 수 없다.
그래서 결과를 높이로 갈라, 테두리는 잘 닫은 쪽에서, 굽다리는 투창을 지킨 쪽에서 가져온다.

경계는 `composite.junction_fraction` — 세로축 둘레 반지름이 가장 좁아지는 높이(= 몸통·굽다리 목).
기하는 만들지 않는다 — 두 composite 산출물의 채움 면을 높이로 고르고 이어 붙이기만 한다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import junction_fraction  # noqa: E402

FILL_NODE = "lascomp_gap_fill"


def fill_of(path: Path) -> trimesh.Trimesh:
    """composite.py 산출 GLB 에서 채움 노드를 월드 좌표로 꺼낸다 (색·질감 유지)."""
    sc = trimesh.load(str(path), process=False, force="scene")
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        if FILL_NODE not in gname and FILL_NODE not in node:
            continue
        g = sc.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        m = g.copy()
        m.apply_transform(T)
        return m
    sys.exit("채움 노드(%s)가 없다: %s" % (FILL_NODE, path))


def slice_by_height(m: trimesh.Trimesh, up: int, y: float, keep_above: bool) -> trimesh.Trimesh:
    """면 중심 높이로 고른다 — 정점을 옮기거나 면을 자르지 않는다."""
    cen = m.triangles_center[:, up]
    sel = np.where(cen > y if keep_above else cen <= y)[0]
    if not len(sel):
        sys.exit("그 높이에 남는 면이 없다")
    return m.submesh([sel], append=True)


ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--above", required=True, help="경계 위(몸통·테두리)를 가져올 composite GLB")
ap.add_argument("--below", required=True, help="경계 아래(굽다리)를 가져올 composite GLB")
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="auto", help="auto 또는 bbox 높이 비율 (0~1)")
a = ap.parse_args()

src = trimesh.load(a.source, force="mesh", process=False)
up = int(np.argmax(src.bounds[1] - src.bounds[0]))
above = fill_of(Path(a.above))
below = fill_of(Path(a.below))
print("위쪽 원본 채움 %d 면 · 아래쪽 원본 채움 %d 면" % (len(above.faces), len(below.faces)))

if a.split == "auto":
    frac = junction_fraction(above, src)
    print("몸통·굽다리 경계(auto) = bbox 높이의 %.3f" % frac)
else:
    frac = float(a.split)
y = src.bounds[0][up] + frac * (src.bounds[1][up] - src.bounds[0][up])
print("경계 높이 %.4f (세로축 %s)" % (y, "xyz"[up]))

top = slice_by_height(above, up, y, True)
bot = slice_by_height(below, up, y, False)
print("가져온 면: 위 %d (전체의 %.1f%%) · 아래 %d (전체의 %.1f%%)"
      % (len(top.faces), 100 * len(top.faces) / len(above.faces),
         len(bot.faces), 100 * len(bot.faces) / len(below.faces)))

merged = trimesh.util.concatenate([top, bot])
print("합친 채움 %d 면 · 면적 = 원본 면적의 %.1f%%" % (len(merged.faces), 100 * merged.area / src.area))

out_scene = trimesh.load(a.source, process=False, force="scene")
out_scene.add_geometry(merged, node_name=FILL_NODE, geom_name=FILL_NODE)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out_scene.export(str(out))
rep = {"source": a.source, "above": a.above, "below": a.below, "out": str(out),
       "split": a.split, "split_frac": round(float(frac), 4), "split_y": round(float(y), 5),
       "up_axis": "xyz"[up],
       "above_faces_total": int(len(above.faces)), "below_faces_total": int(len(below.faces)),
       "taken_above": int(len(top.faces)), "taken_below": int(len(bot.faces)),
       "merged_faces": int(len(merged.faces)),
       "merged_area_frac_of_source": round(float(merged.area / src.area), 4),
       "geometry_changed": False}
out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", out)
