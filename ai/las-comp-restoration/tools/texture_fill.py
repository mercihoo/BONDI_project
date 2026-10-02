#!/usr/bin/env python3
"""채움부에 **원본의 실측 사진 텍스처**를 이어 붙인다 (모델 예측 단색 대신).

    python tools/texture_fill.py --source <원본.glb> --composite <restored_with_original.glb> --out <출력.glb>

왜 필요한가 — 모델(TRELLIS/LaS-Comp)이 내는 정점 색은 조건 이미지에서 나온다. 가이드가 회색 무광 렌더면
예측 색도 무채색 한 톤이 된다(실측: 1~99 백분위 0.216~0.364). 반면 원본 스캔에는 2048² 실측 텍스처가 있다.

어떻게 — 물레로 돌린 그릇은 **같은 높이대의 표면 통계가 회전 방향으로 같다**. 그래서 채움 정점 하나를
원통 좌표 (높이 h, 각 θ) 로 옮기고,
  1) 같은 높이대에서 가장 가까운 **실측** 표면의 각 θ_b 를 찾는다 (= 파단면 경계)
  2) 그 경계를 거울로 삼아 θ_src = 2θ_b - θ 로 되짚어, 인접한 실측 표면을 결손부 안으로 반사해 가져온다
경계에서 연속이라 이음선이 생기지 않고, 같은 높이대라 물레 흔적(가로 띠)이 이어진다. 반사한 자리마저
결손이면 1) 의 최근접 색으로 떨어진다.

회전축은 굽다리 바닥 고리(높이 하위 --base-frac)에 원을 최소제곱으로 맞춰 잡는다 — 굽다리는 대개 온전해서
깨진 몸통의 무게중심보다 축이 정확하다.

기하는 건드리지 않는다 — 정점·면 그대로, 정점 색만 채운다. 원본 노드도 원본 텍스처 그대로 남는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FILL_NODE = "lascomp_gap_fill"


def world_geoms(path: Path):
    """장면의 각 geometry 를 (이름, 월드좌표 메시) 로 돌려준다 — 노드 변환을 정점에 적용해서."""
    sc = trimesh.load(str(path), process=False, force="scene")
    out = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        m = g.copy()
        m.apply_transform(T)
        out.append((gname, node, m))
    return sc, out


def vertex_colors_of(m: trimesh.Trimesh) -> np.ndarray:
    """텍스처든 정점색이든 정점 RGBA(uint8) 로."""
    v = m.visual
    if getattr(v, "kind", None) == "texture":
        return np.asarray(v.to_color().vertex_colors, dtype=np.uint8)
    if getattr(v, "kind", None) == "vertex":
        return np.asarray(v.vertex_colors, dtype=np.uint8)
    sys.exit("원본에서 색을 못 읽었다 (visual.kind=%r)" % getattr(v, "kind", None))


def fit_axis_center(xy: np.ndarray):
    """(x, z) 점들에 원을 최소제곱으로 맞춰 중심을 돌려준다.
    (x-a)^2 + (y-b)^2 = R^2  →  2ax + 2by + c = x^2+y^2,  c = R^2-a^2-b^2  (a,b,c 에 대해 선형)"""
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_, b_, c_ = sol
    R = float(np.sqrt(max(c_ + a_ ** 2 + b_ ** 2, 0.0)))
    return np.array([a_, b_]), R


def cyl(pts: np.ndarray, up: int, center: np.ndarray):
    """월드 좌표 → (높이 h, 각 θ, 반지름 r)."""
    lat = [i for i in range(3) if i != up]
    p = pts[:, lat] - center
    return pts[:, up], np.arctan2(p[:, 1], p[:, 0]), np.hypot(p[:, 0], p[:, 1])


ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="원본 GLB (실측 텍스처가 있는 것)")
ap.add_argument("--composite", required=True, help="composite.py 가 만든 restored_with_original.glb")
ap.add_argument("--out", required=True)
ap.add_argument("--up", default="auto", help="세로축 0/1/2 또는 auto (bbox 최대 변)")
ap.add_argument("--base-frac", type=float, default=0.08, help="회전축을 맞출 굽다리 바닥 높이 비율")
ap.add_argument("--tint", default="", help="예: 1.06,1.02,0.98 — 채움부에만 곱하는 색조 (복원부 구분 표시용)")
ap.add_argument("--mirror", type=int, default=1, help="0 이면 거울 반사 없이 최근접 실측 색만")
ap.add_argument("--mode", default="blend", choices=("mirror", "band", "blend"),
                help="mirror=실측 색 그대로 복사 · band=높이대 평균 톤(얼룩 없음) · blend=톤은 높이대 + 결만 실측")
ap.add_argument("--detail-k", type=int, default=12, help="blend 에서 '결'을 뽑을 때 평활에 쓰는 이웃 수. "
                "키우면 평활 반경이 넓어져 잔차가 저주파가 되고 무늬가 굵어진다 (질감이 도드라지는 대신 얼룩처럼 보일 수 있다).")
ap.add_argument("--detail-gain", type=float, default=0.8, help="blend 에서 결의 세기. 표준편차를 가장 크게 좌우한다 "
                "(k 24 에서 세기 0.8→2.0 이 18.8→37.8; 원본 실측은 40.5).")
ap.add_argument("--bands", type=int, default=200, help="높이대 개수")
a = ap.parse_args()

src = trimesh.load(a.source, force="mesh", process=False)
src_col = vertex_colors_of(src)
sc, geoms = world_geoms(Path(a.composite))
names = [g[0] for g in geoms]
fill_hit = [g for g in geoms if FILL_NODE in g[0] or FILL_NODE in g[1]]
if not fill_hit:
    sys.exit("합성본에서 채움 노드(%s)를 못 찾았다 — 있는 것: %s" % (FILL_NODE, names))
fill_gname, fill_node, fill = fill_hit[0]
print("합성본 geometry %s · 채움 = %s (정점 %d · 면 %d)" % (names, fill_gname, len(fill.vertices), len(fill.faces)))

# 세로축
ext = src.bounds[1] - src.bounds[0]
up = int(np.argmax(ext)) if a.up == "auto" else int(a.up)
print("세로축 = %s (bbox 변 %s)" % ("xyz"[up], ext.round(3).tolist()))

# 회전축 — 굽다리 바닥 고리에 원 맞추기
sv = np.asarray(src.vertices, dtype=np.float64)
lat = [i for i in range(3) if i != up]
h0, h1 = src.bounds[0][up], src.bounds[1][up]
base = sv[sv[:, up] <= h0 + a.base_frac * (h1 - h0)]
center, R = fit_axis_center(base[:, lat])
print("회전축 중심 (%s) = %s · 굽다리 반지름 %.4f · 바닥 정점 %d" %
      ("".join("xyz"[i] for i in lat), center.round(4).tolist(), R, len(base)))

sh, st, sr = cyl(sv, up, center)
fh, ft, fr = cyl(np.asarray(fill.vertices, dtype=np.float64), up, center)

# 원통 펼침 좌표계 — 각을 평균 반지름으로 곱해 높이와 단위를 맞춘다. θ 는 순환이니 ±2π 사본을 붙인다.
s_ang = float(np.median(sr[sr > 1e-6])) if np.any(sr > 1e-6) else 1.0
obs = np.c_[sh, s_ang * st]
obs_all = np.vstack([obs, np.c_[sh, s_ang * (st + 2 * np.pi)], np.c_[sh, s_ang * (st - 2 * np.pi)]])
idx_map = np.tile(np.arange(len(sv)), 3)
tree = cKDTree(obs_all)

# 1) 같은 높이대에서 가장 가까운 실측 표면 = 파단면 경계
d1, i1 = tree.query(np.c_[fh, s_ang * ft])
near = idx_map[i1]
if a.mirror:
    # 2) 경계를 거울로 삼아 인접 실측 표면을 결손부 안으로 반사
    th_b = st[near]
    th_src = 2 * th_b - ft
    th_src = (th_src + np.pi) % (2 * np.pi) - np.pi
    d2, i2 = tree.query(np.c_[fh, s_ang * th_src])
    pick = np.where(d2 <= d1, idx_map[i2], near)
    print("거울 반사 채택 %d / %d = %.1f%% · 경계거리 중앙값 %.4f → 반사 후 %.4f"
          % (int((d2 <= d1).sum()), len(pick), 100 * (d2 <= d1).mean(),
             float(np.median(d1)), float(np.median(np.minimum(d1, d2)))))
else:
    pick = near
    print("최근접 실측 색만 사용 · 경계거리 중앙값 %.4f" % float(np.median(d1)))

def band_tone(h_obs, c_obs, h_query, nbands, smooth=5):
    """높이대별 실측 색의 중앙값을 구해 부드럽게 만든 뒤 질의 높이에서 보간한다.
    얼룩(넓은 저주파 무늬)은 중앙값에 녹아 사라지고, 높이에 따른 자연스러운 톤 변화만 남는다."""
    lo, hi = float(h_obs.min()), float(h_obs.max())
    edges = np.linspace(lo, hi, nbands + 1)
    which = np.clip(np.digitize(h_obs, edges) - 1, 0, nbands - 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    tone = np.zeros((nbands, 3), dtype=np.float64)
    have = np.zeros(nbands, dtype=bool)
    for b in range(nbands):
        sel = c_obs[which == b]
        if len(sel):
            tone[b] = np.median(sel, axis=0)
            have[b] = True
    if not have.any():
        sys.exit("높이대에 실측 색이 하나도 없다")
    # 빈 높이대는 이웃에서 채운다
    for ch in range(3):
        tone[:, ch] = np.interp(mid, mid[have], tone[have, ch])
    if smooth > 1:                       # 높이 방향 이동평균 — 칸 경계의 가로 줄무늬를 없앤다
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        for ch in range(3):
            tone[:, ch] = np.convolve(np.pad(tone[:, ch], pad, mode="edge"), k, mode="valid")[:nbands]
    return np.stack([np.interp(h_query, mid, tone[:, ch]) for ch in range(3)], axis=1)


rgb_obs = src_col[:, :3].astype(np.float64)
if a.mode == "mirror":
    col = src_col[pick].copy()
elif a.mode == "band":
    tone = band_tone(sh, rgb_obs, fh, a.bands)
    col = np.c_[np.clip(tone, 0, 255).astype(np.uint8), np.full((len(fh), 1), 255, np.uint8)]
    print("높이대 톤만 사용 (얼룩 없음) · %d 칸" % a.bands)
else:                                    # blend — 톤은 높이대에서, 결(고주파)만 실측에서
    d_, nb = cKDTree(sv).query(sv, k=a.detail_k)
    smoothed = rgb_obs[nb].mean(axis=1)          # 원본 표면을 국소 평균 → 얼룩·톤은 여기 남는다
    detail = rgb_obs - smoothed                  # 남은 것 = 태토 결·물레 흔적 같은 고주파
    tone = band_tone(sh, rgb_obs, fh, a.bands)
    mix = tone + a.detail_gain * detail[pick]
    col = np.c_[np.clip(mix, 0, 255).astype(np.uint8), np.full((len(fh), 1), 255, np.uint8)]
    print("blend: 톤 = 높이대 중앙값(%d 칸) · 결 = 실측 고주파(이웃 %d, 세기 %.2f) · 결 표준편차 %s"
          % (a.bands, a.detail_k, a.detail_gain, detail.std(axis=0).round(1).tolist()))

if a.tint:
    t = np.array([float(x) for x in a.tint.split(",")], dtype=np.float64)
    col[:, :3] = np.clip(col[:, :3].astype(np.float64) * t, 0, 255).astype(np.uint8)
    print("색조 곱 적용:", t.tolist())
print("채움 색: 원본 실측 텍스처 · 평균 RGB %s · 표준편차 %s"
      % (col[:, :3].mean(0).round(1).tolist(), col[:, :3].std(0).round(1).tolist()))

fill.visual = trimesh.visual.ColorVisuals(mesh=fill, vertex_colors=col)
out_scene = trimesh.load(a.source, process=False, force="scene")   # 원본 노드는 원본 텍스처 그대로
out_scene.add_geometry(fill, node_name=FILL_NODE, geom_name=FILL_NODE)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out_scene.export(str(out))
rep = {"source": a.source, "composite": a.composite, "out": str(out), "up_axis": "xyz"[up],
       "axis_center": center.round(6).tolist(), "base_radius": round(R, 5), "base_frac": a.base_frac,
       "mirror": bool(a.mirror), "mode": a.mode, "detail_k": a.detail_k,
       "detail_gain": a.detail_gain, "bands": a.bands, "tint": a.tint or None,
       "fill_verts": int(len(fill.vertices)), "fill_faces": int(len(fill.faces)),
       "fill_color_mean": col[:, :3].mean(0).round(1).tolist(),
       "fill_color_std": col[:, :3].std(0).round(1).tolist(),
       "source_color_mean": src_col[:, :3].mean(0).round(1).tolist(),
       "source_color_std": src_col[:, :3].std(0).round(1).tolist(),
       "geometry_changed": False}
out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", out, "(기하 변경 없음)")
