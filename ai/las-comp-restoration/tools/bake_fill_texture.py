#!/usr/bin/env python3
"""채움부의 정점 색을 **UV 텍스처로 구워** 넣는다 (정점 색 의존을 없앤다).

    python tools/bake_fill_texture.py --composite <restored_tex.glb> --source <원본.glb> --out <출력.glb> [--size 2048]

왜 필요한가 — 자료 서버는 올린 GLB 를 Blender 로 경량 미리보기로 다시 만든다. 그때 원본(텍스처)과 채움(정점 색)을
한 오브젝트로 합치면서 **텍스처 메시용 흰색 더미 색 속성이 COLOR_0 을 차지하고 실제 채움 색이 COLOR_1 로 밀린다.**
three.js 는 COLOR_0 만 읽으므로 화면에서는 단색으로 보인다 (2026-09-17 실측: COLOR_0 전부 255, COLOR_1 이 [151,144,133]).
채움부를 UV+텍스처로 만들면 색 속성이 아예 없어 이 문제가 사라지고, 텍셀 밀도도 원본과 같아진다.

어떻게 — 물레로 돌린 그릇이라 원통 펼치기가 자연스럽다.
  u = (θ + π) / 2π,  v = (h - h0) / (h1 - h0)
텍셀마다 (u, v) 에서 가장 가까운 채움 정점의 색을 넣는다 (u 는 순환이라 ±1 사본을 붙여 찾는다).
UV 가 (θ, h) 의 함수이고 텍스처도 같은 좌표로 만들었으니 대응이 정의상 정확하다.

이음선 — u 범위가 0.5 를 넘는 면(±π 를 가로지르는 면)은 작은 u 쪽 꼭짓점을 복제해 u+1 로 둔다.
텍스처가 u 방향으로 주기적이고 glTF 기본 래핑이 REPEAT 이라 같은 열을 다시 읽어 이음선이 없다.

견본(--swatch) — 결을 원본에서 가져오는 대신 견본 이미지를 텍셀 단위로 타일링해 찍을 수 있다.
  detail: 톤은 정점 색(texture_fill --mode band 로 높이대 중앙값만) + 결은 견본의 고주파(견본 − 평균)
  full  : 견본 색 그대로 (보존처리의 석고 채움처럼 복원부를 재질로 구분해 보일 때)
  기본(--swatch-synth 1)은 견본을 붙이지 않고 견본의 통계로 결을 합성한다 — 굵기는 방사 평균 파워 스펙트럼,
  명암은 밝기 히스토그램(순위 맞춤). 반복·방향성이 없고 둘레 한 바퀴를 FFT 주기로 잡아 u 순환에서 이음이 없다.
  사진 견본을 그대로 반사 타일링하면(--swatch-synth 0) 견본의 줄 구조가 128px 주기로 반복된다.
  배율은 견본 1px ≈ 텍셀 1px — 작은 견본도 텍셀 밀도 그대로 살린다.

한계 — 굽다리 바닥처럼 수평인 면은 같은 높이가 한 v 줄에 몰려 늘어난다 (면적이 작고 대개 가려진다).
기하는 건드리지 않는다 — 원본 노드도, 채움의 정점·면도 그대로다 (이음선 면의 꼭짓점 복제만 있다).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

FILL_NODE = "lascomp_gap_fill"


def world_geoms(path: Path):
    sc = trimesh.load(str(path), process=False, force="scene")
    out = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        m = g.copy()
        m.apply_transform(T)
        out.append((gname, m))
    return out


def fit_axis_center(xy: np.ndarray):
    """(x, z) 에 원을 최소제곱으로 — texture_fill.py 와 같은 규칙."""
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    sol, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
    a_, b_, c_ = sol
    return np.array([a_, b_]), float(np.sqrt(max(c_ + a_ ** 2 + b_ ** 2, 0.0)))


ap = argparse.ArgumentParser()
ap.add_argument("--composite", required=True, help="texture_fill.py 가 만든 GLB (채움에 정점 색이 있는 것)")
ap.add_argument("--source", required=True, help="원본 GLB — 출력의 원본 노드로 그대로 쓴다")
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--base-frac", type=float, default=0.08)
ap.add_argument("--swatch", default="", help="견본 이미지. 주면 채움 텍스처의 결(또는 색 전체)을 이 이미지를 타일링해 만든다. "
                "텍셀 단위로 찍으므로 정점 밀도와 무관하게 견본의 결이 그대로 살아난다")
ap.add_argument("--swatch-mode", default="detail", choices=("detail", "full"),
                help="detail: 톤은 정점 색에서(texture_fill --mode band 권장) + 결만 견본에서 / full: 견본 색 그대로")
ap.add_argument("--swatch-tiles", type=int, default=0,
                help="둘레에 견본을 몇 장 놓나 (짝수로 맞춘다). 0 이면 견본 1px ≈ 텍셀 1px 이 되게 자동")
ap.add_argument("--swatch-gain", type=float, default=1.0, help="detail 에서 견본 결의 세기")
ap.add_argument("--swatch-highpass", type=float, default=6.0,
                help="견본에서 이 반지름(px)의 가우시안 저주파를 빼고 결만 남긴다. 사진 견본은 조명 기울기·얼룩이 있어 "
                "그대로 타일링하면 주기적 줄무늬가 된다 (51x64 견본 위→아래 161→137 실측). 0 이면 끄기")
ap.add_argument("--swatch-synth", type=int, default=1,
                help="1: 견본을 붙이지 않고 견본의 통계(방사 평균 파워 스펙트럼 + 밝기 히스토그램)로 결을 합성한다 — "
                "반복·방향성이 없고 둘레에서 이음이 없다. 0: 견본을 반사 타일링으로 그대로 붙인다 "
                "(사진 견본은 가로 줄 구조가 있어 128px 주기 줄무늬가 생겼다, 2026-09-18 실측)")
ap.add_argument("--swatch-seed", type=int, default=0, help="합성 결의 난수 시드")
ap.add_argument("--grain-source", type=int, default=0,
                help="1: 결의 통계(굵기·명암 분포)를 외부 견본이 아니라 **원본의 실측 텍스처**에서 뽑아 합성한다 — "
                "채움부가 원본 표면과 같은 결·같은 색을 갖는다. 톤은 정점 색(texture_fill --mode band 권장)에서")
ap.add_argument("--grain-samples", type=int, default=16_000_000, help="원본 표면 표본 점 수 (원통 펼침 관측 이미지용)")
ap.add_argument("--grain-gain", type=float, default=1.0, help="원본 결의 세기 (1 = 원본과 같은 대비)")
ap.add_argument("--tone-side", default="all", choices=("all", "outer"),
                help="톤을 잡을 원본 면. outer: 법선이 회전축 바깥을 향한 면(겉면)만 — 안쪽 면·틈이 중앙값을 끌어내리는 것을 막는다")
ap.add_argument("--tone-source", type=int, default=1,
                help="1(--grain-source 와 함께): 톤도 관측 이미지의 행(높이) 중앙값에서 잡는다 — 면적 가중 실측 색이라 "
                "정점 표본 중앙값(texture_fill --mode band)보다 눈에 보이는 색에 가깝다 (굽다리 높이대에서 13~16 차이, "
                "2026-09-18 실측). 0: 정점 색을 그대로 굽는다")
ap.add_argument("--grain-debug", default="", help="관측 이미지·합성 결을 이 접두로 PNG 저장 (확인용)")
ap.add_argument("--tint", default="", help="예: 1.06,1.02,0.98 — 채움 텍스처 전체에 곱하는 색조")
ap.add_argument("--material", default="source", choices=("source", "matte"),
                help="채움 재질의 PBR 값. source: 원본 재질의 metallic/roughness/baseColorFactor 를 그대로 복사 — "
                "같은 색이 같은 음영으로 보인다 (박물관 스캔은 metallic 1·roughness 1 로 나와 있어 무광 채움과 "
                "색이 달라 보였다, 2026-09-18). matte: metallic 0·roughness 0.85")
a = ap.parse_args()

geoms = dict(world_geoms(Path(a.composite)))
fill = next((m for n, m in geoms.items() if FILL_NODE in n), None)
if fill is None:
    sys.exit("채움 노드(%s)가 없다 — 있는 것: %s" % (FILL_NODE, list(geoms)))
if getattr(fill.visual, "kind", None) != "vertex":
    sys.exit("채움에 정점 색이 없다 (kind=%r) — texture_fill.py 를 먼저 돌려라" % getattr(fill.visual, "kind", None))
col = np.asarray(fill.visual.vertex_colors, dtype=np.uint8)[:, :3]
V = np.asarray(fill.vertices, dtype=np.float64)
F = np.asarray(fill.faces, dtype=np.int64).copy()
print("채움: 정점 %d · 면 %d · 색 평균 %s" % (len(V), len(F), col.mean(0).round(1).tolist()))

src = trimesh.load(a.source, force="mesh", process=False)
ext = src.bounds[1] - src.bounds[0]
up = int(np.argmax(ext))
lat = [i for i in range(3) if i != up]
sv = np.asarray(src.vertices, dtype=np.float64)
h0s, h1s = src.bounds[0][up], src.bounds[1][up]
center, R = fit_axis_center(sv[sv[:, up] <= h0s + a.base_frac * (h1s - h0s)][:, lat])
print("세로축 %s · 회전축 중심 %s · 굽다리 반지름 %.4f" % ("xyz"[up], center.round(4).tolist(), R))

# 원통 펼치기
p = V[:, lat] - center
th = np.arctan2(p[:, 1], p[:, 0])
h = V[:, up]
h0, h1 = float(h.min()), float(h.max())
u = (th + np.pi) / (2 * np.pi)
v = (h - h0) / max(h1 - h0, 1e-12)

# 이음선 — ±π 를 가로지르는 면만 꼭짓점 복제해 u+1
Vl, Ul, Vvl, Cl = list(V), list(u), list(v), list(col)
fu = u[F]
cross = np.where(fu.max(1) - fu.min(1) > 0.5)[0]
remap = {}
for fi in cross:
    for ci in range(3):
        vi = F[fi, ci]
        if u[vi] >= 0.5:
            continue
        if vi not in remap:
            Vl.append(V[vi]); Ul.append(u[vi] + 1.0); Vvl.append(v[vi]); Cl.append(col[vi])
            remap[vi] = len(Vl) - 1
        F[fi, ci] = remap[vi]
print("이음선 면 %d · 복제한 꼭짓점 %d" % (len(cross), len(remap)))
V2 = np.asarray(Vl); U2 = np.asarray(Ul); Vv2 = np.asarray(Vvl); C2 = np.asarray(Cl, dtype=np.uint8)

# 텍스처 굽기 — 텍셀마다 (u, v) 최근접 채움 정점의 색. u 는 순환이라 ±1 사본을 붙인다.
S = a.size
gu, gv = np.meshgrid((np.arange(S) + 0.5) / S, (np.arange(S) + 0.5) / S)
uv0 = np.c_[u, v]                                       # 복제 전 좌표로 찾는다 (색은 같다)
tree = cKDTree(np.vstack([uv0, uv0 + [1, 0], uv0 - [1, 0]]))
idx = tree.query(np.c_[gu.ravel(), gv.ravel()])[1] % len(uv0)
img = col[idx].reshape(S, S, 3)



def grain_stats(lum, mask=None, win=64, hp_sigma=6.0):
    """결 통계 — 굵기는 방사 평균 파워 스펙트럼, 명암 분포는 고역 밝기의 정렬값(히스토그램).
    mask(관측된 화소)가 있으면 완전히 관측된 win×win 창들에서만 스펙트럼을 재고 히스토그램도 관측 화소만 쓴다.
    고역은 정규화 합성곱(구멍을 무시한 가우시안)으로 저주파를 빼서 얻는다 → 톤·얼룩은 빠지고 결만 남는다."""
    from scipy import ndimage
    if mask is None:
        mask = np.ones(lum.shape, bool)
    m = mask.astype(np.float64)
    lo = ndimage.gaussian_filter(lum * m, hp_sigma) / np.maximum(ndimage.gaussian_filter(m, hp_sigma), 1e-6)
    hp = np.where(mask, lum - lo, 0.0)
    H, W = lum.shape
    win = int(min(win, H, W))
    nb = 24
    edges = np.linspace(0, 0.5 * np.sqrt(2), nb + 1)
    kmid = 0.5 * (edges[:-1] + edges[1:])
    hann = np.outer(np.hanning(win), np.hanning(win))
    k = np.hypot(np.fft.fftfreq(win)[:, None], np.fft.fftfreq(win)[None, :]).ravel()
    which = np.clip(np.digitize(k, edges) - 1, 0, nb - 1)
    Pk = np.zeros(nb)
    n = 0
    step = max(win // 2, 1)
    for r in range(0, H - win + 1, step):
        for c in range(0, W - win + 1, step):
            if mask[r:r + win, c:c + win].all():
                P = np.abs(np.fft.fft2(hp[r:r + win, c:c + win] * hann)) ** 2
                Pk += np.bincount(which, P.ravel(), nb)
                n += 1
    if n == 0:
        sys.exit("결을 잴 완전 관측 창(%d×%d)이 하나도 없다" % (win, win))
    Pk /= np.maximum(np.bincount(which, minlength=nb), 1) * n
    return kmid, Pk, np.sort(hp[mask]), n, hp


def synth_from_stats(kmid, Pk, target, rows, cols, seed):
    """통계대로 결을 합성한다 — 스펙트럼을 맞춘 가우시안 잡음에 순위 맞춤으로 명암 분포를 입힌다.
    FFT 로 만들어 cols 방향으로 주기적이니 둘레 한 바퀴를 cols 로 잡으면 u 순환에서 이음이 없다."""
    rng = np.random.default_rng(seed)
    Nf = np.fft.fft2(rng.standard_normal((rows, cols)))
    K = np.hypot(np.fft.fftfreq(rows)[:, None], np.fft.fftfreq(cols)[None, :])
    amp = np.sqrt(np.interp(K, kmid, Pk))
    amp[0, 0] = 0.0
    g = np.real(np.fft.ifft2(Nf * amp))
    ranks = np.argsort(np.argsort(g.ravel()))
    g = target[(ranks.astype(np.int64) * (len(target) - 1)) // max(g.size - 1, 1)]
    return g.reshape(rows, cols)


# 원본 결 합성 — 원본 표면을 표본 추출해 같은 (u, v) 격자에 관측 이미지를 만들고, 그 결의 통계로 채움부 결을 합성한다.
grain_rep = None
if a.grain_source:
    from scipy import ndimage
    vis = src.visual
    if getattr(vis, "kind", None) != "texture" or getattr(vis.material, "baseColorTexture", None) is None:
        sys.exit("원본에 UV 텍스처가 없다 (kind=%r) — --grain-source 는 실측 텍스처가 필요하다" % getattr(vis, "kind", None))
    tex_src = vis.material.baseColorTexture.convert("RGB")
    uv_src = np.asarray(vis.uv, dtype=np.float64)
    faces_src = np.asarray(src.faces, dtype=np.int64)
    interp = getattr(trimesh.visual.color, "uv_to_interpolated_color", None) or trimesh.visual.color.uv_to_color
    acc = np.zeros((S * S, 3))
    cnt = np.zeros(S * S)
    acc_o = np.zeros((S * S, 3))                          # 겉면(법선이 바깥을 향한 표본)만 따로
    cnt_o = np.zeros(S * S)
    fnorm = np.asarray(src.face_normals, dtype=np.float64)
    chunk = 1_000_000
    for done in range(0, a.grain_samples, chunk):
        n = min(chunk, a.grain_samples - done)
        res = trimesh.sample.sample_surface(src, n)
        pts, fid = np.asarray(res[0], dtype=np.float64), np.asarray(res[1])
        bary = trimesh.triangles.points_to_barycentric(src.triangles[fid], pts)
        uvp = (uv_src[faces_src[fid]] * bary[:, :, None]).sum(1)
        colp = np.asarray(interp(uvp, tex_src))[:, :3].astype(np.float64)
        q = pts[:, lat] - center
        cu = np.floor((np.arctan2(q[:, 1], q[:, 0]) + np.pi) / (2 * np.pi) * S).astype(np.int64) % S
        cv = np.floor((pts[:, up] - h0) / max(h1 - h0, 1e-12) * S).astype(np.int64)
        ok = (cv >= 0) & (cv < S)
        li = cv[ok] * S + cu[ok]
        for ch in range(3):
            acc[:, ch] += np.bincount(li, colp[ok, ch], S * S)
        cnt += np.bincount(li, minlength=S * S)
        rdir = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
        outer = ((fnorm[fid][:, lat] * rdir).sum(1) > 0.2) & ok
        lo_ = cv[outer] * S + cu[outer]
        for ch in range(3):
            acc_o[:, ch] += np.bincount(lo_, colp[outer, ch], S * S)
        cnt_o += np.bincount(lo_, minlength=S * S)
    hit = (cnt > 0).reshape(S, S)
    obs = (acc / np.maximum(cnt, 1)[:, None]).reshape(S, S, 3)
    # 표본이 안 떨어진 칸(포아송 구멍)은 원본 표면 영역 안이면 최근접 관측값으로 메운다.
    # 영역 = 관측 밀도가 절반 이상인 곳 → 진짜 결손부(원본이 없는 곳)는 영역 밖으로 남는다.
    mask = ndimage.gaussian_filter(hit.astype(np.float64), 3.0) > 0.5
    holes = mask & ~hit
    if holes.any():
        _, (iy, ix) = ndimage.distance_transform_edt(~hit, return_indices=True)
        obs[holes] = obs[iy[holes], ix[holes]]
    lum = obs.mean(2)
    kmid, Pk, target, nwin, hp_lum = grain_stats(lum, mask)
    m = mask.astype(np.float64)
    den = np.maximum(ndimage.gaussian_filter(m, 6.0), 1e-6)
    hp_rgb = np.stack([np.where(mask, obs[..., c] - ndimage.gaussian_filter(obs[..., c] * m, 6.0) / den, 0.0)
                       for c in range(3)], axis=2)
    ch_std = hp_rgb[mask].std(0) / max(float(hp_lum[mask].std()), 1e-9)      # 채널별 결 대비 비율
    tone_rep = None
    if a.tone_source:
        # 톤 = 관측 이미지의 행(높이) 중앙값. 관측 텍셀이 2% 미만인 행은 이웃에서 보간, 높이 방향 11행 이동평균.
        if a.tone_side == "outer":                          # 겉면 표본만으로 관측값을 다시 만든다
            mask_t = (cnt_o > 0).reshape(S, S)
            obs_t = (acc_o / np.maximum(cnt_o, 1)[:, None]).reshape(S, S, 3)
            print("톤 면 = 겉면만 (겉면 텍셀 %.1f%% / 전체 관측 %.1f%%)" % (100 * mask_t.mean(), 100 * mask.mean()))
        else:
            mask_t, obs_t = mask, obs
        good = mask_t.sum(1) >= max(8, int(0.02 * S))
        rows_i = np.arange(S)
        tone = np.zeros((S, 3))
        for c in range(3):
            med = np.array([np.median(obs_t[r, mask_t[r], c]) if good[r] else np.nan for r in range(S)])
            med = np.interp(rows_i, rows_i[good], med[good])
            k = 11
            tone[:, c] = np.convolve(np.pad(med, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        img = np.clip(np.repeat(tone[:, None, :], S, axis=1), 0, 255).astype(np.uint8)
        tone_rep = {"rows_observed": int(good.sum()), "mean": tone.mean(0).round(1).tolist(), "side": a.tone_side}
        print("톤 = 관측 행 중앙값 (관측 행 %d / %d · 평균 %s)" % (good.sum(), S, tone_rep["mean"]))
    g = synth_from_stats(kmid, Pk, target, S, S, a.swatch_seed)
    img = np.clip(img.astype(np.float64) + a.grain_gain * g[..., None] * ch_std, 0, 255).astype(np.uint8)
    grain_rep = {"from": "source_texture", "samples": int(a.grain_samples), "coverage": round(float(mask.mean()), 4),
                 "holes_filled": int(holes.sum()), "tone_from_observed": tone_rep,
                 "windows": int(nwin), "gain": a.grain_gain, "seed": a.swatch_seed,
                 "source_grain_std": hp_rgb[mask].std(0).round(2).tolist(),
                 "synth_grain_std": (g[..., None] * ch_std).reshape(-1, 3).std(0).round(2).tolist()}
    print("원본 결 합성: 표본 %d · 관측 텍셀 %.1f%% · 분석 창 %d · 원본 결 표준편차 %s → 합성 %s"
          % (a.grain_samples, 100 * mask.mean(), nwin, grain_rep["source_grain_std"], grain_rep["synth_grain_std"]))
    if a.grain_debug:
        dbg = obs.copy()
        dbg[~mask] = [255, 0, 255]                                            # 원본이 없는 텍셀 = 자홍
        Image.fromarray(np.flipud(np.clip(dbg, 0, 255).astype(np.uint8))).save(a.grain_debug + "_observed.png")
        gm = obs[mask].mean(0)
        Image.fromarray(np.flipud(np.clip(gm + g[..., None] * ch_std, 0, 255).astype(np.uint8))).save(a.grain_debug + "_synth.png")
        print("확인용 저장:", a.grain_debug + "_observed.png", a.grain_debug + "_synth.png")

# 견본 타일링 — 텍셀 격자 (행 = v 높이, 열 = u 둘레) 를 견본 픽셀 좌표로 바꿔 찍는다.
swatch_rep = None
if a.swatch:
    sw = Image.open(a.swatch).convert("RGBA")
    swa = np.asarray(sw, dtype=np.float64)
    rgb, alpha = swa[..., :3], swa[..., 3:4] / 255.0
    sw_rgb = rgb * alpha + rgb.reshape(-1, 3).mean(0) * (1 - alpha)      # 투명 픽셀은 평균색으로
    if a.swatch_highpass > 0:                            # 저주파 제거 — 평균은 유지하고 기울기·얼룩만 뺀다
        from PIL import ImageFilter
        lo = np.asarray(Image.fromarray(np.clip(sw_rgb, 0, 255).astype(np.uint8))
                        .filter(ImageFilter.GaussianBlur(a.swatch_highpass)), dtype=np.float64)
        sw_rgb = sw_rgb - lo + sw_rgb.reshape(-1, 3).mean(0)
    sw_h, sw_w = sw_rgb.shape[:2]
    # 둘레 타일 수 — 기본은 견본 1px ≈ 텍셀 1px. 반사 타일링이 u 순환(첫 장↔끝 장)에서도 이어지려면 짝수.
    N = a.swatch_tiles or max(2, int(round(S / sw_w)))
    N += N % 2
    # 텍셀의 실제 폭 = 둘레 / S. 둘레는 채움 정점 반지름의 중앙값으로 잡는다 (넓은 곳·좁은 곳은 ±수십 % 늘거나 줄지만
    # 잔 결이라 눈에 띄지 않는다). 세로도 같은 크기로 → 견본 픽셀이 정방형을 유지한다.
    r_med = float(np.median(np.hypot(p[:, 0], p[:, 1])))
    circ = 2 * np.pi * r_med
    world_per_swpx = (circ / N) / sw_w                  # 견본 1px 의 실제 크기
    xs = (np.arange(S) + 0.5) * (circ / S) / world_per_swpx                 # 텍셀 열 → 견본 x
    ys = (np.arange(S) + 0.5) / S * (h1 - h0) / world_per_swpx               # 텍셀 행 → 견본 y

    def reflect(i, n):                                   # 반사 타일링 인덱스 — 장 경계에서 이음이 없다
        i = np.floor(i).astype(np.int64) % (2 * n)
        return np.where(i < n, i, 2 * n - 1 - i)

    def synth_grain(lum, rows, cols, seed):
        """견본과 같은 굵기·같은 명암 분포의 결을 (rows, cols) 견본 픽셀 격자에 합성한다.
        굵기 = 견본의 방사 평균 파워 스펙트럼(등방성으로 만든다 — 사진의 가로 줄 방향성은 버린다),
        명암 = 견본의 밝기 히스토그램(어두운 점이 박힌 비대칭까지 순위 맞춤으로 그대로).
        FFT 로 만들어 cols 방향으로 주기적이니 둘레 한 바퀴를 cols 로 잡으면 u 순환에서 이음이 없다."""
        h, w = lum.shape
        win = np.outer(np.hanning(h), np.hanning(w))
        P = np.abs(np.fft.fft2((lum - lum.mean()) * win)) ** 2
        k = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]).ravel()
        nb = 24
        edges = np.linspace(0, 0.5 * np.sqrt(2), nb + 1)
        which = np.clip(np.digitize(k, edges) - 1, 0, nb - 1)
        Pk = np.bincount(which, P.ravel(), nb) / np.maximum(np.bincount(which, minlength=nb), 1)
        kmid = 0.5 * (edges[:-1] + edges[1:])
        rng = np.random.default_rng(seed)
        Nf = np.fft.fft2(rng.standard_normal((rows, cols)))
        K = np.hypot(np.fft.fftfreq(rows)[:, None], np.fft.fftfreq(cols)[None, :])
        amp = np.sqrt(np.interp(K, kmid, Pk))
        amp[0, 0] = 0.0
        g = np.real(np.fft.ifft2(Nf * amp))
        ranks = np.argsort(np.argsort(g.ravel()))                 # 순위 맞춤 → 견본 밝기 분포 그대로
        target = np.sort((lum - lum.mean()).ravel())
        g = target[(ranks.astype(np.int64) * (len(target) - 1)) // max(g.size - 1, 1)]
        return g.reshape(rows, cols)

    sw_mean = sw_rgb.reshape(-1, 3).mean(0)
    if a.swatch_synth:
        rows = int(np.ceil(ys.max())) + 1
        cols = int(N * sw_w)                              # 둘레 한 바퀴 = 견본 N 장 폭 → 주기가 정확히 맞는다
        lum = sw_rgb.mean(axis=2)
        g = synth_grain(lum, rows, cols, a.swatch_seed)
        ch_std = sw_rgb.reshape(-1, 3).std(0) / max(float(lum.std()), 1e-9)   # 채널별 대비 비율 (거의 1)
        yi = np.clip(np.floor(ys).astype(np.int64), 0, rows - 1)
        xi = np.floor(xs).astype(np.int64) % cols
        tiled = sw_mean + g[yi[:, None], xi[None, :]][..., None] * ch_std      # (S, S, 3)
        how = "합성 (스펙트럼+히스토그램, 격자 %dx%d, seed %d)" % (rows, cols, a.swatch_seed)
    else:
        tiled = sw_rgb[reflect(ys, sw_h)[:, None], reflect(xs, sw_w)[None, :]]   # (S, S, 3)
        how = "반사 타일링"
    if a.swatch_mode == "full":
        img = np.clip(tiled, 0, 255).astype(np.uint8)
    else:                                                # 톤은 정점 색(높이대 중앙값)에서, 결만 견본에서
        img = np.clip(img.astype(np.float64) + a.swatch_gain * (tiled - sw_mean), 0, 255).astype(np.uint8)
    swatch_rep = {"swatch": a.swatch, "swatch_px": [int(sw_w), int(sw_h)], "mode": a.swatch_mode,
                  "highpass_px": a.swatch_highpass, "synth": bool(a.swatch_synth), "seed": a.swatch_seed,
                  "tiles_around": int(N), "swatch_px_world": round(world_per_swpx, 6),
                  "texel_world": round(circ / S, 6), "gain": a.swatch_gain,
                  "swatch_mean": sw_mean.round(1).tolist(), "swatch_std": sw_rgb.reshape(-1, 3).std(0).round(1).tolist()}
    print("견본 %s %dx%d · %s · %s · 둘레 %d 장 · 견본 1px = %.5f (텍셀 1칸 = %.5f) · 결 표준편차 %s"
          % (a.swatch, sw_w, sw_h, a.swatch_mode, how, N, world_per_swpx, circ / S, swatch_rep["swatch_std"]))
if a.tint:
    t = np.array([float(x) for x in a.tint.split(",")], dtype=np.float64)
    img = np.clip(img.astype(np.float64) * t, 0, 255).astype(np.uint8)
    print("색조 곱 적용:", t.tolist())
# 이미지의 v 축은 위에서 아래로 가고 UV 의 v 는 아래에서 위로 간다
tex = Image.fromarray(np.flipud(img), mode="RGB")
print("텍스처 %d×%d 구움 · 평균 RGB %s" % (S, S, np.asarray(tex).reshape(-1, 3).mean(0).round(1).tolist()))

# 재질 — 원본과 같은 PBR 값이어야 같은 색이 같은 음영으로 보인다
pbr = dict(metallicFactor=0.0, roughnessFactor=0.85, baseColorFactor=None)
if a.material == "source":
    sm = getattr(src.visual, "material", None)
    if isinstance(sm, trimesh.visual.material.PBRMaterial):
        pbr = dict(metallicFactor=sm.metallicFactor if sm.metallicFactor is not None else 1.0,
                   roughnessFactor=sm.roughnessFactor if sm.roughnessFactor is not None else 1.0,
                   baseColorFactor=sm.baseColorFactor)
        # 유효값 = factor × MR 텍스처 평균 (G = roughness, B = metallic). 박물관 스캔은 factor 1·1 인데
        # 텍스처가 B=0 이라 실제로는 무광이다 — factor 만 복사하면 채움이 금속처럼 어둡게 나온다.
        mrt = getattr(sm, "metallicRoughnessTexture", None)
        if mrt is not None:
            arr = np.asarray(mrt.convert("RGB"), dtype=np.float64)
            pbr["roughnessFactor"] = float(pbr["roughnessFactor"] * arr[..., 1].mean() / 255.0)
            pbr["metallicFactor"] = float(pbr["metallicFactor"] * arr[..., 2].mean() / 255.0)
    else:
        print("원본 재질이 PBR 이 아니라(%s) 기본값(metallic 1·roughness 1)을 쓴다" % type(sm).__name__)
        pbr = dict(metallicFactor=1.0, roughnessFactor=1.0, baseColorFactor=None)
print("채움 재질: metallic %.2f · roughness %.2f · baseColorFactor %s (%s)"
      % (pbr["metallicFactor"], pbr["roughnessFactor"],
         None if pbr["baseColorFactor"] is None else list(pbr["baseColorFactor"]), a.material))
mat = trimesh.visual.material.PBRMaterial(name="lascomp_gap_fill_tex", baseColorTexture=tex,
                                          metallicFactor=pbr["metallicFactor"], roughnessFactor=pbr["roughnessFactor"])
if pbr["baseColorFactor"] is not None:
    mat.baseColorFactor = pbr["baseColorFactor"]
fill2 = trimesh.Trimesh(V2, F, process=False)
fill2.visual = trimesh.visual.TextureVisuals(uv=np.c_[U2, Vv2], material=mat)
# 정점 법선을 함께 내보낸다 — 없으면 뷰어마다 다르게 그린다 (three.js·Blender 는 부드럽게 계산해 주지만
# pyrender 는 면 단위 평면 음영으로 그려 채움만 어둡고 각져 보였다, 2026-09-18)
fill2.vertex_normals = fill2.vertex_normals

out_scene = trimesh.load(a.source, process=False, force="scene")   # 원본 노드는 원본 텍스처 그대로
out_scene.add_geometry(fill2, node_name=FILL_NODE, geom_name=FILL_NODE)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
out_scene.export(str(out))
rep = {"composite": a.composite, "source": a.source, "out": str(out), "texture_size": S,
       "up_axis": "xyz"[up], "axis_center": center.round(6).tolist(),
       "fill_verts_before": int(len(V)), "fill_verts_after": int(len(V2)), "fill_faces": int(len(F)),
       "seam_faces": int(len(cross)), "seam_verts_duplicated": int(len(remap)),
       "texture_mean": np.asarray(tex).reshape(-1, 3).mean(0).round(1).tolist(),
       "swatch": swatch_rep, "grain": grain_rep, "tint": a.tint or None,
       "material": {"mode": a.material, "metallicFactor": pbr["metallicFactor"], "roughnessFactor": pbr["roughnessFactor"]},
       "vertex_colors_removed": True, "geometry_changed": False}
out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", out)
