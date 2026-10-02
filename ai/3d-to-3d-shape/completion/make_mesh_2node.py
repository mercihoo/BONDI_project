# -*- coding: utf-8 -*-
"""
make_mesh_2node.py — 관측부를 **재샘플링하지 않는** 복원 메시.

`make_mesh_71489.py` 는 출력 8,192점 전체를 `(y,θ)` 격자로 다시 만든다.
그래서 두 가지를 잃었다.

    투창    원본에는 정확히 있는데 44x64 격자를 거치며 뭉갠다
    해상도  원본 정점 0.71mm → 출력 1.90mm (2.7배 성김)

**둘 다 관측부를 건드려서 생긴 손해다.** 관측부는 모델이 개선할 것이 없다 —
모델이 기여하는 것은 **결손부뿐**이다. 그래서 이렇게 나눈다.

    region_observed   원본 A_normal.glb **무수정** (투창·해상도·텍스처 그대로)
    region_ai         모델 출력 중 **결손 칸만** 격자로 메시화

v32 의 `region_carried`(무수정) / `region_filled` 와 같은 구조다 (`NFR-ETH-003`).

결손 칸을 어떻게 고르나
    입력 조밀 점군(12만점)이 빈 칸 중, **투창이 아닌 것**.
    투창 판정은 파이프라인 기준을 그대로 쓴다 (make_mesh_71489.openwork_cells).

사용
  PY=".../venv/Scripts/python.exe"
  $PY make_mesh_2node.py --src work/pred_71489_A_normal_v2.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

import make_mesh_71489 as M

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
OUT = WORK / "restored_adapointr"
SRC_GLB = ROOT / "gupdari-71489" / "in" / "A_normal.glb"


def load_original(path: Path):
    g = trimesh.load(str(path), process=False)
    return g.to_geometry() if hasattr(g, "to_geometry") else g


def norm_params(mesh, n=120000, seed=0):
    """`run_71489.py` 와 **같은 방식**으로 중심·축척을 되살린다.

    npz 에 c·s 를 안 저장해서 다시 만든다. `sample_surface` 는 seed 가 같으면
    같은 점을 내므로 값이 일치한다.
    """
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    P = np.asarray(pts, np.float64)
    c = P.mean(0)
    s = float(np.linalg.norm(P - c, axis=1).max())
    return c, s


def smooth_keep_border(mesh, iters=12, lam=0.6, orig=None, seam_mm=None,
                       edge_rings=0, free_seed=None):
    """**이음새만 고정**하고 나머지는 평활한다.

    처음엔 열린 모서리를 **전부** 고정했다. 그랬더니 관측면에 안 붙는
    **자유 테두리(아가리 위쪽)까지 굳어 거칠게 남았다.**

    이음새(관측면과 맞닿은 테두리)는 벌어지면 안 되니 고정하고,
    자유 테두리는 풀어서 같이 평활한다.

    `edge_rings > 0` 이면 **자유 테두리에서 그만큼 고리까지만** 평활한다.
    면 전체를 평활하면 `ridge_transfer()` 로 얹은 돌대가 같이 깎인다 —
    실측: 요철 세기가 관측 대비 116% → 65%(평활 12) → 56%(평활 20).
    테두리 근처만 건드리면 **끝 거칠기는 잡고 안쪽 돌대는 남는다.**
    """
    V = mesh.vertices.copy()
    F = mesh.faces
    e = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    border = np.zeros(len(V), bool)
    border[np.unique(uniq[cnt == 1])] = True
    free = np.zeros(len(V), bool)
    if not border.any() and free_seed is not None:
        # **두께 있는 껍질은 닫혀 있어 열린 모서리가 없다.**
        # 그때는 관측 근접으로 이음새를, 테두리 띠로 자유부를 정한다.
        from scipy.spatial import cKDTree as _KD
        d0, _ = _KD(np.asarray(orig.vertices)).query(V)
        thr0 = seam_mm if seam_mm else np.percentile(d0, 12)
        border = d0 <= thr0
        free = free_seed & ~border
        print("  껍질 모드: 이음새 %d 고정 · 테두리 띠 %d 평활"
              % (int(border.sum()), int(free.sum())))
    elif orig is not None:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(orig.vertices)).query(V)
        thr = seam_mm if seam_mm else np.percentile(d[border], 40)
        free = border & (d > thr)
        border = border & ~free  # 자유 테두리는 고정 해제            # 자유 테두리는 고정 해제
        print("  테두리 %d 중 이음새 %d 고정 · 자유 %d 평활"
              % (int((~free & np.zeros(len(V), bool)).sum() + border.sum() + free.sum()),
                 int(border.sum()), int(free.sum())))

    nbr = [[] for _ in range(len(V))]
    for a, b in uniq:
        nbr[a].append(b); nbr[b].append(a)
    idx = [np.array(n, int) if n else np.zeros(0, int) for n in nbr]

    movable = ~border
    w = np.ones(len(V))
    if edge_rings > 0:
        # 자유 테두리에서 고리 거리를 재고, 멀수록 약하게 민다
        dist = np.full(len(V), np.inf)
        seed = np.nonzero(free)[0] if orig is not None else np.nonzero(border)[0]
        dist[seed] = 0
        frontier = list(seed)
        for k in range(1, edge_rings + 1):
            nxt = []
            for i in frontier:
                for j in idx[i]:
                    if not np.isfinite(dist[j]):
                        dist[j] = k
                        nxt.append(j)
            frontier = nxt
            if not nxt:
                break
        movable = movable & np.isfinite(dist)
        w = np.where(np.isfinite(dist), 1.0 - dist / (edge_rings + 1.0), 0.0)
        print("  테두리 근처만 평활: 움직이는 정점 %d / %d (고리 %d)"
              % (int(movable.sum()), len(V), edge_rings))

    mv = np.nonzero(movable)[0]
    for _ in range(iters):
        Vn = V.copy()
        for i in mv:
            if len(idx[i]):
                Vn[i] = V[i] + lam * w[i] * (V[idx[i]].mean(0) - V[i])
        V = Vn
    out = mesh.copy()
    out.vertices = V
    print("  평활: %d회 · 고정한 경계 정점 %d / %d" % (iters, border.sum(), len(V)))
    return out


def transfer_color(patch, orig):
    """관측 메시의 **텍스처 색**을 패치 정점으로 옮긴다.

    모델 출력에는 색이 없다. v32 는 `texture_from_mesh.py` (S10) 로 TRELLIS
    텍스처를 채움면에 심는데, 여기선 가벼운 판을 쓴다 —
    관측 정점 색의 **최근접** 값을 가져온다. 결손부 주변에서 끌어오므로
    이음새가 자연스럽게 이어진다.

    `rtree` 가 없어 표면 최근접점 질의는 못 쓴다. 관측 정점이 94,877개라
    KD 트리 최근접 정점으로 충분하다 (파이프라인도 같은 이유로 그렇게 한다).
    """
    from scipy.spatial import cKDTree
    try:
        cv = orig.visual.to_color().vertex_colors
    except Exception as ex:
        print("  [건너뜀] 관측 색을 못 읽었다: %s" % ex)
        return None
    cv = np.asarray(cv)[:, :3]
    d, i = cKDTree(orig.vertices).query(patch.vertices)
    col = cv[i]
    print("  색 이식: 관측 정점 %d 에서 최근접. 거리 중앙 %.4f" % (len(cv), np.median(d)))
    return np.hstack([col, np.full((len(col), 1), 255, np.uint8)])


def ridge_transfer(R, known, dense_occ, dense, up, c2, h_edges, nh, nt,
                   sigma=2.0, wide=4.0, gain=1.0):
    """**돌대(가로 줄무늬)를 결손부로 잇는다.**

    모델 출력은 전체 형상은 잡지만 요철이 뭉개진다 (8,192점 → 격자). 그런데
    토기는 회전체라 **같은 높이의 돌대는 모든 각도에 있다.** 관측이 남은 각도에서
    `r(h)` 를 읽어 고주파(요철)만 떼어 결손부에 더한다.

        r_patch = 저주파(모델)  +  고주파(관측 프로파일)
                  ^ 어디에 면이 있나      ^ 돌대가 어디서 튀어나오나

    색에 건 회전 복사와 같은 논리를 **기하**에 거는 것이다.
    `bitsal-ssu022891` v5.3 이 요철에 쓴 방법이기도 하다 (이면각 4.22°).

    둘레 전체가 결손인 높이는 관측이 없어 요철을 못 만든다 — 그 행은 건드리지 않는다.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter
    ax = [i for i in range(3) if i != up]
    hd = dense[:, up]
    rd = np.hypot(dense[:, ax[0]] - c2[0], dense[:, ax[1]] - c2[1])
    hb = np.clip(np.digitize(hd, h_edges) - 1, 0, nh - 1)

    prof = np.full(nh, np.nan)
    for i in range(nh):
        m = hb == i
        if m.sum() >= 30:
            prof[i] = np.percentile(rd[m], 90)          # 바깥면
    ok = np.isfinite(prof)
    if ok.sum() < 8:
        print("  [건너뜀] 돌대 이식: 관측 프로파일이 모자란다")
        return R
    f = prof.copy()
    f[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], prof[ok])
    # **`f - lowpass` 는 틀렸다.** 행별 반지름 추정 잡음까지 요철로 집어서
    # 표면이 3배 거칠어졌다 (이면각 5.54° → 18.56°, 관측 6.01°).
    # 진짜 돌대는 폭이 몇 행짜리 구조다 — **대역통과**로 그 대역만 남긴다.
    f = median_filter(f, size=3, mode="nearest")        # 튀는 행 먼저 제거
    detail = gaussian_filter1d(f, sigma) - gaussian_filter1d(f, sigma * wide)
    detail = detail * gain
    detail[~ok] = 0.0                                   # 관측 없는 행은 이식 안 한다

    out = R.copy()
    for i in range(nh):
        if detail[i] == 0.0:
            continue
        row = known[i]
        if row.any():
            base = gaussian_filter1d(R[i], sigma)       # 모델의 저주파만 남기고
            out[i] = base + detail[i]                   # 관측 요철을 얹는다
    n = int((detail != 0).sum())
    print("  돌대 이식: %d/%d 행 · 대역통과 sigma %.1f~%.1f · gain %.2f · 진폭 %.5f"
          % (n, nh, sigma, sigma * wide, gain, float(np.abs(detail).max())))
    return out


def surface_scatter(pred, new, up, c2, nb=24, nt=16,
                    row_min=50, cell_min=5, need=3):
    """모델이 **새로 만든 점이 면을 이루나**. 같은 `(높이, 각도)` 칸 안에서
    반지름의 상대 퍼짐(변동계수)의 중앙값.

    왜 재나
      모델이 점을 많이 만들었다고 잘 만든 것이 아니다. 경주_고적_13472 는
      4,854점을 만들었는데 **그릇 벽이 아니라 파단선을 따라 퍼진 구름**이었다.
      메시화는 그 구름을 성실하게 **없는 넓은 챙**으로 만들었다.
      점 수로는 못 가르고, 이 값으로는 갈린다.

    실측 (파손품 34점 중 측정 가능한 10점, 보충점 200 이상)

        경주_8475   0.026  ┐
        경신_71319  0.028  │ 얇은 면 — 복원해도 된다
        경주_6311   0.030  │
        경신_71488  0.040  ┘
        경신_71489  0.051 ~ 0.061   ← 주 복원 대상. 통과해야 한다
        ─────────────────── 빈틈 ───────────────────
        경주_고적_13472  0.077  ┐
        경주_583        0.179  │ 구름 — 복원하면 안 된다
        경주_8485       0.186  ┘

      0.061 과 0.077 사이가 비어 있어 **0.07** 을 기본 임계로 쓴다.

    못 재면 None. 새 점이 200개 미만이거나 행이 모자란 경우다 —
    그때는 애초에 복원할 것이 없다.
    """
    P = pred[new]
    if len(P) < 200:
        return None
    ax = [i for i in range(3) if i != up]
    r = np.hypot(P[:, ax[0]] - c2[0], P[:, ax[1]] - c2[1])
    th = np.arctan2(P[:, ax[1]] - c2[1], P[:, ax[0]] - c2[0])
    lo, hi = np.percentile(pred[:, up], [0.5, 99.5])
    b = np.clip(((P[:, up] - lo) / max(hi - lo, 1e-9) * nb).astype(int), 0, nb - 1)
    sp = []
    for i in range(nb):
        m = b == i
        if m.sum() < row_min:
            continue
        tb = np.clip(((th[m] + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
        v = [np.std(r[m][tb == j]) / max(np.mean(r[m][tb == j]), 1e-9)
             for j in range(nt) if (tb == j).sum() >= cell_min]
        if len(v) >= 4:
            sp.append(np.median(v))
    return float(np.median(sp)) if len(sp) >= need else None


def wall_thickness(orig, cen, scl, up, c2, h_edges, nh, nt, sigma=2.0,
                   n_sample=600000, min_pts=40):
    """관측 메시에서 **높이별 벽 두께**를 잰다. 두 벽 사이의 **빈 틈**을 찾는다.

    두 번 틀렸다.

      ① `r` 의 분위수 폭(92% − 8%)  — 그 폭에는 두께 말고 **칸 안에서 벽이 굽은
         정도**가 섞인다. 8.2mm 로 나왔는데 실제는 3.6mm 였다.
         **AI 벽이 두 배 두껍게 지어졌다.**
      ② 법선으로 바깥/안쪽을 갈라 중앙 `r` 을 뺀 값 — 이 메시는 열린 조각이라
         파단 테두리에서 두 벽이 맞붙고(용접 후 경계변 38,816), 그 언저리가
         값을 끌어내린다. 0.9mm 로 나왔다.

    맞는 방법은 **단면을 보는 것**이다. `(h, θ) ` 칸 안에서 `r` 을 정렬하면
    바깥벽 무리와 안쪽벽 무리 사이가 **비어 있다.** 그 빈 틈에서 자르고
    양쪽 중앙값을 뺀다 — 굽음에도, 테두리에도 흔들리지 않는다.

        r 정렬:  0.568 0.571 ... 0.589 │ (빈 틈) │ 0.605 ... 0.616
                 └── 안쪽벽 ──┘                   └── 바깥벽 ──┘

    71489 실측 (높이 15cm 환산): 중앙 **3.6mm**, 아가리 쪽이 얇다.
    """
    from scipy.ndimage import gaussian_filter1d, median_filter
    ax = [i for i in range(3) if i != up]
    lo, hi = h_edges[0], h_edges[-1]
    pts, _ = trimesh.sample.sample_surface(orig, n_sample, seed=5)
    V = (np.asarray(pts, np.float64) - cen) / scl          # 격자와 같은 자
    r = np.hypot(V[:, ax[0]] - c2[0], V[:, ax[1]] - c2[1])
    th = np.arctan2(V[:, ax[1]] - c2[1], V[:, ax[0]] - c2[0])

    ch, ct = 48, 72                       # 고운 격자는 칸당 표본이 모자란다
    hb = np.clip(((V[:, up] - lo) / max(hi - lo, 1e-9) * ch).astype(int), 0, ch - 1)
    tb = np.clip(((th + np.pi) / (2 * np.pi) * ct).astype(int), 0, ct - 1)
    order = np.lexsort((r, tb, hb))
    key = (hb * ct + tb)[order]
    rs = r[order]
    bounds = np.searchsorted(key, np.arange(ch * ct + 1))

    prof = np.full(ch, np.nan)
    for i in range(ch):
        vals = []
        for j in range(ct):
            k = i * ct + j
            seg = rs[bounds[k]:bounds[k + 1]]
            if len(seg) < min_pts:
                continue
            # 가운데 60% 안에서 **가장 큰 틈**을 찾는다 (양 끝은 꼬리라 뺀다)
            a0, b0 = int(len(seg) * 0.20), int(len(seg) * 0.80)
            if b0 - a0 < 4:
                continue
            gaps = np.diff(seg[a0:b0])
            g = int(np.argmax(gaps))
            cut = a0 + g + 1
            t = np.median(seg[cut:]) - np.median(seg[:cut])
            # 틈이 무리 안 간격보다 뚜렷할 때만 인정한다
            if t > 0 and gaps[g] > 2.5 * np.median(gaps):
                vals.append(t)
        if len(vals) >= 8:
            prof[i] = np.median(vals)
    ok = np.isfinite(prof)
    if ok.sum() < 6:
        return None
    f = prof.copy()
    f[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], prof[ok])
    # 굽/바리 이음은 22mm 인데 바로 옆 몸통은 7mm 다. 가우시안으로 밀면
    # 그 봉우리가 양옆으로 번져 **결손부 벽이 통째로 두꺼워진다** (실측 +14%).
    # 중앙값 필터는 봉우리를 옮기지 않는다. 그 뒤에만 약하게 다듬는다.
    f = median_filter(f, size=5, mode="nearest")
    f = gaussian_filter1d(f, min(sigma, 1.0))
    out = np.interp(np.linspace(0, ch - 1, nh), np.arange(ch), f)
    print("  벽 두께: 단면 빈틈으로 %dx%d 에서 %d/%d 행 측정 · 중앙 %.4f (정규화)"
          " → %d행 보간" % (ch, ct, ok.sum(), ch, np.median(out), nh))
    return out


def grid_shell(h_edges, nt, R_out, thick, up, c2, ax, keep, seam_cell=None):
    """**두께 있는 껍질**을 만든다 — 바깥면 + 안쪽면 + 자유 테두리 마감.

    한 겹 면이면 안에서 볼 때 얇은 막이다. 실제 도기는 벽이 있으므로
    안쪽면을 `R_out - t(h)` 에 만들고 둘을 테두리에서 잇는다.

    **돌대는 바깥에만 있다.** 물레로 빚은 그릇 안쪽은 매끈하므로
    안쪽면에는 요철을 얹지 않고 매끈한 반지름을 쓴다.

    **이음새는 막지 않는다.** 관측 벽과 맞닿는 쪽까지 마감하면 그 자리에
    벽이 하나 더 서서 "붙는 데가 뚫린" 것처럼 보인다. 자유 테두리만 마감하고
    이음새는 열어 두어 관측 벽에 그대로 붙인다.
    """
    from scipy.ndimage import gaussian_filter1d
    nh = R_out.shape[0]
    hs = 0.5 * (h_edges[:-1] + h_edges[1:])
    ths = (np.arange(nt) + 0.5) / nt * 2 * np.pi - np.pi
    R_in = np.maximum(gaussian_filter1d(R_out, 2.0, axis=0) - thick[:, None], 1e-4)

    def make(Rg):
        V = np.zeros((nh * nt, 3))
        for i in range(nh):
            for j in range(nt):
                v = np.zeros(3)
                v[up] = hs[i]
                v[ax[0]] = c2[0] + Rg[i, j] * np.cos(ths[j])
                v[ax[1]] = c2[1] + Rg[i, j] * np.sin(ths[j])
                V[i * nt + j] = v
        return V

    Vo, Vi = make(R_out), make(R_in)
    V = np.vstack([Vo, Vi])
    off = len(Vo)

    F = []
    quad = []
    for i in range(nh - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            if not (keep[i, j] and keep[i, j2] and keep[i + 1, j] and keep[i + 1, j2]):
                continue
            a, b = i * nt + j, i * nt + j2
            c, d = (i + 1) * nt + j, (i + 1) * nt + j2
            F += [[a, c, b], [b, c, d]]                      # 바깥면
            F += [[off + a, off + b, off + c],
                  [off + b, off + d, off + c]]               # 안쪽면 (뒤집어서)
            quad.append((i, j))

    # 자유 테두리 마감 — 바깥과 안쪽을 잇는 띠
    used = set()
    for i, j in quad:
        j2 = (j + 1) % nt
        for (p, q) in (((i, j), (i, j2)), ((i + 1, j), (i + 1, j2)),
                       ((i, j), (i + 1, j)), ((i, j2), (i + 1, j2))):
            e = tuple(sorted([p[0] * nt + p[1], q[0] * nt + q[1]]))
            used.add(e) if e not in used else used.discard(e)
    band_v = np.zeros(len(V), bool)
    n_cap = n_open = 0
    for a, b in used:                                        # 한 번만 쓰인 변 = 경계
        if seam_cell is not None:
            ia, ja = divmod(a, nt)
            ib, jb = divmod(b, nt)
            if seam_cell[ia, ja] or seam_cell[ib, jb]:
                n_open += 1
                continue                                     # 이음새는 열어 둔다
        F += [[a, b, off + a], [b, off + b, off + a]]
        band_v[[a, b, off + a, off + b]] = True
        n_cap += 1
    print("  테두리 마감 %d변 · 이음새로 열어 둔 %d변" % (n_cap, n_open))
    return V, np.array(F), quad, band_v


def rotation_copy_texture(orig, cen, scl, up, c2, h_edges, nh, nt,
                          tex_h=512, tex_w=1024, n_sample=1500000,
                          donor_k=8, denoise=1.0, clean=0.0, outer=True, blend=0.0):
    """`(y, θ)` 격자를 **UV 로 그대로 쓰고**, 결손부 무늬는 **축 둘레 회전 복사**로 채운다.

    왜 UV 가 공짜인가
      격자 자체가 매개변수다 — `u = θ/2π`, `v = 정규화 h`. 따로 펼칠 필요가 없다.

    왜 회전 복사인가
      토기는 회전체라 **같은 높이의 다른 각도가 같은 무늬**다. 결손부 텍셀을
      같은 행(h)에서 관측이 남아 있는 가장 가까운 θ 에서 가져온다.
      `bitsal-ssu022891` v5.3 이 요철에 같은 방법을 써서 거울(7.91°)·U-Net(7.76°)을
      이면각 **4.22°** 로 눌렀다 — *"회전이 표면을 자기 자신으로 보내는 매끄러운 변환"*.

      **추정이 아니라 같은 유물의 다른 각도**라서 근거가 산다.
      다만 아가리처럼 **둘레 전체가 결손이면** 도너가 없다 — 그때는 아래 행에서 끌어온다.

    반환 (PIL 이미지, 정점 UV 만드는 함수)
    """
    from PIL import Image
    # 정점만 흩뿌리면 텍셀의 12% 밖에 안 찬다 (94,877 정점 / 524,288 텍셀).
    # **표면을 조밀 샘플링**해서 채운다 — 면마다 정점색 평균을 쓴다.
    pts, fid = trimesh.sample.sample_surface(orig, n_sample, seed=2)
    # **원본 텍스처를 직접 읽는다.**
    #   `orig.visual.to_color().vertex_colors` 를 거치면 1024x1024 텍스처가
    #   정점 94,877개로 구워진다 — 디테일이 그 단계에서 죽는다.
    #   실측: 그렇게 만든 텍스처의 색 표준편차가 관측의 55%밖에 안 됐다.
    #   표본점의 무게중심 좌표로 UV 를 보간해 원본 텍셀을 그대로 가져온다.
    ouv = getattr(orig.visual, "uv", None)
    otex = getattr(getattr(orig.visual, "material", None), "baseColorTexture", None)
    col = None
    if ouv is not None and otex is not None and len(ouv) == len(orig.vertices):
        tri = np.asarray(orig.vertices)[orig.faces[fid]]        # (n, 3, 3)
        bc = trimesh.triangles.points_to_barycentric(tri, np.asarray(pts, np.float64))
        uvp = (np.asarray(ouv)[orig.faces[fid]] * bc[:, :, None]).sum(axis=1)
        T = np.asarray(otex.convert("RGB"))
        th_, tw_ = T.shape[:2]
        xs = np.clip((uvp[:, 0] % 1.0) * (tw_ - 1), 0, tw_ - 1).astype(int)
        ys = np.clip((1.0 - (uvp[:, 1] % 1.0)) * (th_ - 1), 0, th_ - 1).astype(int)
        col = T[ys, xs].astype(np.float64)
        print("  원본 텍스처 %dx%d 직접 표집 (정점색 굽기 생략)" % (th_, tw_))
    if col is None:
        vc = np.asarray(orig.visual.to_color().vertex_colors)[:, :3].astype(np.float64)
        col = vc[orig.faces[fid]].mean(axis=1)                  # 면 3정점 평균
    V = (np.asarray(pts, np.float64) - cen) / scl               # 패치와 같은 자로
    h, th, _r, _ax = M.cyl(V, up, c2)

    # **바깥면만 남긴다.**
    #   원본은 두께 있는 닫힌 껍질이라 한 (h, θ) 칸에 바깥벽과 **안쪽벽**이 같이 들어간다.
    #   안쪽은 그늘져 어둡다 — 평균이 내려가 결손부가 관측 띠보다 어둡게 나왔다
    #   ("옆과 자연스럽게 이어지지 않는다"의 실제 원인).
    #   면 법선이 축에서 **바깥쪽**을 보는 표본만 쓴다.
    fn = orig.face_normals[fid]
    rx = V[:, _ax[0]] - c2[0]
    ry = V[:, _ax[1]] - c2[1]
    rr = np.hypot(rx, ry)
    ok_r = rr > 1e-9
    out_dot = np.zeros(len(V))
    out_dot[ok_r] = (fn[ok_r, _ax[0]] * rx[ok_r] + fn[ok_r, _ax[1]] * ry[ok_r]) / rr[ok_r]
    keep_outer = (out_dot > 0.15) if outer else np.zeros(len(V), bool)
    if not outer:
        print("  [옛 동작] 바깥면 거르기 끔 — 안쪽벽이 섞인다")
    outer = keep_outer
    if outer.mean() > 0.2:
        print("  바깥면 표본만 사용: %.0f%% (안쪽벽 %.0f%% 제외 — 그늘이 섞이면 어두워진다)"
              % (100 * outer.mean(), 100 * (1 - outer.mean())))
        V, h, th, col = V[outer], h[outer], th[outer], col[outer]
    else:
        print("  [주의] 바깥면 판정이 %.0f%% 뿐이라 전체를 쓴다" % (100 * outer.mean()))

    lo, hi = h_edges[0], h_edges[-1]
    vv = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1 - 1e-9)
    uu = np.clip((th + np.pi) / (2 * np.pi), 0, 1 - 1e-9)
    ti = (vv * tex_h).astype(int)
    tj = (uu * tex_w).astype(int)

    acc = np.zeros((tex_h, tex_w, 3), np.float64)
    cnt = np.zeros((tex_h, tex_w), np.int32)
    np.add.at(acc, (ti, tj), col)
    np.add.at(cnt, (ti, tj), 1)
    have = cnt > 0
    img = np.zeros((tex_h, tex_w, 3), np.uint8)
    img[have] = (acc[have] / cnt[have, None]).astype(np.uint8)
    print("  텍스처 %dx%d · 관측이 채운 텍셀 %.0f%%" % (tex_h, tex_w, 100 * have.mean()))

    # --- 행마다 회전 복사. θ 는 순환이라 가장 가까운 도너를 양방향으로 찾는다
    n_rot = n_row = 0
    # --- 먼저 **관측 안의 빈 텍셀**만 메운다.
    #     표면을 150만점 흩뿌려도 텍셀이 52만이라 관측부 안에도 구멍이 남는다.
    #     정규화 합성곱 blur(색×있음)/blur(있음) 로 메우되, sigma 를 작게 잡아
    #     큰 결손 호까지 번지지 않게 한다.
    from scipy.ndimage import gaussian_filter
    _m = ("nearest", "wrap")
    wgt = gaussian_filter(have.astype(np.float32), 1.2, mode=_m)
    acc2 = gaussian_filter(img.astype(np.float32) * have[:, :, None], (1.2, 1.2, 0),
                           mode=(_m[0], _m[1], "nearest"))
    tiny = (~have) & (wgt > 0.35)
    img[tiny] = np.clip(acc2[tiny] / wgt[tiny, None], 0, 255).astype(np.uint8)
    have = have | tiny

    # 그리고 **관측 텍셀 자체의 산탄잡음**을 여기서 뺀다.
    # 텍셀 52만에 표본 150만이면 칸당 3점 — 어느 면이 맞았느냐로 색이 튄다.
    # 이걸 두고 타일링하면 그 잡음까지 복사돼 국소 편차가 관측의 1.8배가 됐다.
    # 결손부 전체를 흐리는 것(= 단색)과 달리, 여기는 **관측 안에서만** 다듬는다.
    if clean > 0:
        w2 = gaussian_filter(have.astype(np.float32), clean, mode=_m)
        a2 = gaussian_filter(img.astype(np.float32) * have[:, :, None], (clean, clean, 0),
                             mode=(_m[0], _m[1], "nearest"))
        ok2 = have & (w2 > 1e-3)
        img[ok2] = np.clip(a2[ok2] / w2[ok2, None], 0, 255).astype(np.uint8)
    print("  관측 속 빈 텍셀 %d 메움 → 관측 %.0f%% · 산탄잡음 정리 sigma=%.1f"
          % (int(tiny.sum()), 100 * have.mean(), clean))

    # --- 행마다 회전 복사 = **거울 타일링**
    #
    # 전에는 가까운 도너 k개의 중앙값을 썼다. 그러면 결손 호에서 먼 텍셀도
    # **같은 가장자리 k개**만 보게 돼 호 전체가 한 색으로 눕는다 —
    # 실제로 "단색에 가깝고 옆과 안 이어진다"는 결과가 나왔다.
    # 중앙값은 튀는 값도 지우지만 **무늬도 같이 지우는** 연산이다.
    #
    # 회전체의 무늬는 θ 방향으로 주기적이다. 그래서 관측 띠 `seg` 를 그대로
    # 둘레에 붙여 나간다. 한 주기를 `seg + seg[::-1]`(거울)로 잡으면 이어붙인
    # 자리의 양쪽 값이 같아 **경계가 끊기지 않는다** (reflect 패딩과 같은 원리).
    # 무늬의 잔결이 그대로 살아 옆과 같은 질감이 된다.
    for i in range(tex_h):
        row = have[i]
        if row.all():
            continue
        if not row.any():
            n_row += 1
            continue
        # 가장 긴 연속 관측 구간. θ 는 순환이라 두 바퀴에서 찾는다
        d2 = np.concatenate([row, row])
        best_len = best_a = run = 0
        for j in range(2 * tex_w):
            run = run + 1 if d2[j] else 0
            if run > best_len:
                best_len, best_a = run, j - run + 1
        a0, L = best_a % tex_w, min(best_len, tex_w)
        seg = img[i, (a0 + np.arange(L)) % tex_w]
        per = np.concatenate([seg, seg[::-1]])               # 거울 한 주기 (2L)
        full = np.tile(per, (int(np.ceil(tex_w / len(per))), 1))[:tex_w]
        cols = (a0 + np.arange(tex_w)) % tex_w
        put = ~row[cols]                                     # 관측은 건드리지 않는다
        img[i, cols[put]] = full[put]
        n_rot += int(put.sum())
    # --- **밝기만 이음새에 맞춘다.** 무늬는 건드리지 않는다.
    #
    # 거울 타일링은 무늬를 살리지만 **얼룩까지 그대로 옮긴다.** 이 유물은
    # 한쪽이 밝고 반대쪽이 어둡다 — 같은 높이 안에서도 각도별 밝기 표준편차가
    # 11.4 (전체 밝기 118 대비 10%) 다. 그래서 복사해 온 자리의 톤이
    # 옆과 10% 어긋난다 (칸 기준 실측 -11 ~ +12).
    #
    # 얼룩은 회전대칭이 아니니 **복원할 수 없다.** 대신 결손 호의 양 끝에서
    # 관측 밝기를 읽어 그 사이를 매끄럽게 잇는다 — 가장 약한 가정이다.
    # 무늬(고주파)는 복사본 그대로 두고 **저주파 밝기만** 비율로 맞춘다.
    if blend > 0:
        from scipy.ndimage import gaussian_filter1d as _gb
        n_bl = 0
        for i in range(tex_h):
            row = have[i]
            if not row.any() or row.all():
                continue
            g = img[i].astype(np.float64).mean(1)
            idx = np.nonzero(row)[0]
            # 관측 밝기를 둘레에 따라 순환 보간 → 결손부의 "있어야 할" 톤
            tgt = np.interp(np.arange(tex_w), idx, g[idx], period=tex_w)
            tgt = _gb(tgt, blend, mode="wrap")
            cur = _gb(g, blend, mode="wrap")
            k = np.clip(tgt / np.maximum(cur, 1.0), 0.65, 1.55)
            k[row] = 1.0                       # 관측 텍셀은 절대 안 건드린다
            img[i] = np.clip(img[i].astype(np.float64) * k[:, None],
                             0, 255).astype(np.uint8)
            n_bl += 1
        print("  밝기 이음: %d행 · 저주파 sigma %.0f (무늬는 그대로)" % (n_bl, blend))

    # --- 통째로 빈 행(둘레 전체 결손)은 가장 가까운 채워진 행에서
    filled_rows = np.nonzero(have.any(axis=1))[0]
    if len(filled_rows):
        for i in range(tex_h):
            if not have[i].any():
                img[i] = img[filled_rows[np.argmin(np.abs(filled_rows - i))]]
    if denoise > 0:
        # 점잡음이 남으면 약하게 흐린다. **기본은 0** — 타일링은 무늬를 살리는 것이
        # 목적인데 여기서 흐리면 도로 단색이 된다. 관측부 구멍은 위에서 이미 메웠다.
        img = gaussian_filter(img.astype(np.float32),
                              (denoise, denoise, 0), mode=("nearest", "wrap", "nearest")
                              ).astype(np.uint8)
    print("  회전 복사 텍셀 %d (거울 타일링) · 도너 없는 행 %d · 잡음제거 sigma=%.1f"
          % (n_rot, n_row, denoise))

    def uv_of(nh_, nt_):
        """격자 정점 (i, j) → UV. grid_mesh 의 정점 순서와 같아야 한다."""
        hs = 0.5 * (h_edges[:-1] + h_edges[1:])
        v = np.clip((hs - lo) / max(hi - lo, 1e-9), 0, 1)
        u = (np.arange(nt_) + 0.5) / nt_
        UU, VV = np.meshgrid(u, v)                       # (nh, nt)
        return np.stack([UU.ravel(), VV.ravel()], 1)

    return Image.fromarray(img), uv_of


def force_double_sided(glb: Path) -> None:
    """내보낸 GLB 의 **모든 재질을 양면으로** 바꾼다.

    왜 후처리인가
      - `region_observed` 는 원본 A_normal.glb 의 재질을 그대로 쓴다.
        그게 `doubleSided=False` 라 **안에서 보면 뚫려 보인다.**
      - `region_ai` 를 정점색(ColorVisuals)으로 내보내면 **재질 항목이 아예 안 생긴다.**
        PBRMaterial 을 쓸 때 걸어둔 doubleSided 가 여기서 떨어졌다.

      trimesh 쪽에서 둘 다 고치기 어려워 GLB JSON 청크를 직접 손본다.
      재질이 없는 프리미티브에는 기본 재질을 하나 붙인다.

    메시가 두께 없는 한 겹 면인 한 이 설정이 필요하다.
    """
    import json
    import struct

    b = bytearray(glb.read_bytes())
    jlen = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(bytes(b[20:20 + jlen]).decode("utf-8"))

    mats = js.setdefault("materials", [])
    for m in mats:
        m["doubleSided"] = True

    # 재질 없는 프리미티브에는 **새 민짜 재질**을 붙인다.
    #   기존 재질을 돌려쓰면 안 된다 — 한 번 그렇게 했다가
    #   `region_ai`(UV 없음, 정점색)가 `region_observed` 의 **텍스처 재질**을 물려받았다.
    #   UV 가 없는데 baseColorTexture 를 참조하니 뷰어마다 결과가 달라진다.
    n_fix = 0
    plain_i = None
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" in prim:
                continue
            if plain_i is None:
                mats.append({"name": "region_ai_vertexcolor", "doubleSided": True,
                             "pbrMetallicRoughness": {
                                 "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                                 "metallicFactor": 0.0, "roughnessFactor": 0.9}})
                plain_i = len(mats) - 1
            prim["material"] = plain_i
            n_fix += 1

    raw = json.dumps(js, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((4 - len(raw) % 4) % 4)                 # 4바이트 정렬
    tail = bytes(b[20 + jlen:])                            # BIN 청크 그대로
    out = bytearray(b[:12])
    struct.pack_into("<I", out, 0, 0x46546C67) if False else None
    out += struct.pack("<II", len(raw), 0x4E4F534A) + raw + tail
    struct.pack_into("<I", out, 8, len(out))               # 전체 길이 갱신
    glb.write_bytes(bytes(out))
    print("  양면 설정: 재질 %d개 · 재질 없던 프리미티브 %d개 보정" % (len(mats), n_fix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORK / "pred_71489_A_normal_v2.npz"))
    ap.add_argument("--orig", type=Path, default=SRC_GLB)
    ap.add_argument("--nh", type=int, default=64)
    ap.add_argument("--nt", type=int, default=96, help="파이프라인이 96x144 를 쓴다")
    ap.add_argument("--band-h", type=int, default=10, help="n-fold 띠 두께(칸)")
    ap.add_argument("--outer-shift", type=float, default=0.0,
                    help="""칸 중앙값 R 을 **바깥벽까지** 미는 세기 (1.0 = 전부).

    한 칸에 바깥벽·안쪽벽이 같이 들어가 중앙값이 벽 한가운데에 떨어진다.
    그대로 두면 복원면이 관측면보다 4mm 파묻힌다.""")
    ap.add_argument("--density-keep", type=float, default=0.70,
                    help="""모델이 새로 만든 점 중 **국소 밀도 상위 이만큼만** 쓴다 (0~1).

    빽빽한 심만 남기고 흩뿌려진 후광을 버린다. 1.0 이면 끈다.""")
    ap.add_argument("--min-piece", type=int, default=300,
                    help="""이만큼보다 작은 조각은 날린다 (면 수).

    껍질이라 한 조각이 바깥벽·안쪽벽 둘로 세어진다. 수백 면 미만은 부스러기다.""")
    ap.add_argument("--drop-loose", type=float, default=6.0,
                    help="""관측에 **안 닿는 조각**을 날린다. 관측 점간격의 몇 배까지를 닿았다고 볼지.

    0 이면 끈다. 진짜 결손을 메운 조각은 가장자리가 관측면에 닿는다 —
    안 닿으면 근거가 없는 허공의 파편이다.""")
    ap.add_argument("--fix-strut", action="store_true", default=True,
                    help="""비정상적으로 큰 투창 창에서 **깨진 가로살**을 찾아 결손으로 되돌린다.

    단(tier) 사이는 살이 있어야 하는데, 살이 깨지면 위아래 창이 이어져 한 덩어리가 된다.
    그대로 두면 투창으로 보호되어 영영 안 복원된다.""")
    ap.add_argument("--no-fix-strut", dest="fix_strut", action="store_false")
    ap.add_argument("--strut-ratio", type=float, default=1.6,
                    help="정상 창 높이폭 중앙의 몇 배를 넘으면 '깨진 살'로 볼지")
    ap.add_argument("--strut-min", type=int, default=20,
                    help="이만큼보다 작은 덩어리는 잡티로 보고 창 판정에서 뺀다")
    ap.add_argument("--openwork-hmax", type=float, default=0.55,
                    help="""이 높이 위로는 **투창 판정을 하지 않는다** (0~1, 1.0 이면 끔).

    투창은 굽다리에만 있다. 주기 검사만 쓰면 몸통 결손이 우연히 주기적으로 보여
    투창으로 잡히고, 그러면 **보호되어 영영 안 메워진다.**
    굽다리바리 형식에 기댄 가정이므로 몸통에 투창이 있는 기종에는 꺼야 한다.""")
    ap.add_argument("--mask-close", type=int, default=2,
                    help="결손 마스크의 잔구멍을 메워 조각을 잇는다 (닫기 반복 수, 0 이면 끔)")
    ap.add_argument("--min-island", type=int, default=40,
                    help="이만큼보다 작은 외딴 섬은 버린다 (칸 수). 근거가 약하고 조각만 남긴다")
    ap.add_argument("--density-bands", type=int, default=24,
                    help="""밀도 임계를 **높이 띠마다 따로** 잡는다. 0/1 이면 물체 전체 하나.

    전체 하나로 잡으면 아가리 위 연장부가 몰살당한다 — 거기는 원래 점이 적다.""")
    ap.add_argument("--density-cap", type=float, default=0.92,
                    help="띠별 임계와 별개로 거는 **전역 상한** 백분위. 띠 전체가 후광인 경우를 막는다")
    ap.add_argument("--max-scatter", type=float, default=0.07,
                    help="""모델이 만든 점의 **면 퍼짐** 임계. 넘으면 메시를 안 만든다.

    실측으로 0.061(71489) 과 0.077(고적_13472) 사이가 비어 있어 0.07 로 잡았다.
    자세한 근거는 surface_scatter() 주석.""")
    ap.add_argument("--no-scatter-gate", action="store_true",
                    help="퍼짐 게이트를 끄고 그래도 만든다")
    ap.add_argument("--dump-mask", default=None, metavar="NPZ",
                    help="""(h,θ) 격자 마스크를 **파이프라인이 쓰는 그대로** 내보낸다.

    진단용. 밖에서 dense_occ 를 다시 만들면 정규화 한 줄만 달라도 딴 값이 나온다 —
    실제로 openwork_cells() 에 빈칸 대신 점유 마스크를 넘겨 "투창 0칸" 이라고
    잘못 적은 적이 있다. **진단은 파이프라인 안에서 떠서 밖에서 재지 않는다.**""")
    ap.add_argument("--outer-cov-lo", type=float, default=0.30,
                    help="theta 점유가 이보다 낮으면 관측 대신 **모델**을 목표로 쓴다")
    ap.add_argument("--outer-cov-hi", type=float, default=0.55,
                    help="theta 점유가 이보다 높으면 관측을 그대로 믿는다")
    ap.add_argument("--axisym", type=float, default=0.0,
                    help="""결손부를 **회전체로 되돌리는 세기** (0~1).

    토기는 물레 성형 회전체라 결손부에서 각도마다 반지름이 다를 근거가 없다.
    모델 출력의 theta 변동은 잡음이다 — 실측 변동계수 관측 0.031 · AI 0.052.
    이음새에서는 관측을 따라가야 하므로 거리에 따라 올린다 (--axisym-ramp).""")
    ap.add_argument("--axisym-ramp", type=float, default=6.0,
                    help="이음새에서 몇 칸에 걸쳐 회전체 가중을 최대로 올릴지")
    ap.add_argument("--seam-tuck-ramp", type=float, default=3.0,
                    help="이음새에서 몇 칸까지 안으로 밀지")
    ap.add_argument("--seam-tuck", type=float, default=0.0,
                    help="""이음새 근처에서 패치를 **안쪽으로 밀어 넣는 양** (벽두께 대비).

    패치 테두리와 관측 조각의 실제 가장자리는 격자 한 칸(2~5mm)까지 어긋난다.
    팽창을 키워 겹치게 만들면 틈은 막히지만 패치가 관측면을 뚫고 나온다.
    조금 안으로 밀어 **관측 껍질 밑으로 깔리게** 한다.""")
    ap.add_argument("--dilate", type=int, default=1,
                    help="결손 칸을 이만큼 부풀려 관측면과 겹치게 한다 (이음새)")
    ap.add_argument("--smooth", type=int, default=12,
                    help="패치 평활 반복. 경계는 고정된다. 0 이면 끔")
    ap.add_argument("--match-texture", action="store_true",
                    help="패치 색을 주변 관측 텍스처에서 가져온다 (정점색)")
    ap.add_argument("--donor-k", type=int, default=8,
                    help="(안 쓴다) 거울 타일링으로 바꾸면서 도너 수 개념이 사라졌다")
    ap.add_argument("--no-tex-outer", dest="tex_outer", action="store_false",
                    help="""텍스처에서 **바깥면 거르기를 끈다** (v6~v9 의 옛 동작).

    두께 있는 껍질은 한 (h, theta) 텍셀에 바깥벽과 안쪽벽이 같이 들어간다.
    안쪽은 그늘져 어둡다 — 실측 표본의 52%가 안쪽벽이었다.
    이 플래그는 **옛 버전을 그대로 재현하려고** 남겨 둔다.""")
    ap.add_argument("--tex-blend", type=float, default=0.0,
                    help="""결손부 **밝기만** 이음새에 맞추는 저주파 sigma (0 이면 끔).

    얼룩은 회전대칭이 아니라 복사해 오면 톤이 어긋난다. 무늬는 그대로 두고
    결손 호 양 끝의 관측 밝기 사이를 매끄럽게 잇는다.""")
    ap.add_argument("--tex-samples", type=int, default=1500000,
                    help="텍스처용 표면 표본 수. 텍셀당 표본이 많을수록 산탄잡음이 준다")
    ap.add_argument("--tex-clean", type=float, default=0.8,
                    help="관측 텍셀 산탄잡음 정리 sigma. 결손부가 아니라 **관측 안**만 다듬는다")
    ap.add_argument("--tex-denoise", type=float, default=0.0,
                    help="""텍스처 점잡음 제거 sigma. **기본 0.**

    관측 속 빈 텍셀은 정규화 합성곱으로 따로 메우므로 전역 블러가 필요 없다.
    여기를 올리면 거울 타일링으로 살려 둔 무늬가 도로 뭉개져 단색이 된다.""")
    ap.add_argument("--thickness", action="store_true",
                    help="관측 벽 두께만큼 속을 채운다 (안쪽면은 매끈)")
    ap.add_argument("--mask-smooth", type=float, default=0.0,
                    help="결손 마스크 테두리를 깎는다 (격자 계단 제거). 1.0 권장")
    ap.add_argument("--edge-rings", type=int, default=0,
                    help="0=전체 평활. >0 이면 자유 테두리에서 그 고리까지만")
    ap.add_argument("--ridge", action="store_true",
                    help="관측 프로파일의 요철(돌대)을 결손부로 잇는다")
    ap.add_argument("--r-smooth", type=float, default=0.0,
                    help="격자 R 을 다듬는다 (모델 출력 잡음 제거)")
    ap.add_argument("--ridge-wide", type=float, default=4.0,
                    help="대역통과 윗폭 = sigma x 이 값")
    ap.add_argument("--ridge-gain", type=float, default=1.0,
                    help="요철 세기")
    ap.add_argument("--ridge-sigma", type=float, default=2.0,
                    help="저주파/고주파를 가르는 폭(격자 칸)")
    ap.add_argument("--rot-texture", action="store_true",
                    help="**축 둘레 회전 복사**로 UV+텍스처를 붙인다 (정점색보다 낫다)")
    ap.add_argument("--flat-observed", action="store_true",
                    help="관측부도 단색으로. 기본은 **원본 텍스처 유지**")
    ap.add_argument("--name", default="A_normal_2node")
    args = ap.parse_args()

    d = np.load(Path(args.src), allow_pickle=True)
    P = np.asarray(d["pred"], np.float64)
    new = np.asarray(d["new"], bool)
    dense = np.asarray(d["input_dense"], np.float64)
    up = int(d["up"])

    orig = load_original(args.orig)
    cen, scl = norm_params(orig)
    print("원본 메시 V=%d F=%d  (무수정으로 내보낸다)" % (len(orig.vertices), len(orig.faces)))

    mg = M.load_measure_glb()
    _, c2 = mg.fit_axis(P, up)

    # --- **밀도로 후광을 걷어낸다.**
    #
    # 모델 출력을 보면 빽빽한 심(心)과 흩뿌려진 후광이 나뉜다.
    # 후광까지 격자에 넣으면 그 칸의 반지름이 밖으로 끌려간다 —
    # 경주_고적_13472 의 **없는 넓은 챙**이 그렇게 생겼다.
    #
    # 국소 밀도(k번째 최근접까지 거리)로 상위 일부만 남긴다.
    # 실측: 상위 70% 만 남기면 퍼짐이
    #     고적_13472  0.0765 → 0.0514    (임계 아래로 내려온다)
    #     71489       0.0511 → 0.0383
    # **관측과 겹치는 점은 안 건드린다.** 걷어내는 것은 모델이 새로 만든 점뿐이다.
    if 0 < args.density_keep < 1 and new.sum() >= 200:
        from scipy.spatial import cKDTree as _KD2
        Q = P[new]
        kk = min(12, len(Q) - 1)
        dist, _ = _KD2(Q).query(Q, k=kk + 1)
        dk = dist[:, kk]

        # **임계를 높이 띠마다 따로 잡는다.**
        #
        # 물체 전체에 하나의 임계를 쓰면 **아가리 위로 뻗은 끝단이 몰살당한다.**
        # 거기는 본래 점이 적다 — 후광이라서가 아니라 구조상 그렇다.
        # 실측(71489 v3): 전체 새 점은 70% 살아남는데 관측 위쪽 연장부는
        # **5%**(311 → 17점)만 남았고, 연장 높이가 +17.8mm → +2.6mm 로 잘렸다.
        # 결과물이 모델 능력의 절반만 보여주고 있었다.
        #
        # 띠마다 백분위를 따로 잡으면 "위쪽은 원래 성기다"가 후광으로 안 읽힌다.
        # 몸통처럼 밀도가 고른 구간은 값이 거의 안 바뀐다 — 바뀌는 것은 끝단뿐이다.
        #
        # **안전장치**: 띠 전체가 후광이면(그 높이에 성긴 구름만 있으면)
        # 띠 안에서는 상대적으로 조밀해 보여 70%가 살아남는다.
        # 그래서 전역 상한(`--density-cap` 백분위)을 같이 걸어 최악을 막는다.
        thr = np.full(len(Q), np.inf)
        if args.density_bands >= 2:
            hq = Q[:, up]
            lo_q, hi_q = hq.min(), hq.max()
            bb = np.clip(((hq - lo_q) / max(hi_q - lo_q, 1e-9)
                          * args.density_bands).astype(int), 0, args.density_bands - 1)
            for b in range(args.density_bands):
                m = bb == b
                if m.sum() >= 40:
                    thr[m] = np.percentile(dk[m], 100 * args.density_keep)
                elif m.any():
                    thr[m] = np.inf            # 표본이 적은 띠는 안 거른다
        else:
            thr[:] = np.percentile(dk, 100 * args.density_keep)
        cap = np.percentile(dk, 100 * args.density_cap)
        bad = (dk > thr) | (dk > cap)
        drop = np.zeros(len(P), bool)
        drop[np.nonzero(new)[0][bad]] = True
        print("  밀도 거르기: 새 점 %d 중 %d 버림 (띠 %d개 · 띠별 상위 %.0f%% · 전역 상한 %.0f%%)"
              % (int(new.sum()), int(drop.sum()), args.density_bands,
                 100 * args.density_keep, 100 * args.density_cap))
        P, new = P[~drop], new[~drop]

    # --- **면 퍼짐 게이트** — 모델 출력이 면을 못 이뤘으면 메시로 만들지 않는다.
    #
    # 메시화는 들어온 것을 성실하게 형상으로 바꾼다. 구름이 들어오면
    # **없는 형상을 자신 있게 내놓는다** (경주_고적_13472 의 넓은 챙).
    # 점 수로는 못 가른다 — 그 유물이 4,854점으로 가장 많이 만든 축이었다.
    # **밀도 거르기 뒤에** 잰다 — 걷어내고도 구름이면 그때는 진짜 못 만드는 것이다.
    sc = surface_scatter(P, new, up, c2)
    if sc is None:
        print("  면 퍼짐: 잴 수 없다 (새 점이 적다) — 게이트 통과")
    else:
        verdict = "얇은 면" if sc <= args.max_scatter else "**구름**"
        print("  면 퍼짐 %.4f (임계 %.3f) → %s" % (sc, args.max_scatter, verdict))
        if sc > args.max_scatter and not args.no_scatter_gate:
            print(chr(10) + "[거부] 모델이 만든 점이 면을 이루지 못했다.")
            print("  메시로 만들면 **근거 없는 형상**이 나온다. 실제로 경주_고적_13472 가")
            print("  파단선을 둘레로 돌려 없는 넓은 챙을 만들었다.")
            print("  이 유물은 학습 분포 밖일 가능성이 크다 (몸통 대부분 결실).")
            print("  그래도 만들려면 --no-scatter-gate")
            return 2

    h, th, r, ax = M.cyl(P, up, c2)
    hd, td, _, _ = M.cyl(dense, up, c2)

    nh, nt = args.nh, args.nt
    lo, hi = np.percentile(h, [0.5, 99.5])
    he = np.linspace(lo, hi, nh + 1)

    def cells(hh, tt):
        return (np.clip(np.digitize(hh, he) - 1, 0, nh - 1),
                np.clip(((tt + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1))

    hb, tb = cells(h, th)
    hbd, tbd = cells(hd, td)
    dense_occ = np.zeros((nh, nt), bool)
    dense_occ[hbd, tbd] = True

    ow, per, info = M.openwork_cells_nfold(dense_occ, band_h=args.band_h)

    # --- **투창은 굽다리에만 있다.** 그 위의 투창 판정은 과탐지다.
    #
    # 주기 검사만 쓰면 몸통의 결손도 우연히 주기적으로 보일 수 있다.
    # 실측: 경주_고적_13472 는 h **0.90**(몸통 꼭대기)까지, 71489 는 0.65 까지
    # 투창으로 잡혔다. 그 칸은 **보호되어 영영 안 메워진다** —
    # 결손을 투창으로 오판하는 쪽이 반대보다 조용해서 더 위험하다.
    #
    # 굽/바리 이음이 h 0.4~0.5 이므로 그 위는 몸통이고 투창일 수 없다.
    # **다만 이건 굽다리바리 형식에 기댄 가정이다.** 몸통에 투창이 있는 기종에는
    # `--openwork-hmax 1.0` 으로 꺼야 한다.
    if args.openwork_hmax < 1.0:
        cut = int(nh * args.openwork_hmax)
        n_before = int(ow.sum())
        ow[cut:] = False
        if n_before != int(ow.sum()):
            print("  투창 높이 제한 h<%.2f: %d → %d칸 (위쪽 %d칸은 결손으로 되돌림)"
                  % (args.openwork_hmax, n_before, int(ow.sum()), n_before - int(ow.sum())))
    # --- **깨진 살 복원** — 비정상적으로 큰 창은 투창이 아니라 살이 깨진 것이다.
    #
    # 투창은 단(tier)으로 나뉘고 각 단의 창은 크기가 고른다.
    # 실측(71489): 정상 창 7개가 높이폭 0.08~0.11 인데 하나가 **0.20 · 683칸** 으로
    # 위 단(h 0.27~0.38)과 아래 단(h 0.08~0.19)을 **가로질렀다** — 그 사이 가로살이
    # 깨진 것이다. 그걸 통째로 투창으로 보면 **깨진 살이 영영 안 복원된다.**
    #
    # 정상 창이 한 개도 안 걸치는 행 = 살이 있어야 하는 행이다.
    # 큰 창이 그 행을 지나면 **그 부분만** 결손으로 되돌린다. 창 모양은 안 건드린다.
    #
    # 되돌린 자리는 지어내지 않는다 — **그 높이의 다른 각도에는 살이 멀쩡하다.**
    # 반지름·돌대·색을 회전 복사로 그대로 가져온다.
    if args.fix_strut:
        from scipy import ndimage as _ndi
        lab2, n2 = _ndi.label(np.hstack([ow, ow]))
        wins, seen = [], set()
        for k in range(1, n2 + 1):
            ys, xs = np.nonzero(lab2 == k)
            if len(ys) < args.strut_min:
                continue
            key = (int(ys.min()), int(ys.max()), len(ys))
            if key in seen:
                continue
            seen.add(key)
            wins.append((ys.min(), ys.max(), len(ys), xs % nt))
        if len(wins) >= 4:
            span = np.array([w[1] - w[0] + 1 for w in wins], float)
            med = float(np.median(span))
            normal_rows = np.zeros(nh, bool)
            for (a, b, _n, _x), sp in zip(wins, span):
                if sp <= med * args.strut_ratio:
                    normal_rows[a:b + 1] = True      # 정상 창이 차지하는 행
            back = np.zeros((nh, nt), bool)
            n_win = 0
            for (a, b, _n, xs2), sp in zip(wins, span):
                if sp <= med * args.strut_ratio:
                    continue
                rows = np.arange(a, b + 1)
                bad = rows[~normal_rows[rows]]       # 살이 있어야 하는데 뚫린 행
                if len(bad) == 0:
                    continue
                n_win += 1
                for rr in bad:
                    back[rr, np.unique(xs2)] = True
            back &= ow
            if back.any():
                ow = ow & ~back
                print("  깨진 살 복원: 큰 창 %d개에서 %d칸을 결손으로 되돌림 "
                      "(정상 창 높이폭 중앙 %.0f행)" % (n_win, int(back.sum()), med))

    damage = (~dense_occ) & (~ow)          # 결손 = 빈 칸 중 투창이 아닌 것
    print("격자 %dx%d · 빈칸 %d = 투창 %d + **결손 %d**"
          % (nh, nt, (~dense_occ).sum(), ow.sum(), damage.sum()))
    ys = np.nonzero(per)[0]
    if len(ys):
        print("  주기(n-fold) 띠: h %.2f~%.2f" % (ys.min() / nh, ys.max() / nh))
        for a, b, top, sh in info:
            if top >= 2 and a % max(1, args.band_h // 2) == 0:
                print("    h %.2f~%.2f  %d-fold  %.0f%%" % (a / nh, b / nh, top, 100 * sh))

    # 이음새를 위해 결손을 한 칸 부풀린다. 투창은 절대 안 건드린다
    grow = damage.copy()
    for _ in range(max(0, args.dilate)):
        g = grow.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g |= np.roll(grow, (dy, dx), (0, 1))
        grow = g & ~ow
    print("  이음새용 %d칸 팽창 → 패치 대상 %d칸" % (args.dilate, grow.sum()))

    # 마스크 테두리의 **격자 계단**을 깎는다.
    #   테두리 거칠기는 정점 위치가 아니라 **마스크 모양**에서 온다 —
    #   정점을 아무리 평활해도 안 없어진다 (실측: edge-rings 0/3/6 이 전부 0.034).
    #   (h, θ) 에서 흐린 뒤 0.5 로 자르면 계단 모서리가 둥글어진다. θ 는 순환.
    if args.mask_smooth > 0:
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(grow.astype(float), args.mask_smooth, mode=("nearest", "wrap"))
        before = int(grow.sum())
        grow = (g > 0.5) & ~ow            # 투창은 절대 안 건드린다
        print("  마스크 평활 sigma=%.1f: %d → %d칸" % (args.mask_smooth, before, grow.sum()))

    # --- **패치를 한 덩어리로 만든다.** 다 채우거나 다 깎거나, 중간은 없다.
    #
    # 결손 마스크에는 잔구멍과 외딴 섬이 섞여 있다. 그대로 메시로 만들면
    # 조각이 흩어진다 — 실측으로 `region_ai` 가 **25~46 조각**이었고
    # 경주_고적_13472 는 가장 큰 조각이 80% 밖에 안 됐다. 눈으로는 **면이 끊겨** 보인다.
    #
    #   ① 닫기(closing)  잔구멍을 메워 가까운 조각을 잇는다
    #   ② 작은 섬 제거    이어지지 않는 몇 칸짜리는 버린다 (근거도 약하다)
    #
    # θ 는 순환이라 좌우로 한 바퀴씩 이어 붙여 처리한 뒤 가운데를 꺼낸다.
    # **투창은 어느 단계에서도 안 건드린다.**
    if args.mask_close > 0 or args.min_island > 0:
        from scipy.ndimage import binary_closing, label
        before = int(grow.sum())
        if args.mask_close > 0:
            # θ 는 순환이라 좌우로, h 는 **가장자리를 복제해서** 위아래로 덧댄다.
            # 안 덧대면 닫기의 침식 단계가 맨 위·아래 행을 깎는다 —
            # 아가리가 바로 맨 위 행이라 테두리가 통째로 날아간다.
            pad = 2 * args.mask_close + 1
            t3 = np.concatenate([grow] * 3, axis=1)
            t3 = np.pad(t3, ((pad, pad), (0, 0)), mode="edge")
            t3 = binary_closing(t3, structure=np.ones((3, 3), bool),
                                iterations=args.mask_close)
            grow = t3[pad:pad + nh, nt:2 * nt] & ~ow
        n_isl = 0
        if args.min_island > 0:
            t3 = np.concatenate([grow] * 3, axis=1)
            lab3, _ = label(t3)
            mid = lab3[:, nt:2 * nt]
            sizes = np.bincount(mid.ravel())
            small = np.nonzero(sizes < args.min_island)[0]
            small = small[small > 0]
            if len(small):
                kill = np.isin(mid, small)
                n_isl = int(kill.sum())
                grow = grow & ~kill
        print("  패치 잇기: 닫기 %d회 · 작은 섬 %d칸 제거 → %d → %d칸"
              % (args.mask_close, n_isl, before, int(grow.sum())))

    if args.dump_mask:
        np.savez_compressed(args.dump_mask, occ=dense_occ, ow=ow, damage=damage,
                            grow=grow, h_edges=he, up=up, c2=c2,
                            cen=cen, scl=scl, nh=nh, nt=nt)
        print("  마스크 덤프 → %s (투창 %d · 결손 %d · 패치 %d)"
              % (args.dump_mask, int(ow.sum()), int(damage.sum()), int(grow.sum())))

    # 패치의 반지름 — 모델 출력에서. 비면 이웃 평균
    R = np.full((nh, nt), np.nan)
    known = np.zeros((nh, nt), bool)
    for i, j in zip(*np.nonzero(grow)):
        m = (hb == i) & (tb == j)
        if m.any():
            R[i, j] = np.median(r[m]); known[i, j] = True
    # 관측 칸의 반지름도 채워 넣어야 패치 가장자리가 관측면에 맞는다
    hbo, tbo = cells(hd, td)
    for i, j in zip(*np.nonzero(grow & ~known)):
        m = (hbo == i) & (tbo == j)
        if m.any():
            R[i, j] = np.median(np.hypot(
                dense[m][:, ax[0]] - c2[0], dense[m][:, ax[1]] - c2[1]))
            known[i, j] = True
    if not known.any():
        print("[!] 패치를 만들 점이 없다.")
        return 1
    R = M.fill_unknown(np.nan_to_num(R, nan=np.nanmean(R[known])), known)
    _stage = {"known": known.copy(), "R0_점에서": R.copy()}

    # --- **R 을 바깥벽에 맞춘다.**
    #
    # 이 유물은 두께 있는 껍질이라 한 칸에 **바깥벽과 안쪽벽**이 같이 들어간다.
    # 칸 중앙값은 그 **둘 사이**에 떨어진다 — 실측으로 바깥벽보다 4.67mm 안쪽이고,
    # 행 50분위와는 0.38mm 밖에 차이가 안 났다. 즉 R 은 사실상 벽 한가운데다.
    #
    # 그대로 껍질을 세우면 복원면이 통째로 파묻힌다 — 실제로 관측면보다
    # **중앙 3.8mm, 위쪽 몸통은 8mm** 안으로 들어가 있었다.
    # 모델 출력 자체는 관측과 **-0.28mm** 밖에 차이가 없다. 모델 탓이 아니다.
    #
    # 고정된 보정량을 더하면 아래는 넘치고 위는 모자란다 (실측 +4.4 ~ -4.0mm).
    # 그래서 **행마다** 목표 바깥 반지름에 맞춘다 —
    # 목표는 그 높이의 관측 90분위(없으면 모델 90분위), 회전체니 그 값이 곧 바깥벽이다.
    if args.outer_shift > 0:
        from scipy.ndimage import gaussian_filter1d as _gs
        # 목표는 **법선이 바깥을 보는 면**의 반지름이다.
        # 분위수(90%)로 잡으면 바깥벽의 위쪽 가장자리를 겨냥하게 돼 1.8mm 넘친다.
        _pts, _fid = trimesh.sample.sample_surface(orig, 400000, seed=11)
        _V = (np.asarray(_pts, np.float64) - cen) / scl
        _fn = np.asarray(orig.face_normals)[_fid]
        _rx, _ry = _V[:, ax[0]] - c2[0], _V[:, ax[1]] - c2[1]
        _r = np.hypot(_rx, _ry)
        _g = _r > 1e-9
        _dot = np.zeros(len(_V))
        _dot[_g] = (_fn[_g, ax[0]] * _rx[_g] + _fn[_g, ax[1]] * _ry[_g]) / _r[_g]
        _k = _dot > 0.2
        _ro, _hbo = _r[_k], np.clip(np.digitize(_V[_k, up], he) - 1, 0, nh - 1)
        _tho = np.arctan2(_ry[_k], _rx[_k])

        # **그 높이에 관측이 둘레의 얼마나 남았는지**를 같이 본다.
        #
        # 관측 r 을 그 높이의 바깥면으로 믿으려면 관측이 둘레에 골고루 있어야 한다.
        # 경주_고적_13472 는 몸통이 거의 다 깨져 h 0.7~0.9 에서 **둘레의 21~31%**
        # 만 남았고, 그 조각은 비스듬한 **파단선의 양 끝**이라 반지름이 크게 나온다.
        # 그 값을 둘레 전체에 적용하니 **있지도 않은 넓은 챙**이 생겼다.
        # (71489 는 139/144 행에서 둘레가 많이 남아 이 결함이 안 드러났다.)
        #
        # 점유가 높으면 관측을, 낮으면 모델을 믿는다. 그 사이는 섞는다.
        obs_r = np.full(nh, np.nan)
        mod_r = np.full(nh, np.nan)
        cov = np.zeros(nh)
        NT_C = 72
        for i in range(nh):
            mo = _hbo == i
            if mo.sum() >= 60:
                obs_r[i] = np.median(_ro[mo])
                tb = np.unique(np.clip(((_tho[mo] + np.pi) / (2 * np.pi) * NT_C
                                        ).astype(int), 0, NT_C - 1))
                cov[i] = len(tb) / NT_C
            mp = hb == i
            if mp.sum() >= 40:
                mod_r[i] = np.percentile(r[mp], 90)

        # 모델의 90분위는 관측 바깥면보다 살짝 바깥이다. 둘 다 믿을 만한 행
        # (점유 높은 행)에서 그 차이를 재서 모델 쪽을 같은 자로 옮긴다.
        trust = (cov >= args.outer_cov_hi) & np.isfinite(obs_r) & np.isfinite(mod_r)
        bias = float(np.median(obs_r[trust] - mod_r[trust])) if trust.sum() >= 4 else 0.0
        mod_adj = mod_r + bias

        w = np.clip((cov - args.outer_cov_lo)
                    / max(args.outer_cov_hi - args.outer_cov_lo, 1e-6), 0, 1)
        w[~np.isfinite(obs_r)] = 0.0
        tgt = np.where(np.isfinite(mod_adj), w * np.nan_to_num(obs_r) + (1 - w) * mod_adj,
                       obs_r)
        n_mod = int((w < 0.5).sum())
        print("  바깥벽 목표: 관측 우선 %d행 · **모델 우선 %d행** (theta 점유 %.0f%% 미만)"
              " · 모델 보정 %+.4f" % (int((w >= 0.5).sum()), n_mod,
                                   100 * args.outer_cov_lo, bias))
        lvl = np.array([np.median(R[i, grow[i]]) if grow[i].any() else np.nan
                        for i in range(nh)])
        ok_s = np.isfinite(tgt) & np.isfinite(lvl)
        if ok_s.sum() >= 6:
            off = np.full(nh, np.nan)
            off[ok_s] = tgt[ok_s] - lvl[ok_s]
            off[~ok_s] = np.interp(np.nonzero(~ok_s)[0], np.nonzero(ok_s)[0], off[ok_s])
            off = _gs(off, 2.0) * args.outer_shift
            R = R + off[:, None]
            print("  바깥벽 정렬: x %.2f · 이동 중앙 %.4f (정규화)"
                  % (args.outer_shift, float(np.median(off))))

    # --- **결손부를 회전체로 되돌린다.**
    #
    # 토기는 물레로 빚은 회전체다. 결손부에서 각도마다 반지름이 다를 근거는
    # **없다** — 모델 출력의 θ 방향 변동은 정보가 아니라 잡음이다.
    # 실측: 행별 반지름 변동계수가 관측 0.0307, AI 0.0519 로 **1.7배** 흔들렸다.
    #
    # 그렇다고 통째로 회전체를 만들면 관측면(0.0307 만큼 흔들림)과 만나는
    # **이음새에서 단이 진다.** 그래서 이음새에서 멀어질수록 회전체에 가깝게
    # 섞는다 — 가까이는 관측을 따라가고, 깊은 곳은 깔끔한 회전체가 된다.
    # 이음새(관측이 살아 있는 칸)까지의 칸 거리 — 회전체 되돌림과 tuck 이 같이 쓴다.
    # θ 는 순환이라 좌우로 한 바퀴씩 이어 붙여 거리를 잰다.
    from scipy.ndimage import distance_transform_edt as _edt
    _obs_cell = dense_occ & ~grow
    if _obs_cell.any():
        seam_dist = _edt(~np.concatenate([_obs_cell] * 3, axis=1))[:, nt:2 * nt]
    else:
        seam_dist = np.full((nh, nt), 1e9)

    _stage["R1_바깥벽정렬"] = R.copy()
    if args.axisym > 0:
        from scipy.ndimage import gaussian_filter1d
        w = np.clip(seam_dist / max(args.axisym_ramp, 1e-6), 0, 1) * args.axisym
        # 행별 **중앙값** — 튀는 칸을 빼고 그 높이의 대표 반지름을 잡는다
        med = np.median(R, axis=1)
        med = gaussian_filter1d(med, 1.0)                 # 높이 방향으로만 살짝
        R = (1.0 - w) * R + w * med[:, None]
        print("  회전체로 되돌림: 최대 %.2f · 이음새에서 %d칸에 걸쳐 올림 "
              "(평균 가중 %.2f)" % (args.axisym, args.axisym_ramp, float(w.mean())))

    _stage["R2_회전체블렌드"] = R.copy()
    if args.r_smooth > 0:
        # **모델 출력 자체가 거칠다.** 격자 R 을 높이 방향으로 먼저 다듬는다.
        # 이걸 안 하면 이면각이 28° 까지 뛴다 (관측 6.0°).
        from scipy.ndimage import gaussian_filter1d as _g1
        R = _g1(R, args.r_smooth, axis=0)
        R = _g1(R, args.r_smooth * 0.6, axis=1, mode='wrap')
        print('  R 격자 평활: h %.1f · theta %.1f' % (args.r_smooth, args.r_smooth*0.6))
    _stage["R3_평활"] = R.copy()
    if args.ridge:
        R = ridge_transfer(R, known, dense_occ, dense, up, c2, he, nh, nt,
                           sigma=args.ridge_sigma, wide=args.ridge_wide,
                           gain=args.ridge_gain)

    _stage["R4_돌대이식"] = R.copy()
    if args.dump_mask:
        np.savez_compressed(str(args.dump_mask).replace(".npz", "_R.npz"), **_stage)
        print("  R 단계 덤프 → %s" % str(args.dump_mask).replace(".npz", "_R.npz"))

    band_v = None
    if args.thickness:
        thick = wall_thickness(orig, cen, scl, up, c2, he, nh, nt)
        # --- 이음새를 **관측 껍질 밑으로 깔아** 틈을 막는다
        #
        # 패치 테두리는 144x216 격자에 갇혀 있고 관측 조각의 진짜 가장자리는
        # 그보다 잘다. 그래서 둘 사이가 어긋난다 — 실측 **중앙 2.5mm · 최대 5.1mm**.
        #
        # 팽창(`--dilate`)만 키우면 틈은 줄지만 패치가 관측면을 뚫고 나온다.
        # 이음새 가까운 칸의 반지름을 벽두께의 일부만큼 **안으로** 밀면
        # 겹친 부분이 관측 밑으로 들어가 밖에서는 관측만 보인다.
        if args.seam_tuck > 0 and thick is not None:
            near = np.clip(1.0 - seam_dist / max(args.seam_tuck_ramp, 1e-6), 0, 1)
            R = R - near * args.seam_tuck * thick[:, None]
            print("  이음새 tuck: 벽두께의 %.2f배만큼 %d칸까지 안으로 "
                  "(영향 칸 %d)" % (args.seam_tuck, args.seam_tuck_ramp,
                                  int((near > 0).sum())))
        if thick is None:
            print("  [건너뜀] 두께를 못 쟀다 — 한 겹 면으로 간다")
            V, F, cell = M.grid_mesh(he, nt, R, up, c2, ax, skip=~grow)
        else:
            V, F, cell, band_v = grid_shell(he, nt, R, thick, up, c2, ax, grow,
                                            seam_cell=(grow & dense_occ))
            print("  두께 있는 껍질: 정점 %d · 면 %d (바깥+안쪽+테두리)" % (len(V), len(F)))
    else:
        V, F, cell = M.grid_mesh(he, nt, R, up, c2, ax, skip=~grow)
    if len(F) == 0:
        print("[!] 패치 면이 없다.")
        return 1

    tri = V[F]; cn = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    rad = np.zeros_like(cn)
    rad[:, ax[0]] = cn[:, ax[0]] - c2[0]; rad[:, ax[1]] = cn[:, ax[1]] - c2[1]
    F[(nrm * rad).sum(1) < 0] = F[(nrm * rad).sum(1) < 0][:, ::-1]

    # --- **관측에 안 붙은 조각은 날린다.** (메시를 만들기 **전에** F 를 줄인다)
    #
    # 결손이 흩어져 있으면 패치도 여러 조각이 된다 — 그건 정상이다.
    # 구멍마다 따로 만들어지니까. 문제는 **어디에도 안 붙어 허공에 뜬 조각**이다.
    # 실측(경주_고적_13472): 조각 30개 중 가장 큰 것이 84% 고, 나머지가
    # 굽다리 언저리에 파편처럼 떠 있었다.
    #
    # 진짜 결손을 메운 조각은 **가장자리가 관측면에 닿는다.** 안 닿으면 근거가 없다.
    #
    # UV 는 뒤에서 `keep_v` 로 같은 순서로 솎으므로, **F 를 여기서 줄여야** 어긋나지 않는다.
    if (args.drop_loose > 0 or args.min_piece > 0) and len(F) > 0:
        import trimesh.graph as _tg
        from scipy.spatial import cKDTree as _KD3
        _tmp = trimesh.Trimesh(vertices=V * scl + cen, faces=F, process=False)
        comps = _tg.connected_components(_tmp.face_adjacency,
                                         nodes=np.arange(len(F)))
        if len(comps) > 1:
            ov = np.asarray(orig.vertices)
            tree = _KD3(ov)
            nn, _ = tree.query(ov[:: max(1, len(ov) // 3000)], k=2)
            thr = args.drop_loose * float(np.median(nn[:, 1]))
            fkeep = np.zeros(len(F), bool)
            n_drop = n_face = 0
            for c in comps:
                dd, _ = tree.query(_tmp.vertices[np.unique(F[c])])
                # ① 관측에 닿나  ② 조각이 의미 있는 크기인가
                #   껍질이라 조각 하나가 바깥벽·안쪽벽 **둘로** 세어진다.
                #   실측 크기 분포가 41256 / 1804 / 692·692 / 636·636 / ... / 10
                #   처럼 갈려서, 수백 면 미만은 메시화 부스러기다.
                if dd.min() <= thr and len(c) >= args.min_piece:
                    fkeep[c] = True
                else:
                    n_drop += 1; n_face += len(c)
            if n_drop and fkeep.any():
                F = F[fkeep]
                print("  조각 날리기: %d/%d 조각 · %d면 버림 (관측 미접촉 또는 %d면 미만)"
                      " → 남은 면 %d" % (n_drop, len(comps), n_face,
                                       args.min_piece, len(F)))
            else:
                print("  뜬 조각 날리기: 버릴 것 없음 (조각 %d)" % len(comps))

    patch = trimesh.Trimesh(vertices=V * scl + cen, faces=F, process=False)
    band_keep = band_v if 'band_v' in dir() else None
    keep_v = np.zeros(len(V), bool)
    keep_v[np.unique(F)] = True          # UV 를 같은 순서로 솎으려면 이게 필요하다
    patch.remove_unreferenced_vertices()
    if args.smooth:
        patch = smooth_keep_border(patch, iters=args.smooth, orig=orig,
                                  edge_rings=args.edge_rings,
                                  free_seed=(band_keep[keep_v]
                                             if band_keep is not None else None))

    rgb = M.artifact_color(args.orig)
    if args.rot_texture:
        img, uv_of = rotation_copy_texture(orig, cen, scl, up, c2, he, nh, nt,
                                           donor_k=args.donor_k,
                                           denoise=args.tex_denoise,
                                           clean=args.tex_clean,
                                           n_sample=args.tex_samples,
                                           outer=args.tex_outer,
                                           blend=args.tex_blend)
        uv_all = uv_of(nh, nt)                      # grid_mesh 정점 순서와 같다
        if len(uv_all) < len(keep_v):               # 껍질이면 바깥+안쪽
            uv_all = np.vstack([uv_all, uv_all])
        patch.visual = trimesh.visual.TextureVisuals(
            uv=uv_all[keep_v],
            material=trimesh.visual.material.PBRMaterial(
                name="region_ai", baseColorTexture=img,
                metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True))
        vcol = None
    else:
        vcol = transfer_color(patch, orig) if args.match_texture else None
    if vcol is not None:
        patch.visual = trimesh.visual.ColorVisuals(patch, vertex_colors=vcol)
    elif not args.rot_texture:
        patch.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="region_ai", baseColorFactor=[c / 255 for c in rgb] + [1.0],
                metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True))

    obs = orig.copy()
    if args.flat_observed:
        obs.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="region_observed", baseColorFactor=[c / 255 for c in rgb] + [1.0],
                metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True))

    OUT.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(obs, geom_name="region_observed")
    scene.add_geometry(patch, geom_name="region_ai")
    glb = OUT / (args.name + ".glb")
    scene.export(str(glb))
    force_double_sided(glb)

    print("\n  region_observed  V=%6d F=%6d  **원본 무수정** (%s)"
          % (len(obs.vertices), len(obs.faces),
             "단색" if args.flat_observed else "원본 텍스처"))
    print("  region_ai        V=%6d F=%6d  %s"
          % (len(patch.vertices), len(patch.faces),
             "**UV + 회전 복사 텍스처**" if args.rot_texture
             else ("주변 텍스처 색 이식(정점색)" if vcol is not None
                   else "단색 RGB %s" % rgb)))
    print("\n저장 → " + str(glb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
