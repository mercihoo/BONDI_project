"""방안 ④ — 채움을 SLAT 이 아니라 원통 맵에서 **메시로 직접** 만든다.

왜 바꾸나
---------
v1~v11 에서 조각 수가 22 아래로 내려가지 않았다(손상본은 2). 원인은 SLAT 이다 —
O-Voxel 은 복셀 *안*의 표면 위치를 잠재값이 정하므로 이웃 잠재값이 어긋나면
복셀이 붙어 있어도 면이 안 만난다. 채움 설정 9가지가 전부 복셀 26-연결 1덩어리인데도
메시는 갈라졌다. 조건 이미지를 바꾸면 25 → 168 까지 흔들린다(v11).

`(y, θ)` 격자에서 메시를 직접 만들면 **인접 칸이 정점을 공유하므로 연속성이 정의상 보장**된다.
이 실패 양상 자체가 없다.

동시에 풀리는 것
----------------
- 원본 보존: 관측 메시를 **한 톨도 안 건드린다** (v8 의 2.7e-04 보다 강한 0)
- provenance: `region_carried` / `region_filled` 가 **애초에 별도 노드**다 → `NFR-ETH-003`

쓰지 않는 것
------------
coords 주입도, SLAT 도, 조건 이미지도 쓰지 않는다. TRELLIS 는 A단계에서
손상 메시와 그 PBR 재질을 만드는 역할로만 남는다.

사용:
  python build_mesh_direct.py --glb out/probe/A_normal.glb --out out/v12-cyl+fwd+nn
"""

import sys
import argparse
import json
import os
import struct

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

import skeleton_cyl as S

# 콘솔이 cp949 면 em dash(U+2014) 같은 문자에서 print 가 죽는다.
# 인코딩은 그대로 두고(한글이 깨지므로) errors 만 완화한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass



def log(m):
    print(f"[DIRECT] {m}", flush=True)


# ── 결손 채움 — 격자에서 직접 푼다 ───────────────────────────────

# ── 관측 맵 품질 — 채움의 경계조건을 손질한다 ────────────────────

def radial_grid_median(Vm, sel, cx, cz, y_edges, n_y, n_theta):
    """`ssv.radial_grid` 와 같되 칸값을 평균이 아니라 **중앙값**으로 낸다.

    ssv 쪽은 `sums[valid] / counts[valid]` 라 정점 하나가 내면으로 잘못 분류되면
    (classify_surfaces 는 파단면 근처에서 법선이 흔들려 실제로 그런다) 칸값이 통째로 끌린다.
    그 스파이크가 채움 경계조건이 되어 채움면 안쪽까지 번진다.

    2026-09-11 실측 — 칸당 정점 중앙 3개(3개 이상이 66.3%)라 중앙값이 실제로 다르다.
    평균과 1mm 넘게 다른 칸이 **43.9%**, 최대 **26.3mm**.

    행 MAD 로 정점을 미리 거르는 안도 재봤으나 버렸다 — >5MAD 가 **0개**다.
    행이 한 바퀴를 다 돌아 MAD 자체가 커서 판별력이 없다.

    **그런데 이걸 켜도 최종 메시가 더 매끈해지지 않는다 — 기본값이 mean 인 이유다.**
    2026-09-11 스윕 (채움면 이면각 중앙, 낮을수록 좋다):

        mean   blur0  2.496     median blur0  2.647   <- 중앙값 단독은 오히려 나쁘다
        mean   blur5  2.140     median blur5  2.177
        hybrid(cnt>=8) blur5  2.132  <- min_count 를 올리면 그냥 mean 으로 수렴

    표본이 칸당 중앙 3개뿐이라 중앙값이 **비선형 선택**이 되어 칸끼리 지터를 더한다.
    평균은 칸 안에서 이미 완만한 평활이고, 뒤따르는 obs_blur 가 로버스트 이득을
    어차피 다시 지운다. 이 실험을 다시 하지 말 것.

    반환은 ssv.radial_grid 와 같은 (grid, counts) 다.
    """
    P = Vm[sel]
    th = np.mod(np.arctan2(P[:, 2] - cz, P[:, 0] - cx), 2 * np.pi)
    rad = np.hypot(P[:, 0] - cx, P[:, 2] - cz)
    ti = np.minimum((th / (2 * np.pi) * n_theta).astype(np.int32), n_theta - 1)
    yi = np.clip(np.searchsorted(y_edges, P[:, 1], side="right") - 1, 0, n_y - 1)
    flat = yi * n_theta + ti

    N = n_y * n_theta
    cnt = np.bincount(flat, minlength=N)
    order = np.argsort(flat, kind="stable")          # 같은 칸끼리 붙는다
    rs = rad[order]
    beg = np.concatenate([[0], np.cumsum(cnt)])      # 칸 k 는 rs[beg[k]:beg[k+1]]
    grid = np.full(N, np.nan)
    for k in np.flatnonzero(cnt):
        grid[k] = np.median(rs[beg[k]:beg[k + 1]])
    return grid.reshape(n_y, n_theta), cnt.reshape(n_y, n_theta)


def obs_blur(grid, window, ssv):
    """관측 맵을 **NaN 을 존중하며** 저역통과한다. 결손 칸은 결손으로 남는다.

    파단 테두리는 톱니다 — 2026-09-11 실측으로 θ 인접 관측 칸의 잔차가
    중앙 1.71mm 인데 p95 8.71mm, 최대 45.9mm 다. Dirichlet 풀이는 이 톱니를
    충실히 통과하므로 **경계조건 자체를 펴 주는 것**이 채움면에 직접 듣는다.

    NaN 을 0 으로 먹이면 결손 근처 값이 0 으로 끌리므로
    유한 마스크를 같은 창으로 블러해 나눈다.

    관측 **메시**는 건드리지 않는다 — 맵은 채움을 만들기 위한 중간물이고,
    관측 칸의 값은 patch_mesh 에 들어가지 않는다(사각형은 결손 칸 4개로만 만든다).
    따라서 NFR-ETH-003 과 무관하다.

    창 크기는 5 가 무릎이다 — 2026-09-11 스윕에서 채움면 이면각 중앙이
    blur0 2.496 -> blur3 2.239 -> **blur5 2.140** -> blur7 2.171 로 5에서 포화하고,
    대가(채움 테두리가 관측 메시에서 떨어진 거리)는 3.97 -> 4.26mm 로 0.3mm 뿐이다.
    오히려 그 p95 는 66.21 -> 63.43mm 로 **줄어든다.**
    """
    w = int(window) | 1                              # 짝수면 홀수로 올린다
    v = np.isfinite(grid)
    num = ssv.box_blur_wrap(np.where(v, grid, 0.0), w, w)
    den = ssv.box_blur_wrap(v.astype(float), w, w)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-9)
    moved = np.abs(out - grid)[v]
    log(f"관측 맵 저역통과 {w}x{w} · 관측 칸 이동 중앙 {np.median(moved):.2f}mm "
        f"p95 {np.percentile(moved, 95):.2f}mm 최대 {moved.max():.2f}mm")
    return np.where(v, out, np.nan)


def build_laplacian(n_y, n_theta):
    """(y, θ) 격자의 5점 라플라시안. θ 는 한 바퀴 돌고 y 양 끝은 Neumann(반사)이다.

    반사 경계는 "이웃이 없으면 대각에서 그만큼 뺀다"와 같다 —
    f[-1] = f[0] 을 넣으면 그 항이 상쇄되기 때문이다.
    그래서 대각을 -4 로 고정하지 않고 **실제 이웃 수의 음수**로 둔다.
    """
    N = n_y * n_theta
    k = np.arange(N).reshape(n_y, n_theta)
    rows, cols = [], []
    for sh in (1, -1):                                          # θ — 양쪽이 항상 있다
        rows.append(k.ravel())
        cols.append(np.roll(k, sh, axis=1).ravel())
    rows.append(k[:-1].ravel()); cols.append(k[1:].ravel())      # y 아래 이웃
    rows.append(k[1:].ravel());  cols.append(k[:-1].ravel())     # y 위 이웃
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    # (r,c)=+1 과 (r,r)=-1 을 같이 넣는다. coo 가 중복을 합산하므로
    # 대각이 자동으로 -(이웃 수) 가 된다 — y 끝에서 -3, 나머지는 -4.
    data = np.concatenate([np.ones(len(rows)), -np.ones(len(rows))])
    ii = np.concatenate([rows, rows])
    jj = np.concatenate([cols, rows])
    return coo_matrix((data, (ii, jj)), shape=(N, N)).tocsr()


def fill_solve(r, prof, L, order=2, alpha=0.0, clamp_sigma=0.0):
    """결손 칸을 **한 번에 푼다**. `ssv.fill_grid` 를 대체한다.

    `ssv.fill_grid` 는 확산이 아니라 전진 복사다 — 값이 한 번 들어간 칸은
    `missing` 에서 빠져 두 번 다시 갱신되지 않는다. 그래서 둘이 남는다.

      (1) **방사 줄무늬** — 전선 위 칸은 유한 이웃이 1~2개뿐이라 사실상 복사다.
          파단 테두리의 톱니가 구멍 중심을 향해 행진하고 평균이 한 번만 걸려 안 지워진다
      (2) **합류 능선** — θ폭 200° 결손은 좌우 전선이 40칸씩 들어와 가운데서 만난다.
          만나는 선에서 값이 불연속이라 세로 능선이 하나 생긴다

    여기서는 결손 칸 전부를 미지수로 놓고 선형계를 직접 푼다. **순서 의존이 없다.**

    푸는 대상은 반경이 아니라 **프로파일 잔차 `r - rbar(y)`** 다.
    데이터가 없는 곳의 기준이 회전체가 되므로 manifest 의 "회전대칭 가정" 근거가
    그대로 유지되고, 잔차는 작아서 큰 결손에서도 외삽이 폭주하지 않는다.
    관측 칸은 Dirichlet 경계조건이라 값이 **정확히 보존**된다.

      order=1  조화     Du=0   비눗막. 면적 최소 · 경계 **C0(꺾인다)** · 안으로 꺼진다
      order=2  이중조화 DDu=0  박판.   곡률 최소 · 경계 **C1(접평면을 이어받는다)**

    항아리 벽은 바깥으로 불룩하므로 order=1 은 큰 결손을 안으로 움푹 꺼뜨린다.
    order=2 가 채움면 매끄러움과 이음새 연속성을 같이 준다.

    alpha > 0 이면 `(DD + aI)u = 0` 을 푼다 — 관측에서 멀수록 잔차가 0,
    즉 순수 회전체로 잦아든다. 큰 결손에서 이중조화 외삽을 눌러야 할 때 쓴다.
    """
    n_y, n_theta = r.shape
    N = n_y * n_theta
    A = L if order == 1 else (L @ L).tocsr()
    if alpha > 0:
        d = np.arange(N)
        A = (A + coo_matrix((np.full(N, float(alpha)), (d, d)), shape=(N, N))).tocsr()

    u = (r - prof[:, None]).ravel()                 # 잔차. 결손 칸은 NaN
    mi = np.flatnonzero(~np.isfinite(u))
    ki = np.flatnonzero(np.isfinite(u))
    if not len(mi):
        return r.copy()
    if not len(ki):
        raise SystemExit("관측 칸이 하나도 없다. 축 피팅이나 내·외면 분리를 볼 것")

    x = np.zeros(N)
    x[ki] = u[ki]
    rhs = -(A[mi][:, ki] @ x[ki])
    sol = np.atleast_1d(spsolve(A[mi][:, mi].tocsc(), rhs))
    if not np.isfinite(sol).all():
        raise SystemExit("채움 선형계가 특이행렬이다. --fill-alpha 를 1e-3 쯤 주거나 "
                         "--fill-order 1 로 낮출 것")

    # 안전망 — 기본은 **끔**이다.
    # 2026-09-11 실측: 관측 잔차가 5시그마(20.9mm)를 81칸, 8시그마(33.5mm)를 4칸 넘는다.
    # 4시그마로 자르면 관측 범위보다 낮게 자르는 셈이라 잘린 자리의 평평한 윗면이
    # 오히려 새 꺾임을 만든다 (|Dr| 최대 26.3 -> 36.4mm). 병적인 경우에만 켤 것.
    sig = 1.4826 * np.median(np.abs(u[ki] - np.median(u[ki])))
    cap = clamp_sigma * sig if (clamp_sigma > 0 and sig > 0) else np.inf
    over = np.abs(sol) > cap
    if over.any():
        sol[over] = np.sign(sol[over]) * cap
        log(f"  잔차 상한 {cap:.2f}mm ({clamp_sigma}시그마) 초과 {int(over.sum()):,}칸 잘림")

    x[mi] = sol
    scheme = ("이중조화" if order == 2 else "조화") + (f" + {alpha}I" if alpha > 0 else "")
    log(f"채움 풀이 · {scheme} · 미지수 {len(mi):,}칸 · 관측 잔차 시그마 {sig:.2f}mm "
        f"최대 {np.abs(u[ki]).max():.2f}mm → 채움 잔차 |중앙| {np.median(np.abs(sol)):.2f}mm "
        f"최대 {np.abs(sol).max():.2f}mm")
    return x.reshape(n_y, n_theta) + prof[:, None]


def fill_report(grid, mask, L, tag):
    """채움 칸의 이산 라플라시안 크기. **줄무늬·능선이 있으면 꼬리가 두껍다.**

    눈으로 보는 것보다 확실하다 — 채움 방식을 바꿨을 때 실제로 매끈해졌는지
    숫자로 남는다. 격자가 mm 단위이므로 단위도 mm 다.
    """
    v = np.abs((L @ grid.ravel()).reshape(grid.shape)[mask])
    if not len(v):
        return
    log(f"  [{tag}] 채움 칸 |Dr| 중앙 {np.median(v):.4f} "
        f"p95 {np.percentile(v, 95):.4f} 최대 {v.max():.4f} mm")


def cyl_map(V, F, n_y, n_theta, scale_mm=250.0,
            fill_order=2, fill_alpha=0.0, fill_clamp_sigma=0.0, compare=False,
            obs_stat="mean", obs_blur_w=5):
    """원통 전개 → 외면 반경 맵. skeleton_cyl 이 쓰는 것과 같은 경로."""
    ssv = S.load_ssv()
    ssv.N_Y, ssv.N_THETA = n_y, n_theta
    Vm = V * scale_mm
    cx, cz = ssv.fit_axis(Vm)
    outer, inner, st = ssv.classify_surfaces(Vm, F, cx, cz)
    log(f"축 ({cx/scale_mm:+.4f}, {cz/scale_mm:+.4f}) · 외면 정점 {st['outer_vertices']:,}")

    y = Vm[:, 1]
    y_edges = np.linspace(y.min(), y.max() + 1e-9, n_y + 1)
    if obs_stat == "median":
        r, cnt = radial_grid_median(Vm, outer, cx, cz, y_edges, n_y, n_theta)
    else:
        r, cnt = ssv.radial_grid(Vm, outer, cx, cz, y_edges)
    observed = cnt > 0
    if obs_blur_w:
        r = obs_blur(r, obs_blur_w, ssv)
    rowmed = np.array([np.nanmedian(r[i]) if np.isfinite(r[i]).any() else np.nan
                       for i in range(n_y)])
    prof = ssv.fill_profile(rowmed)
    L = build_laplacian(n_y, n_theta)
    names = {0: "전진복사", 1: "조화", 2: "이중조화"}

    if fill_order == 0:
        filled = ssv.fill_grid(r, prof)
        log("채움 = ssv.fill_grid (옛 전진 복사 경로)")
    else:
        filled = fill_solve(r, prof, L, fill_order, fill_alpha, fill_clamp_sigma)

    fill_report(filled, ~observed, L, names[fill_order])
    if compare and fill_order != 0:
        fill_report(ssv.fill_grid(r, prof), ~observed, L, names[0] + "(대조)")
    log(f"격자 {n_y}x{n_theta} · 관측 {100*observed.mean():.1f}% → 결손 {int((~observed).sum()):,}칸")
    return filled / scale_mm, observed, (cx / scale_mm, cz / scale_mm), y_edges / scale_mm


def select_regions(need, min_cells, ow_max_cells, ow_max_deg):
    """어디를 채울지 **영역 단위로** 정한다.

    지금까지는 빈 칸이면 무조건 채웠다. 그래서 투창(의도된 구멍)까지 메웠다.
    결손 마스크를 연결 성분으로 나눠 판정한다. theta 는 한 바퀴 도므로 좌우로 이어붙여 라벨링한다.

    투창 판정 (셋 다 만족):
      - 좁다        theta 폭이 ow_max_deg 이하
      - 작다        칸 수가 ow_max_cells 이하
      - 갇혀 있다   위아래 끝에 닿지 않는다 (파단은 대개 가장자리까지 이어진다)
    """
    from scipy import ndimage
    n_y, n_theta = need.shape
    big = np.concatenate([need, need, need], axis=1)
    lab, _ = ndimage.label(big, structure=np.ones((3, 3)))
    mid = lab[:, n_theta:2 * n_theta]

    keep = np.zeros_like(need)
    kinds = {"fill": [], "openwork": [], "tiny": []}
    for i in np.unique(mid[mid > 0]):
        m = mid == i
        ys, ts = np.nonzero(m)
        area = int(m.sum())
        occ = np.zeros(n_theta, bool)
        occ[ts] = True
        idx = np.nonzero(occ)[0]
        gaps = np.diff(np.concatenate([idx, [idx[0] + n_theta]]))
        width_deg = (n_theta - gaps.max() + 1) / n_theta * 360
        enclosed = bool(ys.min() > 0 and ys.max() < n_y - 1)
        rec = dict(area=area, width_deg=round(float(width_deg), 1),
                   h=[int(ys.min()), int(ys.max())], enclosed=enclosed,
                   theta_deg=round(float(ts.mean() / n_theta * 360), 1))
        if area < min_cells:
            kinds["tiny"].append(rec)
        elif enclosed and width_deg <= ow_max_deg and area <= ow_max_cells:
            kinds["openwork"].append(rec)
        else:
            kinds["fill"].append(rec)
            keep |= m

    ow = kinds["openwork"]
    log(f"영역 판정 · 채움 {len(kinds['fill'])}개({int(keep.sum()):,}칸) · "
        f"투창 {len(ow)}개 · 미세조각 {len(kinds['tiny'])}개")
    for r in sorted(kinds["fill"], key=lambda r: -r["area"])[:6]:
        log(f"    채움 {r['area']:5d}칸 theta폭 {r['width_deg']:5.1f}도 h {r['h']}")
    if ow:
        angs = sorted(r["theta_deg"] for r in ow)
        d = np.diff(angs + [angs[0] + 360])
        log(f"    투창 theta {['%.0f' % a for a in angs]} · 간격 {['%.0f' % x for x in d]}도")
    return keep, kinds


def upsample_map(filled, need, y_edges, k):
    """**판정 해상도와 면 해상도를 분리한다.**

    선행 문서 §5.3 의 맞바꿈 — 격자를 키우면 면은 촘촘해지지만 칸당 표본이 얕아져
    결손을 과판정하고(96x144 에서 약 5%p) 미세조각이 늘고, 낮추면 사각형을 만들
    칸이 줄어 패치가 성겨진다. 두 요구가 같은 손잡이에 묶여 있던 것이 문제였다.

    결손 판정(select_regions)은 거친 격자에서 끝내고, **그 판정을 그대로 둔 채**
    면만 촘촘하게 만든다. 손상 GLB 를 400k 로 다시 뽑지 않아도 된다.

    θ 는 한 바퀴 도므로 grid-wrap 으로 보간하고(mode='nearest' 로 하면 이음새가
    2.5배 벌어진다), y 는 끝을 연장한다. 마스크는 k×k 블록 복제라
    **채울 영역이 늘지도 줄지도 않는다** — 판정을 건드리지 않는다는 뜻이다.
    """
    from scipy import ndimage
    hi = ndimage.zoom(filled, (k, 1), order=3, mode="nearest", grid_mode=True)
    hi = ndimage.zoom(hi, (1, k), order=3, mode="grid-wrap", grid_mode=True)
    need_hi = np.repeat(np.repeat(need, k, axis=0), k, axis=1)
    y_hi = np.linspace(y_edges[0], y_edges[-1], (len(y_edges) - 1) * k + 1)
    log(f"면 해상도 분리 · 격자 {filled.shape[0]}x{filled.shape[1]} → "
        f"{hi.shape[0]}x{hi.shape[1]} (x{k}) · 채울 칸 {int(need.sum()):,} → "
        f"{int(need_hi.sum()):,}")
    return hi, need_hi, y_hi


def carve_openwork_nfold(need, observed, kinds, n_theta, nmin=3, nmax=8,
                         tol_deg=25.0, mode="auto", min_cells=20):
    """확정 투창으로 **띠**를 찾고, 그 띠의 n-fold 주기에서 빠진 투창 자리를 채움에서 뺀다.

    왜 필요한가 — 문턱(`--openwork-max-deg` 등)으로는 **파단이 겹쳐 커진 투창**을 못 잡는다.
    #84(329칸 θ폭 70도)나 #2(657칸) 안에 들어앉은 투창이 그렇다.
    크기가 아니라 **주기**로 판정하면 잡힌다.

    주기를 어떻게 세나 — 깨진 조각의 θ 중심은 파단이 갉아먹어 흔들린다.
    대신 **띠 안에서 θ를 따라 관측 비율이 오르내리는 신호**에 푸리에를 건다.
    투창이 n개면 n차 조화가 서고, 파단은 비주기라 배경으로 깔린다.

    2026-09-11 실측 (굽다리바리 경주 신수 71489, 96x144):
        띠A h 8-18   4-fold 13.5%(1위)  예측 78/168/258/348  <- 168·348 이 확정 투창과 1.5도 이내
        띠B h 27-38  3-fold  9.9%(1위)  예측 20/140/260      <- 20 은 1.8도, 260 은 #84 와 2.2도
        (대조) 바리 h 55-80 은 1-fold 가 40.1% — 주기가 아니라 큰 파단 하나라는 뜻이다

    **이미 결손인 칸만 뺀다. 관측 살은 절대 뚫지 않는다.**
    그래도 "여기 투창이 있었다"는 추정이므로 manifest 에 `n-fold 가정`으로 남긴다.
    기본은 꺼져 있다 — 주기 peak 이 약한 유물에서 잘못 뚫으면 복원이 틀린다.
    """
    ow = kinds.get("openwork", [])
    if len(ow) < 2:
        log("n-fold 카빙 생략 — 확정 투창이 2개 미만이라 띠를 못 만든다")
        return need, []

    # 1. 확정 투창을 h 겹침으로 묶어 띠를 만든다
    tiers = []
    for r in sorted(ow, key=lambda r: r["h"][0]):
        for t in tiers:
            if r["h"][0] <= t["h1"] and r["h"][1] >= t["h0"]:
                t["h0"] = min(t["h0"], r["h"][0]); t["h1"] = max(t["h1"], r["h"][1])
                t["m"].append(r)
                break
        else:
            tiers.append(dict(h0=r["h"][0], h1=r["h"][1], m=[r]))

    carved = []
    for t in tiers:
        if len(t["m"]) < 2:
            log(f"  띠 h[{t['h0']},{t['h1']}] · 투창 1개뿐이라 주기를 못 센다. 건너뜀")
            continue
        h0, h1 = t["h0"], t["h1"]
        sig = observed[h0:h1 + 1].mean(0)
        sp = np.abs(np.fft.rfft(sig - sig.mean()))
        sp[0] = 0
        if mode == "auto":
            n = int(nmin + np.argmax(sp[nmin:nmax + 1]))
        else:
            n = int(mode)
        share = 100 * sp[n] / max(sp[1:].sum(), 1e-9)

        # 위상 — 관측이 **최소**인 곳이 투창이다
        ph = np.angle(np.fft.rfft(sig - sig.mean())[n])
        t0 = ((np.pi - ph) / n) % (2 * np.pi / n)
        pred = [(t0 + j * 2 * np.pi / n) % (2 * np.pi) for j in range(n)]
        pred_deg = [np.rad2deg(p) for p in pred]

        have = [r["theta_deg"] for r in t["m"]]
        w_cells = int(np.median([r["width_deg"] for r in t["m"]]) / 360 * n_theta)
        w_cells = max(w_cells, 2)

        log(f"  띠 h[{h0},{h1}] · 확정 투창 {len(t['m'])}개 → **{n}-fold** "
            f"(비중 {share:.1f}%) · 예측 θ "
            + " ".join(f"{d:.0f}" for d in pred_deg))

        for d in pred_deg:
            gap = min(abs(((d - hv + 180) % 360) - 180) for hv in have)
            if gap <= tol_deg:
                continue                       # 이미 투창으로 잡혀 있다
            c = int(round(d / 360 * n_theta))
            cols = [(c + k) % n_theta for k in range(-(w_cells // 2), w_cells // 2 + 1)]
            box = np.zeros_like(need)
            box[h0:h1 + 1, cols] = True
            hit = need & box                   # **결손인 칸만** 뺀다
            if hit.sum() < min_cells:
                # 예측 자리에 뺄 칸이 거의 없다 = 거기엔 관측 살이 있다 = 창이 아니다.
                # 몇 칸만 뚫으면 슬리버 구멍이 되므로 아예 건너뛴다.
                # **이 줄이 여러 번 뜨면 그 n 이 틀린 것이다** (2026-09-11 6-fold 검사).
                pct = 100 * observed[h0:h1 + 1][:, cols].mean()
                log(f"    θ {d:5.1f}도 — 건너뜀. 뺄 칸 {int(hit.sum())}개(<{min_cells}) "
                    f"· 그 자리 관측 {pct:.0f}%")
                continue
            need = need & ~hit
            carved.append(dict(tier_h=[h0, h1], n=n, share=round(float(share), 1),
                               theta_deg=round(float(d), 1), cells=int(hit.sum()),
                               nearest_known_deg=round(float(gap), 1)))
            log(f"    θ {d:5.1f}도 — 채움에서 {int(hit.sum()):,}칸 뺌 "
                f"(가장 가까운 확정 투창과 {gap:.0f}도)")
    if not carved:
        log("n-fold 카빙 · 뺄 자리가 없었다")
    else:
        log(f"n-fold 카빙 · 투창 {len(carved)}곳 추가 · "
            f"총 {sum(c['cells'] for c in carved):,}칸을 채움에서 뺐다")
    return need, carved


def patch_mesh(filled, need, cx, cz, y_edges, thickness, split_rim=True):
    """결손 칸 위에 연속 표면을 만든다.

    격자 칸 하나를 정점 하나로 두고 인접 칸끼리 사각형(삼각형 2개)으로 잇는다.
    **인접 칸이 정점을 공유하므로 이 표면은 조각날 수 없다.**
    θ 는 한 바퀴 돌아 이어진다.
    """
    n_y, n_theta = filled.shape
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    th = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi

    idx = -np.ones((n_y, n_theta), np.int64)
    ys, ts = np.nonzero(need)
    idx[ys, ts] = np.arange(len(ys))
    r = filled[ys, ts]
    outer = np.stack([cx + r * np.cos(th[ts]), yc[ys], cz + r * np.sin(th[ts])], 1)
    inner = np.stack([cx + (r - thickness) * np.cos(th[ts]), yc[ys],
                      cz + (r - thickness) * np.sin(th[ts])], 1)
    n = len(ys)
    Vp = np.concatenate([outer, inner], 0)          # 0..n-1 외면, n..2n-1 내면

    quads = []
    for a in range(n_y - 1):
        for b in range(n_theta):
            b2 = (b + 1) % n_theta
            c = (idx[a, b], idx[a, b2], idx[a + 1, b2], idx[a + 1, b])
            if min(c) >= 0:
                quads.append(c)
    if not quads:
        return None
    q = np.array(quads)
    out_f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], 0)
    in_f = np.concatenate([q[:, [2, 1, 0]], q[:, [3, 2, 0]]], 0) + n   # 뒤집어서 안쪽

    # 경계 옆면 — 한 삼각형에만 속한 변을 찾아 바깥·안쪽을 잇는다
    e = np.sort(np.concatenate([out_f[:, [0, 1]], out_f[:, [1, 2]], out_f[:, [2, 0]]]), 1)
    uq, ct = np.unique(e, axis=0, return_counts=True)
    bnd = uq[ct == 1]
    if not len(bnd):
        side = np.zeros((0, 3), np.int64)
        rim = np.array([], np.int64)
    elif split_rim:
        # 옆면에 **자기 정점을 준다.**
        # 외면·내면·옆면이 정점을 공유하면 테두리의 90도 모서리가 정점 법선
        # 평균에 섞여 패치 테두리마다 어두운 띠가 생긴다.
        # 복제본은 원본과 **같은 자리**에 있고 rim 에 같이 들어가므로
        # snap_boundary·smooth_fixed_boundary 에서 함께 움직인다 — 틈이 안 벌어진다.
        # (연결 성분은 위치 기준 용접 후에 세야 한다. main 에서 그렇게 한다)
        uniq = np.unique(bnd)
        remap = -np.ones(n, np.int64)
        remap[uniq] = np.arange(len(uniq))
        m, base = len(uniq), 2 * n
        Vp = np.concatenate([Vp, outer[uniq], inner[uniq]], 0)
        a, b = base + remap[bnd[:, 0]], base + remap[bnd[:, 1]]
        side = np.concatenate([
            np.stack([a, b, b + m], 1),
            np.stack([a, b + m, a + m], 1),
        ], 0)
        rim = np.unique(np.concatenate([uniq, uniq + n,
                                        np.arange(base, base + 2 * m)]))
    else:
        side = np.concatenate([
            np.stack([bnd[:, 0], bnd[:, 1], bnd[:, 1] + n], 1),
            np.stack([bnd[:, 0], bnd[:, 1] + n, bnd[:, 0] + n], 1),
        ], 0)
        rim = np.unique(np.concatenate([bnd.ravel(), bnd.ravel() + n]))
    Fp = np.concatenate([out_f, in_f, side], 0)
    log(f"채움 메시 · 정점 {len(Vp):,} 삼각형 {len(Fp):,} (사각 {len(q):,} · 경계변 {len(bnd):,})")
    return trimesh.Trimesh(Vp, Fp, process=False), rim


def snap_boundary(fill, obs, rim, max_dist):
    """채움 패치의 **테두리 정점만** 관측 표면 최근접점으로 옮긴다.

    관측 메시는 한 톨도 안 건드린다 — 움직이는 것은 채움 쪽뿐이다.
    정점을 공유하지는 않으므로 위상적으로 이어지는 것은 아니지만,
    틈이 시각적으로 사라진다 (기여자 2026-09-08 의 경계 스냅과 같은 방식).
    """
    if not len(rim):
        log("테두리 정점이 없어 스냅 생략")
        return fill
    # trimesh.proximity 는 rtree 를 요구한다(미설치). 관측 메시가 정점 94,877개로 조밀하므로
    # 최근접 *정점* 으로 스냅해도 최근접 표면점과 한 변 길이 안에서 일치한다.
    d, i = cKDTree(obs.vertices).query(fill.vertices[rim])
    move = d <= max_dist
    fill.vertices[rim[move]] = np.asarray(obs.vertices)[i[move]]
    msg = (f"경계 스냅 · 테두리 {len(rim):,}정점 중 {int(move.sum()):,}개 이동 "
           f"(상한 {max_dist*250:.1f}mm) · 이동거리 중앙 {np.median(d[move])*250:.2f}mm")
    if (~move).any():
        msg += f" · 상한 초과로 남긴 {int((~move).sum()):,}개 거리 중앙 {np.median(d[~move])*250:.1f}mm"
    log(msg)
    return fill


def smooth_fixed_boundary(mesh, pinned, iters, lamb, cap):
    """테두리를 고정한 채 채움면 내부만 평활한다 (기여자 2026-09-08 의 경계고정 Laplacian).

    스냅으로 맞춰놓은 테두리는 안 움직이고 격자 단위 주름만 편다.
    관측 메시는 애초에 대상이 아니다 — 이 함수는 채움 메시만 받는다.
    """
    if iters <= 0:
        return mesh
    from scipy.sparse import coo_matrix
    V = np.asarray(mesh.vertices, float).copy()
    F = np.asarray(mesh.faces)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    n = len(V)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel().clip(min=1)
    free = np.ones(n, bool)
    free[pinned] = False
    before = V.copy()

    # Taubin λ|μ — 수축시킨 뒤 되밀어 부피 손실을 상쇄한다.
    # 순수 Laplacian 은 자유 경계 패치를 중심으로 붕괴시킨다
    # (실측: 3회에 최대 이동 147.8mm. 기여자 2026-09-08 도 같은 붕괴를 겪었다).
    mu = -lamb / (1.0 - 0.1 * lamb)
    for _ in range(iters):
        for w in (lamb, mu):
            nb = A @ V / deg[:, None]
            V[free] += w * (nb[free] - V[free])

    # 안전망 — 이동량 상한. 붕괴하면 여기서 잘린다
    d = V - before
    nrm = np.linalg.norm(d, axis=1)
    over = nrm > cap
    if over.any():
        V[over] = before[over] + d[over] / nrm[over, None] * cap
        log(f"  이동 상한 {cap*250:.1f}mm 초과 {int(over.sum()):,}정점 잘림")
    d = np.linalg.norm(V - before, axis=1)
    log(f"경계고정 Taubin 평활 {iters}회 (lambda {lamb}, mu {mu:.3f}) · 고정 {int((~free).sum()):,}정점 · "
        f"이동 중앙 {np.median(d[free])*250:.2f}mm 최대 {d.max()*250:.2f}mm")
    mesh.vertices = V
    return mesh


def surface_samples(obs, n_samples, seed=0):
    """메시 표면에서 **텍스처를 직접 샘플링**한다 → (위치, 법선, RGB).

    정점색만 쓰면 정보를 대부분 버린다 — 이 유물은 baseColor 텍스처가 1024x1024
    (약 105만 텍셀)인데 외면 정점은 47,931개뿐이라 **22배를 버리는 셈**이다.
    2026-09-11 실측: 정점색으로 만든 색 맵은 격자를 키워도 채움 질감이
    관측 대비 53~56% 에서 멈췄고, 192x288 부터는 관측 칸이 41% 로 무너졌다.

    삼각형마다 **면적에 비례해** 무작위 무게중심 좌표를 뽑아 위치와 UV 를 함께 얻고,
    UV 로 텍스처를 찍는다. 표본 수를 격자보다 충분히 크게 잡으면 칸마다 여러 번 맞는다.

    텍스처가 없으면(정점색만 있는 메시) None 을 돌려준다 — 호출부가 정점색으로 떨어진다.
    """
    v = obs.visual
    uv = getattr(v, "uv", None)
    mat = getattr(v, "material", None)
    img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    if uv is None or img is None:
        return None
    uv = np.asarray(uv, float)
    tex = np.asarray(img.convert("RGB"), dtype=np.uint8)
    th_, tw_ = tex.shape[:2]

    V = np.asarray(obs.vertices, float)
    F = np.asarray(obs.faces)
    area = obs.area_faces
    rng = np.random.default_rng(seed)
    # 면적 비례 배분. 아주 작은 삼각형도 최소 1번은 뽑는다
    per = np.maximum(1, np.rint(area / area.sum() * n_samples).astype(np.int64))
    ti = np.repeat(np.arange(len(F)), per)
    r1 = np.sqrt(rng.random(len(ti)))
    r2 = rng.random(len(ti))
    w = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], 1)[:, :, None]

    P_ = (V[F[ti]] * w).sum(1)
    UV = (uv[F[ti]] * w).sum(1)
    # glTF UV 는 좌상단 원점이다. v 를 뒤집어야 텍스처와 맞는다.
    px = np.clip((UV[:, 0] * (tw_ - 1)).astype(np.int64), 0, tw_ - 1)
    py = np.clip(((1.0 - UV[:, 1]) * (th_ - 1)).astype(np.int64), 0, th_ - 1)
    C_ = tex[py, px].astype(float)
    N_ = obs.face_normals[ti]
    log(f"표면 샘플 {len(ti):,}개 (삼각형 {len(F):,} · 텍스처 {tw_}x{th_})")
    return P_, N_, C_


def color_map_inpaint(obs, cx, cz, y_edges, n_y, n_theta, pad_frac=0.25,
                      n_samples=2_000_000):
    """관측 정점색을 (y, θ) 격자에 모으고 **빈 칸을 LaMa 로 채운다.**

    왜 이미지 공간이 아니라 (y, θ) 인가 — 이미지에서는 마스크 바깥이 배경이라
    LaMa 가 배경을 그릴 위험이 있다. 원통 전개 맵에서는 **구멍 사방이 전부 도기 표면**이라
    LaMa 가 가장 잘하는 상황이 된다 (2D-복원-조건이미지-설계.md §3, lama.py 독스트링).

    그리고 이 경로에서는 **형상에 1도 관여하지 않는다.** 채움 형상은 이중조화로 이미 확정돼
    있고 여기서는 색만 얹는다. 선행 문서 §7.10 의 "조건 이미지 질감이 형상을 6.7배 흔든다"는
    위험이 색으로만 한정된다는 뜻이다.

    최근접 복사(`transfer_color`)를 대체한다. 그쪽은 채움 정점마다 **중앙 22.3mm 떨어진**
    관측 정점의 색을 그대로 가져와서, 색은 맞지만 질감이 평평하고 얼룩진다.

    축(cx, cz)과 y_edges 는 `cyl_map` 것을 그대로 쓴다. 다만 정점 배열은 공유하지 않는다 —
    `S.load_glb` 는 74,470개(용접본), trimesh 는 94,877개(색 있음)라 순서가 다르다.
    그래서 외면 판정을 **법선이 축 바깥을 향하는가**로 여기서 다시 한다.

    LaMa 는 생성 모델이다. 쓰면 manifest 의 채움 **색** 근거가 "LaMa 생성 (AI)"으로 올라간다.
    형상 근거(회전대칭 + n-fold)는 그대로다.
    """
    import lama

    got = surface_samples(obs, n_samples)
    if got is None:
        # 텍스처가 없으면 정점색으로 떨어진다
        V = np.asarray(obs.vertices, float)
        try:
            C = np.asarray(obs.visual.to_color().vertex_colors)[:, :3].astype(float)
            if C.shape[0] != len(V):
                raise ValueError(f"정점 {len(V)} vs 색 {C.shape[0]}")
        except Exception as ex:
            log(f"색을 못 읽었다 ({ex}) — 색 인페인팅을 건너뛴다")
            return None
        N = np.asarray(obs.vertex_normals, float)
        log("텍스처가 없어 정점색으로 떨어진다")
    else:
        V, N, C = got

    # 외면만 모은다. 내면은 어두워서 섞이면 채움이 탁해진다.
    rad = np.stack([V[:, 0] - cx, np.zeros(len(V)), V[:, 2] - cz], 1)
    rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
    outer = (N * rad).sum(1) > 0

    th = np.mod(np.arctan2(V[:, 2] - cz, V[:, 0] - cx), 2 * np.pi)
    ti = np.minimum((th / (2 * np.pi) * n_theta).astype(np.int32), n_theta - 1)
    yi = np.clip(np.searchsorted(y_edges, V[:, 1], side="right") - 1, 0, n_y - 1)
    flat = (yi * n_theta + ti)[outer]
    Co = C[outer]

    M = n_y * n_theta
    cnt = np.bincount(flat, minlength=M)
    tex = np.zeros((M, 3))
    for c in range(3):
        tex[:, c] = np.bincount(flat, weights=Co[:, c], minlength=M)
    have = cnt > 0
    tex[have] /= cnt[have][:, None]
    tex, have = tex.reshape(n_y, n_theta, 3), have.reshape(n_y, n_theta)
    log(f"색 맵 {n_y}x{n_theta} · 외면 표본 {int(outer.sum()):,} · "
        f"관측 칸 {100*have.mean():.1f}% · 평균 RGB {tex[have].mean(0).round(1)}")

    # θ 를 좌우로 감아서 넣는다 — 안 그러면 0도/360도 경계에 이음매가 생긴다
    pad = max(int(n_theta * pad_frac), 8)
    def wrap(a):
        return np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    hole = (~have).astype(np.uint8) * 255
    log(f"LaMa 인페인팅 · 입력 {n_theta + 2*pad}x{n_y} · 구멍 {int(have.size - have.sum()):,}칸")
    out = lama.inpaint(wrap(tex).astype(np.uint8), wrap(hole))
    colmap = out[:, pad:pad + n_theta]

    d = np.abs(colmap[have].astype(float) - tex[have])
    log(f"  관측 칸 보존 확인 · 최대 색 변화 {d.max():.0f}/255 (0이어야 한다)")
    log(f"  채움 칸 평균 RGB {colmap[~have].mean(0).round(1)} "
        f"(관측 {tex[have].mean(0).round(1)})")
    return colmap


def transfer_color_map(mesh_fill, colmap, cx, cz, y_edges, n_y, n_theta):
    """채움 정점 색을 (y, θ) 색 맵에서 **쌍선형 보간**으로 가져온다.

    최근접 칸을 그냥 집으면 업샘플한 채움면(정점이 칸보다 촘촘하다)에서 격자 무늬가 보인다.
    θ 는 한 바퀴 도므로 grid-wrap, y 는 끝에서 자른다.
    """
    from scipy import ndimage
    V = np.asarray(mesh_fill.vertices, float)
    th = np.mod(np.arctan2(V[:, 2] - cz, V[:, 0] - cx), 2 * np.pi)
    fy = np.clip((V[:, 1] - y_edges[0]) / (y_edges[-1] - y_edges[0]) * n_y - 0.5,
                 0, n_y - 1)
    fx = th / (2 * np.pi) * n_theta - 0.5
    cols = np.stack([ndimage.map_coordinates(colmap[:, :, c].astype(float), [fy, fx],
                                             order=1, mode="grid-wrap")
                     for c in range(3)], 1)
    cols = np.clip(cols, 0, 255).astype(np.uint8)
    rgba = np.concatenate([cols, np.full((len(cols), 1), 255, np.uint8)], 1)
    mesh_fill.visual = trimesh.visual.ColorVisuals(mesh_fill, vertex_colors=rgba)
    log(f"색 전사(맵) · 채움 정점 {len(V):,} · 평균 RGB {cols.mean(0).round(1)}")
    return mesh_fill


def transfer_color(mesh_fill, mesh_obs):
    """채움 정점 색을 **가장 가까운 잔존 표면점**에서 가져온다. 새 색을 짓지 않는다."""
    obs = mesh_obs.copy()
    try:
        cols = obs.visual.to_color().vertex_colors
    except Exception as ex:
        log(f"경고: 텍스처 → 정점색 변환 실패 ({ex}). 회색으로 채운다")
        cols = np.tile([170, 170, 165, 255], (len(obs.vertices), 1))
    d, i = cKDTree(obs.vertices).query(mesh_fill.vertices)
    log(f"색 전사 · 최근접 거리 중앙 {np.median(d):.4f} (정규화 단위)")
    mesh_fill.visual = trimesh.visual.ColorVisuals(
        mesh_fill, vertex_colors=cols[i].astype(np.uint8))
    return mesh_fill


def patch_material(path, mesh_name, metallic, roughness):
    """내보낸 GLB 에 채움 노드용 PBR 머티리얼을 붙인다.

    trimesh 의 ColorVisuals 는 COLOR_0 만 내보내고 머티리얼을 안 만든다.
    glTF 는 머티리얼이 없으면 **기본값 metallic=1.0, roughness=1.0** 을 쓰므로
    채움면이 완전 금속으로 렌더된다(어둡고 반들거림). 관측 MR 텍스처 실측은
    metallic 0.000 / roughness 0.906 이다.

    JSON 청크만 바꾸고 BIN 청크는 그대로 두므로 지오메트리는 손대지 않는다.
    """
    d = open(path, "rb").read()
    off, js, jrange, binchunk = 12, None, None, b""
    while off < len(d):
        cl, ct = struct.unpack("<II", d[off:off + 8])
        body = d[off + 8:off + 8 + cl]
        if ct == 0x4E4F534A:
            js = json.loads(body.decode("utf-8"))
        else:
            binchunk = body
        off += 8 + cl + ((4 - cl % 4) % 4 if cl % 4 else 0)

    mats = js.setdefault("materials", [])
    mats.append({
        "name": f"{mesh_name}_mat",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0],   # COLOR_0 가 곱해진다
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
        "alphaMode": "OPAQUE",
        "doubleSided": True,     # 채움면은 얇아 안쪽에서도 보여야 한다
    })
    mi = len(mats) - 1
    n = 0
    for m in js["meshes"]:
        if m.get("name") == mesh_name:
            for pr in m["primitives"]:
                pr["material"] = mi
                n += 1
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * ((4 - len(jb) % 4) % 4)
    bb = binchunk + bytes((4 - len(binchunk) % 4) % 4)
    out = (b"glTF" + struct.pack("<II", 2, 12 + 8 + len(jb) + 8 + len(bb))
           + struct.pack("<II", len(jb), 0x4E4F534A) + jb
           + struct.pack("<II", len(bb), 0x004E4942) + bb)
    open(path, "wb").write(out)
    log(f"머티리얼 부여 · {mesh_name} 프리미티브 {n}개 · metallic {metallic} roughness {roughness}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="A단계 손상 GLB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=144)
    ap.add_argument("--relief", default=None,
                    help="unet_relief.py 가 만든 relief.npz. 채움면에 요철을 얹는다 (층 ③)")
    ap.add_argument("--metallic", type=float, default=0.0,
                    help="관측 MR 텍스처 실측 metallic 0.000")
    ap.add_argument("--roughness", type=float, default=0.906,
                    help="관측 MR 텍스처 실측 roughness 0.906")
    ap.add_argument("--smooth-iters", dest="smooth_iters", type=int, default=3,
                    help="경계 고정 Laplacian 반복 횟수 (0 이면 안 함)")
    ap.add_argument("--smooth-lambda", dest="smooth_lambda", type=float, default=0.5)
    ap.add_argument("--smooth-cap-mm", dest="smooth_cap_mm", type=float, default=3.0,
                    help="평활 이동량 상한. 패치 붕괴 안전망")
    ap.add_argument("--min-cells", dest="min_cells", type=int, default=8,
                    help="이보다 작은 결손 조각은 표본 노이즈로 보고 무시")
    ap.add_argument("--openwork-max-cells", dest="ow_cells", type=int, default=400,
                    help="투창 판정 상한 (칸 수)")
    ap.add_argument("--openwork-max-deg", dest="ow_deg", type=float, default=40.0,
                    help="투창 판정 상한 (theta 폭, 도)")
    ap.add_argument("--snap-mm", dest="snap_mm", type=float, default=10.0,
                    help="경계 스냅 상한. 이보다 먼 테두리 정점은 안 옮긴다 (0 이면 스냅 안 함)")
    ap.add_argument("--thickness", type=float, default=0.008,
                    help="채움 벽 두께 (정규화 단위). 관측 벽 실측 중앙 1.84mm = 0.0074")
    ap.add_argument("--fill-order", dest="fill_order", type=int, default=2,
                    choices=[0, 1, 2],
                    help="결손 채움 방식. 2=이중조화(박판, 경계 C1, 기본) "
                         "1=조화(비눗막, 경계 C0) 0=옛 전진복사 ssv.fill_grid")
    ap.add_argument("--fill-alpha", dest="fill_alpha", type=float, default=0.0,
                    help="(DD+aI)u=0 의 a. 클수록 관측에서 먼 곳이 순수 회전체로 잦아든다")
    ap.add_argument("--fill-clamp-sigma", dest="fill_clamp_sigma", type=float,
                    default=0.0,
                    help="채움 잔차 상한(관측 잔차 sigma 배수). 기본 0=끔. "
                         "관측 잔차 자체가 8sigma 까지 가므로 자르면 그 자리가 새 꺾임이 된다")
    ap.add_argument("--fill-compare", dest="fill_compare", action="store_true",
                    help="옛 전진복사 방식의 매끄러움 지표를 나란히 찍는다")
    ap.add_argument("--obs-stat", dest="obs_stat", default="mean",
                    choices=["median", "mean"],
                    help="격자 칸값. mean=ssv.radial_grid 원래 방식(기본). "
                         "median 은 재봤으나 최종 메시가 더 매끈해지지 않는다 "
                         "— radial_grid_median 독스트링 참조")
    ap.add_argument("--obs-blur", dest="obs_blur", type=int, default=5,
                    help="관측 맵 저역통과 창(칸). 파단 테두리 톱니를 펴 채움 "
                         "경계조건을 매끈하게 한다. 5가 무릎. 0=끔")
    ap.add_argument("--mesh-upsample", dest="mesh_upsample", type=int, default=2,
                    help="결손 판정 뒤 면만 k배로 촘촘하게. 판정 해상도는 그대로라 "
                         "과판정이 안 늘고 GLB 재export 도 필요 없다. 삼각형은 k^2 배. "                         "2026-09-11 실측 이면각 중앙 x1 2.04 / x2 1.03 / x3 0.68 / x4 0.51도, "                         "삼각형 18k / 73k / 164k / 291k. 무릎이 없는 순수 손잡이라 "                         "기본 2, 고화질 렌더는 3~4")
    ap.add_argument("--nfold", default="off",
                    help="투창 n-fold 카빙. off(기본) · auto(띠마다 푸리에로 n 탐지) · "
                         "숫자(모든 띠에 그 n 강제). 이미 결손인 칸만 채움에서 뺀다 — "
                         "관측 살은 안 뚫는다. manifest 에 n-fold 가정으로 남는다")
    ap.add_argument("--nfold-tol", dest="nfold_tol", type=float, default=25.0,
                    help="예측 θ가 확정 투창과 이 각도 안이면 이미 잡힌 것으로 본다")
    ap.add_argument("--nfold-min-cells", dest="nfold_min_cells", type=int, default=20,
                    help="예측 자리에서 뺄 칸이 이보다 적으면 건너뛴다. 슬리버 구멍 방지 "
                         "— 이 줄이 여러 번 뜨면 그 n 이 틀린 것이다")
    ap.add_argument("--color-inpaint", dest="color_inpaint", action="store_true",
                    help="채움 색을 (y,theta) 맵에서 LaMa 로 인페인팅한다. 끄면 최근접 "
                         "관측 정점 복사(중앙 22.3mm 떨어진 색). 형상에는 관여하지 않는다")
    ap.add_argument("--color-grid", dest="color_grid", type=int, default=1,
                    help="색 맵을 기하 격자의 k배로. 텍스처에서 뜨므로 정점 밀도에 "
                         "안 묶인다")
    ap.add_argument("--color-samples", dest="color_samples", type=int,
                    default=2_000_000,
                    help="표면에서 텍스처를 뜰 표본 수. 격자 칸 수의 수십 배면 충분하다")
    ap.add_argument("--no-split-rim", dest="no_split_rim", action="store_true",
                    help="테두리 법선 분리를 끈다. 켜 두면(기본) 외면·옆면이 정점을 "
                         "안 나눠 패치 테두리의 어두운 띠가 사라진다")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scene = trimesh.load(args.glb)
    obs = list(scene.geometry.values())[0]
    log(f"관측 메시 정점 {len(obs.vertices):,} 삼각형 {len(obs.faces):,}")

    Vw, Fw = S.load_glb(args.glb)          # 용접본 — 법선 판정과 격자 집계용
    filled, observed, (cx, cz), y_edges = cyl_map(
        Vw, Fw, args.ny, args.ntheta,
        fill_order=args.fill_order, fill_alpha=args.fill_alpha,
        fill_clamp_sigma=args.fill_clamp_sigma, compare=args.fill_compare,
        obs_stat=args.obs_stat, obs_blur_w=args.obs_blur)

    if args.relief:
        # 층 ③ — unet_relief.py 가 만든 요철을 채움 반경에 더한다.
        # 관측부는 relief 가 0 이라 영향이 없다.
        z = np.load(args.relief)
        if (int(z["n_y"]), int(z["n_theta"])) != (args.ny, args.ntheta):
            raise SystemExit(f"격자 불일치: relief {int(z['n_y'])}x{int(z['n_theta'])} "
                             f"vs 요청 {args.ny}x{args.ntheta}")
        rel = z["relief_mm"] / 250.0                      # 정규화 단위
        filled = filled + rel
        log(f"요철 적용 · 결손부 std {rel[~observed].std()*250:.2f}mm "
            f"· 관측부 {np.abs(rel[observed]).max()*250:.3f}mm(0이어야 함)")

    need, kinds = select_regions(~observed, args.min_cells, args.ow_cells, args.ow_deg)
    carved = []
    if args.nfold != "off":
        need, carved = carve_openwork_nfold(need, observed, kinds, args.ntheta,
                                            tol_deg=args.nfold_tol, mode=args.nfold,
                                            min_cells=args.nfold_min_cells)
    if args.mesh_upsample > 1:
        filled, need, y_edges = upsample_map(filled, need, y_edges, args.mesh_upsample)
    fill, rim = patch_mesh(filled, need, cx, cz, y_edges, args.thickness,
                           split_rim=not args.no_split_rim)
    if fill is None:
        raise SystemExit("채울 칸이 인접하지 않아 사각형을 못 만든다. 격자를 키울 것")

    # 위상은 **위치 기준으로 용접한 뒤** 센다.
    # UV 이음새와 #5 의 법선 분리로 정점이 쪼개져 있어 그냥 세면 과다 계수된다
    # (결손-메우는-방법-검토.md §1 과 같은 이유).
    welded = fill.copy()
    welded.merge_vertices()
    comp = welded.split(only_watertight=False)
    log(f"채움 메시 연결 성분 {len(comp)}개 "
        f"(가장 큰 것이 삼각형의 {100*max(len(c.faces) for c in comp)/len(fill.faces):.0f}%)")

    if args.snap_mm > 0:
        fill = snap_boundary(fill, obs, rim, args.snap_mm / 250.0)
    fill = smooth_fixed_boundary(fill, rim, args.smooth_iters, args.smooth_lambda,
                                 args.smooth_cap_mm / 250.0)
    colmap = None
    if args.color_inpaint:
        cny, cnt_ = args.ny * args.color_grid, args.ntheta * args.color_grid
        cye = np.linspace(y_edges[0], y_edges[-1], cny + 1)
        colmap = color_map_inpaint(obs, cx, cz, cye, cny, cnt_,
                                   n_samples=args.color_samples)
    if colmap is not None:
        fill = transfer_color_map(fill, colmap, cx, cz, cye, cny, cnt_)
    else:
        fill = transfer_color(fill, obs)

    out = trimesh.Scene()
    out.add_geometry(obs, node_name="region_carried", geom_name="region_carried")
    out.add_geometry(fill, node_name="region_filled", geom_name="region_filled")
    fill.vertex_normals            # 계산을 강제해 NORMAL 이 내보내지도록
    p = os.path.join(args.out, "restored_direct.glb")
    out.export(p, include_normals=True)
    patch_material(p, "region_filled", args.metallic, args.roughness)
    log(f"GLB → {p}")

    json.dump({
        "method": "cylindrical map -> mesh (no SLAT injection)",
        "grid": [args.ny, args.ntheta],
        "fill": {
            "scheme": {0: "advancing-copy (ssv.fill_grid)", 1: "harmonic",
                       2: "biharmonic"}[args.fill_order],
            "solved_on": ("profile residual r - rbar(y)" if args.fill_order
                          else "raw radius (neighbour copy, order-dependent)"),
            "alpha": args.fill_alpha if args.fill_order else None,
            "clamp_sigma": args.fill_clamp_sigma if args.fill_order else None,
            "cell_statistic": args.obs_stat,
            "obs_blur_cells": args.obs_blur,
            "color_grid": args.color_grid if args.color_inpaint else None,
            "mesh_upsample": args.mesh_upsample,
            "split_rim_normals": not args.no_split_rim,
        },
        "axis_xz": [cx, cz],
        "thickness": args.thickness,
        "observed_cells_frac": float(observed.mean()),
        "regions": {k: len(v) for k, v in kinds.items()},
        "openwork_kept_open": kinds["openwork"],
        "openwork_nfold_carved": carved,
        "snap_mm": args.snap_mm,
        "carried": {"vertices": int(len(obs.vertices)), "faces": int(len(obs.faces)),
                    "modified": 0},
        "filled": {"vertices": int(len(fill.vertices)), "faces": int(len(fill.faces)),
                   "components": len(comp)},
        "provenance": {
            "region_carried": "observed (TRELLIS A단계, 무수정)",
            "region_filled": ("회전대칭 가정 + n-fold 투창 가정 (ai_inferred)"
                              if carved else "회전대칭 가정 (ai_inferred)"),
            "region_filled_color": ("LaMa 인페인팅 (AI 생성)" if colmap is not None
                                    else "최근접 관측 정점 복사"),
        },
    }, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"),
        indent=2, ensure_ascii=False)
    log(f"manifest → {os.path.join(args.out, 'manifest.json')}")


if __name__ == "__main__":
    main()
