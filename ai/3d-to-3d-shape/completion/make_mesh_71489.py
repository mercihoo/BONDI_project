# -*- coding: utf-8 -*-
"""
make_mesh_71489.py — 파인튜닝 AdaPoinTr 의 출력 점군을 **메시**로 만든다.

왜 (y, θ) 격자인가
  마칭큐브를 쓰려면 skimage 나 open3d 가 필요한데 둘 다 venv 에 없고,
  TRELLIS venv 는 본 파이프라인이 쓰는 것이라 건드리지 않는다.
  그리고 v24 가 이미 `(y, θ)` 격자로 메시를 만든다 — **정점 공유가 정의상 보장**되고
  이 유물이 회전체라 잘 맞는다. 같은 표현을 쓰면 v32 결과와 바로 겹쳐 볼 수 있다.

노드를 둘로 나눈다 (`NFR-ETH-003` 규약)
  region_observed   출력점 중 입력 표면 가까이 있던 것에서 나온 면
  region_ai         **보충점**(모델이 새로 만든 점)에서 나온 면
  v32 의 `region_carried` / `region_filled` 와 같은 구조다.

한계 — 미리 적는다
  출력이 8,192점이라 격자 칸당 3점 남짓이다. v32 가 쓰는 관측 94,877 정점에 비하면
  성기고, 빈 칸은 이웃 평균으로 메운다. **형상 확인용이지 납품본이 아니다.**

사용
  PY=".../venv/Scripts/python.exe"
  $PY make_mesh_71489.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
OUT = WORK / "restored_adapointr"


def load_measure_glb():
    p = ROOT / "gupdari-shape-prior" / "measure_glb.py"
    spec = importlib.util.spec_from_file_location("measure_glb", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cyl(P, up, c2):
    ax = [i for i in range(3) if i != up]
    x, y = P[:, ax[0]] - c2[0], P[:, ax[1]] - c2[1]
    return P[:, up], np.arctan2(y, x), np.hypot(x, y), ax


def openwork_cells_nfold(occ, band_h=10, max_deg=50.0, max_cells=400):
    """투창 판정 — **주기 띠 안에서는 θ폭 문턱을 풀어 준다.**

    문턱만 쓰면 넓은 투창(θ폭 64~86°)을 결손으로 오분류해 패치가 덮어버린다.
    실측: 굽다리 투창 두 단 구간의 빈 칸 443 중 **195(44%)만** 잡혔다.

    `nfold_bands()` 가 그 띠를 주기적이라고 확인해 주면, 그 안의 갇힌 빈 칸은
    폭이 넓어도 투창으로 본다. 파이프라인이 크기에서 주기로 옮겨간 이유와 같다.
    """
    from scipy import ndimage as ndi
    empty = ~occ
    nh, nt = empty.shape
    per, info = nfold_bands(occ, band_h=band_h)

    lab, n = ndi.label(np.hstack([empty, empty]))
    out = np.zeros_like(empty)
    seen = set()
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) == 0:
            continue
        key = (int(ys.min()), int(ys.max()), len(ys))
        if key in seen:
            continue
        seen.add(key)
        if ys.min() == 0 or ys.max() == nh - 1:
            continue                                   # 가장자리에 닿음 = 결손
        cols = np.unique(xs % nt)
        deg = len(cols) / nt * 360.0
        in_band = per[ys].mean() > 0.6                 # 성분 대부분이 주기 띠 안
        if in_band:
            if deg > 180.0:                            # 둘레를 거의 다 돌면 결손
                continue
        elif deg > max_deg or len(ys) > max_cells:
            continue
        out[ys, xs % nt] = True
    return out, per, info


def openwork_cells(empty, max_deg=50.0, max_cells=400):
    """빈 칸 중 **투창**(의도된 구멍)을 가려낸다. 결손과 달리 메우면 안 된다.

    기준은 파이프라인 것을 그대로 쓴다
    (`[수학] 결손-메우는-방법-검토.md` §3 표 · `--openwork-max-deg 50`):

        | | 투창 | 파단 결손 |
        | 크기 | 작고 일정 | 크고 제각각 |
        | θ폭 | 좁다 | 넓다 |
        | 갇힘 | 위아래 끝에 안 닿는다 | 대개 가장자리까지 이어진다 |

    θ 는 한 바퀴 도므로 **좌우로 이어붙여** 라벨링한다 — 안 그러면 θ=±180° 를
    지나는 구멍이 둘로 쪼개진다.
    """
    from scipy import ndimage as ndi
    nh, nt = empty.shape
    lab, n = ndi.label(np.hstack([empty, empty]))
    out = np.zeros_like(empty)
    seen = set()
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        if len(ys) == 0:
            continue
        key = (int(ys.min()), int(ys.max()), len(ys))
        if key in seen:              # 복제본이라 같은 성분이 두 번 나온다
            continue
        seen.add(key)
        if ys.min() == 0 or ys.max() == nh - 1:
            continue                 # 가장자리에 닿음 = 결손
        cols = np.unique(xs % nt)
        if len(cols) / nt * 360.0 > max_deg or len(ys) > max_cells:
            continue                 # 너무 넓거나 크다 = 결손
        out[ys, xs % nt] = True
    return out


def nfold_bands(occ, band_h=6, nmax=8):
    """**θ 주기가 서는 높이 띠**를 찾는다 — 투창 띠 판정.

    크기·θ폭 문턱만으로는 못 가른다. 파이프라인이 겪은 그대로다
    (`[수학] 투창-n-fold-판정.md`): *"#84 θ폭 70°, #2 θ폭 80° — 문턱을 70까지
    열면 진짜 파단까지 샌다"*. 그래서 **크기를 버리고 주기로** 판정한다.

    방법도 그 노트 그대로다.
        띠 안에서 θ 를 따라 '관측 비율'이 오르내리는 신호에 푸리에를 건다.
        투창이 n개면 n차 조화가 서고, 파단은 비주기라 배경으로 깔린다.

    그 노트의 실측 (96x144 격자)
        띠 A h 8~18   4-fold 13.5%
        띠 B h 27~38  3-fold  9.9%
        바리 h 55~80  1-fold 40.1%   ← 주기가 아니라 큰 파단 하나
        굽다리 하단   1-fold 18.5%

    반환 (주기띠 마스크, 띠별 진단 리스트)
    """
    nh, nt = occ.shape
    is_per = np.zeros(nh, bool)
    info = []
    for i0 in range(0, max(1, nh - band_h + 1)):
        sig = occ[i0:i0 + band_h].mean(0).astype(float)
        sig = sig - sig.mean()
        if sig.std() < 1e-6:
            continue
        mag = np.abs(np.fft.rfft(sig))[1:nmax + 1]
        if mag.sum() <= 0:
            continue
        share = mag / mag.sum()
        top = int(np.argmax(share)) + 1                 # 1차 조화가 n=1
        info.append((i0, i0 + band_h, top, float(share[top - 1])))
        if top >= 2:                                    # n>=2 가 1위 = 주기적
            is_per[i0:i0 + band_h] = True
    return is_per, info


def fill_unknown(grid, known, iters=1000):
    """빈 칸을 이웃 평균으로 채운다. θ 는 한 바퀴 도므로 좌우를 이어붙인다.

    v24 의 이중조화(`Δ²u=0`)보다 단순한 라플라스 평활이다. 여기서는 격자가 성겨
    빈 칸이 흩어져 있을 뿐이라 이 정도로 충분하다.

    **회전대칭을 가정하지 않는다.** 위·아래·좌·우 네 이웃의 평균일 뿐이라
    높이 방향과 각도 방향을 똑같이 본다. 실측(71489)으로 보간 결과는
    그 행의 중앙 반지름과 **중앙 1.24mm · 90분위 3.62mm · 최대 12.47mm** 다르다
    (회전 가정이면 0 이어야 한다). 다만 결손이 θ 로 넓으면 가장 가까운
    기지값이 같은 높이의 다른 각도라, **결과는 회전체에 가까워진다** —
    가정이 아니라 기하가 그렇게 만든다.

    반복은 400 → **1000**. 야코비 반복이라 경계의 영향이 한 번에 한 칸씩 퍼지는데,
    71489 의 가장 넓은 결손은 θ 로 196칸이다. 400회면 중앙은 이미 맞지만
    가장 깊은 곳이 **최대 1.56mm** 덜 수렴했다 (800회 0.18mm · 2000회 0.01mm).
    비용은 0.08초 → 0.18초라 올리는 편이 낫다.
    """
    g = grid.copy()
    m = ~known
    if not m.any():
        return g
    g[m] = np.nanmean(grid[known])
    for _ in range(iters):
        up_ = np.roll(g, 1, 0); up_[0] = g[0]
        dn = np.roll(g, -1, 0); dn[-1] = g[-1]
        lf = np.roll(g, 1, 1)                 # θ 는 순환
        rt = np.roll(g, -1, 1)
        avg = (up_ + dn + lf + rt) / 4.0
        g[m] = avg[m]
    return g


def grid_mesh(h_edges, th_n, R, up, c2, ax, skip=None):
    """격자 → 삼각형. 정점을 공유하므로 조각나지 않는다."""
    nh, nt = R.shape
    hs = 0.5 * (h_edges[:-1] + h_edges[1:])
    ths = (np.arange(nt) + 0.5) / nt * 2 * np.pi - np.pi

    V = np.zeros((nh * nt, 3))
    for i in range(nh):
        for j in range(nt):
            v = np.zeros(3)
            v[up] = hs[i]
            v[ax[0]] = c2[0] + R[i, j] * np.cos(ths[j])
            v[ax[1]] = c2[1] + R[i, j] * np.sin(ths[j])
            V[i * nt + j] = v

    F, cell = [], []
    for i in range(nh - 1):
        for j in range(nt):
            j2 = (j + 1) % nt
            # 네 모서리 중 하나라도 투창이면 면을 만들지 않는다 → 구멍이 남는다
            if skip is not None and (skip[i, j] or skip[i, j2]
                                     or skip[i + 1, j] or skip[i + 1, j2]):
                continue
            a, b = i * nt + j, i * nt + j2
            c, d = (i + 1) * nt + j, (i + 1) * nt + j2
            F.append([a, c, b]); cell.append((i, j))
            F.append([b, c, d]); cell.append((i, j))
    return V, np.array(F), cell


def artifact_color(glb: Path) -> list:
    """원본 텍스처에서 유물 대표색을 뽑는다.

    모델 출력에는 색이 **아예 없다** — 입출력이 `(x,y,z)` 뿐이다.
    그래서 재질은 원본에서 가져와야 한다. v32 는 `texture_from_mesh.py` 로
    텍스처를 통째로 옮겨 심지만, 여기선 단색이면 충분하다.
    극단 화소(배경 여백·검은 그림자)를 자르고 중앙값을 쓴다.
    """
    if not glb.is_file():
        return [128, 118, 99]
    g = trimesh.load(str(glb), process=False)
    m = g.to_geometry() if hasattr(g, "to_geometry") else g
    img = getattr(getattr(m.visual, "material", None), "baseColorTexture", None)
    if img is None:
        return [128, 118, 99]
    a = np.asarray(img.convert("RGB")).reshape(-1, 3)
    lum = a.mean(1)
    keep = (lum > 25) & (lum < 235)
    sel = a[keep] if keep.sum() > 1000 else a
    return [int(v) for v in np.median(sel, 0)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORK / "pred_71489_A_normal_ft.npz"))
    ap.add_argument("--tex-src", type=Path,
                    default=ROOT / "gupdari-71489" / "in" / "A_normal.glb",
                    help="대표색을 뽑아올 원본 GLB")
    ap.add_argument("--color", default="", help="직접 지정 예: 128,118,99")
    ap.add_argument("--openwork-max-deg", type=float, default=50.0,
                    help="투창 θ폭 상한. 파이프라인 기본값과 같다")
    ap.add_argument("--openwork-max-cells", type=int, default=400)
    ap.add_argument("--nh", type=int, default=44, help="높이 칸")
    ap.add_argument("--nt", type=int, default=64, help="각도 칸")
    ap.add_argument("--name", default="A_normal_adapointr_ft")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print("[!] 없다: " + str(src))
        return 1
    d = np.load(src, allow_pickle=True)
    P = np.asarray(d["pred"], np.float64)
    new = np.asarray(d["new"], bool)
    up = int(d["up"])
    dense = np.asarray(d["input_dense"], np.float64)   # 12만점 — 투창 판정용

    mg = load_measure_glb()
    _, c2 = mg.fit_axis(P, up)
    h, th, r, ax = cyl(P, up, c2)

    lo, hi = np.percentile(h, [0.5, 99.5])
    h_edges = np.linspace(lo, hi, args.nh + 1)
    hb = np.clip(np.digitize(h, h_edges) - 1, 0, args.nh - 1)
    tb = np.clip(((th + np.pi) / (2 * np.pi) * args.nt).astype(int), 0, args.nt - 1)

    R = np.full((args.nh, args.nt), np.nan)
    AI = np.zeros((args.nh, args.nt))
    known = np.zeros((args.nh, args.nt), bool)
    for i in range(args.nh):
        for j in range(args.nt):
            m = (hb == i) & (tb == j)
            if m.any():
                R[i, j] = np.median(r[m])          # 바깥면. 껍질이라 중앙값이 안정적
                AI[i, j] = new[m].mean()
                known[i, j] = True
    print("격자 %dx%d · 채워진 칸 %d (%.0f%%) · 칸당 평균 %.1f점"
          % (args.nh, args.nt, known.sum(), 100 * known.mean(), len(P) / known.sum()))

    # --- 투창은 메우지 않는다 -------------------------------------------
    # 출력 8192점은 성겨서 "빈 칸"이 구멍인지 표본 부족인지 못 가른다.
    # **입력 조밀 점군(12만점)** 으로 판정한다 — 거기서 비면 진짜 구멍이다.
    hd, td, _rd, _ax = cyl(dense, up, c2)
    hbd = np.clip(np.digitize(hd, h_edges) - 1, 0, args.nh - 1)
    tbd = np.clip(((td + np.pi) / (2 * np.pi) * args.nt).astype(int), 0, args.nt - 1)
    dense_occ = np.zeros((args.nh, args.nt), bool)
    dense_occ[hbd, tbd] = True
    ow = openwork_cells(~dense_occ, max_deg=args.openwork_max_deg,
                        max_cells=args.openwork_max_cells)
    print("투창 판정: 입력이 빈 칸 %d 중 **%d칸을 투창으로 보고 뚫어 둔다**"
          % ((~dense_occ).sum(), ow.sum()))

    R = fill_unknown(np.nan_to_num(R, nan=np.nanmean(R)), known)
    V, F, cell = grid_mesh(h_edges, args.nt, R, up, c2, ax, skip=ow)

    # 면을 노드 둘로 가른다 — 보충점에서 나온 면이 region_ai
    ai_face = np.array([AI[i, j] > 0.3 for (i, j) in cell])
    print("면 %d 중 AI 추정 %d (%.1f%%)" % (len(F), ai_face.sum(), 100 * ai_face.mean()))

    # 면 방향을 **축에서 바깥으로** 맞춘다.
    #   이걸 안 하면 뷰어가 뒷면을 잘라내(backface culling) 안쪽만 보이고
    #   바깥은 뚫린 것처럼 투명해진다. 회전체라 판정이 간단하다 —
    #   면 중심의 반지름 방향과 법선의 내적 부호를 본다.
    tri = V[F]
    cen = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    radial = np.zeros_like(cen)
    radial[:, ax[0]] = cen[:, ax[0]] - c2[0]
    radial[:, ax[1]] = cen[:, ax[1]] - c2[1]
    inward = (nrm * radial).sum(1) < 0
    F[inward] = F[inward][:, ::-1]
    print("법선 바깥으로 정렬: %d면 뒤집음 (%.0f%%)" % (inward.sum(), 100 * inward.mean()))

    rgb = artifact_color(args.tex_src) if not args.color else \
        [int(v) for v in args.color.split(",")]
    print("재질 단색 RGB %s" % rgb)

    OUT.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for tag, sel in (("region_observed", ~ai_face), ("region_ai", ai_face)):
        if not sel.any():
            continue
        m = trimesh.Trimesh(vertices=V, faces=F[sel], process=False)
        m.remove_unreferenced_vertices()
        # 두께가 없는 한 겹 면이라 **양면 렌더**로 내보낸다.
        # 안 그러면 뷰어가 뒷면을 잘라내 안쪽에서 볼 때 뚫린 것처럼 보인다.
        # (법선은 이미 바깥으로 맞춰뒀다 — 방향 문제가 아니라 양면 설정이 빠졌던 것)
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name=tag,
                baseColorFactor=[c / 255 for c in rgb] + [1.0],
                metallicFactor=0.0, roughnessFactor=0.9,     # 도기 — 광택 없음
                doubleSided=True))
        scene.add_geometry(m, geom_name=tag)
        print("  %-16s V=%6d F=%6d  (PBR 단색 · 양면)" % (tag, len(m.vertices), len(m.faces)))

    print("  ※ 두 노드는 같은 색이다. 어디가 AI 인지는 **노드 이름**으로 갈린다")

    glb = OUT / (args.name + ".glb")
    scene.export(str(glb))
    print("\n저장 → " + str(glb))
    print("v32 비교본: ../gupdari-71489/out/v32-*/restored_labshift.glb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
