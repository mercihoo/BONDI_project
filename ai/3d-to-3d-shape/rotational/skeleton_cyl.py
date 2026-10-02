"""
B단계 골격 생성 — 원통 전개 판.

기여자 2026-08-27 자기지도 U-Net 파이프라인의 전처리부를 그대로 재사용한다.
기존 restore_run.revolve_fill 은 32x100 θ-h 격자에서 손으로 회전면을 지었고,
그 결과 팽창 → 점선 → 가로 띠 순으로 양자화 인공물이 계속 나왔다.

여기서는 채움을 **조밀한 (y, θ) 격자에서 결정하고 복셀화는 마지막 한 번만** 한다.

재사용 (읽기 전용, 수정하지 않음):
  <PROJECT_ROOT>/restore_pottery_self_supervised.py
    fit_axis          최소제곱 원피팅 + 슬라이스 중앙값 (bbox 중심보다 견고)
    classify_surfaces 면 법선 vs 방사 방향으로 내·외면 분리
                      ← restore_run 의 outer_tol 휴리스틱을 대체하는 정식 해법
    radial_grid       (y, θ) → r 격자
    fill_profile      높이별 프로파일 (보간 + 이동평균)
    fill_grid         결손 칸 반복 채움

아직 안 쓰는 것: train_pottery_unet_v2.PeriodicResidualUNet.
그 가중치는 빗살무늬토기 전용 주기 문양 prior 라 굽다리바리에 전이되지 않는다.
쓰려면 이 유물의 온전한 부위로 자기지도 재학습이 필요하다 (고주파 단계).

사용:
  python skeleton_cyl.py --glb out/probe/A_normal.glb --out out/cyl
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# 콘솔이 cp949 면 em dash(U+2014) 같은 문자에서 print 가 죽는다.
# 인코딩은 그대로 두고(한글이 깨지므로) errors 만 완화한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


# restore_pottery_self_supervised.py 는 같은 폴더에 둔다.
# 다른 곳에 있으면 SSV_DIR 환경변수로 지정한다.
SSV_DIR = os.environ.get("SSV_DIR") or os.path.dirname(os.path.abspath(__file__))


def log(m):
    print(f"[CYL] {m}", flush=True)


def load_ssv():
    """은빈님 모듈을 읽기 전용으로 임포트한다."""
    if SSV_DIR not in sys.path:
        sys.path.insert(0, SSV_DIR)
    import restore_pottery_self_supervised as ssv
    return ssv


# ── GLB 로드 ─────────────────────────────────────────────────────

def load_glb(path):
    d = open(path, "rb").read()
    off, js, bo = 12, None, 0
    while off < len(d):
        cl, ct = struct.unpack("<II", d[off:off + 8])
        if ct == 0x4E4F534A:
            js = json.loads(d[off + 8:off + 8 + cl].decode())
        else:
            bo = off + 8
        off += 8 + cl + ((4 - cl % 4) % 4 if cl % 4 else 0)
    pr = js["meshes"][0]["primitives"][0]
    A, BV = js["accessors"], js["bufferViews"]

    def rd(ai, ncomp, dt):
        a = A[ai]
        bv = BV[a["bufferView"]]
        s = bo + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        w = np.dtype(dt).itemsize * ncomp
        st = bv.get("byteStride") or w
        raw = np.frombuffer(d[s:s + st * a["count"]], np.uint8).reshape(a["count"], st)[:, :w].copy()
        return raw.view(dt).reshape(-1, ncomp)

    V = rd(pr["attributes"]["POSITION"], 3, np.float32).astype(np.float64)
    ci = A[pr["indices"]]["componentType"]
    dt = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[ci]
    F = rd(pr["indices"], 1, dt).reshape(-1, 3).astype(np.int64)

    # GLB 는 UV 때문에 같은 위치의 정점이 쪼개져 있다. 법선 판정과 격자 집계 모두
    # 위치 기준이라 용접해서 쓴다.
    key = np.round(V * 1e6).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    Vw = np.zeros((len(uniq), 3))
    cnt = np.bincount(inv, minlength=len(uniq))
    for k in range(3):
        Vw[:, k] = np.bincount(inv, weights=V[:, k], minlength=len(uniq)) / cnt
    return Vw, inv[F]


# ── 골격 생성 ────────────────────────────────────────────────────

def find_axis_map(V, obs_coords, grid):
    """월드 좌표 → 복셀 인덱스 변환(축 순열 + 반사 8x6=48가지)을 데이터로 찾는다.

    TRELLIS 복셀 인덱스와 GLB 월드 축은 같은 순서가 아니다.
    실측: idx0=+x, idx1=-z, idx2=+y (일치율 98.5%). 그냥 (x,y,z)로 쓰면 16.1%.
    하드코딩하면 파이프라인 설정이 바뀔 때 조용히 틀리므로 매번 찾고 검증한다.
    """
    import itertools
    ko = set(map(int, obs_coords[:, 0] * grid * grid + obs_coords[:, 1] * grid + obs_coords[:, 2]))
    best = (0.0, None, None)
    for p in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            w = V[:, list(p)] * np.array(sg)
            vx = np.floor((w + 0.5) * grid).astype(int).clip(0, grid - 1)
            k = set(map(int, vx[:, 0] * grid * grid + vx[:, 1] * grid + vx[:, 2]))
            hit = len(k & ko) / max(1, len(k))
            if hit > best[0]:
                best = (hit, p, sg)
    hit, p, sg = best
    log(f"축 매핑: 월드{tuple('xyz'[i] for i in p)} 부호{sg} → 복셀 인덱스 · 일치율 {hit*100:.1f}%")
    if hit < 0.85:
        log(f"  경고: 일치율이 낮다. 관측 coords 와 GLB 가 같은 실행에서 나온 것인지 확인할 것")
    return p, np.array(sg)


def build_fill(glb_path, grid=32, n_y=160, n_theta=240, wall=2,
               scale_mm=250.0, min_count=1, obs_coords=None):
    ssv = load_ssv()
    ssv.N_Y, ssv.N_THETA = n_y, n_theta      # 모듈 상수만 바꾼다 (파일 수정 아님)

    V, F = load_glb(glb_path)
    log(f"메시 정점 {len(V):,} · 삼각형 {len(F):,}")

    # fit_axis 에 mm 단위 하드코딩 임계(35)가 있다. 정규화 좌표(±0.5)를 그대로 넣으면
    # 모든 후보가 통과해 버리므로 실물 크기 비슷하게 스케일해서 넘긴다.
    Vm = V * scale_mm

    cx, cz = ssv.fit_axis(Vm)
    log(f"축 (최소제곱 원피팅): ({cx/scale_mm:+.4f}, {cz/scale_mm:+.4f}) 정규화 단위")

    outer, inner, stats = ssv.classify_surfaces(Vm, F, cx, cz)
    log(f"면 법선 분리: 외면 정점 {stats['outer_vertices']:,} / "
        f"내면 {stats['inner_vertices']:,} · 외면 면 {stats['outer_faces']:,}")

    y = Vm[:, 1]
    y_edges = np.linspace(y.min(), y.max() + 1e-9, n_y + 1)
    r_out, counts = ssv.radial_grid(Vm, outer, cx, cz, y_edges)

    observed = counts >= min_count
    log(f"격자 {n_y}x{n_theta} · 외면 관측 칸 {observed.sum():,}/{observed.size:,} "
        f"({100*observed.mean():.1f}%) → 결손 {(~observed).sum():,}")

    row_med = np.array([np.nanmedian(r_out[i]) if np.isfinite(r_out[i]).any() else np.nan
                        for i in range(n_y)])
    profile = ssv.fill_profile(row_med)
    filled = ssv.fill_grid(r_out, profile)
    log(f"채운 뒤 유한값 {np.isfinite(filled).mean()*100:.1f}% · "
        f"반경 {np.nanmin(filled)/scale_mm:.3f}~{np.nanmax(filled)/scale_mm:.3f} (정규화)")

    # ── 복셀화는 여기서 딱 한 번 ────────────────────────────────
    yc = (y_edges[:-1] + y_edges[1:]) / 2
    th = (np.arange(n_theta) + 0.5) / n_theta * 2 * np.pi
    step = scale_mm / grid                      # 복셀 한 칸의 스케일 단위 길이

    iy, it = np.nonzero((~observed) & np.isfinite(filled))
    pts = []
    for dr in range(wall):
        rr = filled[iy, it] - dr * step
        ok = rr > 0
        pts.append(np.stack([
            cx + rr[ok] * np.cos(th[it[ok]]),
            yc[iy[ok]],
            cz + rr[ok] * np.sin(th[it[ok]]),
        ], axis=1))
    pts = np.concatenate(pts, 0) / scale_mm     # 정규화 좌표로 복귀

    # 월드 → 복셀 인덱스. 축 순서·부호가 다르므로 데이터로 찾은 변환을 쓴다.
    if obs_coords is None:
        raise RuntimeError("obs_coords 가 필요하다 (축 매핑 검증용)")
    perm, sign = find_axis_map(V, obs_coords, grid)
    vox = np.floor((pts[:, list(perm)] * sign + 0.5) * grid).astype(np.int64)
    vox = vox[((vox >= 0) & (vox < grid)).all(1)]
    vox = np.unique(vox, axis=0)
    log(f"결손 칸 {len(iy):,} → 복셀 후보 {len(vox):,}")

    meta = dict(axis=[cx / scale_mm, cz / scale_mm], perm=list(perm), sign=sign.tolist(),
                n_y=n_y, n_theta=n_theta,
                wall=wall, observed_frac=float(observed.mean()),
                outer_vertices=stats["outer_vertices"], inner_vertices=stats["inner_vertices"])
    return vox, filled / scale_mm, observed, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="A단계 손상 GLB")
    ap.add_argument("--coords", default=None, help="관측 coords_stage1.npy (겹치는 복셀 제외용)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--ny", type=int, default=160)
    ap.add_argument("--ntheta", type=int, default=240)
    ap.add_argument("--wall", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.coords:
        raise SystemExit("--coords 가 필요하다 (축 매핑 검증에 쓴다)")
    obs = np.load(args.coords)[:, 1:].astype(np.int64)

    vox, filled, observed, meta = build_fill(
        args.glb, grid=args.grid, n_y=args.ny, n_theta=args.ntheta, wall=args.wall,
        obs_coords=obs)

    if True:
        g = args.grid
        ko = set((obs[:, 0] * g * g + obs[:, 1] * g + obs[:, 2]).tolist())
        kv = vox[:, 0] * g * g + vox[:, 1] * g + vox[:, 2]
        keep = np.array([k not in ko for k in kv.tolist()])
        log(f"관측 복셀과 겹치는 {int((~keep).sum()):,}개 제외 → 신규 {int(keep.sum()):,}")
        vox = vox[keep]

    np.savez_compressed(os.path.join(args.out, "fill_cyl.npz"),
                        fill=vox.astype(np.int32), filled_map=filled.astype(np.float32),
                        observed=observed, meta=json.dumps(meta, ensure_ascii=False))
    log(f"저장 → {os.path.join(args.out, 'fill_cyl.npz')}")
    log(f"신규 복셀 {len(vox):,}")


if __name__ == "__main__":
    main()
