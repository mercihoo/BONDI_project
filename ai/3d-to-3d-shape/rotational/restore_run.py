"""
1차 관통 — 회전대칭 골격을 coords 로 주입해 결손을 실제로 채운다.

trellis2-복원-파이프라인-재구성.md §3 [B]+[D]. 목적은 품질이 아니라
"결손이 채워지는가 / 이음새가 없는가 / 재질이 어떻게 나오는가"를 처음 보는 것.

의도적으로 뺀 것 (1차 관통 범위 밖):
  - 투창 보호 (§7.2 ②)     → 굽다리 투창은 메워진다. 알고 하는 것
  - 단계별 cond 분리        → 형상·재질 모두 원본 손상 사진
  - 2D 인페인팅 (C단계)     → 없음
  - 1024_cascade           → 512 경로만. 빠른 확인용

출력 두 벌을 만들어 비교한다.
  fresh    : 합친 coords 에서 그냥 생성 (관측부도 새로 뽑힘)
  spliced  : 관측부 latent 를 A 것으로 덮어씀 (§3 설계의 "obs 복사")
splice 경계에서 불연속이 생기는지가 이번에 처음 확인된다.

사용:
  python restore_run.py --image "...jpg"
"""

import argparse
import os
import sys
import time

# TRELLIS.2 설치 경로. 사람마다 다르므로 환경변수로 덮어쓴다.
#   PowerShell:  $env:TRELLIS2_HOME = "D:\trellis2-stableprojectorz_v22\code"
INSTALL = os.environ.get("TRELLIS2_HOME") or r"C:\Users\<USER>\Downloads\trellis2-stableprojectorz_v22\code"
if not os.path.isdir(INSTALL):
    raise SystemExit(f"TRELLIS.2 설치 폴더를 못 찾았다: {INSTALL}\n"
                     f"환경변수 TRELLIS2_HOME 에 trellis2 의 code 폴더 경로를 지정할 것")
os.environ.setdefault("HF_HOME", os.path.join(INSTALL, "models"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.65"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
sys.path.insert(0, INSTALL)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def log(m):
    print(f"[RESTORE] {m}", flush=True)


# ── B단계: 회전대칭 골격 (복셀 공간) ──────────────────────────────

def pick_vertical_axis(xyz, grid):
    """단면이 가장 원형인 축을 세로축으로 고른다.

    TRELLIS 출력의 축 규약을 가정하지 않고 데이터에서 정한다.
    회전체라면 세로축 기준 단면이 원이고, 나머지 두 축 기준은 아니다.
    """
    scores = []
    for ax in range(3):
        oth = [i for i in range(3) if i != ax]
        h = xyz[:, ax]
        rel = []
        for k in range(grid):
            m = h == k
            if m.sum() < 12:
                continue
            p = xyz[m][:, oth].astype(np.float64)
            c = np.median(p, axis=0)
            r = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
            if r.mean() > 1e-6:
                rel.append(r.std() / r.mean())      # 상대 반경 산포. 원이면 작다
        scores.append(np.mean(rel) if rel else 9.9)
    ax = int(np.argmin(scores))
    log(f"세로축 판정: axis={ax} (상대 반경 산포 {[f'{s:.3f}' for s in scores]})")
    return ax


def revolve_fill(coords_obs, grid, wall=1, pct=95, ang_pad=1.6, mode="outer", outer_tol=1.5):
    """관측 복셀에서 profile(h) 를 뽑아 회전면을 만들고, 비어 있는 자리를 채운다.

    축은 가장 온전한(점유가 많은) 구간에서 잡는다 — 결손이 한쪽에 쏠리면
    전체 중앙값이 밀리기 때문 (문서 §6.4).
    """
    xyz = coords_obs[:, 1:].astype(np.int64)
    ax = pick_vertical_axis(xyz, grid)
    oth = [i for i in range(3) if i != ax]
    h_all = xyz[:, ax]
    p_all = xyz[:, oth].astype(np.float64)

    # 축 중심: 슬라이스별 점유 상위 40% 구간의 중앙값만 사용
    cnt = np.array([(h_all == k).sum() for k in range(grid)])
    solid = np.where(cnt >= np.percentile(cnt[cnt > 0], 60))[0]
    sel = np.isin(h_all, solid)
    cx, cy = np.median(p_all[sel], axis=0)
    log(f"축 중심 (온전 구간 {len(solid)}슬라이스 기준): ({cx:.2f}, {cy:.2f}) / 격자중심 {(grid-1)/2:.1f}")

    r_all = np.hypot(p_all[:, 0] - cx, p_all[:, 1] - cy)

    # profile(h): 높이별 외면 반경
    prof = np.full(grid, np.nan)
    for k in range(grid):
        m = h_all == k
        if m.sum() >= 6:
            prof[k] = np.percentile(r_all[m], pct)
    valid = ~np.isnan(prof)
    if valid.sum() < 4:
        raise RuntimeError("profile 을 뽑을 슬라이스가 부족하다")
    # 빈 높이는 선형 보간, 양끝은 가장 가까운 값으로
    idx = np.arange(grid)
    prof = np.interp(idx, idx[valid], prof[valid])
    lo, hi = idx[valid].min(), idx[valid].max()

    occ = np.zeros((grid, grid, grid), bool)
    occ[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True

    # θ-h 점유 격자. 표면이 이미 있는 (θ, h) 칸은 건드리지 않는다.
    #
    # p95 반경으로 만든 회전면은 실제 표면보다 살짝 바깥이라, 전 각도에 걸어버리면
    # 결손을 채우는 게 아니라 기존 껍질 바깥에 새 껍질을 씌운다(= 팽창).
    # 그래서 "빈 칸에만" 넣는다. 관측 표면은 한 복셀도 건드리지 않는다.
    nt = max(32, int(2 * np.pi * np.nanmax(prof)))         # 한 칸이 약 1복셀
    th_all = (np.degrees(np.arctan2(p_all[:, 1] - cy, p_all[:, 0] - cx)) + 360) % 360
    ti_all = np.clip((th_all / 360 * nt).astype(int), 0, nt - 1)
    filled = np.zeros((grid, nt), bool)
    filled[h_all, ti_all] = True

    # 칸별 외면 반경. 채움 반경을 여기서 각도 보간해 쓴다.
    #
    # 높이별 p95 하나로 채우면 안 된다 — 같은 높이에 대각단부·굽다리축·안팎 벽이
    # 섞여 반경 산포가 크고(관측 중앙 7.8 vs p95 12.1), p95 는 그 바깥이라
    # 채움면이 주변 표면보다 최대 4복셀 튀어나온다. 이음새가 단차로 보이는 원인.
    # 대신 빈 칸의 **각도 이웃 반경**을 보간해서 붙이면 경계가 이어진다.
    rmax = np.full(grid * nt, -1.0)
    np.maximum.at(rmax, h_all * nt + ti_all, r_all)
    rmax = rmax.reshape(grid, nt)
    rmax[rmax < 0] = np.nan

    add = []
    n_cells = n_empty = 0
    steps = []
    idx_t = np.arange(nt)
    for k in range(lo, hi + 1):
        row = rmax[k]
        has_any = ~np.isnan(row)

        if mode == "outer":
            # "복셀이 있나"가 아니라 "**외면**이 있나"로 판정한다.
            #
            # 바깥 벽은 깨져 없고 안쪽 벽만 남은 칸이 점유칸의 54%인데,
            # 단순 존재로 판정하면 그 칸들이 "표면 있음"으로 걸러져 채움에서 빠진다.
            # 채움-건너뜀이 번갈아 나오면서 렌더에 점선 무늬가 생기는 원인.
            if has_any.sum() >= 6:
                R = np.percentile(row[has_any], 75)         # 그 높이의 대표 외면 반경
                anchor = has_any & (row >= R - outer_tol)   # 외면이 실제로 있는 칸
            else:
                anchor = has_any
        else:
            anchor = has_any                                # 기존 동작 (mode="empty")

        if anchor.sum() >= 3:
            # 원주 방향 보간 (양옆으로 한 바퀴씩 늘려 감싼다). 외면이 있는 칸만 기준점으로 쓴다.
            xi = np.concatenate([idx_t[anchor] - nt, idx_t[anchor], idx_t[anchor] + nt])
            yi = np.tile(row[anchor], 3)
            r_row = np.interp(idx_t, xi, yi)
        else:
            r_row = np.full(nt, prof[k])                    # 그 높이가 통째로 비면 profile

        for t in range(nt):
            n_cells += 1
            if anchor[t]:
                continue                                    # 외면 있음 → 보존
            n_empty += 1
            r0 = r_row[t]
            if not np.isfinite(r0) or r0 < 0.5:
                continue
            steps.append(r0 - (prof[k] if np.isfinite(prof[k]) else r0))
            ang = np.radians((t + 0.5) / nt * 360)
            for dr in range(wall):
                rr = r0 - dr
                if rr < 0.5:
                    break
                u = int(round(cx + rr * np.cos(ang)))
                v = int(round(cy + rr * np.sin(ang)))
                if not (0 <= u < grid and 0 <= v < grid):
                    continue
                p = np.empty(3, np.int64)
                p[ax] = k
                p[oth[0]] = u
                p[oth[1]] = v
                add.append(p)
    if steps:
        log(f"  채움 반경: 이웃 보간 사용 (p95 profile 대비 중앙 {np.median(steps):+.1f}복셀)")

    if not add:
        return np.empty((0, 3), np.int64), ax, (cx, cy), prof
    add = np.unique(np.stack(add, 0), axis=0)
    new = add[~occ[add[:, 0], add[:, 1], add[:, 2]]]       # 이미 있는 건 제외
    log(f"θ-h 격자 {grid}x{nt} · 판정={mode} · 채울 칸 {n_empty:,}/{n_cells:,} ({100*n_empty/n_cells:.0f}%) → 신규 복셀 {len(new):,}")
    log(f"  h {lo}~{hi} · profile {prof[lo]:.1f}~{np.nanmax(prof):.1f}복셀 · wall {wall}")
    return new, ax, (cx, cy), prof


def theta_h_map(coords, grid, ax, center, title, nt=48):
    """채워지기 전/후를 눈으로 대조할 점유 맵."""
    xyz = coords[:, 1:].astype(np.int64)
    oth = [i for i in range(3) if i != ax]
    h = xyz[:, ax]
    p = xyz[:, oth].astype(np.float64)
    th = (np.degrees(np.arctan2(p[:, 1] - center[1], p[:, 0] - center[0])) + 360) % 360
    ti = np.clip((th / 360 * nt).astype(int), 0, nt - 1)
    print(f"\n--- {title} ---")
    for k in range(grid - 1, -1, -1):
        m = h == k
        if m.sum() == 0:
            continue
        row = np.zeros(nt, int)
        np.add.at(row, ti[m], 1)
        print(f"{k:3d}|{''.join('#' if c else '.' for c in row)}| {100*(row>0).mean():5.1f}%")


# ── 파이프라인 ────────────────────────────────────────────────────

def load_pipeline():
    try:
        from pipeline_worker import _apply_patches
        _apply_patches()
    except Exception as e:
        log(f"경고: flex_gemm 패치 실패 — {e}")
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 없음")
    log("파이프라인 로드 중")
    t0 = time.time()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipe.cuda()
    log(f"로드 완료 {time.time()-t0:.1f}s")
    return pipe


def get_cond(pipe, image, resolution=512):
    import torch
    if pipe.low_vram:
        pipe.image_cond_model.to(pipe.device)
    cond = pipe._cast_cond(pipe.get_cond([image], resolution))
    if pipe.low_vram:
        pipe.image_cond_model.cpu()
        torch.cuda.empty_cache()
    return cond


def sample_pair(pipe, cond, coords, seed):
    """512 경로. shape/tex 모두 512 모델, 좌표는 준 그대로 쓴다."""
    import torch
    torch.manual_seed(seed)
    t0 = time.time()
    s = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords, {})
    log(f"  shape SLAT {time.time()-t0:.1f}s · {tuple(s.feats.shape)}")
    torch.manual_seed(seed)
    t0 = time.time()
    t = pipe.sample_tex_slat(cond, pipe.models["tex_slat_flow_model_512"], s, {})
    log(f"  tex   SLAT {time.time()-t0:.1f}s")
    return s, t, 512


def sample_pair_cascade(pipe, cond512, cond1024, coords, seed, max_tokens=49152):
    """1024_cascade 경로.

    1단계 격자는 512 경로와 똑같이 32³이라 **주입 좌표는 그대로 쓰인다.**
    달라지는 건 2단계다 — 512 모델로 저해상도 shape SLAT 을 뽑고,
    shape 디코더가 그걸 업샘플해 64³ 좌표를 파생시킨 뒤 1024 모델로 다시 생성한다.

    이 경로에서는 spliced 를 만들 수 없다. latent_A 는 32³ 좌표의 512 잠재값인데
    여기 shape SLAT 은 64³ 에 살아서 행이 대응되지 않는다. fresh 만 나온다.
    """
    import torch
    torch.manual_seed(seed)
    t0 = time.time()
    s, res = pipe.sample_shape_slat_cascade(
        cond512, cond1024,
        pipe.models["shape_slat_flow_model_512"],
        pipe.models["shape_slat_flow_model_1024"],
        512, 1024,
        coords, {}, max_tokens,
    )
    log(f"  shape SLAT cascade {time.time()-t0:.1f}s · {tuple(s.feats.shape)} · res={res}")
    s._spatial_cache = {}
    torch.cuda.empty_cache()

    torch.manual_seed(seed)
    t0 = time.time()
    t = pipe.sample_tex_slat(cond1024, pipe.models["tex_slat_flow_model_1024"], s, {})
    log(f"  tex   SLAT {time.time()-t0:.1f}s")
    torch.cuda.empty_cache()
    return s, t, res


def export(pipe, s, t, res, path, decimation=150_000, texsize=1024):
    import torch
    import o_voxel
    mesh = pipe.decode_and_cleanup(s, t, res)[0]
    mesh.attrs = mesh.attrs.float()
    log(f"  디코딩 verts={len(mesh.vertices):,} faces={len(mesh.faces):,}")
    for m in pipe.models.values():
        m.cpu()
    torch.cuda.empty_cache()
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipe.pbr_attr_layout, grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation, texture_size=texsize,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )
    glb.export(path)
    del mesh, glb
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"  GLB → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wall", type=int, default=2, help="회전면 벽 두께(복셀)")
    ap.add_argument("--pct", type=float, default=95)
    ap.add_argument("--tag", default=None,
                    help="결과를 out/<tag>/ 에 넣는다. 생략하면 실행 시각으로 자동 생성")
    ap.add_argument("--pipeline", default="512", choices=["512", "1024_cascade"],
                    help="1단계 격자는 둘 다 32³이라 주입 좌표는 동일. cascade 는 2단계를 64³로 정밀화")
    ap.add_argument("--masked", action="store_true",
                    help="RePaint 식 마스킹 샘플링. 관측부를 매 스텝 고정하고 결손부만 생성한다")
    ap.add_argument("--fill-npz", dest="fill_npz", default=None,
                    help="skeleton_cyl.py 가 만든 fill_cyl.npz. 주면 revolve_fill 대신 이걸 쓴다")
    ap.add_argument("--fill-mode", dest="fill_mode", default="outer", choices=["empty", "outer"],
                    help="empty=칸에 복셀이 하나라도 있으면 보존(구버전) / outer=외면이 있어야 보존")
    ap.add_argument("--outer-tol", dest="outer_tol", type=float, default=1.5,
                    help="대표 외면 반경에서 이만큼 안쪽이면 '외면 없음'으로 본다 (복셀)")
    ap.add_argument("--decimation", type=int, default=150_000)
    ap.add_argument("--texsize", type=int, default=1024)
    args = ap.parse_args()

    # 캐시(관측 coords·latent)는 out/ 루트에 두고 결과만 버전 폴더로 나눈다.
    # 캐시까지 버전마다 새로 뽑으면 1단계를 매번 다시 돌려야 해서 느리고,
    # 관측 근거가 버전마다 달라지면 비교 자체가 안 된다.
    cache = args.out
    tag = args.tag or time.strftime("v%m%d-%H%M")
    outdir = os.path.join(args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    log(f"캐시 {cache}")
    log(f"결과 {outdir}")

    import torch
    pipe = load_pipeline()
    res, grid = 512, 32

    image = pipe.preprocess_image(Image.open(args.image))
    torch.manual_seed(args.seed)
    cond = get_cond(pipe, image, 512)
    cond1024 = get_cond(pipe, image, 1024) if args.pipeline == "1024_cascade" else None
    log(f"경로 {args.pipeline}")

    # ── 관측 coords + 관측 latent (probe 결과 재사용) ──────────────
    cpath = os.path.join(cache, "coords_stage1.npy")
    if os.path.exists(cpath):
        coords_obs = torch.from_numpy(np.load(cpath)).to(pipe.device).int().contiguous()
        log(f"관측 coords 재사용 {coords_obs.shape[0]:,}")
    else:
        torch.manual_seed(args.seed)
        coords_obs = pipe.sample_sparse_structure(cond, grid, 1, {})
        np.save(cpath, coords_obs.cpu().numpy())
        log(f"관측 coords {coords_obs.shape[0]:,}")

    apath = os.path.join(cache, "latent_A.npz")
    if os.path.exists(apath):
        z = np.load(apath)
        shA = torch.from_numpy(z["shape_feats"]).to(pipe.device)
        txA = torch.from_numpy(z["tex_feats"]).to(pipe.device)
        log("관측 latent 재사용 (latent_A.npz)")
    else:
        log("[A] 관측 latent 생성 (512 경로 — 캐시는 항상 512로 만든다)")
        sA, tA, _ = sample_pair(pipe, cond, coords_obs, args.seed)
        shA, txA = sA.feats.clone(), tA.feats.clone()
        del sA, tA
        torch.cuda.empty_cache()

    obs_np = coords_obs.cpu().numpy()

    # ── B: 회전대칭 골격 ─────────────────────────────────────────
    log("[B] 회전대칭 골격 생성")
    if args.fill_npz:
        # 원통 전개 판 (skeleton_cyl.py). 조밀한 (y,θ) 격자에서 채움을 결정하고
        # 복셀화는 거기서 이미 한 번만 끝냈다. 축·프로파일은 로그용으로만 다시 잡는다.
        z = np.load(args.fill_npz)
        fill_xyz = z["fill"].astype(np.int64)
        log(f"채움을 외부에서 로드: {args.fill_npz} · {len(fill_xyz):,}복셀")
        ax = pick_vertical_axis(obs_np[:, 1:].astype(np.int64), grid)
        oth = [i for i in range(3) if i != ax]
        pp = obs_np[:, 1:].astype(np.float64)[:, oth]
        center = tuple(np.median(pp, axis=0))
        prof = np.full(grid, np.nan)
    else:
        fill_xyz, ax, center, prof = revolve_fill(obs_np, grid, wall=args.wall, pct=args.pct,
                                                  mode=args.fill_mode, outer_tol=args.outer_tol)
    if len(fill_xyz) == 0:
        log("채울 복셀이 없다. 종료")
        return
    fill = np.concatenate([np.zeros((len(fill_xyz), 1), np.int64), fill_xyz], 1)
    all_np = np.concatenate([obs_np.astype(np.int64), fill], 0)
    key = all_np[:, 1] * grid * grid + all_np[:, 2] * grid + all_np[:, 3]
    order = np.argsort(key, kind="stable")            # 1단계와 같은 C-order 규약
    all_np = all_np[order]
    is_obs = np.isin(all_np[:, 1] * grid * grid + all_np[:, 2] * grid + all_np[:, 3],
                     obs_np[:, 1] * grid * grid + obs_np[:, 2] * grid + obs_np[:, 3])
    log(f"coords {len(obs_np):,} → {len(all_np):,} (관측 {is_obs.sum():,} / 추정 {(~is_obs).sum():,})")

    theta_h_map(obs_np, grid, ax, center, "채우기 전 (관측)")
    theta_h_map(all_np, grid, ax, center, "채운 뒤 (관측+추정)")

    np.savez_compressed(os.path.join(outdir, "coords_restored.npz"),
                        coords=all_np.astype(np.int32), is_observed=is_obs,
                        axis=ax, center=np.array(center), profile=prof)

    coords_all = torch.from_numpy(all_np).to(pipe.device).int().contiguous()

    # 관측 행 ↔ A latent 행 대응. splice 와 masked 양쪽이 쓴다.
    rows = np.where(is_obs)[0]
    ka = obs_np[:, 1] * grid * grid + obs_np[:, 2] * grid + obs_np[:, 3]
    kr = all_np[rows][:, 1] * grid * grid + all_np[rows][:, 2] * grid + all_np[rows][:, 3]
    amap = {int(k): i for i, k in enumerate(ka)}
    src = torch.as_tensor([amap[int(k)] for k in kr], device=pipe.device)
    dst = torch.as_tensor(rows, device=pipe.device)

    if args.masked:
        # ── D-masked: 생성 도중 관측부 고정 ────────────────────
        import samplers_masked as MS
        if args.pipeline != "512":
            raise SystemExit("마스킹은 512 경로에서만. cascade 는 좌표 격자가 달라진다")
        ss, ts = MS.attach(pipe)
        n = len(all_np)
        m = torch.zeros(n, 1, dtype=torch.bool, device=pipe.device)
        m[dst] = True

        ks = torch.zeros(n, shA.shape[1], device=pipe.device)
        ks[dst] = MS.normalized_known(shA[src], pipe.shape_slat_normalization, pipe.device)
        ss.set_mask(ks, m)
        kt = torch.zeros(n, txA.shape[1], device=pipe.device)
        kt[dst] = MS.normalized_known(txA[src], pipe.tex_slat_normalization, pipe.device)
        ts.set_mask(kt, m)
        log(f"[D-masked] 관측 {int(m.sum()):,}행 고정 · 결손 {n - int(m.sum()):,}행만 생성")

        sM, tM, res = sample_pair(pipe, cond, coords_all, args.seed)

        # 검증: 관측부가 실제로 A 와 같은가
        got = sM.feats[dst].float().cpu().numpy()
        want = shA[src].float().cpu().numpy()
        d = np.abs(got - want).max()
        log(f"  관측부 보존 확인: A 대비 최대 절대차 {d:.3e} "
            f"({'통과' if d < 5e-2 else '어긋남 — 정규화 왕복 확인'})")

        export(pipe, sM, tM, res, os.path.join(outdir, "R_masked.glb"),
               args.decimation, args.texsize)
        log("완료 (masked)")
        return

    # ── D: 주입 생성 ─────────────────────────────────────────────
    log(f"[D] 합친 coords 로 생성 ({args.pipeline})")
    if args.pipeline == "1024_cascade":
        sR, tR, res = sample_pair_cascade(pipe, cond, cond1024, coords_all, args.seed)
    else:
        sR, tR, res = sample_pair(pipe, cond, coords_all, args.seed)

    log("[D-fresh] 관측부도 새로 뽑힌 그대로 디코딩")
    export(pipe, sR, tR, res, os.path.join(outdir, "R_fresh.glb"),
           args.decimation, args.texsize)

    if args.pipeline == "1024_cascade":
        # cascade 의 shape SLAT 은 64³ 좌표에 산다. latent_A 는 32³ 512 잠재값이라
        # 행이 대응되지 않아 덮어쓸 수 없다. provenance 는 coords_restored.npz 의
        # 32³ 라벨로 남아 있으니, 필요하면 HR 좌표를 32³로 내려서 대조하면 된다.
        log("[D-spliced] cascade 경로에서는 생략 (좌표 격자 불일치)")
        log("완료")
        return

    # 관측부 latent 를 A 것으로 덮어쓴다 (§3 obs 복사)
    log("[D-spliced] 관측부 latent 를 A 로 덮어쓰고 디코딩")
    sf = sR.feats.clone()
    tf = tR.feats.clone()
    sf[dst] = shA[src].to(sf.dtype)
    tf[dst] = txA[src].to(tf.dtype)
    sS, tS = sR.replace(sf), tR.replace(tf)
    del sR, tR
    torch.cuda.empty_cache()
    export(pipe, sS, tS, res, os.path.join(outdir, "R_spliced.glb"))

    log("완료. R_fresh.glb / R_spliced.glb 를 A_normal.glb 와 대조할 것")


if __name__ == "__main__":
    main()
