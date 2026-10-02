"""
완형 메시에 TRELLIS 재질을 입힌다 — 형상은 수학, 재질은 AI (로드맵 S10)

배경
  SLAT 주입(형상 생성)은 종결됐다. 남은 것은 **재질**이다.
  TRELLIS.2 배포본에 `texturing_pipeline.json` 이 있고, 그 구성이 정확히 이 용도다.

    "name": "Trellis2TexturingPipeline"
    models: shape_slat_encoder / tex_slat_decoder / tex_slat_flow_model_512
            ^^^^^^^^^^^^^^^^^^ shape **flow** 가 없다 — 형상을 만들지 않고 읽기만 한다

  클래스 파일(`trellis2_image_to_tex.py`)은 이 포크에서 빠졌으나 부품은 전부 있다.
  `trellis2/` 를 건드리지 않고 밖에서 조립한다.

    v24 완형 메시
      → o_voxel.convert.mesh_to_flexible_dual_grid   coords · dual_vertices · intersected
      → FlexiDualGridVaeEncoder                      shape SLAT (32ch)
      → sample_tex_slat(cond=사진, shape_slat)        tex SLAT
      → decode_tex_slat                              PBR 속성 볼륨
      → to_glb(vertices=v24 정점, faces=v24 면, attr_volume=새 속성)

  마지막 줄이 요점이다. **디코딩된 메시는 버리고 속성만 취한다.**
  인코딩→디코딩은 VAE 왕복이라 형상이 뭉개지는데, v24 의 이면각 0.948° 와
  carried 차이 0.00e+00 을 잃을 이유가 없다.

단계
  encode   메시 → shape SLAT. **왕복 오차를 재서 이 경로가 성립하는지 먼저 본다**
  texture  shape SLAT + 사진 → PBR 속성          (encode 통과 후)

사용
  python texture_from_mesh.py encode  --glb out/v24-cyl+biharm+n6+nn+up2/restored_direct.glb
  python texture_from_mesh.py texture --image in/Mounted_Bow.jpg
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
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
sys.path.insert(0, INSTALL)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = "microsoft/TRELLIS.2-4B"
ENC = f"{REPO}/ckpts/shape_enc_next_dc_f16c32_fp16"
DEC = f"{REPO}/ckpts/shape_dec_next_dc_f16c32_fp16"

try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


def log(m):
    print(f"[TEX] {m}", flush=True)


# ── 메시 로드 ─────────────────────────────────────────────────────

def load_mesh(path, split=False):
    """GLB 를 읽는다.

    split=False → 정점·면을 하나로 합쳐 (V, F). 인코딩용.
    split=True  → [(노드이름, V, F), ...]. 베이크는 노드마다 따로 해야
                  `region_carried` / `region_filled` 분리가 살아남는다 (NFR-ETH-003).
    """
    import trimesh
    s = trimesh.load(path, process=False, force="scene")
    parts = [(n, np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int64))
             for n, g in s.geometry.items()]
    log(f"메시 {os.path.basename(path)} · 노드 {len(parts)} "
        f"[{' + '.join(f'{n}({len(f):,}면)' for n, _v, f in parts)}]")
    if split:
        return parts
    V, F, off = [], [], 0
    for _n, v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    V, F = np.vstack(V), np.vstack(F)
    log(f"  정점 {len(V):,} · 면 {len(F):,} · bbox {(V.max(0) - V.min(0)).round(4)}")
    return V, F


# GLB 는 Y-up 이고 TRELLIS 내부(디코더 출력 · 속성 볼륨 coords)는 Z-up 이다.
# `to_glb` 가 마지막에 Z-up → Y-up 을 넣는다 — 실측으로 확인했다:
#   extents [0.2, 0.6, 0.4] 상자를 넣으면 [0.2, 0.4, 0.6] 이 나온다. (x,y,z) → (x, z, -y)
# 그래서 `A_normal.glb` 도 v24 도 **Y-up** 이다. 그걸 그대로 to_glb 에 넣으면 이중 변환이 되고,
# 인코더에 넣으면 사진 조건과 90° 어긋난 채로 잠재값이 만들어진다.
# 넣기 전에 Z-up 으로 되돌린다. det=+1 인 회전이라 면 winding 은 유지된다.
YUP2ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # (x,y,z) → (x,-z,y)


def yup_to_zup(V):
    return V @ YUP2ZUP.T


def to_unit_cube(V, pad=0.99999, tol=0.01):
    """TRELLIS aabb [-0.5, 0.5] 로 맞춘다.

    tol 만큼의 초과는 무시한다. v24 는 bbox Y 가 1.0021 이라 0.001 벗어나는데,
    그것 때문에 정규화를 걸면 **되돌릴 때 좌표계가 틀어진다** — trimesh Scene 의
    apply_scale/apply_translation 은 graph transform 에 곱해지고, to_glb 가 심는
    Y-up 변환과 섞여 축이 뒤바뀐다(실측: 축 보정 후에도 중앙 1.29% 어긋남).
    무변환으로 두는 쪽이 안전하고, aabb 를 그 정도 넘어도 클리핑 영향이 없다.
    """
    lo, hi = V.min(0), V.max(0)
    over = max(-0.5 - lo.min(), hi.max() - 0.5, 0.0)
    if over <= tol:
        log(f"  단위 큐브 안 (초과 {over:.5f} <= tol {tol}). **무변환**")
        return V, None
    c = (lo + hi) / 2
    sc = pad / (hi - lo).max()
    log(f"  정규화: center {c.round(4)} scale {sc:.6f}  (초과 {over:.5f})")
    return (V - c) * sc, (c, sc)


# ── 1단계: encode ─────────────────────────────────────────────────

def stage_encode(args):
    import torch
    import o_voxel
    from trellis2 import models
    from trellis2.modules.sparse import SparseTensor

    V, F = load_mesh(args.glb)
    V = yup_to_zup(V)                     # GLB(Y-up) → TRELLIS 내부(Z-up)
    log(f"  Z-up 변환 후 bbox {(V.max(0) - V.min(0)).round(4)}")
    V, xform = to_unit_cube(V)

    # 메시 → flexible dual grid. mesh2ovox.py 예제와 같은 가중치·정렬 규약을 쓴다
    t0 = time.time()
    vi, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        torch.from_numpy(V).float(), torch.from_numpy(F).long(),
        grid_size=args.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
    )
    order = torch.argsort(o_voxel.serialize.encode_seq(vi))
    vi, dv, inter = vi[order], dv[order], inter[order]
    log(f"dual grid {time.time() - t0:.1f}s · 복셀 {len(vi):,} · "
        f"intersected {inter.float().mean(0).cpu().numpy().round(3)}")

    # mesh2ovox.py 와 같이 복셀 내 로컬 좌표 [0,1] 로 바꾼다.
    # 인코더가 `vertices.feats - 0.5` 를 하므로 로컬 좌표를 기대한다.
    dv_local = (dv * args.resolution - vi).clamp(0, 1).float()

    dev = "cuda"
    coords = torch.cat([torch.zeros_like(vi[:, :1]), vi], dim=1).int().to(dev).contiguous()
    v_st = SparseTensor(feats=dv_local.to(dev), coords=coords)
    i_st = SparseTensor(feats=inter.float().to(dev), coords=coords)

    # flex_gemm 저VRAM 경로 회피 — 포크 버그.
    # conv_flex_gemm.sparse_conv3d_forward 는 neighbor map 이 256MB 를 넘으면
    # `x.feats = torch.empty(0, ..., device='cpu')` 로 feats 를 잠시 비운 뒤
    # `torch.Size([*x.shape, *x.spatial_shape])` 를 만든다. 이때 `_shape` 가 아직
    # None 이면 __cal_shape 가 **비워진 1D feats** 로 재계산해 (1,) 만 내놓고,
    # spatial 3개와 합쳐도 4개라 `N, C, W, H, D = shape` 가 터진다.
    # `.shape` 를 한 번 건드려 `_shape` 를 확정해두면 replace 가 그것을 전파하므로
    # trellis2/ 를 고치지 않고 피할 수 있다.
    _ = v_st.shape, i_st.shape

    log(f"인코더 로드 {ENC}")
    enc = models.from_pretrained(ENC).to(dev).eval()
    t0 = time.time()
    with torch.no_grad():
        # sample_posterior=False → 사후분포 평균. 결정론적이라 재현된다
        slat = enc(v_st, i_st, sample_posterior=False)
    log(f"encode {time.time() - t0:.1f}s · shape SLAT {tuple(slat.feats.shape)} "
        f"· latent coords {tuple(slat.coords.shape)}")
    log(f"  feats mean {slat.feats.mean().item():+.4f} std {slat.feats.std().item():.4f} "
        f"(latent_A 참고: mean -0.0298 std 5.3698)")

    enc.cpu()
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.slat) or ".", exist_ok=True)
    np.savez_compressed(
        args.slat,
        coords=slat.coords.cpu().numpy().astype(np.int32),
        shape_feats=slat.feats.float().cpu().numpy(),
        normalized=np.bool_(False),      # 인코더 출력 = sample_shape_slat 과 같은 공간
        resolution=np.int64(args.resolution),
        src_glb=np.str_(os.path.abspath(args.glb)),
    )
    log(f"저장 → {args.slat}")

    if args.roundtrip:
        _roundtrip(slat, V, F, args)


def _roundtrip(slat, V, F, args):
    """왕복 오차 — 인코딩이 형상을 얼마나 잃는가.

    이 값이 크면 재질 생성 자체는 되더라도 **속성 볼륨이 v24 형상과 어긋난다**.
    디코딩 메시를 버리고 v24 정점에 속성을 샘플링할 것이므로, 관건은
    '표면이 거의 같은 자리에 있는가' 다.
    """
    import torch
    import torch.nn.functional as Fn
    from scipy.spatial import cKDTree
    from trellis2 import models
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from o_voxel.convert import flexible_dual_grid_to_mesh

    log("")
    log(f"왕복 검증 - 디코더 로드 {DEC}")
    dec = models.from_pretrained(DEC).to("cuda").eval()
    dec.set_resolution(args.resolution)
    # 인코더 출력은 fp16 이지만 디코더 `from_latent` 는 fp32 를 받는다
    # (convert_to_fp16 이 blocks·output_layer 만 바꾼다). probe_intersected 와 같은 이유다.
    slat = slat.replace(slat.feats.float())
    with torch.no_grad():
        h, _subs = SparseUnetVaeDecoder.forward(dec, slat, return_subs=True)
        h._spatial_cache.clear()
        h = h.replace(h.feats.float())
        m = dec.voxel_margin
        verts = h.replace((1 + 2 * m) * torch.sigmoid(h.feats[..., 0:3]) - m)
        inter = h.replace(h.feats[..., 3:6] > 0)
        qlerp = h.replace(Fn.softplus(h.feats[..., 6:7]))
        mv, mf = flexible_dual_grid_to_mesh(
            h.coords[:, 1:], verts.feats, inter.feats, qlerp.feats,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=args.resolution, train=False)
    mv = mv.cpu().numpy()
    mf = mf.cpu().numpy()
    dec.cpu()
    torch.cuda.empty_cache()

    log(f"  복원 메시 정점 {len(mv):,} · 삼각형 {len(mf):,}  (원본 {len(V):,} / {len(F):,})")

    # 표면 대 표면 거리. 정점 샘플로 양방향
    rng = np.random.default_rng(0)
    a = V[rng.choice(len(V), min(60000, len(V)), replace=False)]
    b = mv[rng.choice(len(mv), min(60000, len(mv)), replace=False)]
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    span = float((V.max(0) - V.min(0)).max())
    log(f"  원본→복원  중앙 {np.median(d_ab):.5f}  p95 {np.percentile(d_ab, 95):.5f}  "
        f"최대 {d_ab.max():.5f}")
    log(f"  복원→원본  중앙 {np.median(d_ba):.5f}  p95 {np.percentile(d_ba, 95):.5f}  "
        f"최대 {d_ba.max():.5f}")
    log(f"  (bbox 최대변 {span:.4f} 기준 중앙 {100 * np.median(d_ab) / span:.3f}%)")

    ext_o = (V.max(0) - V.min(0)).round(4)
    ext_r = (mv.max(0) - mv.min(0)).round(4)
    log(f"  bbox 원본 {ext_o} → 복원 {ext_r}")

    if args.roundtrip_glb:
        import trimesh
        trimesh.Trimesh(mv, mf, process=False).export(args.roundtrip_glb)
        log(f"  복원 메시 → {args.roundtrip_glb}")


# ── 2단계: texture ────────────────────────────────────────────────

def stage_texture(args):
    """shape SLAT + 사진 → tex SLAT → PBR 속성 볼륨 → **v24 메시**에 베이크.

    요점은 마지막 줄이다. `decode_and_cleanup` 이 돌려주는 메시(300만 정점)는 버리고
    **속성 볼륨만** 취해 v24 정점·면에 입힌다. `to_glb` 가 메시와 속성을 따로 받고
    UV 언랩·베이크를 해주므로 가능하다. `remesh=False` 로 위상도 안 건드린다.
    """
    import torch
    import o_voxel
    from PIL import Image
    from restore_run import load_pipeline, get_cond
    from trellis2.modules.sparse import SparseTensor

    z = np.load(args.slat, allow_pickle=False)
    src = str(z["src_glb"])
    log(f"shape SLAT {args.slat} · {z['shape_feats'].shape} · src {src}")

    # 베이크 대상 — 인코딩에 쓴 그 메시. 노드를 살려 읽고, 같은 좌표 규약을 쓴다
    mesh_path = src if os.path.exists(src) else args.glb
    parts = [(nm, yup_to_zup(v), f) for nm, v, f in load_mesh(mesh_path, split=True)]
    Vall = np.vstack([v for _n, v, _f in parts])
    log(f"  Z-up 변환 후 bbox {(Vall.max(0) - Vall.min(0)).round(4)}")
    _Vn, xform = to_unit_cube(Vall)

    pipe = load_pipeline()
    image = pipe.preprocess_image(Image.open(args.image))
    torch.manual_seed(args.seed)
    # 해상도별로 flow 모델과 이미지 조건이 짝지어져 있다 (512 / 1024).
    # 1024 는 latent 가 64³ 이라 인코딩도 --resolution 1024 로 해둔 것이어야 한다.
    cond = get_cond(pipe, image, args.resolution)
    flow_key = f"tex_slat_flow_model_{args.resolution}"
    assert flow_key in pipe.models, f"{flow_key} 없음 · 가진 것 {[k for k in pipe.models if 'tex_slat_flow' in k]}"
    log(f"tex flow {flow_key} · cond {args.resolution}")

    enc_res = int(z["resolution"]) if "resolution" in z else 512
    assert enc_res == args.resolution, (
        f"인코딩 해상도 {enc_res} != 요청 {args.resolution}. "
        f"먼저 `encode --resolution {args.resolution}` 을 돌릴 것")
    slat = SparseTensor(
        feats=torch.from_numpy(z["shape_feats"]).to(pipe.device).float(),
        coords=torch.from_numpy(z["coords"]).to(pipe.device).int().contiguous(),
    )
    _ = slat.shape                      # flex_gemm 저VRAM 경로 회피 (stage_encode 주석 참조)

    # sample_tex_slat 이 내부에서 (shape_slat - mean)/std 정규화를 한다.
    # 인코더 출력은 sample_shape_slat 과 같은 공간이므로 그대로 넣는다.
    torch.manual_seed(args.seed)
    t0 = time.time()
    tex = pipe.sample_tex_slat(cond, pipe.models[flow_key], slat, {})
    log(f"tex SLAT {time.time() - t0:.1f}s · {tuple(tex.feats.shape)}")

    # shape 디코더의 subs 가 있어야 tex 를 디코딩할 수 있다.
    # decode_and_cleanup 이 그 순서와 VRAM offload 를 이미 담고 있어 그대로 쓴다.
    t0 = time.time()
    mv = pipe.decode_and_cleanup(slat, tex, args.resolution)[0]
    log(f"decode {time.time() - t0:.1f}s · 속성 복셀 {tuple(mv.attrs.shape)} "
        f"· (버리는 디코딩 메시 {len(mv.vertices):,}정점)")
    log(f"  layout {dict((k, (s.start, s.stop)) for k, s in mv.layout.items())}")

    for m in pipe.models.values():
        m.cpu()
    torch.cuda.empty_cache()

    # 노드마다 따로 베이크한다. to_glb 는 준 메시를 한 덩어리로 내놓으므로
    # 한 번에 넣으면 region_carried / region_filled 분리가 사라진다 (NFR-ETH-003).
    import trimesh
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scene = trimesh.Scene()
    for name, Vi, Fi in parts:
        Vi = Vi if xform is None else (Vi - xform[0]) * xform[1]
        t0 = time.time()
        gi = o_voxel.postprocess.to_glb(
            vertices=torch.from_numpy(Vi).float().cuda(),
            faces=torch.from_numpy(Fi).int().cuda(),
            attr_volume=mv.attrs.float(),
            coords=mv.coords,
            attr_layout=mv.layout,
            grid_size=args.resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation,
            texture_size=args.texsize,
            remesh=False,                  # v24 형상을 건드리지 않는다
            use_tqdm=False,
        )
        geos = list(gi.geometry.values()) if hasattr(gi, "geometry") else [gi]
        nf = sum(len(g.faces) for g in geos)
        log(f"  to_glb [{name}] {time.time() - t0:.1f}s · 면 {len(Fi):,} → {nf:,}")
        for k, g in enumerate(geos):
            scene.add_geometry(g, geom_name=name if len(geos) == 1 else f"{name}_{k}")

    scene.export(args.out)
    log(f"GLB → {args.out}  (노드 {len(scene.geometry)})")
    log("")
    log("아직 안 한 것: 관측부 재질 보존 마스크. 지금은 전 면이 새 재질이다 (로드맵 S10 미정 14)")


def stage_roundtrip(args):
    """저장된 shape SLAT 으로 왕복 오차만 다시 잰다. 인코딩을 다시 하지 않는다."""
    import torch
    from trellis2.modules.sparse import SparseTensor

    z = np.load(args.slat, allow_pickle=False)
    src = str(z["src_glb"])
    log(f"shape SLAT {args.slat} · {z['shape_feats'].shape} · src {src}")
    slat = SparseTensor(
        feats=torch.from_numpy(z["shape_feats"]).cuda().float(),
        coords=torch.from_numpy(z["coords"]).cuda().int().contiguous(),
    )
    V, F = load_mesh(src if os.path.exists(src) else args.glb)
    V = yup_to_zup(V)
    V, _ = to_unit_cube(V)
    _roundtrip(slat, V, F, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "roundtrip", "texture"])
    ap.add_argument("--glb", default=os.path.join(HERE, "out", "v24-cyl+biharm+n6+nn+up2",
                                                  "restored_direct.glb"))
    ap.add_argument("--image", default=os.path.join(HERE, "in", "Mounted_Bow.jpg"))
    ap.add_argument("--slat", default=os.path.join(HERE, "out", "probe", "slat_from_mesh.npz"))
    # 버전 폴더 이름에 방법을 적는다 — cyl(원통전개) biharm(이중조화) n6(6-fold) trellistex(TRELLIS 재질)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "v27-cyl+biharm+n6+trellistex+up2",
                                                  "restored_trellistex.glb"))
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--decimation", type=int, default=1_000_000)
    ap.add_argument("--texsize", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--roundtrip", action="store_true", default=True,
                    help="인코딩 왕복 오차를 잰다 (기본 켜짐)")
    ap.add_argument("--no-roundtrip", dest="roundtrip", action="store_false")
    ap.add_argument("--roundtrip-glb", dest="roundtrip_glb", default=None)
    args = ap.parse_args()

    if args.stage == "encode":
        stage_encode(args)
    elif args.stage == "roundtrip":
        stage_roundtrip(args)
    else:
        stage_texture(args)


if __name__ == "__main__":
    main()
