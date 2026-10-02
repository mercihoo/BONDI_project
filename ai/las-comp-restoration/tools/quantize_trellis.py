"""TRELLIS 파이프라인의 DiT 만 골라 양자화한다.

왜 골라서 하나 — 계획 §5.2:
  · DiT(`slat_flow_*`, `ss_flow_*`)  : 가중치 대부분이 여기. 내린다.
  · VAE·디코더(`slat_dec_*`, `ss_dec_*`): 기하를 실제로 만들어내는 곳. **FP16 유지.**
  · FFN down-projection               : 확산 모델 양자화에서 품질 유지의 핵심 지렛대. 제외.

RTX 4070 Laptop 은 sm_89 라 FP8 이 하드웨어로 지원된다(torchao 요구선 ≥ 8.9).

쓰는 법 — 환경변수로 켠다. 끄면(기본) 아무것도 안 건드린다.

    LASCOMP_QUANT=fp8   python run_lascomp_image_condition_single.py ...
    LASCOMP_QUANT=int8  python run_lascomp_image_condition_single.py ...
    LASCOMP_QUANT=off   (기본)

실행 스크립트를 고치지 않고 붙이려면 sitecustomize 대신 measure_vram.py 에서
이 모듈의 `apply_from_env(pipeline)` 를 부르면 된다.
"""
from __future__ import annotations

import os
import re
import sys

import torch


# DiT 만 고른다. 디코더/인코더는 건드리지 않는다.
#
# pipeline.json 의 실제 키 이름 (image-large 기준):
#   sparse_structure_flow_model  ← DiT   (ss_flow_img_dit_L_16l8_fp16)
#   slat_flow_model              ← DiT   (slat_flow_img_dit_L_64l8p2_fp16)
#   sparse_structure_decoder     ← 디코더 (ss_dec_conv3d_16l8_fp16)
#   slat_decoder_gs / _rf / _mesh ← 디코더
# 'dit' 는 **파일명에만** 있고 키 이름에는 없다. 키로 걸러야 하므로 'flow' 로 잡는다.
DIT_NAME_PAT = re.compile(r"flow", re.IGNORECASE)
KEEP_FP16_PAT = re.compile(r"(_dec_|_enc_|decoder|encoder|vae)", re.IGNORECASE)

# 이 이름이 들어간 Linear 는 양자화에서 뺀다 — 어텐션 출력 투영(to_out) 등.
SKIP_MODULE_PAT = re.compile(r"(down_proj|fc2|proj_out|to_out|out_proj)", re.IGNORECASE)

# FFN down-projection 은 **이름이 아니라 모양**으로 잡는다.
# TRELLIS DiT 의 FFN 은 1024→4096→1024 이고 층 이름이 `mlp.2` 라 위 이름 규칙에 안 걸린다.
# 첫 FP8 실행(13:3x)은 그래서 FFN down-proj 를 양자화하고 어텐션 to_out 만 뺐다 — 계획 §5.2 와 반대.
# CPU 진단 기준 DiT 하나에 (4096,1024) 층이 24개 = 약 100 MiB 의 절감을 포기하는 대신 품질을 지킨다.
# 비교 실험용으로 LASCOMP_QUANT_AGGRESSIVE=1 이면 이 규칙을 끈다.
_AGGRESSIVE = os.environ.get("LASCOMP_QUANT_AGGRESSIVE", "").strip() in ("1", "true", "yes")


def _is_ffn_down_proj(module: torch.nn.Linear) -> bool:
    return module.in_features >= 2 * module.out_features


def _sm_version() -> tuple[int, int]:
    p = torch.cuda.get_device_properties(0)
    return p.major, p.minor


def _linear_filter(module: torch.nn.Module, fqn: str) -> bool:
    """torchao 에 넘길 필터 — True 면 이 Linear 를 양자화한다."""
    if not isinstance(module, torch.nn.Linear):
        return False
    if SKIP_MODULE_PAT.search(fqn):
        return False
    if not _AGGRESSIVE and _is_ffn_down_proj(module):
        return False
    # 아주 작은 층은 이득이 없고 오차만 는다
    if module.in_features < 256 or module.out_features < 256:
        return False
    return True


def _param_bytes(t: torch.Tensor) -> int:
    """실제 저장 바이트. torchao 의 AffineQuantizedTensor 는 겉 dtype 이 fp16 처럼 보여서
    numel*element_size 로 재면 **양자화 전과 같은 값**이 나온다 — 그래서 절감이 0 으로 찍혔다.
    서브클래스면 안쪽 저장 텐서(int_data/float8_data + scale)를 직접 더한다."""
    impl = getattr(t, "tensor_impl", None)
    if impl is not None:
        tot = 0
        for a in ("int_data", "float8_data", "scale", "zero_point"):
            x = getattr(impl, a, None)
            if torch.is_tensor(x):
                tot += x.numel() * x.element_size()
        if tot:
            return tot
    return t.numel() * t.element_size()


def _is_quantized(m: torch.nn.Module) -> bool:
    w = getattr(m, "weight", None)
    return w is not None and type(w).__name__ not in ("Parameter", "Tensor")


def quantize_module(mod: torch.nn.Module, mode: str, tag: str = "") -> dict:
    """mod 안의 Linear 를 양자화한다. 무엇을 바꿨는지 돌려준다."""
    from torchao.quantization import quantize_

    on_cuda = any(p.is_cuda for p in mod.parameters())
    if on_cuda:
        torch.cuda.synchronize()
    cuda_before = torch.cuda.memory_allocated() if on_cuda else None
    before = sum(_param_bytes(p) for p in mod.parameters())

    # torchao 는 버전마다 이름이 다르다. 0.5.x 는 함수형(float8_weight_only),
    # 최신은 Config 클래스형(Float8WeightOnlyConfig). 둘 다 받는다.
    import torchao.quantization as q

    def _cfg(new_name, old_name):
        if hasattr(q, new_name):
            return getattr(q, new_name)()
        if hasattr(q, old_name):
            return getattr(q, old_name)()
        raise RuntimeError(
            "torchao 에서 %s / %s 를 못 찾았다 — 버전을 확인해야 한다 (설치본: %s)"
            % (new_name, old_name, getattr(__import__("torchao"), "__version__", "?"))
        )

    if mode == "fp8":
        major, minor = _sm_version()
        if (major, minor) < (8, 9):
            raise RuntimeError(
                "FP8 은 compute capability 8.9 이상이 필요하다 (현재 sm_%d%d). "
                "int8 로 대신 돌려라." % (major, minor)
            )
        cfg = _cfg("Float8WeightOnlyConfig", "float8_weight_only")
    elif mode == "int8":
        cfg = _cfg("Int8WeightOnlyConfig", "int8_weight_only")
    else:
        raise ValueError("모르는 양자화 모드: %s" % mode)

    linears = [m for m in mod.modules() if isinstance(m, torch.nn.Linear)]
    n_eligible = sum(1 for n, m in mod.named_modules()
                     if isinstance(m, torch.nn.Linear) and _linear_filter(m, n))
    quantize_(mod, cfg, filter_fn=_linear_filter)
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    after = sum(_param_bytes(p) for p in mod.parameters())
    n_quantized = sum(1 for m in mod.modules()
                      if isinstance(m, torch.nn.Linear) and _is_quantized(m))
    cuda_after = torch.cuda.memory_allocated() if on_cuda else None

    info = {
        "tag": tag,
        "mode": mode,
        "linear_count": len(linears),
        "eligible_count": n_eligible,
        "quantized_count": n_quantized,          # 0 이면 필터가 전부 걸러낸 것 — 그게 진짜 실패다
        "bytes_before_mib": round(before / 1024 ** 2, 1),
        "bytes_after_mib": round(after / 1024 ** 2, 1),
        "saved_mib": round((before - after) / 1024 ** 2, 1),
    }
    if on_cuda:
        info["cuda_alloc_before_mib"] = round(cuda_before / 1024 ** 2, 1)
        info["cuda_alloc_after_mib"] = round(cuda_after / 1024 ** 2, 1)
        info["cuda_saved_mib"] = round((cuda_before - cuda_after) / 1024 ** 2, 1)
    return info


def apply(pipeline, mode: str) -> list[dict]:
    """파이프라인의 DiT 들만 양자화한다. 바꾼 내역을 돌려준다."""
    if mode in ("off", "", None):
        return []

    models = getattr(pipeline, "models", None)
    if not isinstance(models, dict):
        raise RuntimeError(
            "pipeline.models 를 못 찾았다 — TRELLIS 구조가 바뀌었는지 확인해야 한다"
        )

    changed = []
    for name, mod in models.items():
        if not isinstance(mod, torch.nn.Module):
            continue
        if KEEP_FP16_PAT.search(name):
            print("  [유지] %s — 디코더/인코더는 FP16 그대로" % name)
            continue
        if not DIT_NAME_PAT.search(name):
            print("  [건너뜀] %s — DiT 아님" % name)
            continue
        info = quantize_module(mod, mode, tag=name)
        changed.append(info)
        extra = ""
        if "cuda_saved_mib" in info:
            extra = "  | GPU 할당 %.0f → %.0f (-%.0f)" % (
                info["cuda_alloc_before_mib"], info["cuda_alloc_after_mib"], info["cuda_saved_mib"])
        print("  [양자화] %-28s %s  Linear %d/%d 변환  %.0f → %.0f MiB (-%.0f)%s"
              % (name, mode, info["quantized_count"], info["linear_count"],
                 info["bytes_before_mib"], info["bytes_after_mib"], info["saved_mib"], extra))
        if info["quantized_count"] == 0:
            print("  경고: %s 에서 변환된 Linear 가 0개 — 필터가 전부 걸러냈다" % name, file=sys.stderr)

    if not changed:
        print("  경고: 양자화된 모듈이 없다. models 키 이름을 확인해라:",
              list(models.keys()), file=sys.stderr)
    else:
        total = sum(c["saved_mib"] for c in changed)
        print("  합계 절감: %.1f MiB" % total)
    return changed


def apply_from_env(pipeline) -> list[dict]:
    mode = os.environ.get("LASCOMP_QUANT", "off").strip().lower()
    if mode in ("off", ""):
        return []
    print("=== 양자화 적용: %s ===" % mode)
    return apply(pipeline, mode)


# 마지막으로 적용한 내역 — 보고서에 싣는다
LAST_REPORT: list[dict] = []
OFFLOAD_REPORT: dict = {}


# ─────────────────────────── 디코딩 직전 오프로드 ───────────────────────────
# FP8 실행에서 OOM 이 난 곳은 DiT 샘플링이 아니라 **메시 디코더**(decoder_mesh.py → group_norm)였다.
# 샘플링 중 GPU 는 57% 였고, decode_slat 에 들어가며 90% 를 넘겨 터졌다.
# 그 시점에 DiT 둘·DINOv2·희소구조 인코더/디코더는 할 일이 끝났는데 GPU 에 그대로 있다.
# decode_slat 을 가로채 그것들을 CPU 로 내리고 캐시를 비운 뒤 원래 디코딩을 부른다.
#   LASCOMP_OFFLOAD=1  로 켠다. 끄면(기본) 아무것도 안 건드린다.
KEEP_ON_GPU_FOR_DECODE = ("slat_decoder_mesh",)


def _offload_for_decode(pipeline) -> dict:
    import time
    models = getattr(pipeline, "models", {})
    if not isinstance(models, dict):
        return {}
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()      # 디코딩 구간의 피크만 따로 재기 위해
    moved = []
    for name, mod in models.items():
        if name in KEEP_ON_GPU_FOR_DECODE or not isinstance(mod, torch.nn.Module):
            continue
        if any(p.is_cuda for p in mod.parameters()):
            mod.to("cpu")
            moved.append(name)
    tcm = getattr(pipeline, "text_cond_model", None)      # 텍스트 파이프라인: CLIP 텍스트 인코더는 models 밖 dict 에 있다
    if isinstance(tcm, dict) and isinstance(tcm.get("model"), torch.nn.Module) \
            and any(p.is_cuda for p in tcm["model"].parameters()):
        tcm["model"].to("cpu")
        moved.append("text_cond_model")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    rep = {
        "moved_to_cpu": moved,
        "cuda_alloc_before_mib": round(before / 2**20),
        "cuda_alloc_after_mib": round(after / 2**20),
        "freed_mib": round((before - after) / 2**20),
    }
    print("[offload] 디코딩 전 CPU 로 내림: %s  | GPU 할당 %d → %d MiB (-%d)"
          % (", ".join(moved) or "(없음)", rep["cuda_alloc_before_mib"],
             rep["cuda_alloc_after_mib"], rep["freed_mib"]))
    return rep


# ─────────────────────────── 메시 추출을 CPU 로 ───────────────────────────
# 오프로드까지 했는데도 FlexiCubes 에서 OOM (torch 예약 7054 MiB). 원인은 grad 가 아니라(run_lascomp 는 이미
# no_grad) **dense 격자**다: SLatMeshDecoder 가 SparseFeatures2Mesh(res=resolution*4=256) 를 만들고,
# 그 __init__ 이 construct_dense_grid(256) 으로 큐브 1,670 만 개의 인덱스(int64)·정점을 **GPU 에 상주**시킨다
# (약 1.5 GB — 샘플링 중에도!). 추출 때는 get_dense_attrs 가 257³×F fp32 dense 를 또 만든다.
# 디코더 신경망은 GPU 에 두고, 격자·추출만 CPU 로 옮긴다. RAM 31 GB 라 여유가 있다. 느려지지만 확실하다.
#   LASCOMP_MESH_CPU=1  로 켠다.
MESH_CPU_REPORT: dict = {}


def _move_mesh_extraction_to_cpu(pipeline) -> dict:
    from trellis.representations.mesh import SparseFeatures2Mesh
    models = getattr(pipeline, "models", {})
    dec = models.get("slat_decoder_mesh") if isinstance(models, dict) else None
    if dec is None or not hasattr(dec, "mesh_extractor"):
        print("  경고: slat_decoder_mesh / mesh_extractor 를 못 찾았다 — CPU 추출 훅 미적용", file=sys.stderr)
        return {}
    old = dec.mesh_extractor
    res, use_color = int(old.res), bool(getattr(old, "use_color", True))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    cpu_ext = SparseFeatures2Mesh(device="cpu", res=res, use_color=use_color)
    dec.mesh_extractor = cpu_ext
    del old
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()

    def to_representation_cpu(x):
        # 디코더 출력(fp16, GPU) → CPU fp32 로 옮겨 추출. CPU 는 half 연산이 빠지는 게 많아 float 로 올린다.
        ret = []
        for i in range(x.shape[0]):
            xi = x[i].cpu()
            xi = xi.replace(xi.feats.float())
            ret.append(cpu_ext(xi, training=dec.training))
        return ret

    dec.to_representation = to_representation_cpu
    rep = {"res": res, "use_color": use_color,
           "cuda_alloc_before_mib": round(before / 2**20), "cuda_alloc_after_mib": round(after / 2**20),
           "freed_mib": round((before - after) / 2**20)}
    print("[mesh-cpu] FlexiCubes 격자(res=%d)·메시 추출을 CPU 로 — GPU 할당 %d → %d MiB (-%d)"
          % (res, rep["cuda_alloc_before_mib"], rep["cuda_alloc_after_mib"], rep["freed_mib"]))
    return rep


def _install_mesh_cpu_hook_for(P) -> bool:
    on = os.environ.get("LASCOMP_MESH_CPU", "").strip() in ("1", "true", "yes")
    if not on:
        return False
    if getattr(P, "_lascomp_meshcpu_hooked", False):
        return True
    current = getattr(P, "from_pretrained")     # 양자화 훅이 먼저 걸려 있으면 그것을 감싼다

    def patched(*a, **kw):
        pipe = current(*a, **kw)
        global MESH_CPU_REPORT
        MESH_CPU_REPORT = _move_mesh_extraction_to_cpu(pipe)
        return pipe

    P.from_pretrained = staticmethod(patched)
    P._lascomp_meshcpu_hooked = True
    print("[quantize_trellis] 메시 추출 CPU 훅 설치됨 (LASCOMP_MESH_CPU=1)")
    return True


def _install_offload_hook_for(P) -> bool:
    on = os.environ.get("LASCOMP_OFFLOAD", "").strip() in ("1", "true", "yes")
    if not on:
        return False
    if getattr(P, "_lascomp_offload_hooked", False):
        return True
    original = P.decode_slat

    def patched(self, slat, *a, **kw):
        global OFFLOAD_REPORT
        OFFLOAD_REPORT = _offload_for_decode(self)
        # 디코딩은 추론이다 — 기울기가 필요 없다. run_lascomp 는 IAS 때문에 grad 를 켠 채 내려올 수 있고,
        # 그러면 디코더·FlexiCubes 의 중간 활성값이 전부 남아 메모리를 몇 배로 먹는다.
        # (오프로드로 1607 MiB 에서 출발했는데도 FlexiCubes 에서 torch 예약 7054 MiB 로 OOM 났다.)
        # slat 에 붙은 그래프도 끊는다.
        if hasattr(slat, "feats") and torch.is_tensor(slat.feats) and slat.feats.requires_grad:
            slat = slat.replace(slat.feats.detach()) if hasattr(slat, "replace") else slat
        torch.cuda.empty_cache()
        # 텍스트 파이프라인의 decode_slat 은 gaussian·radiance_field 도 디코드한다(이미지 쪽은 주석 처리됨). 그 디코더는
        # pipeline.json 에서 뺐고, 남았더라도 오프로드로 CPU 에 있다. 저자 스크립트는 ["mesh"] 만 쓰므로 메시만 디코드한다.
        with torch.no_grad():
            out = original(self, slat, ["mesh"])
        torch.cuda.synchronize()
        OFFLOAD_REPORT["cuda_alloc_after_decode_mib"] = round(torch.cuda.memory_allocated() / 2**20)
        OFFLOAD_REPORT["cuda_peak_during_decode_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        print("[offload] 디코딩 후 GPU 할당 %d MiB · 디코딩 구간 torch 피크 %d MiB"
              % (OFFLOAD_REPORT["cuda_alloc_after_decode_mib"], OFFLOAD_REPORT["cuda_peak_during_decode_mib"]))
        return out

    P.decode_slat = patched
    P._lascomp_offload_hooked = True
    print("[quantize_trellis] 디코딩 전 오프로드 훅 설치됨 (LASCOMP_OFFLOAD=1)")
    return True


def _install_hook_for(P) -> bool:
    """저자 스크립트를 고치지 않고 붙인다.

    `from_pretrained` 가 가중치를 다 올린 직후를 가로채 DiT 만 양자화한다.
    LASCOMP_QUANT 가 off 면 훅을 걸지 않는다 — 기준선 측정이 오염되지 않게.
    """
    mode = os.environ.get("LASCOMP_QUANT", "off").strip().lower()
    if mode in ("off", ""):
        return False


    if getattr(P, "_lascomp_quant_hooked", False):
        return True

    # TRELLIS 의 from_pretrained 는 **staticmethod** 다 (cls 를 받지 않는다).
    # classmethod 로 가정하고 .__func__ 를 찍었다가 'function' object has no attribute '__func__' 로 터졌다.
    # getattr 로 얻은 호출 가능 객체는 static 이면 그 함수, class 면 이미 cls 가 묶인 메서드라
    # 어느 쪽이든 인자를 그대로 넘겨 부르면 된다. 갈아 끼울 때는 staticmethod 로 둔다.
    original = getattr(P, "from_pretrained")

    def patched(*a, **kw):
        pipe = original(*a, **kw)
        global LAST_REPORT
        LAST_REPORT = apply_from_env(pipe)
        return pipe

    P.from_pretrained = staticmethod(patched)
    P._lascomp_quant_hooked = True
    print("[quantize_trellis] 훅 설치됨 (LASCOMP_QUANT=%s)" % mode)
    return True


# ── 디코딩 메모리 절약 — 정확한 재구성 (LASCOMP_DECODE_LEAN=1) ─────────────────────────────
# 굽다리바리를 온전한 참조 이미지(팀 기하 복원본 렌더)로 조건 걸자 예측 복셀이 34,345 → 47,740 개로 늘었고
# (256³ 해상도에서 ×64 = 3.05M 복셀) 메시 디코더 업샘플 2단(SparseSubdivideBlock3d 192→96 채널)의
# SparseGroupNorm32 에서 OOM 났다 (torch 예약 7350 MiB, 산출물 없음).
# 원인은 계산량이 아니라 **복사**다:
#   · SparseGroupNorm32.forward = super().forward(x.float()).type(x.dtype)
#       x.float() 복사 → zeros_like → permute/reshape 복사 → GroupNorm 출력 → fp16 으로 되돌림
#       = 입력(fp16 0.59 GB) 의 ~8 배가 한순간에 산다 (≈4.7 GB).
#   · SparseSubdivideBlock3d.forward 는 sub(x)(192채널·전해상도 1.17 GB) 를 out_layers 가 끝날 때까지 들고 있고,
#       nn.Sequential 을 통과하는 동안 호출자의 h(=sub(h) 1.17 GB) 도 끝까지 산다.
# 아래 두 함수는 **같은 수식**을 순서와 수명만 바꿔 계산한다 — 저자 코드는 손대지 않고 클래스 메서드를 갈아 끼운다.
# 원본과의 일치는 tools/test_decode_lean.py 로 확인한다 (fp16 반올림 범위 안에서 동일).
DECODE_LEAN_REPORT: dict = {}
LEAN_NORM_CHUNK_ELEMS = 1 << 22      # 조각당 원소 수 (fp32 16 MiB) — 위치 축을 이만큼씩 잘라 통계를 내고 정규화한다. 3.05M×96 이면 70 조각


def _lean_group_norm_forward(self, x):
    """SparseGroupNorm32.forward 와 같은 결과 (그룹별 평균·biased 분산·eps·채널별 affine), 메모리는 입력+출력(fp16)+조각 하나.

    통계는 fp64 로 누적한다 (조각 합산이라 fp32 누적은 3M 위치에서 자릿수를 잃을 수 있다).
    정규화는 채널별 스케일 a 와 이동 sh 로 접어 y = x·a + sh 를 제자리 연산으로 — 원본은 (x-mean)·rstd·w + b 를 fp32 로 계산하니
    fp32 안에서 반올림 순서만 다르고, fp16 으로 내리면 거의 전부 같은 값이 된다 (시험: 동일 원소 100 %, 최대차 fp16 한 눈금).
    조각 임시는 하나만 만든다 — 처음엔 `(rows.float()*a + sh)` 로 fp32 임시 3개를 만들어 피크가 거의 안 줄었다.
    """
    feats = x.feats
    N, C = feats.shape[0], feats.shape[1]
    G = self.num_groups
    cg = C // G
    rows_per = max(1, LEAN_NORM_CHUNK_ELEMS // C)
    out = torch.empty_like(feats)
    w = self.weight.float() if self.affine else None
    b = self.bias.float() if self.affine else None
    for k in range(x.shape[0]):
        sl = x.layout[k]
        rows, orow = feats[sl], out[sl]
        n = rows.shape[0]
        s = torch.zeros(G, dtype=torch.float64, device=feats.device)
        ss = torch.zeros_like(s)
        for i in range(0, n, rows_per):
            c = rows[i:i + rows_per].float().view(-1, G, cg)
            s += c.sum(dim=(0, 2), dtype=torch.float64)
            ss += c.square_().sum(dim=(0, 2), dtype=torch.float64)
        del c
        cnt = n * cg
        mean = s / cnt
        var = (ss / cnt - mean * mean).clamp_(min=0)
        rstd = torch.rsqrt(var + self.eps)
        a = rstd.repeat_interleave(cg).float()             # 채널별 스케일
        sh = (-mean.repeat_interleave(cg)).float() * a     # 채널별 이동
        if w is not None:
            sh = sh * w + b
            a = a * w
        for i in range(0, n, rows_per):
            c = rows[i:i + rows_per].float()
            c.mul_(a).add_(sh)
            orow[i:i + rows_per] = c.to(out.dtype)
        del c
    DECODE_LEAN_REPORT["norm_calls"] = DECODE_LEAN_REPORT.get("norm_calls", 0) + 1
    DECODE_LEAN_REPORT["norm_max_rows"] = max(DECODE_LEAN_REPORT.get("norm_max_rows", 0), int(N))
    return x.replace(out)


def _lean_subdivide_forward(self, x):
    """SparseSubdivideBlock3d.forward 와 같은 계산, 순서와 수명만 다르다.

    · skip 을 먼저 계산해 sub(x)(192채널·전해상도) 를 곧바로 버린다 (skip 이 Identity 면 원본과 같다)
    · out_layers 를 한 층씩 불러 sub(h) 가 첫 컨볼루션 뒤에 바로 풀리게 한다
    """
    h = self.act_layers(x)
    h = self.sub(h)
    s = self.skip_connection(self.sub(x))
    del x
    for layer in self.out_layers:
        h = layer(h)
    return h + s


def install_decode_lean_hook(force: bool = False) -> bool:
    on = force or os.environ.get("LASCOMP_DECODE_LEAN", "").strip() in ("1", "true", "yes")
    if not on:
        return False
    from trellis.modules.sparse.norm import SparseGroupNorm32
    from trellis.models.structured_latent_vae.decoder_mesh import SparseSubdivideBlock3d
    if getattr(SparseGroupNorm32, "_lascomp_lean", False):
        return True
    SparseGroupNorm32._lascomp_orig_forward = SparseGroupNorm32.forward
    SparseSubdivideBlock3d._lascomp_orig_forward = SparseSubdivideBlock3d.forward
    SparseGroupNorm32.forward = _lean_group_norm_forward
    SparseSubdivideBlock3d.forward = _lean_subdivide_forward
    SparseGroupNorm32._lascomp_lean = True
    DECODE_LEAN_REPORT.update({"enabled": True, "norm_chunk_elems": LEAN_NORM_CHUNK_ELEMS})
    print("[quantize_trellis] 디코딩 절약 재구성 훅 설치됨 (LASCOMP_DECODE_LEAN=1): GroupNorm32 조각 정규화 · Subdivide 블록 수명 정리")
    return True


# ── 훅을 두 파이프라인(이미지·텍스트)에 모두 건다 ─────────────────────────────────────────
# 텍스트 조건 실행(run_lascomp_text_condition_single.py, ckpt text-xlarge)도 같은 절감책이 필요하다.
# DiT 이름 규칙('flow')·오프로드·CPU 추출·디코더 절약 훅은 파이프라인에 무관하고, 붙이는 자리(클래스)만 둘이다.
def _pipeline_classes():
    from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
    return (TrellisImageTo3DPipeline, TrellisTextTo3DPipeline)


def install_hook() -> bool:
    return any([_install_hook_for(P) for P in _pipeline_classes()])


def install_mesh_cpu_hook() -> bool:
    return any([_install_mesh_cpu_hook_for(P) for P in _pipeline_classes()])


def install_offload_hook() -> bool:
    return any([_install_offload_hook_for(P) for P in _pipeline_classes()])


# ── 모델이 예측한 색 건져 오기 (LASCOMP_COLOR_OUT=<경로.npz>) ─────────────────────────────
# 메시 디코더는 색을 **만든다**: SparseFeatures2Mesh(use_color=True) 가 v_attrs 의 [4:] 를 FlexiCubes 에
# voxelgrid_colors 로 넘기고, 결과를 MeshExtractResult(vertex_attrs=colors) 로 담는다
# (`trellis/representations/mesh/cube2mesh.py`: LAYOUTS['color'] = (8, 6) — 색 3 + 법선 3).
# 그런데 저자 저장 코드는 vertices·faces 만 꺼내 GLB 로 내보내서(run_lascomp_*.py) **색이 버려진다**.
# 이 훅은 decode_slat 직후 vertex_attrs 를 npz 로 남긴다. 좌표 역정규화는 저자 코드가 뒤에서 하지만
# 정점 순서·개수는 그대로이므로 나중에 tools/apply_colors.py 가 GLB 에 붙일 수 있다.
COLOR_REPORT: dict = {}


def _save_decoded_colors(out) -> None:
    import numpy as np
    path = os.environ.get("LASCOMP_COLOR_OUT", "").strip()
    if not path:
        return
    meshes = out.get("mesh") if isinstance(out, dict) else None
    if not meshes:
        return
    m = meshes[0]
    attrs = getattr(m, "vertex_attrs", None)
    if attrs is None:
        print("  경고: vertex_attrs 가 없다 — 색 저장 생략", file=sys.stderr)
        return
    a = attrs.detach().float().cpu().numpy()
    v = m.vertices.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, vertex_attrs=a, n_verts=len(v))
    COLOR_REPORT.update({"path": path, "attrs_shape": list(a.shape), "n_verts": int(len(v)),
                         "attr_min": float(a.min()), "attr_max": float(a.max())})
    print("[color] 모델 예측 색 저장: %s · vertex_attrs %s · 값 범위 %.3f~%.3f"
          % (path, tuple(a.shape), a.min(), a.max()))


def install_color_hook() -> bool:
    """decode_slat 결과에서 vertex_attrs(색+법선)를 꺼내 저장한다. 오프로드 훅이 이미 decode_slat 을 감쌌으면 그 위에 덧씌운다."""
    if not os.environ.get("LASCOMP_COLOR_OUT", "").strip():
        return False
    installed = False
    for P in _pipeline_classes():
        if getattr(P, "_lascomp_color_hooked", False):
            installed = True
            continue
        current = P.decode_slat

        def patched(self, slat, *a, __cur=current, **kw):
            out = __cur(self, slat, *a, **kw)
            try:
                _save_decoded_colors(out)
            except Exception as e:            # 색은 부가 기능이다 — 실패해도 복원을 멈추지 않는다
                print("  경고: 색 저장 실패 (%s: %s)" % (type(e).__name__, e), file=sys.stderr)
            return out

        P.decode_slat = patched
        P._lascomp_color_hooked = True
        installed = True
    if installed:
        print("[quantize_trellis] 색 저장 훅 설치됨 (LASCOMP_COLOR_OUT)")
    return installed


# ── ERS 를 양쪽 방향으로 (LASCOMP_KEEP_EMPTY=1) ──────────────────────────────────
#
# 저자의 ERS 는 `pred_voxel[voxel_mask] = 1.0` 한 줄, IAS 의 BCE 도 관측 복셀에만 걸린다.
# **빈 공간에는 제약이 없어서** 그 자리를 TRELLIS 의 사전분포가 채운다 — 굽다리 투창이 이렇게 사라진다
# (2026-09-18 실측: 입력 점군 창 7 개 → 1 단계 복셀 2 개. 투창은 7~16 복셀 폭이라 해상도 문제가 아니다).
#
# 저자 코드를 고치지 않고 양쪽으로 만드는 방법: ERS 는 `(decoder(x_0) > 0.0)` 으로 점유를 정하므로
# **희소 구조 디코더의 로짓을 '비어야 하는 칸'에서만 큰 음수로 눌러** 두면 매 스텝 0 이 된다.
# 마스크는 tools/ers_keep_empty.py 가 입력 격자(ss)에서 만든다 — 굽다리 벽을 원통으로 펼쳐
# **사방이 관측면으로 둘러싸인 빈 덩어리(= 투창)** 만 고른다. 결손부(파단면으로 열린 곳)는 건드리지 않는다.
#
# 마스크는 관측 복셀과 서로소라 ERS 의 1 고정과 충돌하지 않는다. 샘플링이 끝난 뒤
# `argwhere(decoder(z_s) > 0)` 에도 같은 억제가 걸려 최종 구조까지 투창이 남는다.
KEEP_EMPTY_REPORT: dict = {}
_KEEP_EMPTY_MASK = None


def _keep_empty_build(ss):
    """입력 격자에서 '비어야 하는' 마스크를 만들어 전역에 둔다. ss: (1,1,64,64,64) 또는 (1,64,64,64)."""
    global _KEEP_EMPTY_MASK, KEEP_EMPTY_REPORT
    _KEEP_EMPTY_MASK, KEEP_EMPTY_REPORT = None, {}   # 중간에 실패해도 앞 유물 마스크가 남지 않게 먼저 비운다
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ers_keep_empty import keep_empty_mask

    t = ss.detach()
    while t.dim() > 3:
        t = t[0]
    occ = (t > 0.5).cpu().numpy()
    pad = float(os.environ.get("LASCOMP_KEEP_EMPTY_PAD", "2.0"))
    minc = int(os.environ.get("LASCOMP_KEEP_EMPTY_MINCELLS", "4"))
    mask, info = keep_empty_mask(occ, rad_pad=pad, min_cells=minc)
    if not mask.any():
        _KEEP_EMPTY_MASK = None
        KEEP_EMPTY_REPORT = dict(info, enabled=True, applied=False)
        return
    _KEEP_EMPTY_MASK = torch.from_numpy(np.ascontiguousarray(mask)).to(ss.device)
    KEEP_EMPTY_REPORT = dict(info, enabled=True, applied=True)


def _keep_empty_apply(logits):
    """희소 구조 디코더 로짓을 마스크 칸에서만 큰 음수로. 모양은 (N,C,64,64,64)."""
    m = _KEEP_EMPTY_MASK
    if m is None or not torch.is_tensor(logits):
        return logits
    if logits.dim() < 3 or tuple(logits.shape[-3:]) != tuple(m.shape):
        return logits                                  # 다른 해상도면 손대지 않는다
    out = logits.clone()
    out[..., m] = -1e4
    return out



def _keep_empty_wrap(current):
    """희소 구조 샘플링을 감싼다 — 입력 격자로 마스크를 만들고 디코더 로짓을 누르게 한다."""
    def patched(self, *a, __cur=current, **kw):
        ss = kw.get("ss")
        if ss is None:
            for x in a:                                # 위치 인자로 올 수도 있다 (mask, ss, …)
                if torch.is_tensor(x) and x.dim() >= 4 and min(x.shape[-3:]) == 64:
                    ss = x
                    break
        # 호출마다 다시 만든다 — 한 프로세스에서 유물을 둘 이상 돌려도 앞 유물의 마스크가 남지 않는다.
        # 생성 1 회당 1 회뿐이고 샘플링에 견주면 무시할 만하다.
        if ss is not None:
            try:
                _keep_empty_build(ss)
            except Exception as e:
                print("  경고: 빈칸 마스크 생성 실패 (%s: %s)" % (type(e).__name__, e), file=sys.stderr)
        dec = self.models.get("sparse_structure_decoder")
        if dec is not None and not getattr(dec, "_lascomp_keep_empty", False):
            dec_fwd = dec.forward

            def dec_patched(*da, __f=dec_fwd, **dkw):
                return _keep_empty_apply(__f(*da, **dkw))

            dec.forward = dec_patched
            dec._lascomp_keep_empty = True
        return __cur(self, *a, **kw)
    return patched


def install_ers_empty_hook() -> bool:
    if os.environ.get("LASCOMP_KEEP_EMPTY", "").strip() not in ("1", "true", "yes"):
        return False
    installed = False
    names = ("lascomp_sample_sparse_structure", "sample_sparse_structure")
    for P in _pipeline_classes():
        if getattr(P, "_lascomp_keep_empty_hooked", False):
            installed = True
            continue
        hooked_any = False
        for nm in names:
            current = getattr(P, nm, None)
            if current is None:
                continue
            setattr(P, nm, _keep_empty_wrap(current))
            hooked_any = True
        if not hooked_any:
            continue

        P._lascomp_keep_empty_hooked = True
        installed = True
    if installed:
        print("[quantize_trellis] ERS 양방향 훅 설치됨 (LASCOMP_KEEP_EMPTY=1) — 투창 자리를 매 스텝 0 으로 고정")
    return installed
