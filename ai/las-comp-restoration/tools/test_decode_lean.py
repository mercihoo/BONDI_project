"""디코딩 절약 재구성(quantize_trellis.install_decode_lean_hook) 이 원본과 같은 값을 내는지 — GPU 에서 무작위 희소 텐서로.

    python tools/test_decode_lean.py

원본 GroupNorm32 는 fp32 로 계산해 fp16 으로 내리고, 절약판은 fp64 통계 + fp32 정규화를 fp16 으로 내린다.
그래서 두 결과의 차이는 fp16 반올림 한 눈금 안이어야 한다. 블록은 컨볼루션(fp16)을 두 번 거치니 조금 더 느슨하게 본다.
피크 메모리도 같은 입력으로 원본·절약판을 재어 비율을 낸다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATTN_BACKEND", "xformers")
import torch

import quantize_trellis as qt
from trellis.modules import sparse as sp
from trellis.modules.sparse.norm import SparseGroupNorm32
from trellis.models.structured_latent_vae.decoder_mesh import SparseSubdivideBlock3d
from trellis.modules.utils import convert_module_to_f16

torch.manual_seed(0)
dev = "cuda"


def rand_sparse(n, C, res, dtype):
    coords = torch.randint(0, res, (n * 2, 3), device=dev)
    coords = torch.unique(coords, dim=0)[:n]
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=dev), coords], 1).int()
    feats = (torch.randn(coords.shape[0], C, device=dev) * 3 + 0.7).to(dtype)
    return sp.SparseTensor(feats, coords)


def cmp(name, got, ref):
    d = (got.float() - ref.float()).abs()
    same = (got == ref).float().mean().item()
    print(f"  {name:22s} 최대차 {d.max().item():.3e}  평균차 {d.mean().item():.3e}  동일원소 {same*100:5.1f}%  (기준 |값| 평균 {ref.float().abs().mean().item():.3f})")
    return d.max().item()


orig_gn, orig_blk = SparseGroupNorm32.forward, SparseSubdivideBlock3d.forward
assert qt.install_decode_lean_hook(force=True)
lean_gn, lean_blk = SparseGroupNorm32.forward, SparseSubdivideBlock3d.forward
assert lean_gn is not orig_gn and lean_blk is not orig_blk


def use(gn, blk):
    SparseGroupNorm32.forward, SparseSubdivideBlock3d.forward = gn, blk


# 1) GroupNorm32 — 디코더 2단 입력과 같은 192 채널·32 그룹
gn = SparseGroupNorm32(32, 192).to(dev)
with torch.no_grad():
    gn.weight.normal_(1, 0.2)
    gn.bias.normal_(0, 0.3)
x = rand_sparse(20000, 192, 128, torch.float16)
with torch.no_grad():
    ref = orig_gn(gn, x).feats
    got = lean_gn(gn, x).feats
    ref32 = orig_gn(gn, x.replace(x.feats.float())).feats      # fp32 입력 정답
print("[GroupNorm32 192ch/32g, fp16 입력 20k 위치]")
m1 = cmp("원본 vs 절약", got, ref)
cmp("원본 vs fp32정답", ref, ref32.half())
cmp("절약 vs fp32정답", got, ref32.half())

# 2) 블록 — 실제 디코더 2단과 같은 모양 (192@128 → 96@256), 가중치 fp16 (use_fp16 과 같게)
blk = SparseSubdivideBlock3d(192, 128, out_channels=96).to(dev)
with torch.no_grad():
    for p in blk.parameters():
        if p.dim() > 1:
            p.normal_(0, 0.05)          # zero_module 된 마지막 conv 도 채운다 — skip 만 남는 시험이 되지 않게
blk.apply(convert_module_to_f16)
x = rand_sparse(6000, 192, 128, torch.float16)
with torch.no_grad():
    use(orig_gn, orig_blk); ref = blk(x)
    use(lean_gn, lean_blk); got = blk(x)
assert torch.equal(ref.coords, got.coords), "출력 좌표가 다르다"
print("[SparseSubdivideBlock3d 192@128→96@256, 6k→48k 위치]")
m2 = cmp("원본 vs 절약", got.feats, ref.feats)

# 3) 피크 메모리 — 같은 입력으로 원본·절약판
x = rand_sparse(40000, 192, 128, torch.float16)


def peak(gn_f, blk_f):
    use(gn_f, blk_f)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        y = blk(x)
    torch.cuda.synchronize()
    p = torch.cuda.max_memory_allocated() - base
    del y
    return p / 2**20


p_orig = peak(orig_gn, orig_blk)
p_lean = peak(lean_gn, lean_blk)
print(f"[피크 메모리, 40k→320k 위치] 원본 {p_orig:.0f} MiB → 절약 {p_lean:.0f} MiB  (×{p_lean/p_orig:.2f})")

ok = m1 < 5e-3 and m2 < 5e-2 and p_lean < p_orig * 0.6
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
