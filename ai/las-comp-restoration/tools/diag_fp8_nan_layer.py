#!/usr/bin/env python3
"""FP8 양자화가 text-xlarge DiT 를 NaN 으로 만드는 층을 찾는다 — 양자화 전 가중치 통계(0 행·극단 범위)와 양자화 뒤 첫 NaN 모듈.

    python tools/diag_fp8_nan_layer.py [ss|slat]

GPU 를 쓴다 (모델 하나 fp16 ~2 GB). 실행 중인 복원과 겹치지 않게 GPU 가 빌 때 돌린다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATTN_BACKEND", "xformers")
import torch

import quantize_trellis as qt
from trellis.pipelines import TrellisTextTo3DPipeline

which = sys.argv[1] if len(sys.argv) > 1 else "ss"
pipe = TrellisTextTo3DPipeline.from_pretrained("ckpt/text-xlarge")
pipe.cuda()
cond = pipe.get_cond(["a gray stoneware pedestal bowl"])["cond"]
flow = pipe.models["sparse_structure_flow_model" if which == "ss" else "slat_flow_model"]

# 1) 양자화 전: 대상 Linear 의 가중치 통계 — 행 전체가 0 이거나(스케일 0 → 0/0), 범위가 극단인 층
print("=== 양자화 대상 Linear 가중치 통계 (문제 후보만) ===")
cands = []
for name, m in flow.named_modules():
    if isinstance(m, torch.nn.Linear) and qt._linear_filter(m, name):
        w = m.weight.detach().float()
        rowmax = w.abs().amax(dim=1)
        zero_rows = int((rowmax == 0).sum())
        if zero_rows or w.abs().max() > 100 or not torch.isfinite(w).all():
            cands.append(name)
            print("  %-60s shape %s · dtype %s · 0인 행 %d/%d · |max| %.3g · finite %s" % (
                name, tuple(w.shape), m.weight.dtype, zero_rows, w.shape[0], w.abs().max().item(), torch.isfinite(w).all().item()))
print("  후보 %d 개" % len(cands))

# 2) 양자화 뒤: 전진하며 처음 NaN 을 내는 모듈
qt.quantize_module(flow, "fp8", tag=which)
first = []
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(o) and torch.isnan(o).any() and not first:
            i = inp[0] if inp and torch.is_tensor(inp[0]) else None
            first.append(name)
            print("  첫 NaN 모듈: %s (%s) · 입력 NaN %s" % (name, type(mod).__name__, torch.isnan(i).any().item() if i is not None else "?"))
    return f
hs = [m.register_forward_hook(hook(n)) for n, m in flow.named_modules()]
torch.manual_seed(0)
x = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution, device="cuda") if which == "ss" else None
with torch.no_grad():
    if which == "ss":
        y = flow(x, torch.tensor([1.0], device="cuda"), cond)
        print("출력 NaN:", torch.isnan(y).any().item())
    else:
        print("slat 은 희소 입력이 필요해 전진 시험은 생략 — 가중치 통계만")
for h in hs:
    h.remove()
