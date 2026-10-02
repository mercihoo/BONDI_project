#!/usr/bin/env python3
"""텍스트 조건 실행이 첫 스텝부터 NaN 을 낸 원인 분리 — CLIP 조건 임베딩인가, XL DiT 자체인가, FP8 양자화인가.

    python tools/diag_text_nan.py

훅 없이 text-xlarge 를 그대로 올려 (1) CLIP 임베딩 NaN 여부 (2) fp16 DiT 한 번 전진 (3) 같은 입력을 FP8 양자화 뒤 전진.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATTN_BACKEND", "xformers")
import torch

import quantize_trellis as qt
from trellis.pipelines import TrellisTextTo3DPipeline

PROMPT = ("a complete intact Korean Silla gray stoneware pedestal bowl: a deep round bowl with horizontal grooved bands, "
          "on a tall trumpet-shaped stand pierced with two tiers of rectangular openwork windows, symmetric, museum artifact")


def stat(name, t):
    t = t.float()
    print("  %-22s dtype→ shape %s · NaN %d · inf %d · |max| %.3g · 평균 %.4g" % (
        name, tuple(t.shape), torch.isnan(t).sum().item(), torch.isinf(t).sum().item(),
        t.abs().nan_to_num(0).max().item(), t.nan_to_num(0).mean().item()))


pipe = TrellisTextTo3DPipeline.from_pretrained("ckpt/text-xlarge")
pipe.cuda()
clip = pipe.text_cond_model["model"]
print("CLIP: device %s · dtype %s" % (next(clip.parameters()).device, next(clip.parameters()).dtype))
cond = pipe.get_cond([PROMPT])
stat("cond", cond["cond"]); stat("neg_cond", cond["neg_cond"])

flow = pipe.models["sparse_structure_flow_model"]
p = next(flow.parameters())
print("SS flow DiT: dtype %s · in_channels %d · reso %d" % (p.dtype, flow.in_channels, flow.resolution))
torch.manual_seed(0)
x = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution, device="cuda")
t = torch.tensor([1.0], device="cuda")
with torch.no_grad():
    y = flow(x, t, cond["cond"]); stat("fp16 DiT 출력", y)
    y2 = flow(x, t, cond["neg_cond"]); stat("fp16 DiT 출력(neg)", y2)
    rep = qt.quantize_module(flow, "fp8", tag="ss_flow_txt")
    print("  양자화:", {k: v for k, v in rep.items() if k in ("converted", "total", "saved_mib", "mode")})
    y3 = flow(x, t, cond["cond"]); stat("FP8 DiT 출력", y3)
    if not torch.isnan(y3).any():
        print("  fp16 vs FP8 차이: 평균 |Δ| %.4g (|y| 평균 %.4g)" % ((y3.float() - y.float()).abs().mean().item(), y.float().abs().mean().item()))
