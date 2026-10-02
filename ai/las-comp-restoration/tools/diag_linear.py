"""양자화가 왜 0 MiB 였는지 가른다 — GPU 를 안 쓰고 CPU 에서만.

두 가설:
  (a) 내 filter_fn 이 Linear 를 전부 걸러냈다 (이름·크기 규칙, 또는 nn.Linear 가 아님)
  (b) 양자화는 됐는데, element_size() 합산이 텐서 서브클래스(AffineQuantizedTensor)의
      실제 저장 크기를 못 본다 — 겉 dtype 기준으로 셈

DiT 두 개만 ckpts 에서 직접 올린다 (DINOv2·GPU 필요 없음).
"""
import sys, collections
sys.path.insert(0, ".")
import torch
from torch import nn
from trellis import models
import quantize_trellis as qt

CKPTS = [
    ("sparse_structure_flow_model", "ckpt/image-large/ckpts/ss_flow_img_dit_L_16l8_fp16"),
    ("slat_flow_model",             "ckpt/image-large/ckpts/slat_flow_img_dit_L_64l8p2_fp16"),
]

for tag, path in CKPTS:
    print("=" * 64)
    print(tag, "←", path)
    m = models.from_pretrained(path)          # CPU
    m.eval()

    linears = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
    leaf_cls = collections.Counter(type(mod).__name__ for _, mod in m.named_modules()
                                   if hasattr(mod, "weight") and len(list(mod.children())) == 0)
    print("  가중치 가진 잎 모듈 종류:", dict(leaf_cls.most_common(8)))
    print("  nn.Linear 개수:", len(linears))

    eligible = [(n, mod) for n, mod in linears if qt._linear_filter(mod, n)]
    by_name  = [n for n, mod in linears if qt.SKIP_MODULE_PAT.search(n)]
    by_size  = [n for n, mod in linears if mod.in_features < 256 or mod.out_features < 256]
    print("  필터 통과:", len(eligible), "| 이름으로 제외:", len(by_name), "| 크기로 제외:", len(by_size))
    if linears:
        shapes = collections.Counter((mod.in_features, mod.out_features) for _, mod in linears)
        print("  (in,out) 분포 상위:", shapes.most_common(5))
        print("  이름 예:", [n for n, _ in linears[:4]])
    if by_name:
        print("  이름 제외 예:", by_name[:4])

    # 양자화 전후 — 가중치 클래스가 바뀌는지, 실제 바이트가 바뀌는지 (CPU 에서 int8 로 기전만 확인)
    w_cls_before = collections.Counter(type(mod.weight).__name__ for _, mod in linears)
    fp16_bytes = sum(p.numel() * p.element_size() for p in m.parameters())
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(m, int8_weight_only(), filter_fn=qt._linear_filter)
        err = None
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, str(e)[:120])
    w_cls_after = collections.Counter(type(mod.weight).__name__ for _, mod in m.named_modules()
                                      if isinstance(mod, nn.Linear))
    naive_after = sum(p.numel() * p.element_size() for p in m.parameters())

    def real_bytes(t):
        # 서브클래스면 안쪽 저장 텐서들을 더한다
        if hasattr(t, "tensor_impl"):
            impl = t.tensor_impl
            tot = 0
            for a in ("int_data", "float8_data", "scale", "zero_point"):
                x = getattr(impl, a, None)
                if torch.is_tensor(x):
                    tot += x.numel() * x.element_size()
            if tot: return tot
        return t.numel() * t.element_size()
    real_after = sum(real_bytes(p) for p in m.parameters())

    print("  양자화 시도(int8, CPU):", "OK" if not err else "실패 → " + err)
    print("  weight 클래스 전:", dict(w_cls_before), "| 후:", dict(w_cls_after))
    print("  바이트 — 원래 %.0f MiB | element_size 합산 %.0f MiB | 실제 저장 %.0f MiB"
          % (fp16_bytes / 2**20, naive_after / 2**20, real_after / 2**20))
    del m
