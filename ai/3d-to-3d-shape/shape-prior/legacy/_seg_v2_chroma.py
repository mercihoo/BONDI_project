"""배경 분리 — v2. 그림자를 빼는 게 핵심이다.

v1 이 실패한 이유
-----------------
몽타주를 보면 유물 외곽선 자체는 대체로 맞았다. 그런데 **바닥에 드리운 그림자**가
통째로 전경에 들어와 실루엣이 옆으로 퍼졌다. 그래서 최대폭이 33~38cm 로 나왔다 —
높이 14cm 짜리 토기가 그럴 리 없다.

두 가지를 고친다.

1. **배경이 평평하지 않다.** 박물관 사진은 은은한 그라데이션이 있어서 테두리 중앙값
   하나로는 반대쪽 구석을 설명하지 못한다. 채널마다 2차 다항식을 테두리에 맞춰
   배경면 B(x,y) 를 만든다.

2. **그림자는 '색은 그대로, 밝기만 어두워진 것'이다.** 비율 a/B 를 보면
   그림자는 (k,k,k) 꼴로 세 채널이 비슷하게 줄어든다. 반면 유물은 채널마다 다르게
   바뀌거나 훨씬 어둡다. 이 차이로 거른다.

   shadow  =  채널 간 비율 차 작음  &  너무 어둡지 않음  &  배경과 크게 안 다름
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _poly_basis(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / w - 0.5).ravel(); y = (yy / h - 0.5).ravel()
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def background(a: np.ndarray, frac=0.035, rounds=2):
    """테두리 화소로 채널별 2차 곡면을 맞춘다. 유물이 테두리에 걸치면 한 번 걸러낸다."""
    h, w = a.shape[:2]
    m = max(3, round(frac * min(h, w)))
    sel = np.zeros((h, w), bool)
    sel[:m] = sel[-m:] = True; sel[:, :m] = sel[:, -m:] = True
    A = _poly_basis(h, w)
    for _ in range(rounds):
        idx = np.flatnonzero(sel.ravel())
        C = np.linalg.lstsq(A[idx], a.reshape(-1, 3)[idx], rcond=None)[0]
        B = (A @ C).reshape(h, w, 3)
        r = np.linalg.norm(a - B, axis=2).ravel()[idx]
        keep = r <= np.median(r) + 3 * (np.median(np.abs(r - np.median(r))) + 1e-6)
        s2 = np.zeros(h * w, bool); s2[idx[keep]] = True
        sel = s2.reshape(h, w)
    return B, float(np.median(np.linalg.norm(a - B, axis=2)[sel]))


def segment(a: np.ndarray):
    h, w = a.shape[:2]
    B, noise = background(a)
    d = np.linalg.norm(a - B, axis=2)
    ratio = a / np.maximum(B, 1e-3)
    k = ratio.mean(2)                              # 밝기가 몇 배가 됐나
    chroma = ratio.max(2) - ratio.min(2)           # 채널 간 차 — 그림자면 작다
    shadow = (chroma < 0.10) & (k > 0.62) & (k < 1.06) & (d < 0.26)

    thr = max(6.0 * noise, 0.045)
    fg = (d > thr) & ~shadow
    fg = ndi.binary_opening(fg, np.ones((3, 3)), iterations=2)
    fg = ndi.binary_closing(fg, np.ones((7, 7)), iterations=2)
    fg = ndi.binary_fill_holes(fg)

    lab, n = ndi.label(fg)
    if n == 0:
        return fg, dict(area=0.0, thr=thr, noise=noise, shadow_px=0.0, comps=0, touched=False)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    inner = [i for i in range(1, n + 1) if i not in edge]
    pick = (max(inner, key=lambda i: sizes[i - 1]) if inner
            else int(np.argmax(sizes)) + 1)          # 테두리에 안 닿는 것 우선
    big = lab == pick
    big = ndi.binary_fill_holes(ndi.binary_closing(big, np.ones((5, 5)), iterations=2))
    return big, dict(area=float(big.mean()), thr=float(thr), noise=float(noise),
                     shadow_px=float(shadow.mean()), comps=int(n), touched=pick in edge)
