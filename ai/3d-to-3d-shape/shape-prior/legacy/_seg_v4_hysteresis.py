"""배경 분리 — v3.

v1: 테두리 중앙값 하나로 배경을 잡았다. 그라데이션을 못 따라가고 **그림자가 통째로**
    전경에 들어왔다. 실루엣 최대폭이 33~38cm 로 나왔다 (높이 14cm 토기가).
v2: 그림자를 '색은 그대로 어두워진 것'으로 보고 a/B 의 채널 간 차로 걸렀다.
    **회색 토기가 회색 배경 앞에 있으면 토기도 무채색이라 같이 지워졌다.**
    박물관 사진들의 덩어리 수가 11~18 로 튀었다 = 유물이 조각남.
v3: 색으로 거르지 않는다. **기하로 거른다.**
    고배는 아가리가 최대폭이다 (기록에서도 입지름 18.1 vs 받침지름 11.2).
    그림자는 바닥에 옆으로 퍼지므로 아래쪽 줄을 아가리보다 넓게 만든다.
    그 줄을 잘라내고 **몇 줄 잘랐는지 보고**한다.

색 기반 판별은 `legacy/_seg_v2_chroma.py` 에 남겨 뒀다 — 배경과 유물의 색이 다른
자료에서는 유효하다.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _basis(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / w - 0.5).ravel(); y = (yy / h - 0.5).ravel()
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def background(a, frac=0.035, rounds=2):
    """테두리 화소로 채널별 2차 곡면을 맞춘다 — 박물관 사진의 은은한 그라데이션 때문."""
    h, w = a.shape[:2]
    m = max(3, round(frac * min(h, w)))
    sel = np.zeros((h, w), bool)
    sel[:m] = sel[-m:] = True; sel[:, :m] = sel[:, -m:] = True
    A = _basis(h, w)
    for _ in range(rounds):
        idx = np.flatnonzero(sel.ravel())
        C = np.linalg.lstsq(A[idx], a.reshape(-1, 3)[idx], rcond=None)[0]
        B = (A @ C).reshape(h, w, 3)
        r = np.linalg.norm(a - B, axis=2).ravel()[idx]
        med = np.median(r)
        keep = r <= med + 3 * (np.median(np.abs(r - med)) + 1e-6)
        s2 = np.zeros(h * w, bool); s2[idx[keep]] = True
        sel = s2.reshape(h, w)
    return B, float(np.median(np.linalg.norm(a - B, axis=2)[sel]))


def segment(a):
    h, w = a.shape[:2]
    B, noise = background(a)
    d = np.linalg.norm(a - B, axis=2)
    # 이력(hysteresis) 문턱 — 회색 유물이 회색 배경 앞에 있으면 대비가 낮다.
    # 단일 문턱을 낮추면 배경이 새고, 높이면 유물이 조각난다 (v3 에서 덩어리 17~21).
    # 그래서 확실한 씨앗을 높은 문턱으로 잡고, 낮은 문턱으로 그 씨앗만 키운다.
    # 낮은 문턱은 그림자를 같이 데려오지만 그건 뒤에서 기하로 걷어낸다.
    hi = max(6.0 * noise, 0.070)
    lo = max(2.5 * noise, 0.022)
    seed = ndi.binary_opening(d > hi, np.ones((3, 3)), iterations=2)
    weak = ndi.binary_closing(d > lo, np.ones((5, 5)), iterations=2)
    lw, nw = ndi.label(weak)
    keep = set(np.unique(lw[seed])) - {0}
    fg = np.isin(lw, list(keep)) if keep else seed
    fg = ndi.binary_fill_holes(ndi.binary_closing(fg, np.ones((7, 7)), iterations=2))
    thr = hi

    lab, n = ndi.label(fg)
    if n == 0:
        return fg, dict(area=0.0, thr=thr, noise=noise, comps=0, touched=False)
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    inner = [i for i in range(1, n + 1) if i not in edge]
    pick = (max(inner, key=lambda i: sizes[i - 1]) if inner else int(np.argmax(sizes)) + 1)
    big = ndi.binary_fill_holes(ndi.binary_closing(lab == pick, np.ones((5, 5)), iterations=2))
    return big, dict(area=float(big.mean()), thr=float(thr), noise=float(noise),
                     comps=int(n), touched=pick in edge)
