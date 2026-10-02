"""배경 분리 — v5. 문턱값을 **고르지 않고 찾는다.**

여기까지 온 경위
----------------
v1 테두리 중앙값 하나 -> 그라데이션 못 따라감 + 그림자 포함. 최대폭 33~38cm.
v2 그림자를 '무채색 어두워짐'으로 판별 -> **회색 유물이 회색 배경 앞이면 같이 지워짐**
   (덩어리 11~18 로 조각남).
v3 색 판별을 버리고 기하로 그림자 제거 + 단일 문턱 -> 대비 낮은 사진에서 유물이 조각남.
v4 이력 문턱(낮은 문턱으로 씨앗 키우기) -> 반대로 **배경이 샘** (+22%, 좌우차 0.0 = 전체 프레임).

교훈이 분명하다. 사진 43장의 배경이 제각각(스튜디오 회색 그라데이션 · 크림색 · 어두운 실내)
이고 유물 색과 겹치기도 한다. **어떤 고정 문턱도 43장 전부에 맞지 않는다.**

그래서 문턱을 고르는 대신, **후보를 훑고 '합격 조건'에 맞는 것을 고른다.**
합격 조건은 유물 사진이면 당연히 만족해야 하는 것들이다.

    면적이 프레임의 1~45%          너무 작으면 분리 실패, 너무 크면 배경이 샘
    가로폭 < 프레임의 92%          꽉 차면 배경을 잡은 것
    세로폭 < 프레임의 95%
    무게중심이 가운데 60% 안        유물은 가운데 놓고 찍는다

합격한 것 중 **가장 엄격한(문턱이 높은)** 것을 쓴다. 어느 문턱이 뽑혔는지 `q` 로 보고하므로
나중에 어떤 사진이 아슬아슬했는지 추적할 수 있다.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _basis(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx / w - 0.5).ravel(); y = (yy / h - 0.5).ravel()
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)


def background(a, frac=0.035, rounds=2):
    """테두리 화소로 채널별 2차 곡면 — 박물관 사진의 은은한 그라데이션 때문."""
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


def _otsu(x, bins=256):
    hist, e = np.histogram(x, bins=bins)
    p = hist / max(hist.sum(), 1)
    w0 = np.cumsum(p); w1 = 1 - w0
    c = (e[:-1] + e[1:]) / 2
    m0 = np.cumsum(p * c) / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((p * c)[::-1])[::-1]) / np.maximum(w1, 1e-12)
    v = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.nanargmax(v))])


def _blob(d, t, h, w):
    fg = ndi.binary_opening(d > t, np.ones((3, 3)), iterations=2)
    fg = ndi.binary_fill_holes(ndi.binary_closing(fg, np.ones((7, 7)), iterations=2))
    lab, n = ndi.label(fg)
    if n == 0:
        return None, 0
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    inner = [i for i in range(1, n + 1) if i not in edge]
    pick = (max(inner, key=lambda i: sizes[i - 1]) if inner else int(np.argmax(sizes)) + 1)
    big = ndi.binary_fill_holes(ndi.binary_closing(lab == pick, np.ones((5, 5)), iterations=2))
    return big, n


def _accept(m, h, w):
    if m is None or not m.any():
        return False
    area = m.mean()
    if not (0.010 <= area <= 0.45):
        return False
    ys, xs = np.nonzero(m)
    if (xs.max() - xs.min()) > 0.92 * w or (ys.max() - ys.min()) > 0.95 * h:
        return False
    cx = xs.mean() / w
    return 0.20 <= cx <= 0.80


def segment(a):
    h, w = a.shape[:2]
    B, noise = background(a)
    d = np.linalg.norm(a - B, axis=2)

    cands = [("otsu", _otsu(d))]
    cands += [(f"q{int(100*q)}", float(np.quantile(d, q)))
              for q in (0.55, 0.62, 0.68, 0.74, 0.80, 0.85, 0.89, 0.93)]
    cands += [(f"n{k:g}", max(k * noise, 0.02 * k)) for k in (3, 5, 8, 12)]
    cands = sorted({round(t, 5): (n, t) for n, t in cands if t > 1e-4}.values(),
                   key=lambda x: -x[1])                 # 엄격한 것부터

    for name, t in cands:
        m, n = _blob(d, t, h, w)
        if _accept(m, h, w):
            return m, dict(area=float(m.mean()), thr=float(t), noise=float(noise),
                           comps=int(n), rule=name, tried=len(cands))
    m, n = _blob(d, max(5 * noise, 0.05), h, w)         # 아무것도 합격 못 하면
    if m is None:
        m = np.zeros((h, w), bool)
    return m, dict(area=float(m.mean()), thr=float(max(5 * noise, 0.05)), noise=float(noise),
                   comps=int(n), rule="fallback", tried=len(cands))
