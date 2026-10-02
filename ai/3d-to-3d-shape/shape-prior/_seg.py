"""배경 분리. 채택본 = v3 (다항 배경면 + 단일 문턱 + 테두리에 안 닿는 최대 덩어리).

시도한 것과 결과 — 여기 남기는 이유는 다음에 같은 길을 또 걷지 않기 위해서다.

| 판 | 방법 | 결과 |
| --- | --- | --- |
| v1 | 테두리 중앙값 배경 | 그라데이션 못 따라감 + **그림자 포함**. 최대폭 33~38cm |
| v2 | 그림자를 '무채색 어두워짐'으로 판별 | **회색 유물이 회색 배경 앞이면 같이 지워짐** (덩어리 11~18) |
| v3 | 색 판별 버리고 **기하로** 그림자 제거 | 부호오차 중앙 **-16.3%**, 앙각 10.2도로 설명됨. **채택** |
| v4 | 이력 문턱 | 반대로 배경이 샘 (+22%, 좌우차 0.0 = 프레임 전체) |
| v5 | 문턱 후보 훑기 + 합격조건 | 엄격한 쪽부터 골라 조각을 잡음 (좌우차 290%) |

v4·v5 는 `legacy/` 에 있다.

**배운 것.** 배경이 제각각인 사진 43장에 맞는 단일 문턱은 없다. v3 도 전부는 못 맞힌다.
그래서 문턱을 더 만지는 대신 **못 맞힌 것을 골라내는 관문**을 뒀다 (`extract_silhouette.py`).
자동으로 안 되는 것을 억지로 되게 만드는 것보다, **안 된 것을 안다고 말하는 편이 낫다.**
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


USE_OTSU_SHADOW = False


def _otsu(x, bins=256):
    hist, e = np.histogram(x, bins=bins)
    q = hist / max(hist.sum(), 1)
    w0 = np.cumsum(q); w1 = 1 - w0
    c = (e[:-1] + e[1:]) / 2
    m0 = np.cumsum(q * c) / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((q * c)[::-1])[::-1]) / np.maximum(w1, 1e-12)
    v = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.nanargmax(v))])


def drop_shadow(blob, d):
    """덩어리 안에서 유물과 그림자를 가른다 — **대비로**.

    왜 이게 되나
    ------------
    앞서 색(무채색 어두워짐)으로 가르려다 실패했다. 회색 유물이 회색 배경 앞에 있으면
    유물도 무채색이라 같이 지워졌기 때문이다.

    그런데 **덩어리 안에서만** 보면 얘기가 다르다. 그림자는 '배경이 조금 어두워진 것'이라
    배경과의 거리 d 가 작고, 유물은 크다. 회색이든 아니든 그렇다.
    그래서 덩어리 내부의 d 에 Otsu 를 걸면 둘이 갈린다.

    무턱대고 믿지 않는다 — 그림자가 없는 사진에서는 이 분할이 **유물 자신을**
    밝은 부분/어두운 부분으로 쪼갤 수 있다. 그래서 받아들이는 조건을 둔다:

        남는 면적 >= 원래의 35%        너무 많이 깎으면 유물을 자른 것
        폭/높이 비가 줄어들 것          그림자를 뗐다면 반드시 홀쭉해진다

    조건을 못 맞추면 원래 덩어리를 그대로 쓴다.
    """
    if not blob.any():
        return blob, "none"
    inside = d[blob]
    if inside.size < 50:
        return blob, "none"
    t = _otsu(inside)
    core = blob & (d > t)
    core = ndi.binary_opening(core, np.ones((3, 3)), iterations=1)
    lab, n = ndi.label(core)
    if n == 0:
        return blob, "keep"
    sizes = ndi.sum(core, lab, range(1, n + 1))
    core = ndi.binary_fill_holes(
        ndi.binary_closing(lab == int(np.argmax(sizes)) + 1, np.ones((7, 7)), iterations=2))

    def aspect(m):
        ys, xs = np.nonzero(m)
        return (xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1)

    if core.sum() < 0.35 * blob.sum():
        return blob, "keep(작아짐)"
    if aspect(core) > aspect(blob) * 0.98:
        return blob, "keep(안홀쭉)"
    return core, "cut"


def segment(a):
    h, w = a.shape[:2]
    B, noise = background(a)
    d = np.linalg.norm(a - B, axis=2)
    thr = max(5.0 * noise, 0.050)
    fg = ndi.binary_opening(d > thr, np.ones((3, 3)), iterations=2)
    fg = ndi.binary_fill_holes(ndi.binary_closing(fg, np.ones((7, 7)), iterations=2))

    lab, n = ndi.label(fg)
    if n == 0:
        return fg, dict(area=0.0, thr=thr, noise=noise, comps=0, rule="none")
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    edge = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
    inner = [i for i in range(1, n + 1) if i not in edge]
    pick = (max(inner, key=lambda i: sizes[i - 1]) if inner else int(np.argmax(sizes)) + 1)
    big = ndi.binary_fill_holes(ndi.binary_closing(lab == pick, np.ones((5, 5)), iterations=2))
    a0 = big.sum()
    # v6 (덩어리 내부 Otsu 로 유물/그림자 가르기) 는 **채택하지 않았다**.
    # 대부분 조건 미달로 거부됐고 (남는 면적 < 35%), 작동한 경우는 오히려 나빠졌다
    # (증 873 좌우차 5.6 -> 33.2%, 경주 5756 -7.9 -> -34.8%). 그림자의 d 분포가
    # 유물의 어두운 부분과 겹쳐서 한 문턱으로 안 갈린다. 함수는 근거로 남겨 둔다.
    how = "off"
    if USE_OTSU_SHADOW:
        big, how = drop_shadow(big, d)
    return big, dict(area=float(big.mean()), thr=float(thr), noise=float(noise),
                     comps=int(n), rule="v6", shadow=how,
                     cut_frac=round(1 - big.sum() / max(a0, 1), 3))
