"""배경 분리 — 학습 모델판 (rembg / U^2-Net).

왜 이걸 쓰나
------------
`_seg.py` 를 v1 부터 v7 까지 손으로 짰는데 그림자를 못 떼어냈다.
정리하면 이렇다.

    · 색으로 가르기      -> 회색 유물이 회색 배경 앞이면 같이 지워진다
    · 문턱 하나로 가르기 -> 배경이 제각각인 43장에 맞는 값이 없다
    · 폭으로 가르기      -> 그림자를 없애는 게 아니라 줄일 뿐, 굽이 통짜로 부푼다
    · 덩어리 내부 Otsu   -> 그림자의 배경거리가 유물 어두운 부분과 겹쳐 안 갈린다

공통점이 있다. **"그림자냐 도자기냐"를 가르는 국소 규칙이 존재하지 않는다.**
한 화소만 보면 둘 다 '배경보다 조금 어두운 회색'이다. 구별하려면 **무엇이 물체인지**를
알아야 하는데, 그건 규칙으로 못 쓰고 데이터에서 배워야 하는 종류의 지식이다.

빗살무늬토기에서는 반대였다. 거기선 규칙(축 회전)이 U-Net 을 이겼다 — 회전체는
**회전에 대해 자기 자신**이라는 강한 구조가 있었기 때문이다. 여기에는 그런 구조가 없다.

같은 지표(`입지름 오차`·`foot_bad`)로 재서 규칙판과 직접 비교한다.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        from rembg import new_session
        _SESSION = new_session("u2net")
    return _SESSION


def segment(a: np.ndarray):
    """a: float32 RGB (0~1). `_seg.segment` 와 같은 (mask, info) 를 돌려준다."""
    from PIL import Image
    from rembg import remove
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    cut = remove(im, session=_session())                   # RGBA, 알파가 물체
    alpha = np.asarray(cut)[:, :, 3].astype(np.float32) / 255.0
    m = alpha > 0.5
    if not m.any():
        return m, dict(area=0.0, thr=0.5, noise=0.0, comps=0, rule="rembg", shadow="",
                       cut_frac=0.0)
    m = ndi.binary_fill_holes(ndi.binary_closing(m, np.ones((5, 5)), iterations=2))
    lab, n = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = ndi.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
    # 알파가 애매한 화소 비율 — 모델이 자신 없어 한 곳 (그림자 경계에서 뜬다)
    soft = float(((alpha > 0.1) & (alpha < 0.9)).mean())
    return m, dict(area=float(m.mean()), thr=0.5, noise=soft, comps=int(n),
                   rule="rembg", shadow="", cut_frac=0.0)
