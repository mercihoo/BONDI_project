"""사진에서 투창을 세어 **문헌 기록과 맞춰 본다.**

왜 이게 되나
------------
이 프로젝트에서 계속 걸린 것이 "정답이 없다" 였다. 그런데 투창 개수만은 **정답이 있다** —
e뮤지엄 `설명` 에 한자로 적혀 있고, 그 유물들의 사진도 전부 있다.

    "臺脚에는 3개의 透窓이 2단으로 엇갈림"        (경주 6311·6312·6313)
    "臺脚은 4個의 透窓을 2段 交列로 配置하였음"   (경주 5758)

합성으로 지어낸 라벨이 아니라 **박물관이 적어 둔 기록**이다.

무엇을 세는가
-------------
투창은 뚫린 구멍이라 **배경이 비쳐 보인다.** rembg 알파에서 그 자리는 물체가 아니다.
그래서 `binary_fill_holes` 를 **끄면** 투창이 구멍으로 남는다 (기존 파이프라인은 이걸 메운다).

다만 사진 한 장에서는 **앞쪽 절반**만 보인다. 뒤쪽 투창은 앞 구멍을 통해 비치거나 안 보인다.
그래서 이 스크립트가 내는 것은 N 이 아니라 **`보이는 투창 수`** 이고,
N 과의 관계는 기록과 맞춰 보면서 알아내야 한다. 그게 이 스크립트의 목적이다.

결과 — **실패했다. 쓰지 말 것.**
-------------------------------------
기록에 개수가 적힌 10점에 돌렸더니 **구멍이 잡힌 것 4점, 그나마 최대 1개**였다.

    경주 6311  기록 3개 -> 보임 1
    경주 5758  기록 4개 -> 보임 0
    경주 8477  기록 3개 -> 보임 0

가정이 틀렸다. **투창 너머는 배경이 아니라 그릇 안쪽의 어둠이다.**
(71489 근접 렌더에서 이미 봤다 — 투창으로 보이던 '찢어진 자국' 이 반대편 내벽이었다.)
rembg 알파 입장에서는 그것도 전부 물체라 안 뚫린다. 모델이 틀린 게 아니라 전제가 틀렸다.

투창은 실루엣의 **구멍**이 아니라 몸체 안의 **어두운 사각형**이다.
세려면 실루엣이 아니라 **내부 밝기·질감**을 봐야 하고, 그건 다른 도구다.
게다가 사진 한 장은 앞쪽 절반만 보여 주므로 N 을 직접 세는 데는 애초에 한계가 있다.

**그래도 문헌 13건은 살아 있다.** 사전(N 후보 좁히기)에는 사진이 필요 없다.
방법을 채점하려면 그 유물들의 **3D** 가 있어야 한다 (TRELLIS 실행 필요).

사용:
  python count_tuchang.py --all      # 음성 결과 재현용
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
H = Path(__file__).resolve().parent
MAXDIM = 1400

HAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}


def documented(rec) -> int | None:
    """설명문에서 '단당 몇 개' 를 읽는다. parse_meta 의 정규식이 놓친 한글 표현까지."""
    s = rec.get("설명") or ""
    if "透窓" not in s:
        return None
    for pat in (r"각각\s*([0-9一二三四五六])\s*개",          # 上下段에 각각 3개
                r"각\s*([0-9一二三四五六])\s*개",            # 각 3개씩
                r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓",   # 3개의 透窓 / 4個의 透窓
                r"透窓이?\s*([0-9一二三四五六])\s*[개個]"):  # 透窓이 3개가
        m = re.search(pat, s)
        if m:
            g = m.group(1)
            return HAN.get(g, int(g) if g.isdigit() else None)
    return None


def holes(path: Path, band=(0.00, 0.55), min_frac=2.5e-4):
    """물체 안에 갇힌 배경 = 뚫린 구멍. 굽다리 높이대만 본다."""
    from rembg import new_session, remove
    global _S
    try:
        _S
    except NameError:
        _S = new_session("u2net")
    im = Image.open(path).convert("RGB")
    s = min(1.0, MAXDIM / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    a = np.asarray(remove(im, session=_S))[:, :, 3] / 255.0
    fg = a > 0.5
    lab, n = ndi.label(fg)
    if n == 0:
        return None, None, 0, []
    sz = ndi.sum(fg, lab, range(1, n + 1))
    body = lab == int(np.argmax(sz)) + 1
    filled = ndi.binary_fill_holes(body)
    hole = filled & ~body                                   # 갇힌 배경
    rows = np.flatnonzero(filled.any(1))
    y0, y1 = rows[0], rows[-1]
    Hh = y1 - y0 + 1
    lo = y1 - int(band[1] * Hh)                             # 굽다리 = 아래쪽
    hi = y1 - int(band[0] * Hh)
    lab2, m = ndi.label(hole)
    out = []
    for i in range(1, m + 1):
        ys, xs = np.nonzero(lab2 == i)
        if len(ys) < min_frac * filled.sum():
            continue
        cy = ys.mean()
        if lo <= cy <= hi:
            out.append(dict(area=len(ys), cy=(y1 - cy) / Hh,
                            cx=(xs.mean() - xs.min()) / max(filled.sum() ** .5, 1)))
    return np.asarray(im), body, len(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="work/figures/fig_tuchang_count.png")
    a = ap.parse_args()

    recs = json.loads((H / "meta/artifacts.json").read_text("utf-8"))
    tgt = []
    for r in recs:
        d = documented(r)
        if d and (a.all or r.get("class") == "plain"):
            tgt.append((r, d))
    print(f"기록에 개수가 적힌 유물 {len(tgt)}점 — 이것이 정답표다\n")
    print(f"{'소장품':<12}{'분류':<9}{'기록 단당':>9}{'사진에서 보이는 구멍':>20}")
    print("-" * 54)

    panels, rowsum = [], []
    for r, d in tgt:
        img, body, k, det = holes(H / "images" / r["file"])
        if img is None:
            print(f"{r['소장품번호']:<12}{r['class']:<9}{d:>9}      분리 실패")
            continue
        print(f"{r['소장품번호']:<12}{r['class']:<9}{d:>9}{k:>20}")
        rowsum.append((r["소장품번호"], d, k))
        panels.append((r["소장품번호"], d, k, img, body))

    if rowsum:
        dd = np.array([x[1] for x in rowsum]); kk = np.array([x[2] for x in rowsum])
        print(f"\n기록 단당 {dd.min()}~{dd.max()} · 보이는 구멍 {kk.min()}~{kk.max()}")
        ok = kk > 0
        print(f"구멍이 하나라도 보인 것 {ok.sum()}/{len(kk)}")
        if ok.sum() > 2:
            print(f"기록 대 관측 상관 r = {np.corrcoef(dd[ok], kk[ok])[0,1]:+.3f}")
            for v in sorted(set(dd.tolist())):
                m = (dd == v) & ok
                if m.any():
                    print(f"  기록 {v}개 -> 보이는 구멍 {kk[m].tolist()}")

    if panels:
        nc = min(6, len(panels)); nr = -(-len(panels) // nc)
        fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.3, nr * 2.7))
        for a_, (num, d, k, img, body) in zip(np.ravel(np.atleast_1d(ax)), panels):
            a_.imshow(img)
            a_.contour(body, [0.5], colors="#00d2ff", linewidths=0.8)
            a_.set_title(f"{num}\n기록 {d}개 · 보임 {k}", fontsize=8)
            a_.axis("off")
        for a_ in np.ravel(np.atleast_1d(ax))[len(panels):]:
            a_.axis("off")
        fig.suptitle("투창 — 기록(정답) 대 사진에서 보이는 구멍", fontsize=12, y=1.0)
        fig.tight_layout()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.out, dpi=130, bbox_inches="tight")
        print(f"\n그림: {a.out}")


if __name__ == "__main__":
    main()
