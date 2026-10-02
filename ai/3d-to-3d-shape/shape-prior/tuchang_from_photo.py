"""사진에서 투창을 **밝기로** 찾고 **각도로** 되돌린다.

앞선 실패
---------
`count_tuchang.py` 는 투창을 rembg 알파의 **구멍**으로 찾으려 했다. 10점 중 4점,
최대 1개밖에 못 잡았다. 투창 너머로 보이는 것은 배경이 아니라 **그릇 안쪽 어둠**이라
알파가 안 뚫린다. 신호를 잘못 고른 것이다.

여기서는 그 어둠을 **그대로 신호로 쓴다.** 투창은 주변 기벽보다 뚜렷하게 어둡다.

각도로 되돌리기
---------------
개수만 세면 정보가 반쯤 버려진다. 사진 한 장에는 **앞쪽 절반**만 보이므로
"보이는 개수"와 N 의 관계가 흐리다. 대신 **위치**를 쓰면 주기를 직접 잴 수 있다.

회전체를 옆에서 보면, 반지름 r 인 높이에서 각 θ 의 점은

    x = cx + r * sin(θ)          (cx = 화면상의 축)

로 찍힌다. 그래서 어두운 덩어리의 가로 위치를 되돌리면 **θ 를 복원**할 수 있다.

    θ = asin( (x - cx) / r )

투창 두 개의 θ 간격이 Δ 면 **N = 360 / Δ** 다. 두 개만 보여도 N 이 나온다.
(가장자리 |x-cx|/r -> 1 에서는 asin 이 불안정하므로 0.85 를 넘으면 버린다.)

정답은 문헌이 준다 — `설명` 에 개수가 적힌 12점이 그대로 평가셋이다.

사용:
  python tuchang_from_photo.py
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
OUTFIG = H / "work" / "figures"
MAXDIM = 1400
HAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}
_S = None


def documented(rec):
    s = rec.get("설명") or ""
    if "透窓" not in s:
        return None
    for pat in (r"각각\s*([0-9一二三四五六])\s*개", r"각\s*([0-9一二三四五六])\s*개",
                r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓",
                r"透窓이?\s*([0-9一二三四五六])\s*[개個]"):
        m = re.search(pat, s)
        if m:
            g = m.group(1)
            return HAN.get(g, int(g) if g.isdigit() else None)
    return None


def body_mask(path: Path):
    global _S
    from rembg import new_session, remove
    if _S is None:
        _S = new_session("u2net")
    im = Image.open(path).convert("RGB")
    s = min(1.0, MAXDIM / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    a = np.asarray(remove(im, session=_S))[:, :, 3] / 255.0
    fg = ndi.binary_fill_holes(a > 0.5)
    lab, n = ndi.label(fg)
    if n:
        sz = ndi.sum(fg, lab, range(1, n + 1))
        fg = lab == int(np.argmax(sz)) + 1
    return np.asarray(im).astype(np.float32) / 255.0, fg


def geometry(fg):
    """줄마다 좌/우 끝 -> 축과 반지름. 허리(최소 반지름) 아래가 굽다리다."""
    xs = np.arange(fg.shape[1])
    rows = np.flatnonzero(fg.any(1))
    L = np.full(fg.shape[0], np.nan); R = np.full(fg.shape[0], np.nan)
    for y in rows:
        v = xs[fg[y]]
        L[y], R[y] = v[0], v[-1]
    r = (R - L) / 2
    cx = np.nanmedian((L + R) / 2)
    y0, y1 = rows[0], rows[-1]
    hh = y1 - y0
    mid = slice(y0 + int(.35 * hh), y0 + int(.75 * hh))     # 허리는 중간쯤
    waist = y0 + int(.35 * hh) + int(np.nanargmin(r[mid]))
    return cx, r, y0, y1, waist


def find_windows(img, fg, cx, r, waist, y1, dark_q=0.22, min_frac=6e-4):
    """굽다리 구간에서 주변보다 어두운 덩어리 = 투창."""
    g = img.mean(2)
    band = np.zeros_like(fg)
    band[waist:y1 + 1] = True
    sel = fg & band
    if sel.sum() < 200:
        return [], sel
    thr = np.quantile(g[sel], dark_q)
    dark = sel & (g < thr)
    dark = ndi.binary_opening(dark, np.ones((5, 5)), iterations=1)
    lab, n = ndi.label(dark)
    out = []
    for i in range(1, n + 1):
        ys, xs_ = np.nonzero(lab == i)
        if len(ys) < min_frac * fg.sum():
            continue
        cy = int(round(ys.mean()))
        rr = r[cy]
        if not np.isfinite(rr) or rr <= 1:
            continue
        u = (xs_.mean() - cx) / rr
        if abs(u) > 0.85:                        # 가장자리는 asin 이 불안정
            continue
        out.append(dict(theta=float(np.degrees(np.arcsin(np.clip(u, -1, 1)))),
                        cy=cy, cx=float(xs_.mean()), area=int(len(ys)),
                        h=(y1 - cy) / max(y1 - waist, 1)))
    out.sort(key=lambda d: d["theta"])
    return out, sel


def estimate_n(wins, tol=0.28):
    """보이는 투창들의 θ 간격에서 N = 360/Δ. 같은 단에 있는 것끼리만 짝짓는다."""
    cand = []
    for i in range(len(wins)):
        for j in range(i + 1, len(wins)):
            a, b = wins[i], wins[j]
            if abs(a["h"] - b["h"]) > 0.22:      # 다른 단이면 건너뛴다
                continue
            d = abs(b["theta"] - a["theta"])
            if d < 12:
                continue
            for k in (1, 2):                     # 사이에 창이 하나 낀 경우까지
                n = 360.0 * k / d
                if 2.4 <= n <= 8.5:
                    cand.append(n)
    if not cand:
        return None, cand
    cand = np.array(cand)
    best, bestcnt = None, -1
    for n in (3, 4, 5, 6):
        c = int((np.abs(cand - n) <= tol * n).sum())
        if c > bestcnt:
            bestcnt, best = c, n
    return (best if bestcnt > 0 else None), cand.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTFIG / "fig_tuchang_photo.png"))
    a = ap.parse_args()
    recs = json.loads((H / "meta/artifacts.json").read_text("utf-8"))
    tgt = [(r, documented(r)) for r in recs]
    tgt = [(r, d) for r, d in tgt if d]
    print(f"문헌에 개수가 적힌 유물 {len(tgt)}점 = 평가셋\n")
    print(f"{'소장품':<12}{'기록':>5}{'보인 창':>8}{'추정 N':>8}{'θ (도)':>34}")
    print("-" * 70)

    panels, rows = [], []
    for r, d in tgt:
        img, fg = body_mask(H / "images" / r["file"])
        cx, rr, y0, y1, waist = geometry(fg)
        wins, sel = find_windows(img, fg, cx, rr, waist, y1)
        n, cand = estimate_n(wins)
        th = " ".join(f"{w['theta']:+5.0f}" for w in wins[:6])
        print(f"{r['소장품번호']:<12}{d:>5}{len(wins):>8}{(str(n) if n else '-'):>8}{th:>34}")
        rows.append((r["소장품번호"], d, len(wins), n))
        panels.append((r["소장품번호"], d, n, img, fg, wins, cx, rr, waist, y1))

    ok = [(d, n) for _, d, _, n in rows if n]
    if ok:
        dd = np.array([x[0] for x in ok]); nn = np.array([x[1] for x in ok])
        hit = int((dd == nn).sum())
        print(f"\nN 을 추정한 것 {len(ok)}/{len(rows)} · 기록과 일치 **{hit}/{len(ok)}**")
        for v in sorted(set(dd.tolist())):
            m = dd == v
            print(f"  기록 {v}개 -> 추정 {nn[m].tolist()}")
    else:
        print("\nN 을 하나도 추정하지 못했다")

    nc = 4; nr = -(-len(panels) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 3.4))
    for a_, (num, d, n, img, fg, wins, cx, rr, waist, y1) in zip(np.ravel(np.atleast_1d(ax)), panels):
        a_.imshow(img)
        a_.contour(fg, [0.5], colors="#00d2ff", linewidths=0.6)
        a_.axhline(waist, color="#ffe600", lw=0.7, ls="--")
        for w in wins:
            a_.plot(w["cx"], w["cy"], "o", ms=7, mfc="none", mec="#ff2d2d", mew=1.6)
            a_.text(w["cx"], w["cy"] - 14, f"{w['theta']:+.0f}", color="#ff2d2d",
                    fontsize=7, ha="center")
        a_.set_title(f"{num}  기록 {d} · 추정 {n if n else '-'}", fontsize=9)
        a_.set_ylim(y1 + 20, waist - 30); a_.axis("off")
    for a_ in np.ravel(np.atleast_1d(ax))[len(panels):]:
        a_.axis("off")
    fig.suptitle("사진에서 투창을 밝기로 찾고 θ 로 되돌리기  (빨강=검출, 숫자=복원한 θ)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight")
    print(f"\n그림: {a.out}")


if __name__ == "__main__":
    main()
