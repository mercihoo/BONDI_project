"""문헌에 투창 개수가 적힌 유물들의 **굽다리 사진**을 모아 한 장으로.

`[수학] 투창-n-fold-판정.md` §7 에 쓴다. 이 그림이 보여 줄 것은 둘이다.

  1. 투창은 실루엣의 **구멍이 아니라 몸체 안의 어두운 사각형**이다
     (그래서 알파 채널로는 못 센다 — `count_tuchang.py` 의 음성 결과)
  2. 그런데 **문헌에는 개수가 적혀 있다.** 눈으로 세는 대신 읽으면 된다

굽다리만 잘라 낸다. rembg 로 몸체를 잡고 bbox 아래쪽 구간만 취한다.
"""
from __future__ import annotations

import json
import os
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
# 이 그림은 별도 분석 보고서용이다. 기본 출력은 이 폴더의 work/.
#   PowerShell:  $env:TUCHANG_FIG_OUT = "C:\path\to\Study\기여자\images"
OUT = Path(os.environ.get("TUCHANG_FIG_OUT") or (H / "work" / "figures"))
OUT.mkdir(parents=True, exist_ok=True)

# (소장품번호, 굽다리 구간 상단 비율, 표제)
PICK = [
    ("경주 5757", 0.52, "1단 · 4개",      "臺脚에는 4個의 長方形 一段透窓이 있고"),
    ("경주 5758", 0.55, "2단 · 4개 交列",  "臺脚은 4個의 透窓을 2段 交列로 配置하였음"),
    ("경주 6311", 0.55, "2단 · 3개 엇갈림", "臺脚에는 3개의 透窓이 2단으로 엇갈림"),
    ("경주 8475", 0.50, "2단 · 각 3개",    "上段에는 方形, 下段에는 長方形의 透窓이 각 3개씩 貫通됨"),
    ("경주 8476", 0.50, "2단 · 각 3개",    "上段에는 長方形, 下段에는 方形의 透窓이 각 3개 貫通되었음"),
    ("경주 8477", 0.50, "1단 · 3개",      "臺脚에는 1段의 長方形 透窓이 3개가 貫通되었됨"),
]
_S = None


def foot_crop(path: Path, top_frac: float, w=1400):
    global _S
    from rembg import new_session, remove
    if _S is None:
        _S = new_session("u2net")
    im = Image.open(path).convert("RGB")
    s = min(1.0, w / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    a = np.asarray(remove(im, session=_S))[:, :, 3] / 255.0
    fg = a > 0.5
    lab, n = ndi.label(fg)
    if n:
        sz = ndi.sum(fg, lab, range(1, n + 1))
        fg = lab == int(np.argmax(sz)) + 1
    ys, xs = np.nonzero(fg)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    Hh = y1 - y0
    ty = int(y0 + top_frac * Hh)
    m = int(0.04 * (x1 - x0))
    box = (max(x0 - m, 0), max(ty - m, 0),
           min(x1 + m, im.width), min(y1 + m, im.height))
    return np.asarray(im.crop(box))


def main():
    recs = {r.get("소장품번호"): r for r in
            json.loads((H / "meta/artifacts.json").read_text("utf-8"))}
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 8.0))
    for a_, (num, tf, cap, han) in zip(np.ravel(ax), PICK):
        r = recs.get(num)
        if r is None:
            a_.axis("off"); continue
        a_.imshow(foot_crop(H / "images" / r["file"], tf))
        a_.set_title(f"{num}  —  기록: {cap}", fontsize=12, pad=6)
        a_.set_xlabel(han, fontsize=8.5, color="#555", wrap=True)
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color("#bbb")
    fig.suptitle("투창 개수는 사진에서 세는 것이 아니라 문헌에서 읽는다"
                 "  —  e뮤지엄 `설명` 에 개수가 적힌 유물들의 굽다리",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.012,
             "투창은 실루엣의 구멍이 아니라 몸체 안의 어두운 사각형이다 "
             "— 너머로 보이는 것이 배경이 아니라 그릇 안쪽이라, 알파 채널로는 안 잡힌다 "
             "(10점 중 4점만, 최대 1개)",
             ha="center", fontsize=9.5, color="#8a3b3b")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    p = OUT / "투창-문헌-예시.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"{p.stat().st_size/1024:.0f}KB  {p.name}")

    # ── 개수 분포 ──
    from collections import Counter
    HAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}

    def cnt(rec):
        s = rec.get("설명") or ""
        if "透窓" not in s:
            return None
        for pat in (r"각각\s*([0-9一二三四五六])\s*개", r"각\s*([0-9一二三四五六])\s*개",
                    r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓", r"透窓이?\s*([0-9一二三四五六])\s*[개個]"):
            m = re.search(pat, s)
            if m:
                g = m.group(1)
                return HAN.get(g, int(g) if g.isdigit() else None)
        return None

    c = Counter(v for v in (cnt(r) for r in recs.values()) if v)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ks = [1, 2, 3, 4, 5, 6]
    vs = [c.get(k, 0) for k in ks]
    bars = ax.bar([str(k) for k in ks], vs,
                  color=["#b0b0b0", "#b0b0b0", "#2f7d32", "#2f7d32", "#b0b0b0", "#c62828"])
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width() / 2, v + .12, str(v), ha="center", fontsize=11)
    ax.annotate("v24 가 택한 값\n문헌 지지 0건", xy=(5, 0.15), xytext=(4.6, 4.2),
                fontsize=10, color="#c62828", ha="center",
                arrowprops=dict(arrowstyle="->", color="#c62828"))
    ax.set_xlabel("단당 투창 개수"); ax.set_ylabel("유물 수")
    ax.set_ylim(0, max(vs) + 2); ax.grid(axis="y", alpha=.25)
    ax.set_title(f"e뮤지엄 설명문에서 읽은 단당 투창 개수  (굽다리바리 {sum(vs)}점)", fontsize=11)
    fig.tight_layout()
    p2 = OUT / "투창-문헌-분포.png"
    fig.savefig(p2, dpi=130, bbox_inches="tight")
    print(f"{p2.stat().st_size/1024:.0f}KB  {p2.name}")
    print("분포:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
