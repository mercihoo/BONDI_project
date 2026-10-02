"""정답 없이 채점한다 — 갈래의 분포 안에 들어오나.

무엇을 하는가
-------------
사진 27점에서 뽑은 프로파일을 정규화해 **중앙 형태 + 백분위 밴드**를 만든다.
라벨도 정답도 필요 없다. 그냥 여러 개를 겹쳐 놓고 "보통 이렇다" 를 읽는 것이다.
(비지도 학습이 하는 일이 이것이다. PCA 도 같은 갈래다.)

그 밴드 위에 **오늘 복원한 결과**를 얹으면, 정답이 없어도 물을 수 있다.

    이 복원본은 고배답게 생겼나? 어디가 갈래에서 벗어나나?

원근을 피하는 법
----------------
사진 프로파일에는 촬영 앙각 때문에 세로가 부풀어 있다 (측정 -13~-15%).
그런데 **반지름을 그 유물 자신의 최대 반지름으로 나누면** 그 배율이 약분된다.
축척이 k 배 틀렸어도 r/r_max 는 그대로다. 그래서 **모양만** 비교한다.

세로(h)는 여전히 영향을 받는다. 겉보기 높이 A = H cos t + D sin t 이고, 늘어난 몫의
대부분은 **아가리 타원이 위로 솟은 것**이다. 그래서 밴드의 맨 위쪽은 실제보다
납작하게 눌려 있다 — 아가리 근처 비교는 그만큼 감안해서 봐야 한다.

사용:
  python score_against_band.py --glb <복원본.glb> [--glb <입력.glb>] [--label ...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_glb_profiles import profile as glb_profile

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
H = Path(__file__).resolve().parent
NB = 120
GRID = np.linspace(0, 1, NB)


def norm(r, h):
    """h 0~1 격자 위로 옮기고 r 을 그 유물의 최대 반지름으로 나눈다 (축척 약분)."""
    o = np.argsort(h)
    v = np.interp(GRID, h[o], r[o])
    return v / max(v.max(), 1e-9)


def photo_band(tag="rembg"):
    f = H / "work/profiles" / f"step1_summary_{tag}.csv"
    keep = []
    for row in csv.DictReader(f.open(encoding="utf-8-sig")):
        if row["ok"] != "True" or row["cls"] != "plain" or row["pass"] != "True":
            continue
        p = H / "work/profiles" / f"{Path(row['file']).stem}__{tag}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        z, r = d["z"], d["r"]
        if z.max() <= 0:
            continue
        keep.append((norm(r, z / z.max()), row["num"]))
    return np.array([k[0] for k in keep]), [k[1] for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--out", default="work/figures/fig_band_score.png")
    a = ap.parse_args()

    P, names = photo_band()
    med = np.median(P, 0)
    lo, hi = np.percentile(P, 10, 0), np.percentile(P, 90, 0)
    p25, p75 = np.percentile(P, 25, 0), np.percentile(P, 75, 0)
    print(f"밴드 = 사진 {len(P)}점 (관문 통과분) · 라벨 없음 · 정답 없음\n")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0))
    ax[0].fill_betweenx(GRID, lo, hi, color="#9dc3e0", alpha=.40, label="10~90 백분위")
    ax[0].fill_betweenx(GRID, p25, p75, color="#5b91bb", alpha=.35, label="25~75 백분위")
    ax[0].plot(med, GRID, color="#14406b", lw=2.4, label=f"중앙 형태 (n={len(P)})")

    cols = ["#c62828", "#e8a33d", "#2f7d32"]
    for i, g in enumerate(a.glb):
        lab = a.label[i] if i < len(a.label) else Path(g).stem
        pr, n, _ = glb_profile(Path(g))
        v = pr / max(pr.max(), 1e-9)
        ax[0].plot(v, GRID, color=cols[i % 3], lw=2.2, ls="--", label=f"{lab} ({n:,}정점)")

        # 밴드 밖으로 나간 곳
        out_hi = v > hi
        out_lo = v < lo
        z = (v - med) / np.maximum(hi - lo, 1e-9)          # 밴드 폭 기준 편차
        ax[1].plot(GRID, z, color=cols[i % 3], lw=2, label=lab)
        print(f"{lab}")
        print(f"  밴드(10~90) 밖 {100*(out_hi|out_lo).mean():.0f}% "
              f"· 위로 {100*out_hi.mean():.0f}% · 아래로 {100*out_lo.mean():.0f}%")
        print(f"  편차 |중앙| {np.median(np.abs(z)):.2f} 밴드폭 · 최대 {np.abs(z).max():.2f}")
        for name, m in (("굽 0.00~0.15", (GRID < .15)), ("대각 0.15~0.35", (GRID >= .15) & (GRID < .35)),
                        ("허리 0.35~0.50", (GRID >= .35) & (GRID < .50)),
                        ("배신 0.50~0.85", (GRID >= .50) & (GRID < .85)),
                        ("아가리 0.85~1.0", GRID >= .85)):
            flag = "  <-- 밖" if np.abs(z[m]).max() > 1.0 else ""
            print(f"    {name:<16} 편차 {np.median(z[m]):+.2f}  최대 {z[m][np.argmax(np.abs(z[m]))]:+.2f}{flag}")
        print()

    ax[0].set_xlabel("반지름 / 그 유물의 최대 반지름  (축척 약분)")
    ax[0].set_ylabel("높이 (0=굽, 1=아가리)")
    ax[0].legend(fontsize=8.5, loc="upper left"); ax[0].grid(alpha=.25)
    ax[0].set_title("(a) 갈래의 분포 위에 복원본 얹기", fontsize=11)

    for y, c in ((1, "#c62828"), (-1, "#c62828"), (0, "#888")):
        ax[1].axhline(y, color=c, lw=1, ls=":" if y else "-")
    ax[1].fill_between(GRID, -1, 1, color="#9dc3e0", alpha=.25)
    ax[1].set_xlabel("높이 (0=굽, 1=아가리)")
    ax[1].set_ylabel("중앙 형태에서 벗어난 정도 (밴드폭 단위)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
    ax[1].set_title("(b) ±1 밖이면 갈래에서 벗어난 것", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"그림: {a.out}")


if __name__ == "__main__":
    main()
