"""두 GLB 의 **높이 정규화 프로파일**을 맞춰 본다.

무엇을 재는가 — 그리고 무엇은 못 재는가
--------------------------------------
두 메시의 모양이 얼마나 같은지를 잰다. **그뿐이다.**

처음 이 스크립트를 쓸 때 `굽다리바리회손유물.glb` 를 실물 3D 로 착각하고
"TRELLIS 가 실물을 얼마나 살렸나" 를 잰다고 적었다. **둘 다 같은 모델의 산출물이었다.**
그래서 나온 r=0.9952 는 정확도가 아니라 **같은 모델 두 실행이 서로 얼마나 닮았나**,
즉 재현성이다.

정확도를 재려면 **모델 바깥의 정답**이 있어야 한다. 이 유물에는 없다
(71489 는 `입지름` 기록도 없다).

비교는 **높이로 정규화한 프로파일** r(h)/H 로 한다. 두 파일 다 정규화 좌표라
절대 크기는 의미가 없고 비율만 비교 대상이다.

경고 신호가 하나 있다 — 같은 계열인 `geometry_0` 는 오늘 아침 형태가 완전히
무너져 있었다 (갈비뼈 원통). 두 산출물이 서로 닮았다고 해서 **둘 다 맞다는
보장은 없다.** 같은 방향으로 같이 틀릴 수 있다.

사용:
  python compare_glb_profiles.py <a.glb> <b.glb> [--height-cm 15] [--label-a ..] [--label-b ..]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from measure_glb import fit_axis, up_axis, vertices

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
NB = 120


def profile(path: Path):
    """높이로 정규화한 바깥 프로파일 r(h)/H. h 는 0(굽) ~ 1(아가리)."""
    V = vertices(path)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    H = float(z.max() - z.min())
    h = (z - z.min()) / H
    ib = np.clip((h * NB).astype(int), 0, NB - 1)
    out = np.full(NB, np.nan)
    for i in range(NB):
        m = ib == i
        if m.sum() >= 30:
            out[i] = np.quantile(r[m], 0.97) / H      # 바깥 껍질
    g = ~np.isnan(out)
    return np.interp(np.linspace(0, 1, NB), np.linspace(0, 1, NB)[g], out[g]), len(V), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_glb"); ap.add_argument("b_glb")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--height-cm", type=float, default=15.0)
    ap.add_argument("--out", default="work/figures/fig_glb_compare.png")
    a = ap.parse_args()

    pr, nr, Hr = profile(Path(a.a_glb))
    pt, nt, Ht = profile(Path(a.b_glb))
    h = np.linspace(0, 1, NB)
    d = pt - pr
    rel = d / np.maximum(pr, 1e-9)
    k = a.height_cm                                   # 정규화 높이 1 = 기록 높이

    print(f"A  {Path(a.a_glb).name:<30} 정점 {nr:,}  ({a.label_a})")
    print(f"B  {Path(a.b_glb).name:<30} 정점 {nt:,}  ({a.label_b})")
    print("\n※ 둘 다 같은 계열 산출물이면 이 값은 정확도가 아니라 **재현성**이다.\n")
    print(f"{'구간':<16}{'A r/H':>10}{'B r/H':>10}{'차이':>9}{'차이cm':>9}{'상대%':>8}")
    print("-" * 62)
    for lo, hi, lab in ((0.00, 0.15, "굽 바닥"), (0.15, 0.35, "대각"),
                        (0.35, 0.50, "허리"), (0.50, 0.75, "배신 아래"),
                        (0.75, 0.92, "배신 위"), (0.92, 1.00, "아가리")):
        m = (h >= lo) & (h < hi)
        print(f"{lab:<16}{pr[m].mean():>10.4f}{pt[m].mean():>10.4f}"
              f"{d[m].mean():>+9.4f}{d[m].mean()*k:>+9.2f}{100*rel[m].mean():>+8.1f}")
    print("-" * 62)
    print(f"{'전체':<16}{pr.mean():>10.4f}{pt.mean():>10.4f}"
          f"{np.median(d):>+9.4f}{np.median(d)*k:>+9.2f}{100*np.median(rel):>+8.1f}")
    print(f"\n절대차 중앙 {np.median(np.abs(d))*k:.2f}cm · p90 {np.percentile(np.abs(d),90)*k:.2f}cm"
          f" · 최대 {np.abs(d).max()*k:.2f}cm")
    print(f"상대차 중앙 {100*np.median(np.abs(rel)):.1f}% · p90 {100*np.percentile(np.abs(rel),90):.1f}%")
    print(f"모양 상관계수 r = {np.corrcoef(pr, pt)[0,1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(pr * k, h, lw=2.2, color="#14406b", label=f"{a.label_a} ({nr:,}정점)")
    ax[0].plot(pt * k, h, lw=2.2, color="#e07b39", ls="--", label=f"{a.label_b} ({nt:,}정점)")
    ax[0].set_xlabel(f"반지름 (cm · 높이 {a.height_cm}cm 환산)")
    ax[0].set_ylabel("높이 (0=굽, 1=아가리)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
    ax[0].set_title("(a) 단면 프로파일", fontsize=11)

    ax[1].axhline(0, color="#888", lw=1)
    ax[1].plot(h, d * k, lw=2, color="#c62828")
    ax[1].fill_between(h, 0, d * k, color="#c62828", alpha=.18)
    ax[1].set_xlabel("높이 (0=굽, 1=아가리)"); ax[1].set_ylabel(f"{a.label_b} - {a.label_a} (cm)")
    ax[1].grid(alpha=.25)
    ax[1].set_title(f"(b) 차이 — 절대 중앙 {np.median(np.abs(d))*k:.2f}cm", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"\n그림: {a.out}")


if __name__ == "__main__":
    main()
