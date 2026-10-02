"""TRELLIS 3D 가 **실물 비율**을 살렸나 — 박물관 기록으로 채점한다.

이게 왜 이제서야 되나
--------------------
앞서 `compare_glb_profiles.py` 로 "TRELLIS 가 실물을 얼마나 살렸나" 를 쟀다고 썼는데
**틀렸다.** 비교 대상 둘 다 같은 모델의 산출물이었다. 그래서 r=0.9952 는 정확도가 아니라
재현성이었다. 정확도를 재려면 **모델 바깥의 정답**이 필요한데 71489 에는 없었다.

이제 있다. 사진 43장을 전부 3D 로 만들었고, e뮤지엄 `설명` 에는
**높이와 입지름이 둘 다 적힌 굽다리바리가 20점** 있다. 그 둘의 비를 맞춰 보면 된다.

    기록:  입지름 / 높이          <- 사람이 자로 잰 값
    3D  :  아가리지름 / 높이      <- 모델이 만든 값

비(比) 로 보는 이유는 TRELLIS 출력이 정규화돼 있어 **절대 크기가 없기** 때문이다.
비는 절대 크기와 무관하므로 그대로 비교된다.

주의 — 깨진 유물은 빼야 한다
---------------------------
아가리가 결실된 유물은 3D 의 '윗변' 이 원래 아가리가 아니다. 기록에 `입지름` 이
없는 것들이 대개 그렇고, `설명` 에 `缺失/破損` 이 적힌 것도 조심해야 한다.
그래서 **입지름이 기록된 것** 만 쓴다 — 박물관이 잴 수 있었다는 뜻이니까.

사용:
  python validate_trellis_ratio.py [--all]
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

from measure_glb import fit_axis, up_axis, vertices

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
H = Path(__file__).resolve().parent
G3 = H / "trellis3d"


def ratios(path: Path):
    V = vertices(path)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    Hh = float(z.max() - z.min())
    h = (z - z.min()) / Hh
    top = h >= 0.93
    d_rim = 2 * float(np.quantile(r[top], 0.92)) if top.sum() > 50 else np.nan
    d_max = 2 * float(np.quantile(r, 0.999))
    d_foot = 2 * float(np.quantile(r[h <= 0.05], 0.92)) if (h <= 0.05).sum() > 50 else np.nan
    return dict(H=Hh, rim=d_rim / Hh, mx=d_max / Hh, foot=d_foot / Hh, nv=len(V))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="handled 까지 포함")
    ap.add_argument("--out", default="work/figures/fig_trellis_ratio_check.png")
    a = ap.parse_args()

    recs = json.loads((H / "meta/artifacts.json").read_text("utf-8"))
    rows = []
    for rec in recs:
        if not a.all and rec.get("class") != "plain":
            continue
        hcm, dcm = rec.get("height_cm"), rec.get("mouth_cm")
        if not (hcm and dcm):
            continue                                   # 입지름 없는 것 = 아가리 결실
        s = re.sub(r"[^0-9A-Za-z가-힣]+", "_", (rec.get("소장품번호") or "")).strip("_")
        p = G3 / f"{s}.glb"
        if not p.exists():
            continue
        try:
            m = ratios(p)
        except Exception as e:
            print(f"  {s}: 실패 {e}")
            continue
        rows.append(dict(num=rec["소장품번호"], cls=rec["class"], hcm=hcm, dcm=dcm,
                         rec_ratio=dcm / hcm, **m))

    if not rows:
        print("대상 없음"); return

    rr = np.array([x["rec_ratio"] for x in rows])
    t_rim = np.array([x["rim"] for x in rows])
    t_max = np.array([x["mx"] for x in rows])

    print(f"대상 {len(rows)}점 — 높이·입지름이 **둘 다 기록된** 굽다리바리\n")
    print(f"{'소장품':<13}{'높이cm':>7}{'입지름cm':>9}{'기록 D/H':>9}"
          f"{'3D 아가리/H':>12}{'오차%':>8}{'3D 최대/H':>10}{'오차%':>8}")
    print("-" * 76)
    for x, a1, a2 in zip(rows, t_rim, t_max):
        e1 = 100 * (a1 - x["rec_ratio"]) / x["rec_ratio"]
        e2 = 100 * (a2 - x["rec_ratio"]) / x["rec_ratio"]
        print(f"{x['num']:<13}{x['hcm']:>7.1f}{x['dcm']:>9.1f}{x['rec_ratio']:>9.3f}"
              f"{a1:>12.3f}{e1:>+8.1f}{a2:>10.3f}{e2:>+8.1f}")

    for lab, t in (("아가리지름/높이", t_rim), ("최대지름/높이", t_max)):
        e = (t - rr) / rr
        print(f"\n[{lab}]")
        print(f"  상관 r = {np.corrcoef(rr, t)[0,1]:+.3f}")
        print(f"  부호 있는 오차 중앙 {100*np.median(e):+.1f}%  (양수면 3D 가 더 벌어졌다)")
        print(f"  절대 오차 중앙 {100*np.median(np.abs(e)):.1f}% · p90 {100*np.percentile(np.abs(e),90):.1f}%")
        print(f"  10% 이내 {int((np.abs(e)<0.10).sum())}/{len(e)} · "
              f"20% 이내 {int((np.abs(e)<0.20).sum())}/{len(e)}")

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.8))
    lim = [min(rr.min(), t_rim.min(), t_max.min()) * 0.9,
           max(rr.max(), t_rim.max(), t_max.max()) * 1.05]
    for a_, t, lab in ((ax[0], t_rim, "3D 아가리지름 / 높이"), (ax[1], t_max, "3D 최대지름 / 높이")):
        a_.plot(lim, lim, "k--", lw=1, label="일치선")
        a_.scatter(rr, t, s=40, c="#14406b", zorder=3)
        for x, v in zip(rows, t):
            a_.annotate(x["num"].split()[-1], (x["rec_ratio"], v), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", color="#555")
        e = np.median((t - rr) / rr)
        a_.set_xlabel("기록 입지름 / 높이"); a_.set_ylabel(lab)
        a_.set_xlim(lim); a_.set_ylim(lim); a_.grid(alpha=.25); a_.legend(fontsize=9)
        a_.set_title(f"{lab}\n부호 오차 중앙 {100*e:+.1f}% · r={np.corrcoef(rr,t)[0,1]:+.3f}",
                     fontsize=11)
    fig.suptitle("TRELLIS 3D 의 비율을 박물관 기록으로 채점 — 모델 바깥의 정답", fontsize=13)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"\n그림: {a.out}")


if __name__ == "__main__":
    main()
