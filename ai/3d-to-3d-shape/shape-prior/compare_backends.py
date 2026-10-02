"""규칙 기반 분리 vs 학습 모델 분리 — 같은 잣대로.

`_seg.py` 를 v1~v7 까지 손으로 짜고도 그림자를 못 뗐다. 그래서 사전학습 모델
(rembg / U^2-Net) 을 같은 자리에 끼우고 **같은 지표**로 쟀다.

지표는 1단계에서 이미 만들어 둔 것을 그대로 쓴다 — 새로 만들면 비교가 안 된다.
  · 입지름 오차   실루엣 최대폭 vs 박물관 기록 (온전한 유물은 아가리가 최대폭이다)
  · foot_bad     아래쪽 35% 에서 잘라내기가 일어난 비율 = 굽이 그림자에 오염된 정도
  · 좌우차       회전체인데 좌우 반폭이 다른 정도
  · 덩어리 수     마스크가 조각났는지
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent


def load(tag):
    f = H / "work" / "profiles" / f"step1_summary_{tag}.csv"
    rows = [r for r in csv.DictReader(f.open(encoding="utf-8-sig"))
            if r["ok"] == "True" and r["cls"] == "plain"]
    for r in rows:
        for k in ("asym", "foot_bad", "wmax_cm", "mouth_cm", "comps", "elev"):
            r[k] = float(r[k]) if r[k] not in ("", "None") else None
    return {r["num"]: r for r in rows}


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def main():
    A, B = load("rule"), load("rembg")
    keys = sorted(set(A) & set(B))
    print(f"굽다리바리 {len(keys)}점 · 같은 사진 · 같은 측정 코드 · 분리 방식만 교체\n")
    print(f"{'':<22}{'규칙 v3':>12}{'rembg':>12}")
    print("-" * 46)

    def line(lab, f, fmt="{:>12.1f}"):
        print(f"{lab:<22}" + fmt.format(f(A)) + fmt.format(f(B)))

    line("관문 통과", lambda D: sum(D[k]["pass"] == "True" for k in keys), "{:>12.0f}")
    line("굽 오염 0% 인 것", lambda D: sum(D[k]["foot_bad"] == 0 for k in keys), "{:>12.0f}")
    line("덩어리 1개인 것", lambda D: sum(D[k]["comps"] == 1 for k in keys), "{:>12.0f}")
    line("좌우차 중앙 %", lambda D: med([D[k]["asym"] for k in keys]))
    line("굽오염 중앙 %", lambda D: med([D[k]["foot_bad"] for k in keys]))
    err = lambda D: med([abs(D[k]["wmax_cm"] - D[k]["mouth_cm"]) / D[k]["mouth_cm"] * 100
                         for k in keys if D[k]["mouth_cm"]])
    line("입지름 오차 중앙 %", err)
    sgn = lambda D: med([(D[k]["wmax_cm"] - D[k]["mouth_cm"]) / D[k]["mouth_cm"] * 100
                         for k in keys if D[k]["mouth_cm"]])
    line("  부호 있는 오차 %", sgn)
    ev = lambda D: med([D[k]["elev"] for k in keys])
    line("앙각 추정 중앙 도", ev)
    iqr = lambda D: (lambda v: np.percentile(v, 75) - np.percentile(v, 25))(
        [D[k]["elev"] for k in keys if D[k]["elev"] is not None])
    line("  앙각 사분위폭 도", iqr)

    print("\n유물별 굽 오염 (0 이면 굽까지 믿을 수 있다)")
    print(f"{'소장품':<12}{'규칙':>8}{'rembg':>8}   {'좌우차 규칙->rembg':>22}")
    for k in sorted(keys, key=lambda k: -A[k]["foot_bad"]):
        print(f"{k:<12}{A[k]['foot_bad']:>8.1f}{B[k]['foot_bad']:>8.1f}"
              f"{A[k]['asym']:>14.1f} -> {B[k]['asym']:.1f}")


if __name__ == "__main__":
    main()
