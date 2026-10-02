#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""가중치 여러 개를 **같은 쌍**에 재서 한 표로 낸다.

왜
--
`eval_completion.py` 는 한 번에 하나만 잰다. 그래서 v1 → v2 → v3 를 비교하려면
표를 손으로 옮겨 붙여야 했고, 실제로 **평가 쌍이 다른 표를 나란히 놓은 적**이 있다
(v2 는 900쌍, v3 는 1,920쌍 — 결과.md §5.11). 그걸 막는다.

무엇을 재나
-----------
    재현율(커버리지)   결손 정답점 중 예측이 δ 안에 있는 비율
    바닥값             아무것도 안 메운 모델도 공짜로 받는 몫
    순증               (재현율 − 바닥값) / (1 − 바닥값)
    정밀도             보충점 중 정답 δ 안에 있는 비율
    F생 / F순증        위 둘의 조화평균. 바닥값을 빼기 전/후

**정밀도가 없으면 재현율만으로는 속는다.** 무작위로 뿌린 대조군이
`side` 에서 순증 1.000 을 받는다 (`eval_completion.py --self-test` 의 `random`).
그때 정밀도는 0.042 다. 둘을 같이 봐야 한다.

쓰는 법
-------
    $PY eval_all_models.py \\
        base=work/pred_cmp_base v1=work/pred_cmp_v1 \\
        v2=work/_보관/pred_cmp_v2 v3=work/_보관/pred_cmp_v3
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("ADAPOINTR_WORK", HERE / "work"))

KEYS = [("miss_cover", "재현율"), ("cover_floor", "바닥값"), ("cover_gain", "순증"),
        ("precision", "정밀도"), ("fscore", "F생"), ("fscore_gain", "F순증")]


def main() -> int:
    # **검증 형상만** 볼 수 있어야 한다. 학습 형상을 섞어 보고하면 낙관적이다.
    only_val = "--only-val" in sys.argv
    # **쌍 폴더를 넘겨야 한다.** 안 넘기면 eval_completion 이 기본 work/pairs 를 보고
    # 테스트 예측과 이름이 안 맞아 조용히 아무것도 못 잰다 (실제로 겪었다).
    pairs = next((a.split("=", 1)[1] for a in sys.argv[1:]
                  if a.startswith("--pairs=")), str(WORK / "pairs"))
    VAL_SOURCES = ("경주_887", "대구대_1834")
    items = []
    for a in [x for x in sys.argv[1:]
              if x != "--only-val" and not x.startswith("--pairs=")]:
        if "=" not in a:
            print("[!] 이름=경로 꼴로 준다: " + a)
            return 1
        n, p = a.split("=", 1)
        items.append((n, p))
    if not items:
        print(__doc__)
        return 1

    got: dict[str, list[dict]] = {}
    for name, pred in items:
        out = WORK / ("eval_%s.csv" % name)
        if not (WORK / pred.split("/")[-1]).exists() and not Path(pred).exists():
            print("  [건너뜀] %s — 예측 폴더가 없다: %s" % (name, pred))
            continue
        print("── %s  재는 중 …" % name, flush=True)
        r = subprocess.run([sys.executable, "eval_completion.py",
                            "--pred", pred, "--out", str(out),
                            "--pairs", pairs],
                           cwd=str(HERE), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-800:])
            continue
        rr = list(csv.DictReader(out.open(encoding="utf-8-sig")))
        got[name] = [r for r in rr if r["source"] in VAL_SOURCES] if only_val else rr

    if not got:
        print("[!] 잰 것이 없다.")
        return 1

    pats = []
    for rows in got.values():
        for r in rows:
            if r["pattern"] not in pats:
                pats.append(r["pattern"])
    order = ["none", "bottom", "rim", "side", "ragged", "slant"]
    pats = [p for p in order if p in pats] + [p for p in pats if p not in order]
    pats = ["전체"] + pats          # 합산을 맨 위에

    def agg(rows, pat, key):
        """pat 이 "전체" 면 **결손이 있는 패턴 전부**를 한 무더기로 본다.

        `none` 은 뺀다 — 결손이 없어 재현율·정밀도가 정의되지 않는다.
        대신 과채움 대조군으로 따로 본다 (보충점이 0 에 가까워야 한다).
        """
        v = [float(r[key]) for r in rows
             if (r["pattern"] == pat if pat != "전체" else r["pattern"] != "none")
             and r.get(key) not in ("", None)
             and r[key] == r[key] and r[key] != "nan"]
        return float(np.median(v)) if v else float("nan")

    # --- **결손 깊이별로도 낸다.**
    #
    # 패턴별 중앙값 하나로 요약하면 속는다. 검증 쌍의 절반이 결손률 0.20 미만이라
    # **중앙값이 쉬운 구간에 앉는다** — 실제로 v1 이 전체 0.902 로 보였는데
    # 깊이별로 가르면 0.50~0.65 에서 **0.236** 이었다 (결과.md §5.14).
    depth = {}
    for f in Path(pairs).glob("*.npz"):
        try:
            depth[f.stem] = float(np.load(f, allow_pickle=True)["complete_removed"].mean())
        except Exception:
            pass
    BANDS = [(0.0, .20), (.20, .35), (.35, .50), (.50, .65), (.65, 1.01)]

    def agg_d(rows, lo, hi, key):
        v = [float(r[key]) for r in rows
             if r["pattern"] != "none" and r.get(key) not in ("", None)
             and r[key] == r[key] and r[key] != "nan"
             and lo <= depth.get(r["name"], -1) < hi]
        return float(np.median(v)) if v else float("nan")

    names = list(got)
    for key, label in KEYS:
        print("\n%s" % label)
        print("  %-8s %s" % ("패턴", " ".join("%8s" % n for n in names)))
        for p in pats:
            vals = [agg(got[n], p, key) for n in names]
            if all(not np.isfinite(v) for v in vals):
                continue
            print("  %-8s %s" % (p, " ".join(
                "%8.3f" % v if np.isfinite(v) else "%8s" % "-" for v in vals)))

    # --- 깊이별 (제일 중요한 표라 뒤에 크게 낸다)
    for key, label in (("cover_gain", "순증"), ("precision", "정밀도"),
                       ("fscore_gain", "F순증")):
        print("\n깊이별 %s  (결손률 구간 · none 제외)" % label)
        print("  %-12s %5s %s" % ("결손률", "쌍", " ".join("%8s" % n for n in names)))
        for lo, hi in BANDS:
            n0 = sum(1 for r in got[names[0]]
                     if r["pattern"] != "none" and lo <= depth.get(r["name"], -1) < hi)
            if n0 == 0:
                continue
            print("  %.2f~%.2f %5d %s" % (lo, hi, n0, " ".join(
                "%8.3f" % agg_d(got[n], lo, hi, key) if np.isfinite(
                    agg_d(got[n], lo, hi, key)) else "%8s" % "-" for n in names)))

    # 과채움 대조군 — `none` 쌍에 보충점을 얼마나 만드나. 0 에 가까워야 한다.
    print("\n과채움 (none 쌍 · 보충점 수 — 낮을수록 좋다)")
    print("  %-8s %s" % ("", " ".join("%8s" % n for n in names)))
    print("  %-8s %s" % ("none", " ".join(
        "%8.0f" % agg(got[n], "none", "n_new") for n in names)))

    out = WORK / "eval_all_models.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["지표", "패턴"] + names)
        for key, label in KEYS:
            for lo, hi in BANDS:
                vals = [agg_d(got[n], lo, hi, key) for n in names]
                if all(not np.isfinite(v) for v in vals):
                    continue
                w.writerow([label, "깊이 %.2f~%.2f" % (lo, hi)]
                           + [round(v, 4) if np.isfinite(v) else "" for v in vals])
            for p in pats:
                vals = [agg(got[n], p, key) for n in names]
                if all(not np.isfinite(v) for v in vals):
                    continue
                w.writerow([label, p] + [round(v, 4) if np.isfinite(v) else "" for v in vals])
    print("\n표 → %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
