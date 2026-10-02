# -*- coding: utf-8 -*-
"""
compare_models.py — 두 가중치를 **같은 쌍에** 돌린 결과를 나란히 놓는다.

왜 따로 만드나
  학습 로그의 순증은 **검증 쌍이 바뀌면 비교가 안 된다.**
  v2 는 900쌍 중 200쌍(5패턴), v3 는 1,920쌍 중 240쌍(6패턴)으로 잘랐다.
  분모가 다르니 0.913 과 0.9658 은 **다른 것을 잰 값**이다.

  그래서 같은 쌍 전체에 둘 다 돌리고, **결손 깊이로 갈라** 비교한다.
  깊이가 중요한 이유는 v2 의 약점이 정확히 거기였기 때문이다 —
  0.40~0.50 에서 0.913 인데 0.55~0.65 에서 0.433 으로 무너졌다.

무엇을 재나
  `eval_completion.py` 와 같은 **커버리지 순증**이다.

      cover      결손부 중 예측이 덮은 비율
      floor      입력만으로 이미 덮이는 비율 (바닥값)
      **순증**    (cover - floor) / (1 - floor)    0=안 메움 ~ 1=완벽

  바닥값을 빼는 이유는 결손이 얕으면 아무것도 안 해도 cover 가 높게 나오기 때문이다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY compare_models.py --a work/pred_cmp_v2 --b work/pred_cmp_v3 --a-name v2 --b-name v3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))


def load_pred(d: Path, stem: str):
    for ext in (".npy", ".npz"):
        p = d / (stem + ext)
        if p.is_file():
            a = np.load(p, allow_pickle=True)
            return np.asarray(a["pred"] if ext == ".npz" else a, np.float64)
    return None


def gain(pred, comp, removed, part, k=3.0) -> float:
    """커버리지 순증. `eval_completion.coverage_gain` 과 같은 정의."""
    if removed.sum() < 10:
        return float("nan")
    nn, _ = cKDTree(part).query(part[:: max(1, len(part) // 2000)], k=2)
    tau = k * float(np.median(nn[:, 1]))
    gt = comp[removed]
    d_pred, _ = cKDTree(pred).query(gt)
    d_part, _ = cKDTree(part).query(gt)
    cover = float((d_pred < tau).mean())
    floor = float((d_part < tau).mean())
    return (cover - floor) / max(1e-9, 1.0 - floor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(WORK / "pairs"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(WORK / "compare_models.csv"))
    args = ap.parse_args()

    files = sorted(Path(args.pairs).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    A, B = Path(args.a), Path(args.b)
    rows = []
    for i, f in enumerate(files, 1):
        d = np.load(f, allow_pickle=True)
        part = np.asarray(d["partial"], np.float64)
        comp = np.asarray(d["complete"], np.float64)
        rem = np.asarray(d["complete_removed"], bool)
        pa, pb = load_pred(A, f.stem), load_pred(B, f.stem)
        if pa is None or pb is None:
            continue
        rows.append({
            "name": f.stem, "pattern": str(d["pattern"]),
            "removed": float(rem.mean()),
            "a": gain(pa, comp, rem, part), "b": gain(pb, comp, rem, part)})
        if i % 200 == 0:
            print("  %d/%d" % (i, len(files)), flush=True)

    if not rows:
        print("[!] 겹치는 예측이 없다.")
        return 1
    with Path(args.out).open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    R = [r for r in rows if np.isfinite(r["a"]) and np.isfinite(r["b"])]
    an, bn = args.a_name, args.b_name

    def line(lab, sel):
        if not sel:
            return
        a = np.median([r["a"] for r in sel]); b = np.median([r["b"] for r in sel])
        print("  %-22s %5d %9.3f %9.3f   %+.3f" % (lab, len(sel), a, b, b - a))

    print("\n%-24s %5s %9s %9s   %s" % ("", "쌍", an, bn, "차이"))
    print("  " + "-" * 56)
    print("  [패턴별]")
    for p in ("none", "bottom", "rim", "side", "ragged", "slant"):
        line(p, [r for r in R if r["pattern"] == p])
    print("  [결손 깊이별]  ← v2 의 약점이 여기였다")
    for lo, hi in ((0.0, 0.10), (0.10, 0.20), (0.25, 0.35), (0.40, 0.50),
                   (0.55, 0.65), (0.65, 1.01)):
        line("%.2f~%.2f" % (lo, hi),
             [r for r in R if lo <= r["removed"] < hi and r["pattern"] != "none"])
    print("  " + "-" * 56)
    line("전체", R)
    print("\n표 → " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
