# -*- coding: utf-8 -*-
"""
run_ea028.py — [1] 기여자 08-28 과 **같은 입력·같은 지표로** AdaPoinTr 를 돌린다.

바꾸는 변수는 하나다: PoinTr → AdaPoinTr. 나머지는 08-28 그대로다.

  대상   RR_07_01_EA_028  토기 항아리 179,294점 · 0.65×0.62×0.35m · 점간격 1.63mm
         하부 몸통 전체 결실 (08-28 판독 · `scan_dataset.py` 의 bot_closed 0.000 과 일치)
  자세   원자세(Z-up) · 정준(Y-up) 2종 — ShapeNet 은 정준 자세로 학습됐다
  가중치 ShapeNet-55 사전학습. 파인튜닝 없음

08-28 기준선
  결실 하부로 나간 출력점      **0점**
  표면 12mm 초과 "보충점"      ~960점 (전부 재구성 오차로 판정됨)
  잔존부 표면 정합 오차 중앙    3.5mm
  출력 점간격                  ~12mm (원본 1.63mm)

laspy 를 안 쓴다
  원천데이터는 .las 지만 라벨링데이터의 **CSV 가 같은 점군(X,Y,Z,R,G,B)** 이다.
  TRELLIS venv 에 laspy 가 없고, 그 venv 는 본 파이프라인이 쓰는 것이라 건드리지 않는다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY run_ea028.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import run_adapointr as R

HERE = Path(__file__).resolve().parent
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
AIHUB = Path(r"C:\Users\<USER>\Desktop\211-2.문화유산 유적 3D 데이터\01-1.정식개방데이터")

SURF_MM = 12.0          # 08-28 이 쓴 "표면 초과" 문턱


def find_csv(pid: str) -> Path:
    for sp in ("Training", "Validation"):
        p = AIHUB / sp / "02.라벨링데이터" / (pid + ".csv")
        if p.is_file():
            return p
    raise SystemExit("[!] 못 찾았다: " + pid)


def load_points(pid: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(find_csv(pid), usecols=[0, 1, 2])
    P = df.to_numpy(np.float64)
    return P[np.isfinite(P).all(1)]


def fps_torch(P: np.ndarray, n: int, device) -> np.ndarray:
    """08-28 은 open3d 의 FPS 를 썼다. 여기선 shim 의 순수 PyTorch FPS 를 쓴다.
    (open3d 가 venv 에 없다. 같은 알고리즘이다.)"""
    from pointnet2_ops import pointnet2_utils
    x = torch.from_numpy(P).float().unsqueeze(0).to(device)
    idx = pointnet2_utils.furthest_point_sample(x, n).long().squeeze(0).cpu().numpy()
    return P[idx]


def to_yup(P: np.ndarray) -> np.ndarray:
    """Z-up → Y-up 정준. ShapeNet 학습 자세."""
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def from_yup(P: np.ndarray) -> np.ndarray:
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)


def measure(pred_m: np.ndarray, full_m: np.ndarray) -> dict:
    """전부 m 단위. 08-28 과 같은 축으로 잰다."""
    t = cKDTree(full_m)
    d, _ = t.query(pred_m)
    d_mm = d * 1000.0
    new = d_mm > SURF_MM                       # 보충점

    z_min = full_m[:, 2].min()
    below = new & (pred_m[:, 2] < z_min)       # 결실 하부로 나간 점

    dn, _ = cKDTree(pred_m).query(pred_m, k=2)
    return {
        "n_pred": len(pred_m),
        "n_new": int(new.sum()),
        "n_below": int(below.sum()),
        "below_depth_mm": float((z_min - pred_m[below, 2]).max() * 1000) if below.any() else 0.0,
        "resid_mm": float(np.median(d_mm[~new])) if (~new).any() else float("nan"),
        "spacing_mm": float(np.median(dn[:, 1]) * 1000),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="RR_07_01_EA_028")
    ap.add_argument("--n-input", type=int, default=2048)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    R.install_shims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("디바이스:", device)
    model = R.build(device, args.ckpt or None)

    full = load_points(args.id)
    ext = full.max(0) - full.min(0)
    print("\n%s  %d점  bbox %s m" % (args.id, len(full), np.round(ext, 3)))

    out_dir = WORK / "pred_ea028"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for pose in ("raw", "yup"):
        P = full if pose == "raw" else to_yup(full)
        # 08-28 과 같은 정규화: centroid 빼고 최대 노름으로 나눈다
        c = P.mean(0)
        Q = P - c
        s = float(np.linalg.norm(Q, axis=1).max())
        Q /= s

        inp = fps_torch(Q, args.n_input, device)
        pred = R.infer(model, inp.astype(np.float32), device)

        back = pred.astype(np.float64) * s + c          # 원좌표(m)로
        if pose == "yup":
            back = from_yup(back)
        np.save(out_dir / (args.id + "_" + pose + ".npy"), back.astype(np.float32))

        m = measure(back, full)
        m["pose"] = pose
        rows.append(m)
        print("\n=== %s ===" % pose)
        print("  출력 %d점 · 점간격 %.2fmm  (원본 1.63mm)" % (m["n_pred"], m["spacing_mm"]))
        print("  잔존부 정합 오차 중앙 %.2fmm  (08-28: 3.5mm)" % m["resid_mm"])
        print("  보충점(표면 %.0fmm 초과) %d개  (08-28: ~960개)" % (SURF_MM, m["n_new"]))
        print("  **결실 하부로 나간 출력점 %d개**  (08-28: 0개)" % m["n_below"])
        if m["n_below"]:
            print("     아래로 최대 %.1fmm 연장" % m["below_depth_mm"])

    best = max(rows, key=lambda r: r["n_below"])
    print("\n=== [1] 판정 ===")
    print("  08-28 PoinTr  결실 하부 출력점 0개")
    print("  AdaPoinTr     결실 하부 출력점 %d개 (%s 자세)" % (best["n_below"], best["pose"]))
    if best["n_below"] < 50:
        print("  → **08-28 이 재현됐다.** 확장판도 하부 결실을 못 읽는다")
    else:
        print("  → 08-28 과 다르다. 확장판이 개선했다는 뜻 — 로드맵 §2.1 을 고쳐야 한다")

    import csv
    with (WORK / "eval_ea028.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n저장 → %s · %s" % (out_dir, WORK / "eval_ea028.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
