# -*- coding: utf-8 -*-
"""
run_71489.py — **실제 복원 대상**을 AdaPoinTr 에 넣어본다.

[1][2] 가 남긴 질문은 이것이다.
    [2] 굽다리 잘라낸 고배 → 메운다 (커버리지 순증 0.785, rim 은 0.961)
    [1] 바닥 없는 돔       → 0개
  차이는 "남은 형상이 학습 분포에서 부분으로 읽히나" 였다.

  **71489 는 어느 쪽인가?** 아가리가 둘레 전체 톱니로 깨진 고배다 —
  [2] 의 `rim` 과 닮았다(0.961). 그렇다면 될 수도 있다.
  반대로 손상 GLB 가 **경계변 0 의 닫힌 껍질**이라 [1] 처럼 완형으로 읽힐 수도 있다.
  안 넣어보고는 모른다.

정답이 없다. 그래서 재는 것은 둘뿐이다.
    보충점 수      입력 표면에서 멀리 떨어진 출력점 = 모델이 새로 만든 것
    어디로 갔나    특히 **아가리 위쪽**으로 연장했나 (알려진 결손 방향)

사용
  PY=".../venv/Scripts/python.exe"
  $PY run_71489.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import run_adapointr as R

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
TARGET = ROOT / "gupdari-71489" / "in" / "A_normal.glb"
# 비교용: 같은 유물의 e뮤지엄 사진에서 만든 3D (역시 깨진 상태다)
ALT = ROOT / "gupdari-shape-prior" / "trellis3d" / "경신_71489.glb"


def sample(path: Path, n: int, seed: int = 0) -> np.ndarray:
    g = trimesh.load(str(path), process=False)
    m = g.to_geometry() if hasattr(g, "to_geometry") else g
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts, np.float64)


def up_axis_of(P: np.ndarray) -> int:
    import importlib.util
    p = ROOT / "gupdari-shape-prior" / "measure_glb.py"
    spec = importlib.util.spec_from_file_location("measure_glb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.up_axis(P))


def run_one(model, device, path: Path, tag: str, n_in: int, runs: int = 1) -> dict:
    dense = sample(path, 120000)
    up = up_axis_of(dense)

    c = dense.mean(0)
    Q = dense - c
    s = float(np.linalg.norm(Q, axis=1).max())
    Q /= s

    from pointnet2_ops import pointnet2_utils
    x = torch.from_numpy(Q).float().unsqueeze(0).to(device)
    idx = pointnet2_utils.furthest_point_sample(x, n_in).long().squeeze(0).cpu().numpy()
    inp = Q[idx]

    pred = (R.infer_tta(model, inp.astype(np.float32), device, up, runs)
            if runs > 1 else R.infer(model, inp.astype(np.float32), device)
            ).astype(np.float64)

    # 입력(조밀본) 표면에서 얼마나 떨어졌나. τ 는 입력 점간격의 3배 — [2] 와 같은 기준
    t_in = cKDTree(Q)
    dnn, _ = cKDTree(inp).query(inp, k=2)
    tau = 3.0 * float(np.median(dnn[:, 1]))
    d, _ = t_in.query(pred)
    new = d > tau

    hn = (pred[:, up] - Q[:, up].min()) / max(1e-9, np.ptp(Q[:, up]))
    above = new & (pred[:, up] > Q[:, up].max())          # 아가리 위로 연장한 점

    np.savez_compressed(WORK / ("pred_71489_" + tag + ".npz"),
                        input_dense=Q.astype(np.float32),
                        input_fps=inp.astype(np.float32),
                        pred=pred.astype(np.float32),
                        new=new, up=np.int32(up))

    res = {"tag": tag, "n_new": int(new.sum()), "n_above": int(above.sum()),
           "tau": tau, "up": up,
           "new_hn_median": float(np.median(hn[new])) if new.any() else float("nan")}
    print("\n=== %s (%s) ===" % (tag, path.name))
    print("  보충점 %d / %d  (문턱 τ = 입력 점간격 3배)" % (res["n_new"], len(pred)))
    print("  아가리 위로 연장한 점 %d개" % res["n_above"])
    if new.any():
        print("  보충점의 높이 중앙 %.2f  (0=바닥 1=아가리)" % res["new_hn_median"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-input", type=int, default=2048)
    ap.add_argument("--ckpt", default="", help="파인튜닝 가중치")
    ap.add_argument("--tag", default="", help="출력 파일 접미사")
    ap.add_argument("--runs", type=int, default=1, help="축 둘레 회전 TTA")
    args = ap.parse_args()

    R.install_shims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("디바이스:", device)
    model = R.build(device, args.ckpt or None)
    WORK.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, tag in ((TARGET, "A_normal"), (ALT, "emuseum")):
        if path.is_file():
            rows.append(run_one(model, device, path, tag + args.tag, args.n_input, args.runs))
        else:
            print("[건너뜀] 없다: " + str(path))

    print("\n=== 판정 ===")
    for r in rows:
        verdict = ("거의 안 만든다 — [1] 쪽" if r["n_new"] < 100
                   else "무언가 만든다 — [2] 쪽. 어디인지 그림으로 볼 것")
        print("  %-10s 보충점 %5d · 아가리 위 %4d  → %s"
              % (r["tag"], r["n_new"], r["n_above"], verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
