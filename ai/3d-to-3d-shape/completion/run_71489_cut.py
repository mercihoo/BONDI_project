# -*- coding: utf-8 -*-
"""
run_71489_cut.py — 결손 마스크로 껍질을 잘라 **진짜 구멍**을 만들고 넣는다.

왜
  71489 를 그냥 넣었더니 보충점 20/8192 였다 (`run_71489.py`).
  모델이 못 한 게 아니라 **입력에 구멍이 없었다** — TRELLIS 가 이미 메워놨고
  `region_carried` 도 경계변 0 의 닫힌 껍질이다.
  [2] 의 `rim` 은 순증 0.961 이었는데 그 입력은 **실제로 떼어낸** 것이었다.

  **구멍을 만들어 주면 되는가.** 그게 이 실험이다.

마스크를 어디서 얻나
  v32 결과 GLB 가 `region_carried` / `region_filled` 두 노드로 나뉘어 있다
  (`NFR-ETH-003` 노드 분리). **`region_filled` 이 결손 영역 그 자체다.**
  carried 를 샘플링해서 filled 표면 가까이 있는 점을 지우면 진짜 부분 점군이 된다.

무엇과 비교하나
  정답은 없다. 대신 **v32 의 회전대칭 채움과 대조**한다 —
  AdaPoinTr 가 만든 것이 기하 방법이 만든 것과 얼마나 겹치나.
  이건 "정답 대비 정확도"가 아니라 **"두 방법이 같은 말을 하나"** 다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY run_71489_cut.py               # 문턱 자동
  $PY run_71489_cut.py --tau 0.012   # 직접 지정
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
V32 = (ROOT / "gupdari-71489" / "out"
       / "v32-cyl+biharm+n6+trellisfill+labshift+up2" / "restored_labshift.glb")


def nodes(path: Path):
    s = trimesh.load(str(path), process=False)
    if not hasattr(s, "geometry"):
        raise SystemExit("[!] Scene 이 아니다: " + str(path))
    return s.geometry


def sample(mesh, n, seed=0):
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=int, default=120000)
    ap.add_argument("--n-input", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="carried 를 깎을 거리. 0 이면 결손 면적비로 자동")
    args = ap.parse_args()

    geo = nodes(V32)
    carried, filled = geo["region_carried"], geo["region_filled"]
    print("carried F=%d · filled F=%d" % (len(carried.faces), len(filled.faces)))

    C = sample(carried, args.dense, 1)
    F = sample(filled, args.dense // 2, 2)

    # 정규화 — carried 기준 (모델 입력이 될 것이므로)
    c0 = C.mean(0)
    s0 = float(np.linalg.norm(C - c0, axis=1).max())
    Cn, Fn = (C - c0) / s0, (F - c0) / s0

    # 결손 영역 = filled 표면 가까이 있는 carried 점
    d_fill, _ = cKDTree(Fn).query(Cn)
    if args.tau > 0:
        tau = args.tau
    else:
        # filled 의 면적 비율만큼 깎는다. 그게 파이프라인이 결손이라 판정한 몫이다
        frac = filled.area / (carried.area + filled.area)
        tau = float(np.quantile(d_fill, frac))
        print("filled 면적비 %.3f → 문턱 %.4f (정규화 단위)" % (frac, tau))

    in_gap = d_fill <= tau
    part = Cn[~in_gap]
    print("carried %d점 중 **%d점(%.1f%%) 을 결손으로 보고 지웠다** → 입력 %d점"
          % (len(Cn), int(in_gap.sum()), 100 * in_gap.mean(), len(part)))

    R.install_shims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R.build(device)

    from pointnet2_ops import pointnet2_utils
    x = torch.from_numpy(part).float().unsqueeze(0).to(device)
    idx = pointnet2_utils.furthest_point_sample(x, args.n_input).long().squeeze(0).cpu().numpy()
    inp = part[idx]
    pred = R.infer(model, inp.astype(np.float32), device).astype(np.float64)

    # --- 판정 : [2] 와 같은 축
    dnn, _ = cKDTree(inp).query(inp, k=2)
    sp = float(np.median(dnn[:, 1]))
    tau_new, delta = 3 * sp, 3 * sp

    d_part, _ = cKDTree(part).query(pred)
    new = d_part > tau_new                       # 보충점

    gap_pts = Cn[in_gap]                         # 우리가 지운 자리
    t_pred = cKDTree(pred)
    dg, _ = t_pred.query(gap_pts)
    cover = float((dg <= delta).mean())
    dgf, _ = cKDTree(part).query(gap_pts)
    floor = float((dgf <= delta).mean())
    gain = (cover - floor) / max(1e-9, 1 - floor)

    dn_valid, _ = cKDTree(gap_pts).query(pred[new]) if new.any() else (np.array([]), None)
    n_valid = int((dn_valid <= delta).sum()) if new.any() else 0

    # v32 회전대칭 채움과 대조
    d_v32, _ = cKDTree(Fn).query(pred[new]) if new.any() else (np.array([]), None)
    agree = float((d_v32 <= delta).mean()) if new.any() else float("nan")

    print("\n=== 판정 ===")
    print("  보충점            %d / 8192" % int(new.sum()))
    print("  그중 결손 자리에   %d" % n_valid)
    print("  결손 커버리지      %.3f  (바닥값 %.3f · **순증 %.3f**)" % (cover, floor, gain))
    print("  v32 채움과 일치    %.3f  (보충점 중 v32 표면 δ 이내 비율)" % agree)
    print("\n  [비교] 그냥 넣었을 때(구멍 없음) 보충점 20 · 순증 해당없음")
    print("  [비교] [2] rim 순증 0.961 · bottom 0.785")

    np.savez_compressed(WORK / "pred_71489_cut.npz",
                        carried=Cn.astype(np.float32), in_gap=in_gap,
                        part=part.astype(np.float32), inp=inp.astype(np.float32),
                        pred=pred.astype(np.float32), new=new,
                        v32_fill=Fn.astype(np.float32))
    print("\n저장 → " + str(WORK / "pred_71489_cut.npz"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
