# -*- coding: utf-8 -*-
"""
scan_dataset.py — AI-Hub 문화유산 3D 데이터(211-2)에서 완형 토기 후보를 고른다.

왜 필요한가
  이 데이터셋은 유적 실측 스캔이라 EA(토기) 695점이 전부 완형이 아니다.
  기여자 2026-08-28 의 대상이던 RR_07_01_EA_028 자체가 하부 전체 결실 항아리였다.
  라벨 JSON 에는 완형/결손 정보가 없다 (bbox + 분류 코드뿐). 그래서 재서 골라야 한다.

  08-28 교훈 그대로다 — "모양만 보고 판단하지 말고 테두리 규칙성을 측정할 것".
  거기서 나온 실측 기준을 그대로 쓴다:
      마감된 구연부  반경편차 1.7mm · 높이편차 1.4mm
      파단면        반경편차 37.1mm  (구연부의 22배)

무엇을 재나 (유물 1점당)
  1. 회전축        PCA 3축 중 단면이 가장 둥근 것
  2. 둥근정도 cv    회전체가 아니면 이 파이프라인 대상이 아니다
  3. 위/아래 테두리 반경편차 · 높이편차 · 각도 커버리지
  4. 격자 빈칸률    (h, theta) 격자의 빈칸 비율 — 옆구리 구멍

출력  work/scan_EA.csv   유물 1행

사용
  PY=".../venv/Scripts/python.exe"
  $PY scan_dataset.py --limit 20          # 맛보기
  $PY scan_dataset.py                     # EA 전체
  $PY scan_dataset.py --label PA          # 다른 분류
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(r"C:\Users\<USER>\Desktop\211-2.문화유산 유적 3D 데이터\01-1.정식개방데이터")
HERE = Path(__file__).resolve().parent
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))

# 08-28 실측 기준을 m 로 환산. 이 데이터의 좌표 단위는 m 다 —
# RR_07_01_EA_028 의 bbox 가 0.65 x 0.62 x 0.35 로 기여자이 적은 치수와 일치한다.
RIM_REGULAR_M = 0.005   # 반경편차 5mm 이하 = 마감된 테두리 (실측 1.7mm 의 3배 여유)
RIM_BROKEN_M = 0.020    # 20mm 이상 = 파단면 (실측 37.1mm)


# ---------------------------------------------------------------- 입출력

def find_files(label, splits):
    """라벨링데이터의 CSV 를 쓴다. 원천데이터는 .las 라 laspy 가 필요한데
    CSV 가 같은 점군(X,Y,Z,R,G,B)을 그대로 담고 있어 의존을 하나 줄인다."""
    out = []
    for sp in splits:
        d = DATA_ROOT / sp / "02.라벨링데이터"
        if not d.is_dir():
            continue
        out += sorted(d.glob("*_" + label + "_*.csv"))
    return out


def load_xyz(csv_path, max_pts, seed=0):
    df = pd.read_csv(csv_path, usecols=[0, 1, 2])
    P = df.to_numpy(np.float64)
    P = P[np.isfinite(P).all(1)]
    if max_pts and len(P) > max_pts:
        idx = np.random.default_rng(seed).choice(len(P), max_pts, replace=False)
        P = P[idx]
    return P


# ---------------------------------------------------------------- 기하

def frame_for(axis):
    """axis 에 직교하는 정규직교 두 축."""
    a = axis / np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


def cylindrical(P, c, axis):
    """(h, theta, r) 로 편다."""
    e1, e2 = frame_for(axis)
    Q = P - c
    h = Q @ axis
    x, y = Q @ e1, Q @ e2
    return h, np.arctan2(y, x), np.hypot(x, y)


def outer_radius_by_angle(th, r, mask, nt):
    """각도칸마다 바깥반경(최대). 껍질이라 안쪽 면도 섞여 들어오므로 max 를 쓴다."""
    out = np.full(nt, np.nan)
    tb = np.clip(((th[mask] + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
    rr = r[mask]
    order = np.argsort(tb)
    tb, rr = tb[order], rr[order]
    edges = np.searchsorted(tb, np.arange(nt + 1))
    for j in range(nt):
        a, b = edges[j], edges[j + 1]
        if b > a:
            out[j] = rr[a:b].max()
    return out


def roundness_cv(P, c, axis, nh=16, nt=36):
    """높이 띠마다 '각도별 바깥반경'의 변동계수를 재고 중앙값을 돌려준다.
    회전체면 작다. 축이 틀렸으면 커진다 — 축 고르기에 그대로 쓴다."""
    h, th, r = cylindrical(P, c, axis)
    lo, hi = np.percentile(h, [2, 98])
    if hi - lo < 1e-9:
        return np.inf
    hb = np.clip(((h - lo) / (hi - lo) * nh).astype(int), 0, nh - 1)
    cvs = []
    for i in range(nh):
        m = hb == i
        if m.sum() < nt * 4:
            continue
        outer = outer_radius_by_angle(th, r, m, nt)
        ok = np.isfinite(outer)
        if ok.sum() < nt * 0.7 or outer[ok].mean() < 1e-9:
            continue
        cvs.append(outer[ok].std() / outer[ok].mean())
    return float(np.median(cvs)) if cvs else np.inf


def pick_axis(P):
    """PCA 세 축 중 단면이 가장 둥근 것을 회전축으로 본다."""
    c = P.mean(0)
    C = P - c
    _, V = np.linalg.eigh(C.T @ C / len(C))
    best, best_cv = None, np.inf
    for k in range(3):
        a = V[:, k] / np.linalg.norm(V[:, k])
        cv = roundness_cv(P, c, a)
        if cv < best_cv:
            best, best_cv = a, cv
    if best is None:
        best = V[:, 2] / np.linalg.norm(V[:, 2])
    # PCA 축은 부호가 임의다. 그대로 두면 위/아래가 유물마다 뒤집힌다.
    # 이 데이터는 지상 스캔이라 세계좌표 +Z 가 위다 — 거기에 맞춘다.
    if best[2] < 0:
        best = -best
    return c, best, best_cv


def bottom_closed_ratio(h, th, r, nr=6, nt=24, band_frac=0.08):
    """아래쪽 끝이 '막혀 있나'.

    완형 토기의 바닥은 원반이라 축 근처(r≈0)까지 점이 찬다.
    하부가 통째로 결실되면 테두리만 남아 고리 모양이 된다 — 08-28 의 EA_028 이 그랬다.
    안쪽 원반을 극좌표 칸으로 나눠 채워진 비율을 돌려준다. 1 에 가까우면 막힌 바닥.
    """
    lo, hi = h.min(), h.max()
    span = hi - lo
    if span < 1e-9:
        return np.nan
    sel = h <= lo + span * band_frac
    if sel.sum() < 50:
        return np.nan
    rs, ts = r[sel], th[sel]
    rmax = np.percentile(rs, 98)
    if rmax < 1e-9:
        return np.nan
    inner = rs <= rmax * 0.6          # 바깥 테두리는 빼고 안쪽만 본다
    if inner.sum() == 0:
        return 0.0
    rb = np.clip((rs[inner] / (rmax * 0.6) * nr).astype(int), 0, nr - 1)
    tb = np.clip(((ts[inner] + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
    occ = np.zeros((nr, nt), bool)
    occ[rb, tb] = True
    return float(occ.mean())


def rim_stats(h, th, r, top, nt=72, band_frac=0.03):
    """한쪽 끝 테두리의 규칙성.

    반환  (반경편차, 높이편차, 각도커버리지, 평균반경)
      반경편차      각도별 테두리 반경의 표준편차  — 08-28 의 주지표
      높이편차      각도별 테두리 높이의 표준편차
      각도커버리지   테두리가 있는 각도칸 비율. 낮으면 한쪽만 남은 것
    """
    lo, hi = h.min(), h.max()
    span = hi - lo
    if span < 1e-9:
        return np.nan, np.nan, 0.0, np.nan
    sel = (h >= hi - span * band_frac) if top else (h <= lo + span * band_frac)
    if sel.sum() < nt:
        return np.nan, np.nan, 0.0, np.nan

    tb = np.clip(((th[sel] + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
    hs, rs = h[sel], r[sel]
    rim_h = np.full(nt, np.nan)
    rim_r = np.full(nt, np.nan)
    for j in range(nt):
        m = tb == j
        if m.any():
            rim_h[j] = hs[m].max() if top else hs[m].min()
            rim_r[j] = rs[m].max()
    ok = np.isfinite(rim_h)
    cover = float(ok.mean())
    if ok.sum() < 4:
        return np.nan, np.nan, cover, np.nan
    return (float(np.std(rim_r[ok])), float(np.std(rim_h[ok])),
            cover, float(np.mean(rim_r[ok])))


def grid_gap_ratio(h, th, r, nh=48, nt=72):
    """(h, theta) 격자 빈칸 비율. 굽다리-71489 의 결손 판정과 같은 표현이다.
    투창(의도된 구멍)도 같이 잡히므로 '결손률'이 아니라 '빈칸률'로 읽을 것."""
    lo, hi = np.percentile(h, [1, 99])
    if hi - lo < 1e-9:
        return np.nan
    m = (h >= lo) & (h <= hi)
    hb = np.clip(((h[m] - lo) / (hi - lo) * nh).astype(int), 0, nh - 1)
    tb = np.clip(((th[m] + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)
    occ = np.zeros((nh, nt), bool)
    occ[hb, tb] = True
    return float(1.0 - occ.mean())


# ---------------------------------------------------------------- 1점 처리

def measure(csv_path, max_pts):
    P = load_xyz(csv_path, max_pts)
    rec = {"id": csv_path.stem, "n_points": int(len(P))}
    if len(P) < 2000:
        rec["note"] = "점이 너무 적다"
        return rec

    c, axis, cv = pick_axis(P)
    h, th, r = cylindrical(P, c, axis)
    ext = P.max(0) - P.min(0)

    rec.update(
        size_x=float(ext[0]), size_y=float(ext[1]), size_z=float(ext[2]),
        height_m=float(h.max() - h.min()),
        radius_m=float(np.percentile(r, 98)),
        round_cv=float(cv),
        gap_ratio=grid_gap_ratio(h, th, r),
        axis_z=float(abs(axis[2])),          # 1 에 가까우면 수직축 = 세워진 상태
        bot_closed=bottom_closed_ratio(h, th, r),
    )
    for side, is_top in (("top", True), ("bot", False)):
        sr, sh, cov, rr = rim_stats(h, th, r, is_top)
        rec[side + "_r_std"] = sr
        rec[side + "_h_std"] = sh
        rec[side + "_cover"] = cov
        rec[side + "_r_mean"] = rr
    return rec


def classify(rec):
    """완형후보 / 파단의심 / 회전체아님 / 보류 / 판정불가.

    완형의 조건 두 가지를 따로 본다.
      위  마감된 구연부      = 반경편차가 작고 둘레 전체에 있다
      아래 막힌 바닥         = 축 근처까지 점이 찬다 (고리면 하부 결실)
    """
    def num(key, default=np.nan):
        v = rec.get(key, default)
        return v if v is not None else np.nan

    cv = num("round_cv", np.inf)
    if not np.isfinite(cv) or cv > 0.45:
        return "회전체아님"

    tr, cover = num("top_r_std"), num("top_cover", 0.0)
    if not np.isfinite(tr):
        return "판정불가"

    # 위 테두리가 파단이면 그것만으로 탈락 (구연부 결손 = 08-28 사례)
    if tr >= RIM_BROKEN_M or cover < 0.85:
        return "파단의심"

    # 아래는 '막힌 바닥'이 정상이다. 고리로 열려 있으면 하부 결실을 의심한다.
    closed = num("bot_closed")
    if np.isfinite(closed) and closed < 0.35:
        return "파단의심"

    if tr <= RIM_REGULAR_M:
        return "완형후보"
    return "보류"


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="EA", help="분류코드 (EA 토기 · PA · ST · VE ...)")
    ap.add_argument("--splits", default="Training,Validation")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N점만")
    ap.add_argument("--max-pts", type=int, default=120000, help="유물당 사용할 점 수")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(","))
    files = find_files(args.label, splits)
    if not files:
        print("[!] " + args.label + " 파일을 못 찾았다. DATA_ROOT 확인: " + str(DATA_ROOT))
        return 1
    if args.limit:
        files = files[: args.limit]

    WORK.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else WORK / ("scan_" + args.label + ".csv")

    print("대상 %d점 · 유물당 최대 %s점 · 출력 %s"
          % (len(files), format(args.max_pts, ","), out_path))
    rows, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        try:
            rec = measure(f, args.max_pts)
            rec["verdict"] = classify(rec)
        except Exception as e:      # 한 점 때문에 전체가 죽지 않게
            rec = {"id": f.stem, "verdict": "오류",
                   "note": type(e).__name__ + ": " + str(e)}
        rows.append(rec)
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            print("  %d/%d  %6.1fs  (%.2fs/점)" % (i, len(files), el, el / i), flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== 판정 분포 ===")
    print(df["verdict"].value_counts().to_string())

    cand = df[df["verdict"] == "완형후보"].sort_values("top_r_std")
    print("\n=== 완형후보 %d점 (상위 15) ===" % len(cand))
    cols = [c for c in ["id", "height_m", "radius_m", "round_cv", "axis_z",
                        "top_r_std", "top_h_std", "top_cover", "bot_closed",
                        "gap_ratio"]
            if c in cand.columns]
    if len(cand):
        print(cand[cols].head(15).to_string(
            index=False, float_format=lambda v: "%.4f" % v))
    print("\n저장 → " + str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
