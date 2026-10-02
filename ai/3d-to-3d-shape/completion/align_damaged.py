# -*- coding: utf-8 -*-
"""
align_damaged.py — 훼손 3D 와 원본 3D 를 **같은 자에 올린다.**

왜 필요한가
  TRELLIS 는 결과를 매번 단위 정육면체로 정규화한다. 아가리를 잘라낸 쪽은
  bbox 가 달라져 **축척과 높이 원점이 어긋난다.** 그대로 쌍을 만들면
  `complete_removed` 마스크가 통째로 틀린다.

무엇을 기준으로 맞추나
  **굽다리는 양쪽 다 온전하다.** 그래서

    1. 회전축을 찾아 +Z 로 세우고, 축 위로 중심을 옮긴다
    2. 바닥(min h)을 0 으로 둔다 — 굽 바닥은 안 깨졌으니 공통 원점이다
    3. 남은 자유도는 **축척 하나**다. 훼손본의 h 범위 안에서
       `r(h)` 프로파일이 가장 잘 겹치는 s 를 찾는다

  회전체라 `r(h)` 가 형상을 거의 다 담는다. 굽다리의 굽 벌어짐이 뚜렷한
  랜드마크라 축척이 잘 잡힌다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY align_damaged.py               # 전부 맞추고 진단 CSV 를 쓴다
  $PY align_damaged.py --figure      # 프로파일 겹침 그림 (matplotlib venv 필요)
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PRIOR = ROOT / "gupdari-shape-prior"
ORIG_DIR = PRIOR / "trellis3d"
DMG_DIR = WORK / "trellis3d_damaged"
OUT = WORK / "aligned"


def load_measure_glb():
    p = PRIOR / "measure_glb.py"
    spec = importlib.util.spec_from_file_location("measure_glb", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sample(path: Path, n: int, seed: int = 0) -> np.ndarray:
    g = trimesh.load(str(path), process=False)
    m = g.to_geometry() if hasattr(g, "to_geometry") else g
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts, np.float64)


def canonical(P: np.ndarray, mg) -> np.ndarray:
    """회전축을 +Z 로 세우고 축 위에 중심을, 바닥을 0 에 둔다."""
    up = mg.up_axis(P)
    ax, c2 = mg.fit_axis(P, up)
    Q = np.empty_like(P)
    Q[:, 0] = P[:, ax[0]] - c2[0]
    Q[:, 1] = P[:, ax[1]] - c2[1]
    Q[:, 2] = P[:, up]
    if np.mean(np.hypot(Q[Q[:, 2] > np.median(Q[:, 2]), 0],
                        Q[Q[:, 2] > np.median(Q[:, 2]), 1])) < \
       np.mean(np.hypot(Q[Q[:, 2] <= np.median(Q[:, 2]), 0],
                        Q[Q[:, 2] <= np.median(Q[:, 2]), 1])):
        Q[:, 2] = -Q[:, 2]          # 넓은 쪽(바리)이 위로 오게
    Q[:, 2] -= Q[:, 2].min()        # 바닥을 0 으로
    return Q


def profile(Q: np.ndarray, hmax: float, nb: int = 48) -> np.ndarray:
    """높이 띠별 바깥 반지름. 껍질이라 상위 분위수를 쓴다."""
    r = np.hypot(Q[:, 0], Q[:, 1])
    b = np.clip((Q[:, 2] / max(hmax, 1e-9) * nb).astype(int), 0, nb - 1)
    out = np.full(nb, np.nan)
    for i in range(nb):
        m = b == i
        if m.sum() >= 20:
            out[i] = np.percentile(r[m], 90)
    return out


def waist(Q: np.ndarray, nb: int = 60, hi_frac: float = 0.70):
    """굽과 바리 사이 **잘록한 허리의 높이**. 축척을 박는 물리적 기준점이다.

    왜 필요한가 — 굽다리의 굽은 원뿔이고 **원뿔은 자기닮음**이다. 프로파일
    RMSE 만 보면 축척을 키워도 똑같이 잘 맞아 최소가 평평해진다. 실제로
    깊은 세트에서 경주_8478 이 탐색 상한 1.600 까지 밀려 올라갔다.

    허리는 길이 자체를 갖는 지형지물이고, 위를 깊게 잘라내도 남는다.
    """
    r = np.hypot(Q[:, 0], Q[:, 1])
    h = Q[:, 2]
    hmax = h.max()
    b = np.clip((h / hmax * nb).astype(int), 0, nb - 1)
    prof = np.array([np.percentile(r[b == i], 90) if (b == i).sum() >= 20 else np.nan
                     for i in range(nb)])
    lim = int(nb * hi_frac)
    seg = prof[:lim]
    ok = np.isfinite(seg)
    if ok.sum() < 6:
        return None
    idx = np.nonzero(ok)[0]
    v = np.convolve(seg[ok], np.ones(3) / 3, mode="same")
    j = int(np.argmin(v[1:-1])) + 1 if len(v) > 2 else int(np.argmin(v))
    return float((idx[j] + 0.5) / nb * hmax)


def foot_ok(Q: np.ndarray, nb: int = 60):
    """훼손본에 **굽이 제대로 섰는지** 본다. (합격여부, 이유)

    깊게 깨진 사진에서 TRELLIS 가 굽다리를 통째로 놓치는 일이 있었다.
    경주_8478 은 반지름이 아래부터 88 → 501 로 **단조증가**했다 — 굽 벌어짐이
    없는 원뿔이다. 사진에서는 위만 지웠으니 굽은 멀쩡해야 맞다.
    즉 상류(TRELLIS)가 깨진 것이라 정렬로는 못 고친다. 쌍에서 뺀다.

    굽이 없으면 허리도 없고, 허리가 없으면 축척 기준점이 사라진다 —
    실제로 그 점들이 축척 22~25 로 튀었다.
    """
    r = np.hypot(Q[:, 0], Q[:, 1])
    h = Q[:, 2]
    hmax = h.max()
    w = waist(Q, nb)
    if w is None:
        return False, "허리 없음"
    if w < 0.06 * hmax:
        return False, "허리가 바닥(%.0f%%)" % (100 * w / hmax)
    lo = r[h <= 0.04 * hmax]
    mid = r[np.abs(h - w) <= 0.04 * hmax]
    if len(lo) < 20 or len(mid) < 20:
        return False, "표본 부족"
    fb, fw = np.percentile(lo, 90), np.percentile(mid, 90)
    if fb < fw * 1.05:
        return False, "굽 벌어짐 없음(바닥/허리 %.2f)" % (fb / fw)
    return True, "굽 %.2f배 · 허리 %.0f%%" % (fb / fw, 100 * w / hmax)


def best_scale(Qd, Qo, frac=0.75, nb=48, lo=0.5, hi=1.6, steps=221, span=0.18):
    """훼손본을 s 배 했을 때 원본 프로파일과 가장 잘 겹치는 s.

    훼손본의 **아래 frac** 구간만 본다 — 위쪽은 잘려 나가 비교할 것이 없다.

    허리로 축척을 먼저 박고 그 **둘레 ±span 안에서만** 프로파일을 맞춘다.
    허리 없이 전 구간을 훑으면 원뿔 자기닮음 때문에 최소가 평평하다.
    """
    wd, wo = waist(Qd), waist(Qo)
    s0 = None
    if wd and wo and wd > 1e-6:
        s0 = wo / wd
        lo, hi = s0 * (1 - span), s0 * (1 + span)
        steps = 121
    hd = Qd[:, 2].max() * frac
    pd = profile(Qd[Qd[:, 2] <= hd], hd, nb)
    best, bs = np.inf, (s0 or 1.0)
    for s in np.linspace(lo, hi, steps):
        po = profile(Qo[Qo[:, 2] <= hd * s], hd * s, nb) / s
        m = np.isfinite(pd) & np.isfinite(po)
        if m.sum() < nb * 0.5:
            continue
        e = float(np.sqrt(np.mean((pd[m] - po[m]) ** 2)))
        if e < best:
            best, bs = e, float(s)
    return bs, best, s0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=int, default=120000)
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--max-rmse", type=float, default=0.06,
                    help="""정렬 후 프로파일이 이만큼도 안 겹치면 버린다.

    굽 검사를 통과해도 **비율 자체가 바뀐** 경우가 있다. 깊은 세트에서
    경주_887 은 훼손본 허리가 원본보다 **더 낮게**(31%% vs 36%%) 나왔다 —
    위만 지웠는데 그럴 수는 없다. TRELLIS 가 남은 부분을 다시 빚은 것이다.
    얕은 세트 RMSE 는 0.004~0.048, 깊은 세트 불량은 0.12~0.24 로 갈린다.""")
    ap.add_argument("--no-qc", action="store_true", help="굽 검사를 끄고 전부 쓴다")
    ap.add_argument("--tag", default="", help="세트 접미사. make_photo_damage 의 --tag 와 같게")
    args = ap.parse_args()

    global DMG_DIR, OUT
    if args.tag:
        DMG_DIR = WORK / ("trellis3d_damaged" + args.tag)
        OUT = WORK / ("aligned" + args.tag)

    mg = load_measure_glb()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, dropped = [], []
    for dmg in sorted(DMG_DIR.glob("*.glb")):
        orig = ORIG_DIR / dmg.name
        if not orig.is_file():
            print("  [건너뜀] 원본 없음: " + dmg.name)
            continue
        Pd = canonical(sample(dmg, args.dense, 1), mg)
        Po = canonical(sample(orig, args.dense, 1), mg)
        ok, why = foot_ok(Pd)
        if not ok and not args.no_qc:
            print("  [탈락] %-14s %s — TRELLIS 가 굽을 못 세웠다" % (dmg.stem, why))
            dropped.append({"slug": dmg.stem, "reason": why})
            continue
        s, err, s0 = best_scale(Pd, Po)
        if err > args.max_rmse and not args.no_qc:
            print("  [탈락] %-14s 정합 RMSE %.4f > %.3f — 비율이 달라졌다"
                  % (dmg.stem, err, args.max_rmse))
            dropped.append({"slug": dmg.stem, "reason": "정합 RMSE %.4f" % err})
            continue
        Pd2 = Pd * s                                   # 훼손본을 원본 자에 올린다

        h_ratio = Pd2[:, 2].max() / Po[:, 2].max()
        np.savez_compressed(OUT / (dmg.stem + ".npz"),
                            partial=Pd2.astype(np.float32),
                            complete=Po.astype(np.float32),
                            scale=np.float64(s))
        rows.append({"slug": dmg.stem, "scale": round(s, 4),
                     "waist_scale": round(s0, 4) if s0 else "",
                     "profile_rmse": round(err, 5),
                     "height_ratio": round(float(h_ratio), 3)})
        print("  %-14s 축척 %.3f (허리 %s) · RMSE %.4f · 남은 높이 %.0f%%"
              % (dmg.stem, s, ("%.3f" % s0) if s0 else "없음", err, 100 * h_ratio))

    if not rows:
        print("[!] 맞출 것이 없다. TRELLIS 가 아직 안 끝났을 수 있다.")
        return 1
    with (WORK / ("align%s.csv" % args.tag)).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n%d점 정렬 → %s" % (len(rows), OUT))
    if dropped:
        print(chr(10) + "[탈락 %d점] 상류 TRELLIS 가 굽다리를 재현 못 했다 — 쌍에서 뺀다" % len(dropped))
        for d in dropped:
            print("   %-14s %s" % (d["slug"], d["reason"]))
    print("남은 높이 중앙 %.0f%%  (사진에서 위 35%% 를 지웠으니 65%% 근처여야 맞다)"
          % (100 * np.median([r["height_ratio"] for r in rows])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
