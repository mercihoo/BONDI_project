# -*- coding: utf-8 -*-
"""
make_pairs_photo.py — 정렬된 **사진 훼손** 점군을 학습 쌍으로 바꾼다.

    work/aligned*/*.npz  ──▶  work/pairs_photo/*.npz   (make_pairs 와 같은 형식)

합성 쌍과 무엇이 다른가
  `make_pairs` 는 **완형 3D 를 3D 에서 깎는다.** 그래서 학습 입력은
  "TRELLIS 가 온전한 사진에서 만든 것을 잘라낸 것"이다.
  71489 는 **깨진 사진**에서 TRELLIS 가 만든 것이다. 상류 경로가 다르다.
  §5.6.2 에서 파단면·손상 복잡도·표면 잡음을 다 기각하고 **이것만 남았다.**

  여기 쌍은 사진을 먼저 깨고 TRELLIS 를 태운 것이라 **71489 와 같은 경로**다.

결손 마스크를 어떻게 정하나
  합성 쌍은 깎을 때 마스크를 같이 저장했다. 여기는 손상이 **이미지 단계**에서
  일어나 3D 대응이 없다. 그래서 거리로 정한다 —
  complete 의 점에서 partial 최근접까지가 `tau` 보다 멀면 결실이다.
  `tau` 는 partial 자신의 최근접 이웃 간격 중앙값의 `--tau-k` 배로 잡는다.
  (고정 상수를 쓰면 유물 크기에 따라 뜻이 달라진다.)

왜 회전으로 불리나
  정렬을 통과한 쌍이 11개뿐이다(합성은 900). 이 유물은 회전체라 **축 둘레
  회전은 강체 변환**이고, partial·complete 를 같이 돌리면 쌍이 그대로 유효하다.
  추론 때 쓴 TTA 와 같은 논리를 학습 쪽에 거는 것이다.
  ragged 한 아가리 경계는 회전해도 모양이 유지되므로 정보가 늘지는 않지만,
  모델이 각도에 덜 민감해진다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY make_pairs_photo.py --aug 40
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
OUT = WORK / "pairs_photo"

N_PART, N_COMP = 2048, 8192


def stable_seed(*parts) -> int:
    """`hash()` 는 프로세스마다 달라진다 — 재현되는 씨앗을 쓴다."""
    h = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def fps(P: np.ndarray, n: int, seed: int) -> np.ndarray:
    """최원점 표집. 균일 무작위보다 얇은 부분(굽·아가리)을 덜 놓친다."""
    if len(P) <= n:
        return np.arange(len(P))
    rng = np.random.default_rng(seed)
    # 12만점 전체에 FPS 를 돌리면 느리다. 먼저 넉넉히 솎고 그 안에서 고른다.
    cand = rng.choice(len(P), min(len(P), n * 6), replace=False)
    Q = P[cand]
    idx = np.empty(n, np.int64)
    d = np.full(len(Q), np.inf)
    cur = int(rng.integers(len(Q)))
    for i in range(n):
        idx[i] = cur
        d = np.minimum(d, np.sum((Q - Q[cur]) ** 2, axis=1))
        cur = int(np.argmax(d))
    return cand[idx]


def rot_z(P: np.ndarray, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    Q = P.copy()
    Q[:, 0] = P[:, 0] * c - P[:, 1] * s
    Q[:, 1] = P[:, 0] * s + P[:, 1] * c
    return Q


def build_one(src: Path, tag: str, aug: int, tau_k: float, rows: list) -> int:
    d = np.load(src, allow_pickle=True)
    Pd = np.asarray(d["partial"], np.float64)
    Po = np.asarray(d["complete"], np.float64)

    # tau — partial 자신의 점 간격을 자로 쓴다
    tp = cKDTree(Pd)
    nn, _ = tp.query(Pd[:: max(1, len(Pd) // 4000)], k=2)
    tau = float(np.median(nn[:, 1])) * tau_k

    made = 0
    for a in range(aug):
        seed = stable_seed(src.stem, tag, a)
        ang = 2 * np.pi * a / aug
        ip = fps(Pd, N_PART, seed)
        ic = fps(Po, N_COMP, seed + 1)
        p, c = rot_z(Pd[ip], ang), rot_z(Po[ic], ang)

        # 결손 마스크 — 회전 전 좌표로 재도 같다. 한 번만 재서 쓴다.
        dist, _ = tp.query(Po[ic])
        removed = dist > tau
        frac = float(removed.mean())
        if frac < 0.05 or frac > 0.80:
            if a == 0:
                print("  [건너뜀] %-14s 결손률 %.2f 가 범위 밖" % (src.stem, frac))
            return 0

        cp = p.mean(0)
        sp = float(np.linalg.norm(p - cp, axis=1).max()) or 1.0
        np.savez_compressed(
            OUT / ("%s%s__r%02d.npz" % (src.stem, tag, a)),
            partial=p.astype(np.float32), complete=c.astype(np.float32),
            complete_removed=removed, center=cp, scale=np.float64(sp),
            up_axis=np.int32(2), pattern=np.str_("photo" + tag))
        made += 1
        if a == 0:
            rows.append({"slug": src.stem, "set": tag or "_shallow",
                         "removed_frac": round(frac, 3),
                         "tau": round(tau, 5), "aug": aug})
            print("  %-14s%-8s 결손률 %.2f · tau %.4f · %d각도"
                  % (src.stem, tag or "_shallow", frac, tau, aug))
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=int, default=40, help="유물당 회전 각도 수")
    ap.add_argument("--tau-k", type=float, default=3.0,
                    help="결손 판정 거리 = partial 점 간격 중앙값 × 이 값")
    ap.add_argument("--sets", default="_shallow,_deep",
                    help="work/aligned* 중 쓸 것. _shallow 는 접미사 없는 폴더")
    ap.add_argument("--out", default=str(WORK / "pairs_photo"))
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.npz"):
        f.unlink()

    total, rows = 0, []
    for tag in args.sets.split(","):
        d = WORK / ("aligned" if tag == "_shallow" else "aligned" + tag)
        if not d.is_dir():
            print("[건너뜀] 없는 폴더: " + str(d))
            continue
        files = sorted(d.glob("*.npz"))
        print("\n%s — %d점" % (d.name, len(files)))
        for f in files:
            total += build_one(f, "" if tag == "_shallow" else tag,
                               args.aug, args.tau_k, rows)

    if not rows:
        print("[!] 만든 쌍이 없다.")
        return 1
    with (WORK / "pairs_photo_manifest.csv").open("w", encoding="utf-8-sig",
                                                  newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("\n쌍 %d개 (형상 %d) → %s" % (total, len(rows), OUT))
    print("결손률 중앙 %.2f" % np.median([r["removed_frac"] for r in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
