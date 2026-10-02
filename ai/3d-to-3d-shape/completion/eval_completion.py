# -*- coding: utf-8 -*-
"""
eval_completion.py — 완성 결과를 08-28 과 같은 축으로 잰다.

주지표는 하나다.
    **결실 영역을 실제로 메웠는가.**
기여자 08-28 의 PoinTr 결과가 *"결실 하부로 나간 출력점 0점"* 이었다. 그 0 과 비교한다.

무엇을 재나
  보충점        출력점 중 입력 partial 에서 τ 넘게 떨어진 것 = 모델이 새로 만든 점
  유효 보충점    그 보충점 중 **결실 영역 정답**에 δ 이내로 붙은 것
  결실 커버리지   결실 영역 정답점 중 출력이 δ 이내에 있는 비율  ← 가장 읽기 쉬운 값
  잔존부 오차     출력 ↔ partial 최근접 거리 중앙값        (08-28 기준선 3.5mm)
  CD             출력 ↔ complete 양방향 챔퍼
  점간격          출력의 최근접이웃 거리 중앙값              (08-28 기준선 ~12mm)

결실 영역은 **`make_pairs.py` 가 저장해 둔 `complete_removed` 마스크**다.
거리로 역추정하면 partial 이 성긴 곳까지 결손으로 잡혀 판정이 오염된다.

단위
  npz 좌표는 단위구 정규화다. `meta/artifacts.csv` 의 실제 높이로 **mm 로 환산**해
  08-28 의 숫자(3.5mm · 12mm)와 바로 대조되게 한다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY eval_completion.py --self-test            # 모델 없이 판정기 자체를 검증
  $PY eval_completion.py --pred out/            # out/<쌍이름>.npy 를 읽어 채점
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PAIRS = WORK / "pairs"
META = ROOT / "gupdari-shape-prior" / "meta" / "artifacts.csv"

# G-A 게이트 (README §5)
#
# ① 은 처음에 **유효 보충점 개수 1,000** 으로 잡았다. 틀린 기준이었다.
#    완벽한 모델(pred=complete)이 결실부에 2,315점을 넣는다는 자체 테스트에서
#    그 43% 를 통과선으로 삼았는데, 모델의 출력 예산은 8,192점으로 고정이고
#    그걸 유물 전체에 퍼뜨린다. 결실부를 **성기게라도 다 덮으면** 개수는 적어도
#    형상은 복원된 것이다. 실제로 그런 결과가 나왔다 — 보충점 777개로 78.5% 를 덮었다.
#    개수가 아니라 **커버리지 순증**이 답해야 할 질문에 맞는 통계다.
#    순증은 자체 테스트로 이미 0(안 메움)~1(완벽) 에 보정돼 있다.
GATE_COVER = 0.50       # bottom 쌍 커버리지 순증이 이 이상이어야 ①
GATE_NOADD = 200        # none 쌍에서 보충점이 이 미만이어야 ②
GATE_FILL_REF = 1000    # 참고용으로만 같이 찍는다 (옛 기준)


# ------------------------------------------------------------ 단위

def mm_per_unit(comp: np.ndarray, up: int, height_cm: float | None) -> float | None:
    """정규화 좌표 1 이 몇 mm 인가. 유물 실제 높이를 자로 쓴다."""
    if not height_cm:
        return None
    span = float(comp[:, up].max() - comp[:, up].min())
    if span < 1e-9:
        return None
    return height_cm * 10.0 / span


def load_heights() -> dict:
    if not META.is_file():
        return {}
    out = {}
    with META.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            s = re.sub(r"\s+", "_", unicodedata.normalize("NFC", r["소장품번호"]).strip())
            try:
                out[s] = float(r["height_cm"])
            except (TypeError, ValueError):
                pass
    return out


# ------------------------------------------------------------ 지표

def nn_spacing(P: np.ndarray) -> float:
    """점군 자체의 최근접이웃 거리 중앙값. τ 를 데이터에서 정하는 데 쓴다."""
    d, _ = cKDTree(P).query(P, k=2)
    return float(np.median(d[:, 1]))


def evaluate(pred: np.ndarray, d, tau_k: float, delta_k: float,
             delta_src: str = "partial") -> dict:
    part = np.asarray(d["partial"], np.float64)
    comp = np.asarray(d["complete"], np.float64)
    removed = np.asarray(d["complete_removed"], bool)
    up = int(d["up_axis"])

    # τ·δ 를 점간격에서 정한다. 절대값을 박으면 유물 크기마다 뜻이 달라진다.
    #
    # **입력에 잡음이 섞이면 `partial` 기준이 무너진다.** 잡음이 점간격을 키워
    # δ 가 헐거워지고 점수가 저절로 오른다 — 실측으로 확인했다:
    #     잡음 없음 δ=0.0739 → σ=0.015 에서 δ=0.0898 (+22%)
    # 그때는 잡음이 없는 `complete` 에서 뽑아야 공정하다 (`--delta-src complete`).
    sp = nn_spacing(comp if delta_src == "complete" else part)
    tau, delta = tau_k * sp, delta_k * sp

    t_part = cKDTree(part)
    d_pred_part, _ = t_part.query(pred)
    is_new = d_pred_part > tau                     # 보충점

    res = {
        "n_pred": len(pred),
        "n_new": int(is_new.sum()),
        "tau": tau, "delta": delta,
        "resid_err": float(np.median(d_pred_part[~is_new])) if (~is_new).any() else np.nan,
        "pred_spacing": nn_spacing(pred),
    }

    # 결실 영역과의 관계
    gt_miss = comp[removed]
    if len(gt_miss):
        t_miss = cKDTree(gt_miss)
        if is_new.any():
            dn, _ = t_miss.query(pred[is_new])
            res["n_new_valid"] = int((dn <= delta).sum())    # 유효 보충점
        else:
            res["n_new_valid"] = 0
        dm, _ = cKDTree(pred).query(gt_miss)
        cover = float((dm <= delta).mean())                  # 결실 커버리지
        # **바닥값(floor).** 아무것도 안 메운 모델(pred=partial)도 0 이 아니다 —
        # 결손 띠가 얇으면 정답점이 남은 표면 δ 안에 들어간다.
        # 자체 테스트 실측: rim 0.345 · side 0.367 · bottom 0.081.
        # 빼지 않으면 "아무것도 안 하고 35% 메웠다" 가 된다.
        df, _ = t_part.query(gt_miss)
        floor = float((df <= delta).mean())
        res["miss_cover"] = cover
        res["cover_floor"] = floor
        res["cover_gain"] = (cover - floor) / max(1e-9, 1.0 - floor)  # 0~1 로 정규화
        res["n_gt_miss"] = int(len(gt_miss))

        # --- **정밀도.** 커버리지만 보면 점을 사방에 뿌릴수록 올라간다.
        #
        # 실제로 그 실패가 있었다 — 경주_고적_13472 가 파단선을 둘레로 돌려
        # **없는 넓은 챙**을 만들었는데, 그 유물이 점을 가장 많이 만든 축이었다
        # (결과.md §5.7). 지금은 메시화의 면 퍼짐 게이트가 막지만
        # **모델 평가에는 안 들어가 있었다.**
        #
        #   재현율(recall)   = 정답점 중 예측이 δ 안에 있는 비율   ← miss_cover
        #   정밀도(precision)= 보충점 중 정답 δ 안에 있는 비율     ← 이것
        #
        # `n_new_valid` 를 이미 세고 있었는데 **비율로 안 쓰고 있었다.**
        res["precision"] = (res["n_new_valid"] / res["n_new"]) if res["n_new"] else np.nan
        # F-Score 둘. **바닥값을 빼느냐 마느냐**로 갈린다.
        #
        #   fscore     생 커버리지로. 점군 완성 문헌(AdaPoinTr 논문)과 같은 꼴이라
        #              **밖과 비교할 때** 쓴다. 다만 바닥값이 높은 패턴에서는 후하다 —
        #              rim 의 재현율 0.998 안에 공짜 0.347 이 섞여 있다.
        #   fscore_gain 순증(바닥값 보정)으로. **버전끼리 비교할 때** 이쪽이 맞다.
        pr, rc = res["precision"], cover
        res["fscore"] = (2 * pr * rc / (pr + rc)) if (np.isfinite(pr) and pr + rc > 0) else np.nan
        rg = res["cover_gain"]
        res["fscore_gain"] = (2 * pr * rg / (pr + rg))             if (np.isfinite(pr) and np.isfinite(rg) and pr + rg > 0) else np.nan
    else:
        res.update(n_new_valid=0, miss_cover=np.nan, cover_floor=np.nan,
                   cover_gain=np.nan, n_gt_miss=0)

    # 정답 대비 CD (양방향 평균 거리)
    dpc, _ = cKDTree(comp).query(pred)
    dcp, _ = cKDTree(pred).query(comp)
    res["cd"] = float(dpc.mean() + dcp.mean())
    res["up"] = up
    return res


# ------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="", help="예측 폴더. <쌍이름>.npy 또는 .npz")
    ap.add_argument("--self-test", action="store_true",
                    help="모델 없이 판정기를 검증한다 (§아래)")
    ap.add_argument("--tau-k", type=float, default=3.0,
                    help="보충점 문턱 = tau_k × 입력 점간격")
    ap.add_argument("--delta-k", type=float, default=3.0,
                    help="정답 근접 문턱 = delta_k × 입력 점간격")
    ap.add_argument("--pairs", default=str(PAIRS), help="쌍 폴더")
    ap.add_argument("--delta-src", choices=["partial", "complete"],
                    default="partial",
                    help="문턱을 어느 점군의 간격에서 뽑나. 입력에 잡음이 있으면 complete")
    ap.add_argument("--out", default=str(WORK / "eval.csv"))
    args = ap.parse_args()

    pairs = sorted(Path(args.pairs).glob("*.npz"))
    if not pairs:
        print("[!] 쌍이 없다. make_pairs.py 를 먼저 돌릴 것: " + str(PAIRS))
        return 1

    heights = load_heights()
    rows, modes = [], []

    if args.self_test:
        # 판정기가 맞는지 정답을 아는 두 극단으로 확인한다.
        #   pred = partial   → 아무것도 안 메운 모델. 커버리지 0 이어야 한다
        #   pred = complete  → 완벽히 메운 모델. 커버리지 1 이어야 한다
        # **셋으로 잰다.**
        #   partial  아무것도 안 메운 모델  → 커버리지 순증 0 이어야 한다
        #   complete 완벽히 메운 모델      → 순증 1 · 정밀도 1 이어야 한다
        #   random   **아무렇게나 뿌린 모델** → 정밀도의 **바닥값**을 준다
        #
        # 셋째가 없으면 "정밀도 0.9" 가 잘한 건지 알 수 없다. 커버리지는 floor 를
        # 재서 빼는데 정밀도에는 그 기준선이 없었다.
        modes = ["partial", "complete", "random"]
    elif args.pred:
        modes = ["pred"]
    else:
        print("[!] --pred 나 --self-test 중 하나가 필요하다.")
        return 1

    for mode in modes:
        for f in pairs:
            d = np.load(f, allow_pickle=True)
            if mode == "partial":
                pred = np.asarray(d["partial"], np.float64)
            elif mode == "complete":
                pred = np.asarray(d["complete"], np.float64)
            elif mode == "random":
                # 입력의 바운딩 상자 안에 고르게 뿌린다. 출력 점 수는 같게.
                base = np.asarray(d["partial"], np.float64)
                comp_ = np.asarray(d["complete"], np.float64)
                rs = np.random.RandomState(abs(hash(f.stem)) % (2 ** 31))
                lo_, hi_ = base.min(0), base.max(0)
                pred = np.vstack([base, lo_ + rs.rand(len(comp_) - len(base), 3)
                                  * (hi_ - lo_)])
            else:
                p = Path(args.pred)
                cand = [p / (f.stem + ".npy"), p / (f.stem + ".npz")]
                hit = next((c for c in cand if c.is_file()), None)
                if hit is None:
                    continue
                arr = np.load(hit)
                pred = np.asarray(arr if hit.suffix == ".npy" else arr["pred"], np.float64)

            r = evaluate(pred, d, args.tau_k, args.delta_k, args.delta_src)
            src = str(d["pattern"]), f.stem.split("__")[0]
            k = mm_per_unit(np.asarray(d["complete"], np.float64), r["up"],
                            heights.get(src[1]))
            rows.append({
                "mode": mode, "name": f.stem, "source": src[1], "pattern": src[0],
                "n_pred": r["n_pred"], "n_new": r["n_new"],
                "n_new_valid": r["n_new_valid"], "n_gt_miss": r["n_gt_miss"],
                "precision": r.get("precision", float("nan")),
                "fscore": r.get("fscore", float("nan")),
                "fscore_gain": r.get("fscore_gain", float("nan")),
                "miss_cover": round(r["miss_cover"], 4) if np.isfinite(r["miss_cover"]) else "",
                "cover_floor": round(r["cover_floor"], 4) if np.isfinite(r["cover_floor"]) else "",
                "cover_gain": round(r["cover_gain"], 4) if np.isfinite(r["cover_gain"]) else "",
                "resid_err_mm": round(r["resid_err"] * k, 3) if k and np.isfinite(r["resid_err"]) else "",
                "spacing_mm": round(r["pred_spacing"] * k, 3) if k else "",
                "cd": round(r["cd"], 5),
            })

    if not rows:
        print("[!] 채점한 것이 없다. --pred 폴더에 <쌍이름>.npy 가 있는지 확인할 것.")
        return 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- 요약
    def agg(sel, key):
        v = [r[key] for r in rows if sel(r) and r[key] != ""]
        return float(np.mean(v)) if v else float("nan")

    for mode in modes:
        print("\n=== %s ===" % mode)
        print("%-8s %5s %9s %11s %9s %8s %9s | %8s %6s %6s"
              % ("패턴", "쌍", "보충점", "유효보충점", "커버리지", "바닥값", "순증",
                 "정밀도", "F생", "F순증"))
        # 패턴 목록을 박아 두면 새 패턴이 표에서 빠진다. 실제로 `slant`(v3 의 핵심)가
        # 안 보이고 있었다. 데이터에서 딴다.
        PAT0 = ["none", "bottom", "rim", "side", "ragged", "slant"]
        pats = [q for q in PAT0 if any(r["pattern"] == q for r in rows)]
        pats += sorted({r["pattern"] for r in rows} - set(PAT0))
        for pat in pats:
            sel = lambda r, p=pat, m=mode: r["mode"] == m and r["pattern"] == p
            n = sum(1 for r in rows if sel(r))
            if not n:
                continue
            print("%-8s %5d %9.0f %11.0f %9.3f %8.3f %9.3f | %8.3f %6.3f %6.3f"
                  % (pat, n, agg(sel, "n_new"), agg(sel, "n_new_valid"),
                     agg(sel, "miss_cover"), agg(sel, "cover_floor"),
                     agg(sel, "cover_gain"),
                     agg(sel, "precision"), agg(sel, "fscore"),
                     agg(sel, "fscore_gain")))
        print("  잔존부 오차 중앙 %.2fmm · 출력 점간격 %.2fmm"
              % (agg(lambda r, m=mode: r["mode"] == m, "resid_err_mm"),
                 agg(lambda r, m=mode: r["mode"] == m, "spacing_mm")))

    # ---- G-A 게이트
    if "pred" in modes:
        bot = lambda r: r["mode"] == "pred" and r["pattern"] == "bottom"
        cover = agg(bot, "cover_gain")
        fill = agg(bot, "n_new_valid")
        noadd = agg(lambda r: r["mode"] == "pred" and r["pattern"] == "none", "n_new")
        ok1, ok2 = cover >= GATE_COVER, noadd < GATE_NOADD
        print("\n=== G-A 게이트 ===")
        print("  ① bottom 커버리지 순증 %.3f  (통과선 %.2f 이상)  → %s"
              % (cover, GATE_COVER, "통과" if ok1 else "미달"))
        print("     (참고 · 옛 개수 기준: 유효보충점 %.0f vs %d — 기준이 틀렸다. 주석 참조)"
              % (fill, GATE_FILL_REF))
        print("  ② none 보충점 %.0f  (통과선 %d 미만)  → %s"
              % (noadd, GATE_NOADD, "통과" if ok2 else "미달"))
        if ok1 and ok2:
            print("  → 둘 다 통과. **이 데이터에서는 메운다.**")
            print("     다만 [2] 는 '메울 능력이 있나'에만 답한다. 08-28 과의 차이가")
            print("     구조인지 입력인지는 **[1] EA_028 이 갈린다.** 거기까지 보고 판정할 것")
        elif ok1:
            print("  → 과채움 모델. S11 대비 이득 없음 → 닫는다")
        else:
            print("  → 결실부를 못 메운다. 08-28 결과가 재현됐다")

    print("\n저장 → " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
