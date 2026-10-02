# -*- coding: utf-8 -*-
"""
make_pairs.py — 완형 메시를 합성으로 깨서 (부분, 완형) 쌍을 만든다.

입력은 `gupdari-shape-prior/trellis3d/*.glb` (완형 굽다리바리 42점)를 기본으로 쓴다.
AI-Hub 211-2 는 유적 발굴 스캔이라 완형이 거의 없다 — `scan_dataset.py` 결과 참조.

쓰는 데가 둘이다.
  [2] 진단   완형 하나를 깨서 사전학습 AdaPoinTr 에 그대로 넣는다.
             정답 완형을 우리가 갖고 있으니 처음으로 정량 측정이 된다.
  [3] 학습   파인튜닝 데이터셋. [2] 결과가 좋을 때만 간다.

결손 패턴 — 08-28 실패를 겨냥해 고른 것이다
  none    결손 없음 (완형 → 완형)
          **반드시 넣는다.** 이게 빠지면 모델이 "입력은 항상 부족하다"를 배워
          멀쩡한 유물에 살을 붙인다 (로드맵 §1.1 의 과채움).
  bottom  하부 전체 결실 — 08-28 에서 출력점 0점이 나온 바로 그 형태
  rim     구연부 일부 결손 — 71489 가 이 상태다 (둘레 전체가 톱니로 깨짐)
  side    측면 파단 — θ 구간 하나가 위아래로 관통
  ragged  **둘레 전체 불규칙 침식** — 71489 가 이 상태. [1][2] 가 갈라낸 실패 유형

회전축은 `gupdari-shape-prior/measure_glb.up_axis()` 를 그대로 쓴다.
직접 구현하면 거기서 두 번 틀린 것을 또 틀린다 (argmax(bbox) · 축길이 동점).

출력  work/pairs/<유물>__<패턴><번호>.npz
        partial          (n_partial, 3)   정규화됨. 모델 입력
        complete         (n_complete, 3)  정규화됨. 정답
        complete_removed (n_complete,) bool
              **떼어낸 영역의 정답점.** 판정의 "결실 영역"이 이것이다.
              거리 문턱으로 역추정하면 partial 이 성긴 곳을 결손으로 잘못 잡는다
              — 결손 0% 인 `none` 에서도 빨간 점이 흩어져 나왔다.
        center/scale             원래 좌표로 되돌리는 값
      work/pairs_manifest.csv

사용
  PY=".../venv/Scripts/python.exe"
  $PY make_pairs.py --limit 2                 # 맛보기
  $PY make_pairs.py --patterns none,bottom    # [2] 진단용
  $PY make_pairs.py --per-pattern 4           # [3] 학습용
"""
from __future__ import annotations

import argparse
import hashlib
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
PRIOR_DIR = ROOT / "gupdari-shape-prior"
TRELLIS3D = PRIOR_DIR / "trellis3d"

PATTERNS = ("none", "bottom", "rim", "side", "ragged", "slant")

# ragged 깎임 깊이 범위. --ragged-base 로 덮어쓴다.
# 71489 는 위쪽 약 48% 가 결손이다 (gupdari-shape-prior PIPELINE.md §12.3 —
# "둘레 80% 이상 남은 높이 h <= 0.52"). 기본값 0.06~0.22 는 그보다 훨씬 얕다.
RAGGED_BASE = (0.06, 0.22)

MIN_REMOVED = 0.02      # 이보다 적게 떼어냈으면 결손 패턴이라 할 수 없다
MAX_RETRY = 8


def stable_seed(*parts) -> int:
    """실행마다 같은 쌍이 나오게 한다.

    파이썬의 `hash()` 는 문자열에 대해 프로세스마다 달라진다(PYTHONHASHSEED).
    그걸 쓰면 데이터셋이 재현되지 않는다.
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest()[:8], 16)


# ------------------------------------------------- 완형 선별

# 메타 기준 9점에 **손으로 더한 7점**. 43점 전수 측정 + 사진 확인으로 골랐다.
# 값은 실측 결손률 (`(h,θ)` 빈 칸 비율, 투창 제외).
EXTRA = {
    "경주_5758":      0.050,   # plain
    "경주_6313":      0.051,   # plain
    "경신_71853":     0.062,   # plain
    "경주_4437":      0.084,   # handled  ← 손잡이 완형
    "경주_569":       0.091,   # handled  ← 흑백 사진이지만 형상은 온전
    "경주_고적_11504":  0.093,   # handled
    "인천소장_445":     0.043,   # dish     ← 굽다리접시. 형식 1점뿐이라 효과는 지켜볼 것
}


def complete_slugs(meta_csv: Path) -> tuple[set, dict]:
    """메타에서 **완형이라고 확인된** 유물만 고른다.

    trellis3d 42점을 전부 완형으로 쓰면 안 된다. 깨진 것이 섞여 있고,
    그중 하나가 **복원 대상인 경신 71489 자신**이다 (아가리가 둘레 전체 톱니).

    조건 셋 (메타 기준) — 9점
      class == plain    handled 는 손잡이가 회전대칭을 깨뜨린다
                        (gupdari-shape-prior README 가 1차 사용에서 이미 제외)
      is_broken == False
      mouth_cm 있음      입지름이 없다 = 박물관도 못 잰 것 = 아가리 결손
                        경신 계열 7점이 여기 걸린다. 71489 포함

    **그리고 손으로 고른 7점** (`EXTRA`) — 메타가 틀렸다

      메타의 `is_broken` 은 못 믿는다. 경주_고적_13472 는 몸통이 35% 없는데
      `is_broken=False` 다. 반대로 경주_5758 은 `is_broken=True` 인데
      실측 결손률이 **5.0%** 로 지금 쓰는 경주_5757(14.0%) 보다 훨씬 온전하다.

      그래서 43점을 전수 측정하고(`(h,θ)` 격자의 빈 칸 비율, 투창 제외)
      **사진을 같이 보며** 골랐다. 지표만으로는 안 된다 —
      경주_8480(1.6%)·8477(1.9%) 은 지표가 가장 깨끗한데 사진을 보면
      아가리가 크게 이빠져 있다. 아가리는 격자에서 맨 위 몇 행뿐이라
      전체 비율로는 작게 나온다.

      **handled 3점이 들어가는 것이 이번 확대의 핵심이다.** 지금 완형 풀에
      손잡이 형식이 0점인데, 파손품 34점 중 14점이 handled 다.
    """
    import pandas as pd
    import re
    import unicodedata

    df = pd.read_csv(meta_csv, encoding="utf-8-sig")
    df["slug"] = df["소장품번호"].map(
        lambda s: re.sub(r"\s+", "_", unicodedata.normalize("NFC", str(s)).strip()))
    df = df.drop_duplicates("slug")
    ok = df[(df["class"] == "plain") & (df["is_broken"] == False)
            & df["mouth_cm"].notna()]
    info = {r.slug: {"height_cm": r.height_cm, "mouth_cm": r.mouth_cm}
            for r in ok.itertuples()}

    # 손으로 고른 추가분. (실측 결손률, class) 는 고른 근거다
    for r in df[df["slug"].isin(EXTRA)].itertuples():
        info.setdefault(r.slug, {"height_cm": r.height_cm, "mouth_cm": r.mouth_cm})
    return set(info), info


# ------------------------------------------------- 이웃 폴더 함수 재사용

def load_measure_glb():
    """gupdari-shape-prior/measure_glb.py 를 경로로 불러온다.
    restore/ 를 통째로 옮겨도 따라오도록 스크립트 기준 상대경로다
    (restore/README.md 의 SSV_DIR 규약과 같은 방식)."""
    p = PRIOR_DIR / "measure_glb.py"
    if not p.is_file():
        raise SystemExit("[!] 없다: " + str(p))
    spec = importlib.util.spec_from_file_location("measure_glb", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------- 샘플링

def sample_surface(path: Path, n: int, seed: int) -> np.ndarray:
    """표면 균일 샘플. 정점을 그냥 쓰면 밀도가 곡률에 쏠린다."""
    g = trimesh.load(str(path), process=False)
    mesh = g.to_geometry() if hasattr(g, "to_geometry") else g
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float64)


def cyl_frame(P: np.ndarray, up: int, center2d: np.ndarray):
    """회전축 기준 (h, theta, r)."""
    ax = [i for i in range(3) if i != up]
    h = P[:, up]
    x = P[:, ax[0]] - center2d[0]
    y = P[:, ax[1]] - center2d[1]
    return h, np.arctan2(y, x), np.hypot(x, y)


# ------------------------------------------------- 결손 패턴

def pick_theta_in(th: np.ndarray, band: np.ndarray, rng: np.random.Generator) -> float:
    """쐐기 중심을 **그 띠에 실제로 점이 있는 각도**에서 고른다.

    전 각도에서 균일하게 뽑으면 빗나간다. 경주_고적_13472 가 그랬다 — 축 상단 띠가
    θ -141°~-46° 에만 있어서 `rim` 이 아무것도 못 떼어냈다 (결손률 0.000).
    """
    if band.any():
        return float(th[band][rng.integers(band.sum())])
    return float(rng.uniform(-np.pi, np.pi))


def mixed_mask(h, th, r, rng, parts=("ragged", "rim", "side")):
    """여러 패턴을 **한 유물에 동시에** 적용한다 (가설 3 검증용).

    실제 71489 는 V자 노치 + 측면 결실 + 둘레 톱니가 **섞여** 있는데,
    내 합성은 한 번에 한 패턴만 낸다. 그 복잡도 차이가 sim-to-real 격차의
    원인인지 보려고 만들었다.
    """
    keep = np.ones(len(h), bool)
    prm = {}
    for q in parts:
        k, pp = damage_mask(h, th, r, q, rng)
        keep &= k
        prm.update({q[:4] + "_" + kk: vv for kk, vv in pp.items()})
    return keep, prm


def damage_mask(h, th, r, pattern: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """남길 점 True. 반환 (mask, 파라미터)."""
    lo, hi = h.min(), h.max()
    span = max(hi - lo, 1e-12)
    hn = (h - lo) / span                       # 0 바닥 ~ 1 아가리

    if pattern == "none":
        return np.ones(len(h), bool), {}

    if pattern == "bottom":
        # 아래에서 f 만큼 잘라낸다. 08-28 의 "하부 전체 결실"
        f = float(rng.uniform(0.25, 0.55))
        return hn > f, {"cut_frac": round(f, 3)}

    if pattern == "rim":
        # 위쪽 f 구간에서 각도 w 만큼 떼어낸다 (구연부 결손)
        f = float(rng.uniform(0.10, 0.30))
        w = float(rng.uniform(np.pi / 3, np.pi * 1.2))
        band = hn > 1.0 - f
        c = pick_theta_in(th, band, rng)
        d = np.abs(np.angle(np.exp(1j * (th - c))))
        return ~(band & (d < w / 2)), {
            "band_frac": round(f, 3), "theta_deg": round(np.degrees(w), 1),
            "theta_c_deg": round(np.degrees(c), 1)}

    if pattern == "ragged":
        # 둘레 **전체**가 불규칙하게 깎인 아가리 — 71489 가 이 상태다.
        # [1][2] 가 갈라낸 실패 유형이 이것이라, 파인튜닝으로 가르치려면 반드시 넣어야 한다.
        # 각도마다 다른 높이까지 깎되, 저주파 잡음이라 톱니가 매끄럽게 이어진다.
        lo_b, hi_b = RAGGED_BASE                        # 스윕용 전역 (기본 0.06~0.22)
        base = float(rng.uniform(lo_b, hi_b))          # 평균적으로 얼마나 깎이나
        amp = float(rng.uniform(0.03, 0.12))           # 각도별 들쭉날쭉 정도
        k = rng.integers(2, 6)                         # 저주파 성분 수
        phase = rng.uniform(0, 2 * np.pi, size=k)
        freq = rng.integers(1, 5, size=k)
        w = rng.uniform(0.4, 1.0, size=k)
        w = w / w.sum()
        wave = sum(wi * np.sin(fi * th + pi)
                   for wi, fi, pi in zip(w, freq, phase))
        cut = 1.0 - base - amp * wave                  # 각도별 남길 높이 상한
        return hn < cut, {
            "base": round(base, 3), "amp": round(amp, 3), "n_freq": int(k)}

    if pattern == "slant":
        # **비스듬한 평면으로 자른다** — 한쪽은 남고 반대쪽은 거의 다 없어진다.
        #
        # 왜 필요한가
        #   `ragged` 는 둘레 전체를 깎지만 **깎는 높이가 거의 일정**하다
        #   (base 에 amp 0.03~0.12 만 흔들린다). 남는 것은 온전한 아래 그릇이다.
        #   그런데 실제 파손품에는 **몸통이 비스듬히 날아가 초승달 조각만** 남은
        #   것들이 있다 (경주_고적_13472 결손 35%, 경신_71488 33%).
        #   모델이 그런 입력을 한 번도 못 봐서 §5.10 ⑥ 의 구름이 나왔다.
        #
        #   `ragged` 가 여러 저주파의 합으로 **잔물결**을 만든다면,
        #   여기는 `cos(θ - c)` **한 주기**로 크게 기울인다.
        #
        # 굽다리는 남긴다
        #   실물에서 굽다리는 두껍고 낮아 거의 안 깨진다 — 고적_13472 도 굽은 온전하다.
        #   `h_keep` 아래는 무조건 남겨 **굽까지 날아가는 비현실적인 쌍**을 막는다.
        tilt = float(rng.uniform(0.35, 0.90))          # 기울기
        c = float(rng.uniform(0, 2 * np.pi))           # 어느 방향으로 기우나
        base = float(rng.uniform(0.15, 0.55))          # 평균 남기는 높이
        h_keep = float(rng.uniform(0.30, 0.50))        # 굽다리 보존선
        cut = np.maximum(base + tilt * np.cos(th - c), h_keep)
        return hn < cut, {
            "tilt": round(tilt, 3), "base": round(base, 3),
            "theta_c_deg": round(np.degrees(c), 1), "h_keep": round(h_keep, 3)}

    if pattern == "side":
        # 각도 w 구간을 위아래로 관통해 떼어낸다 (측면 파단)
        w = float(rng.uniform(np.pi / 4, np.pi * 0.9))
        h0 = float(rng.uniform(0.10, 0.40))
        h1 = h0 + float(rng.uniform(0.30, 0.55))
        band = (hn > h0) & (hn < h1)
        c = pick_theta_in(th, band, rng)
        d = np.abs(np.angle(np.exp(1j * (th - c))))
        return ~(band & (d < w / 2)), {
            "theta_deg": round(np.degrees(w), 1),
            "theta_c_deg": round(np.degrees(c), 1),
            "h_lo": round(h0, 3), "h_hi": round(min(h1, 1.0), 3)}

    raise ValueError("모르는 패턴: " + pattern)


# ------------------------------------------------- 파단면

def fracture_points(dense, keep, h, th, r, rng, nh=56, nt=80, per_cell=52):
    """결손 경계에 **파단면**을 세운다.

    왜 필요한가 — sim-to-real 격차 가설
      지금 합성 결손은 점을 **지우기만** 한다. 그러면 경계가 두께 0 의 깨끗한 절단이다.
      실제 도기는 두께가 있어(71489 manifest `thickness 0.008`) 깨지면
      **안쪽벽과 바깥벽을 잇는 단면이 드러난다.** 모델이 보는 국소 기하가 다르다.

      이 함수가 그 단면을 만든다. 같은 결손 파라미터로 **켜고/끄고** 비교하면
      파단면이 원인인지 아닌지가 갈린다.

    어떻게
      TRELLIS 메시는 두께 있는 닫힌 껍질이라, 한 (h, θ) 칸에 **바깥벽과 안쪽벽**
      점이 같이 있다. 칸마다 r 의 최대/최소가 그 두 벽이다.
      결손과 맞닿은 **남은 칸**에서 그 구간을 채워 벽을 세운다.
    """
    lo, hi = h.min(), h.max()
    span = max(hi - lo, 1e-12)
    hb = np.clip(((h - lo) / span * nh).astype(int), 0, nh - 1)
    tb = np.clip(((th + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1)

    kept_cell = np.zeros((nh, nt), bool)
    gone_cell = np.zeros((nh, nt), bool)
    np.logical_or.at(kept_cell, (hb[keep], tb[keep]), True)
    np.logical_or.at(gone_cell, (hb[~keep], tb[~keep]), True)

    # 결손과 맞닿은 남은 칸 = 경계
    nb = np.zeros((nh, nt), bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb |= np.roll(gone_cell, (dy, dx), (0, 1))
    border = kept_cell & nb & ~gone_cell
    if not border.any():
        return np.zeros((0, 3))

    hs = lo + (np.arange(nh) + 0.5) / nh * span
    ths = (np.arange(nt) + 0.5) / nt * 2 * np.pi - np.pi

    out = []
    for i, j in zip(*np.nonzero(border)):
        m = keep & (hb == i) & (tb == j)
        if m.sum() < 4:
            continue
        r_in, r_out = r[m].min(), r[m].max()
        if r_out - r_in < 1e-6:
            continue
        t = rng.random(per_cell)
        rr = r_in + t * (r_out - r_in)
        hh = hs[i] + (rng.random(per_cell) - 0.5) * (span / nh)
        tt = ths[j] + (rng.random(per_cell) - 0.5) * (2 * np.pi / nt)
        out.append(np.stack([hh, tt, rr], 1))
    if not out:
        return np.zeros((0, 3))
    return np.concatenate(out)


def htr_to_xyz(htr, up, center2d):
    ax = [i for i in range(3) if i != up]
    P = np.zeros((len(htr), 3))
    P[:, up] = htr[:, 0]
    P[:, ax[0]] = center2d[0] + htr[:, 2] * np.cos(htr[:, 1])
    P[:, ax[1]] = center2d[1] + htr[:, 2] * np.sin(htr[:, 1])
    return P


# ------------------------------------------------- 정규화

def normalize(complete: np.ndarray, partial: np.ndarray):
    """완형 기준 단위구로. 부분 기준으로 하면 결손 정도에 따라 축척이 흔들린다."""
    c = complete.mean(0)
    s = float(np.linalg.norm(complete - c, axis=1).max())
    s = s if s > 1e-12 else 1.0
    return (complete - c) / s, (partial - c) / s, c, s


def resample_idx(n_src: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """뽑은 **인덱스**를 준다. 인덱스를 들고 있어야 결손 마스크를 옮길 수 있다."""
    if n_src == 0:
        return np.zeros(n, int)
    return rng.choice(n_src, n, replace=n_src < n)


# ------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TRELLIS3D), help="완형 GLB 폴더")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--per-pattern", type=int, default=1, help="유물·패턴당 쌍 수")
    ap.add_argument("--n-complete", type=int, default=8192)
    ap.add_argument("--n-partial", type=int, default=2048)
    ap.add_argument("--dense", type=int, default=120000, help="깨기 전 조밀 샘플 수")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-meshes", action="store_true",
                    help="완형 필터를 끄고 GLB 전부를 쓴다 (권장하지 않음)")
    ap.add_argument("--meta", default=str(PRIOR_DIR / "meta" / "artifacts.csv"))
    ap.add_argument("--ragged-base", default="", help="예: 0.35,0.50")
    ap.add_argument("--pairs-out", default="", help="쌍을 다른 폴더에 쓴다")
    ap.add_argument("--only", default="", metavar="SLUG,SLUG", help="""이 유물만 쓴다.

    **테스트 세트를 따로 만들 때** 쓴다. 학습 16점은 에폭 선택(VAL_SOURCES)에도
    쓰였으므로 진짜 홀드아웃이 아니다 — 학습에도 에폭 선택에도 안 쓴 형상으로
    따로 세트를 만들어야 편향 없는 숫자가 나온다 (결과.md §5.14).

    완형 필터(`complete_slugs`)를 **건너뛴다.** 메타가 완형으로 안 잡아도
    실측이 깨끗하면 쓸 수 있다 — 경주_4614 가 그렇다 (전체 결손 0.1%).""")
    ap.add_argument("--mix", default="", help="패턴 동시 적용 예: ragged,rim,side")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="입력 점에 가우스 잡음 (정규화 단위. 유물 반지름=1)")
    ap.add_argument("--frac-per-cell", type=int, default=52,
                    help="파단면 칸당 점 수. 52 면 실제 71489 비중(~9%)에 맞는다")
    ap.add_argument("--fracture", action="store_true",
                    help="결손 경계에 파단면(두께 벽)을 세운다 — sim-to-real 비교용")
    args = ap.parse_args()

    global RAGGED_BASE
    if args.ragged_base:
        a, b = (float(v) for v in args.ragged_base.split(","))
        RAGGED_BASE = (a, b)
        print("ragged 깎임 깊이 %.2f~%.2f" % RAGGED_BASE)

    pats = [p.strip() for p in args.patterns.split(",") if p.strip()]
    bad = [p for p in pats if p not in PATTERNS]
    if bad:
        print("[!] 모르는 패턴: " + ", ".join(bad))
        return 1
    if "none" not in pats:
        print("[경고] 'none'(완형→완형)이 빠졌다. 학습에 쓰면 모델이 과채움을 배운다.")

    src = Path(args.src)
    files = sorted(src.glob("*.glb"))
    all_files = list(files)          # --only 는 완형 필터 전 목록에서 고른다
    if not files:
        print("[!] GLB 가 없다: " + str(src))
        return 1

    if args.all_meshes:
        print("[경고] 완형 필터를 껐다. 깨진 유물이 '완형 정답'으로 들어간다.")
    else:
        meta = Path(args.meta)
        if not meta.is_file():
            print("[!] 메타가 없다: " + str(meta) + "  (--all-meshes 로 건너뛸 수 있다)")
            return 1
        keep, info = complete_slugs(meta)
        import unicodedata
        before = len(files)
        files = [f for f in files
                 if unicodedata.normalize("NFC", f.stem) in keep]
        print("완형 필터: GLB %d개 → **%d개** (plain · 안깨짐 · 입지름 있음)"
              % (before, len(files)))
        if not files:
            print("[!] 남은 게 없다. 메타의 소장품번호와 GLB 파일명이 어긋났을 수 있다.")
            return 1

    if args.only:
        import unicodedata
        want = [unicodedata.normalize("NFC", x.strip())
                for x in args.only.split(",") if x.strip()]
        by = {unicodedata.normalize("NFC", f.stem): f for f in all_files}
        miss = [w for w in want if w not in by]
        if miss:
            print("[!] GLB 가 없다: %s" % ", ".join(miss))
            return 1
        files = [by[w] for w in want]
        print("--only: %d점만 쓴다 (완형 필터 건너뜀) — %s"
              % (len(files), ", ".join(want)))

    if args.limit:
        files = files[: args.limit]

    mg = load_measure_glb()
    out_dir = Path(args.pairs_out) if args.pairs_out else WORK / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("완형 %d점 × 패턴 %d × %d = 쌍 %d개"
          % (len(files), len(pats), args.per_pattern,
             len(files) * len(pats) * args.per_pattern))

    for fi, f in enumerate(files, 1):
        try:
            dense = sample_surface(f, args.dense, seed=args.seed + fi)
            up = mg.up_axis(dense)
            _, c2 = mg.fit_axis(dense, up)
            h, th, r = cyl_frame(dense, up, c2)
        except Exception as e:
            print("  [건너뜀] %s — %s: %s" % (f.name, type(e).__name__, e))
            continue

        for p in pats:
            for k in range(args.per_pattern):
                # 결손이 거의 없으면 'none' 과 같은 표본에 패턴 이름만 붙은 꼴이 된다.
                # 라벨이 틀린 학습 표본은 표본이 하나 빠지는 것보다 나쁘다 — 다시 뽑는다.
                keep, prm = None, None
                for attempt in range(MAX_RETRY):
                    rng = np.random.default_rng(
                        stable_seed(f.stem, p, k, attempt, args.seed))
                    if args.mix:
                        keep, prm = mixed_mask(
                            h, th, r, rng,
                            tuple(x.strip() for x in args.mix.split(",")))
                    else:
                        keep, prm = damage_mask(h, th, r, p, rng)
                    removed = 1.0 - float(keep.mean())
                    if p == "none" or removed >= MIN_REMOVED:
                        break
                else:
                    print("  [건너뜀] %s %s%d — %d회 시도에도 결손 %.3f"
                          % (f.stem, p, k, MAX_RETRY, removed))
                    continue

                kept = dense[keep]
                n_frac = 0
                if args.fracture:
                    htr = fracture_points(dense, keep, h, th, r, rng,
                                          per_cell=args.frac_per_cell)
                    if len(htr):
                        fp = htr_to_xyz(htr, up, c2)
                        kept = np.vstack([kept, fp])
                        n_frac = len(fp)
                if len(kept) < args.n_partial // 4:
                    print("  [건너뜀] %s %s%d — 남은 점이 너무 적다" % (f.stem, p, k))
                    continue

                ci = resample_idx(len(dense), args.n_complete, rng)
                pi = resample_idx(len(kept), args.n_partial, rng)
                comp, part = dense[ci], kept[pi]
                # complete 점마다 '떼어낸 영역에서 왔나'. 판정의 정답 마스크가 된다.
                # 거리 문턱으로 역추정하면 partial 이 성긴 곳을 결손으로 잘못 잡는다.
                comp_removed = ~keep[ci]
                comp_n, part_n, cen, sc = normalize(comp, part)
                if args.noise:
                    # 정답(comp)은 건드리지 않는다. 실제 잡음은 관측에만 있다
                    part_n = part_n + rng.normal(0, args.noise, part_n.shape)

                name = "%s__%s%d" % (f.stem, p, k)
                np.savez_compressed(
                    out_dir / (name + ".npz"),
                    partial=part_n.astype(np.float32),
                    complete=comp_n.astype(np.float32),
                    complete_removed=comp_removed,      # (n_complete,) bool
                    center=cen.astype(np.float64),
                    scale=np.float64(sc),
                    up_axis=np.int32(up),
                    pattern=p,
                )
                rows.append({
                    "name": name, "source": f.stem, "pattern": p, "rep": k,
                    "up_axis": up,
                    "kept_ratio": round(float(keep.mean()), 4),
                    "removed_ratio": round(float(1 - keep.mean()), 4),
                    "scale": round(sc, 6),
                    "comp_removed_ratio": round(float(comp_removed.mean()), 4),
                    "n_fracture": n_frac,
                    **{("prm_" + kk): vv for kk, vv in prm.items()},
                })
        print("  %d/%d  %s" % (fi, len(files), f.stem), flush=True)

    if not rows:
        print("[!] 만들어진 쌍이 없다.")
        return 1

    import pandas as pd
    df = pd.DataFrame(rows)
    man = out_dir.parent / (out_dir.name + "_manifest.csv")
    df.to_csv(man, index=False, encoding="utf-8-sig")

    print("\n=== 패턴별 결손 비율 ===")
    print(df.groupby("pattern")["removed_ratio"]
            .agg(["count", "mean", "min", "max"])
            .round(3).to_string())
    print("\n쌍 %d개 → %s" % (len(df), out_dir))
    print("목록 → " + str(man))
    return 0


if __name__ == "__main__":
    sys.exit(main())
