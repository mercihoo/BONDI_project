"""3D 프로파일로 형상 사전(PCA)을 만들고 **가린 뒤 맞혀서** 채점한다.

왜 이제 3D 인가
---------------
사진 실루엣으로 만들려던 것을 3D 로 바꾼다. 두 가지가 동시에 풀린다.

| | 사진 | 3D |
| --- | --- | --- |
| 원근 | 앙각 때문에 **-13.2% 계통 오차** (보정 필요) | 투영을 안 거친다. **-1.4%** (편향 없음) |
| 그림자 | 굽이 오염돼 23점 중 5점만 성했다 | 없다 |

비율은 박물관 기록 20점으로 채점해 뒀다 (`validate_trellis_ratio.py`,
절대오차 중앙 3.9%). 그래서 이 3D 를 형상 사전의 재료로 써도 된다.

무엇을 학습하나 — 라벨이 없다
-----------------------------
PCA 는 비지도다. 정답도 라벨도 필요 없고, **여러 개를 겹쳐 놓고 "보통 이렇다" 를 뽑는다.**

    어떤 고배든  ≈  평균 + a1*(1번 변형) + ... + ak*(k번 변형)

프로파일 120개 숫자가 k 개로 준다. 그래서 **깨져서 일부만 남아도**, 남은 부분으로
a 를 맞추면 나머지가 따라 나온다. 3DMM 이 얼굴에서 하는 일과 같은 자리다.

어떻게 채점하나 — leave-one-out
-------------------------------
20점으로 만든 식으로 그 20점을 맞히면 당연히 잘 맞는다. 하나씩 빼고 맞혀야
**실제로 쓸 때의 오차**가 나온다. 게다가 가린 부분의 정답은 **실제 데이터**다 —
합성으로 지어낸 것이 아니다.

    한 점을 빼고 나머지로 PCA -> 그 점의 위쪽 X% 를 가림
    -> 남은 부분으로 계수 맞춤 -> 가린 부분 예측 -> 실제와 mm 비교

바닥선(baseline)과 같이 잰다. **마지막 반지름을 그대로 연장**하는 것보다 나아야
PCA 가 값어치가 있다.

사용:
  python shape_prior_3d.py --build      # 3D -> 프로파일 캐시
  python shape_prior_3d.py              # PCA + leave-one-out 채점
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from measure_glb import fit_axis, up_axis, vertices

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
H = Path(__file__).resolve().parent
G3 = H / "trellis3d"
CACHE = H / "work" / "profiles_3d.npz"
NB = 120
GRID = np.linspace(0, 1, NB)

# 사진에 유물이 둘 찍혀 3D 가 엉킨 장. 비율 채점에서 -64% 로 혼자 튀었다.
EXCLUDE = {"경주 583"}


def profile_cm(path: Path, height_cm: float):
    """높이 격자 위의 바깥 반지름 (cm). 기록 높이로 축척을 준다."""
    V = vertices(path)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    Hh = float(z.max() - z.min())
    h = (z - z.min()) / Hh
    ib = np.clip((h * NB).astype(int), 0, NB - 1)
    out = np.full(NB, np.nan)
    for i in range(NB):
        m = ib == i
        if m.sum() >= 30:
            out[i] = np.quantile(r[m], 0.97) / Hh * height_cm
    g = ~np.isnan(out)
    return np.interp(GRID, GRID[g], out[g])


def build():
    recs = json.loads((H / "meta/artifacts.json").read_text("utf-8"))
    P, names, heights = [], [], []
    for rec in recs:
        if rec.get("class") != "plain":
            continue
        num = (rec.get("소장품번호") or "").strip()
        # 아가리가 기록된 것 = 박물관이 잴 수 있었다 = 아가리가 온전하다
        if num in EXCLUDE or not rec.get("height_cm") or not rec.get("mouth_cm"):
            continue
        p = G3 / (re.sub(r"[^0-9A-Za-z가-힣]+", "_", num).strip("_") + ".glb")
        if not p.exists():
            continue
        try:
            P.append(profile_cm(p, rec["height_cm"]))
            names.append(num); heights.append(rec["height_cm"])
            print(f"  {num:<13} 높이 {rec['height_cm']:>5.1f}cm  최대반지름 {P[-1].max():.2f}cm")
        except Exception as e:
            print(f"  {num}: 실패 {e}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, P=np.array(P), names=np.array(names), heights=np.array(heights))
    print(f"\n프로파일 {len(P)}점 → {CACHE}")


def pca(X, k):
    m = X.mean(0)
    U, S, Vt = np.linalg.svd(X - m, full_matrices=False)
    return m, Vt[:k], S[:k] / np.sqrt(max(len(X) - 1, 1))


def fit_partial(m, W, sd, x, obs, clip=2.0, ridge=1e-3):
    """남은 부분으로 계수를 맞춘다. 표본이 적으므로 ±2σ 로 묶는다."""
    A = W[:, obs].T
    b = x[obs] - m[obs]
    a = np.linalg.lstsq(A.T @ A + ridge * np.eye(len(W)), A.T @ b, rcond=None)[0]
    a = np.clip(a, -clip * sd, clip * sd)
    return m + W.T @ a


def completeness(path: Path, NT=120):
    """높이대마다 **둘레의 몇 %에 바깥 벽이 있나**. 깨진 데는 낮다."""
    V = vertices(path)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    th = np.degrees(np.arctan2(V[:, ax[1]] - c[1], V[:, ax[0]] - c[0])) % 360
    Hh = float(z.max() - z.min())
    h = (z - z.min()) / Hh
    ib = np.clip((h * NB).astype(int), 0, NB - 1)
    it = np.clip((th / 360 * NT).astype(int), 0, NT - 1)
    rmax = np.zeros(NB * NT)
    np.maximum.at(rmax, ib * NT + it, r)
    rmax = rmax.reshape(NB, NT)
    prof = np.array([np.quantile(r[ib == k], 0.97) if (ib == k).sum() > 20 else np.nan
                     for k in range(NB)])
    with np.errstate(invalid="ignore"):
        shell = rmax >= 0.85 * prof[:, None]
    return shell.mean(1), prof, Hh


def fit_unknown_height(m, W, sd, obs_cm, obs_top_cm, mesh_cm, Ts=None):
    """온전했을 때의 높이 T 를 **같이 찾는다.**

    아가리가 깨지면 남은 것이 원래의 얼마인지 모른다. 그래서 후보 T 마다
    관측을 [0, obs_top_cm/T] 구간에 얹고 맞춰 본 뒤 **잔차가 가장 작은 T** 를 고른다.

    제약이 하나 있다 — **T 는 남은 메시 높이보다 작을 수 없다.** 조각이 거기까지
    올라와 있으니까. 처음엔 이걸 안 걸어서 T=11.8cm 라는, 메시(15.0cm)보다 낮은
    답이 나왔다.
    """
    if Ts is None:
        Ts = np.arange(mesh_cm, mesh_cm * 1.8, 0.1)
    n = len(obs_cm)
    best = None
    for T in Ts:
        f = obs_top_cm / T
        if f >= 0.995:
            continue
        idx = np.unique(np.clip((np.linspace(0, f, n) * (NB - 1)).astype(int), 0, NB - 1))
        oo = np.interp(np.linspace(0, 1, len(idx)), np.linspace(0, 1, n), obs_cm)
        x = np.zeros(NB); x[idx] = oo
        pred = fit_partial(m, W, sd, x, idx)
        res = float(np.sqrt(np.mean((pred[idx] - oo) ** 2)))
        if best is None or res < best[0]:
            best = (res, float(T), float(f), pred)
    return best


def predict_broken(name, height_cm, X, names, k=5, cover=0.80):
    p = G3 / (re.sub(r"[^0-9A-Za-z가-힣]+", "_", name).strip("_") + ".glb")
    cov, prof_unit, Hh = completeness(p)
    good = np.flatnonzero(cov >= cover)
    top = int(good.max()) if len(good) else NB - 1
    frac = (top + 1) / NB
    obs_top_cm = frac * height_cm
    print(f"\n=== {name} ===")
    print(f"  둘레 {int(100*cover)}% 이상 남은 높이  h <= {frac:.2f}  "
          f"(= 아래 {obs_top_cm:.1f}cm)")
    print(f"  그 위 {100*(1-frac):.0f}% 는 조각만 남아 프로파일로 못 쓴다")

    m, W, sd = pca(X, k)
    obs_cm = prof_unit[:top + 1] / Hh * height_cm
    g = ~np.isnan(obs_cm)
    obs_cm = np.interp(np.linspace(0, 1, g.sum()), np.linspace(0, 1, g.sum()), obs_cm[g])
    res, T, f, pred = fit_unknown_height(m, W, sd, obs_cm, obs_top_cm, height_cm)

    rim, mx = 2 * pred[-1], 2 * pred.max()
    print(f"\n잔차 {res:.3f}cm   (leave-one-out 중앙오차 0.11cm 와 견줄 것)")
    print(f"  -> 온전했을 때 높이 T ~ **{T:.1f}cm**  (기록 {height_cm}cm 는 남은 높이)")
    print(f"  -> 남은 것은 원래의 {100*f:.0f}%")
    print(f"  -> **아가리 지름 ~ {rim:.2f}cm**  · 최대지름 {mx:.2f}cm")
    print(f"  -> 입지름/높이 = {rim/T:.3f}   (갈래 20점 중앙 1.183 · 범위 0.93~1.51)")
    return T, f, pred, obs_cm, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--predict", default=None, help="깨진 유물 이름 (예: 경신 71489)")
    ap.add_argument("--height-cm", type=float, default=15.0)
    ap.add_argument("--out", default="work/figures/fig_shape_prior_loo.png")
    a = ap.parse_args()
    if a.build:
        build(); return

    d = np.load(CACHE, allow_pickle=True)
    X, names = d["P"], list(d["names"])
    n = len(X)
    if a.predict:
        predict_broken(a.predict, a.height_cm, X, names)
        return
    print(f"프로파일 {n}점 · 각 {NB}점 격자 · 단위 cm\n")

    # ── 주성분이 무엇을 나타내나 ──
    m0, W0, sd0 = pca(X, 6)
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    print("주성분 설명력 :", " ".join(f"PC{i+1} {100*v/ev.sum():.0f}%" for i, v in enumerate(ev[:6])))
    print(f"누적 6개      : {100*ev[:6].sum()/ev.sum():.1f}%\n")

    # ── leave-one-out 완성 채점 ──
    print(f"{'가린 구간':<12}{'성분':>5}{'PCA 중앙':>10}{'PCA p90':>10}"
          f"{'연장 중앙':>11}{'개선':>8}")
    print("-" * 58)
    best = None
    for hide in (0.20, 0.30, 0.40):
        cut = int((1 - hide) * NB)
        obs = np.arange(cut)
        hid = np.arange(cut, NB)
        for k in (3, 5, 8):
            ep, eb = [], []
            for i in range(n):
                tr = np.delete(X, i, 0)
                mm, WW, ss = pca(tr, k)
                pred = fit_partial(mm, WW, ss, X[i], obs)
                ep.append(np.abs(pred[hid] - X[i][hid]))
                eb.append(np.abs(X[i][cut - 1] - X[i][hid]))     # 마지막 값 연장
            ep = np.concatenate(ep); eb = np.concatenate(eb)
            imp = 100 * (1 - np.median(ep) / np.median(eb))
            print(f"{'위 ' + str(int(hide*100)) + '%':<12}{k:>5}{np.median(ep):>10.2f}"
                  f"{np.percentile(ep,90):>10.2f}{np.median(eb):>11.2f}{imp:>7.0f}%")
            if best is None or np.median(ep) < best[0]:
                best = (np.median(ep), hide, k)
    print(f"\n최적 — 가림 {int(best[1]*100)}% · 성분 {best[2]}개 · 중앙오차 {best[0]:.2f}cm")

    # ── 그림 ──
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].plot(m0, GRID, color="#14406b", lw=2.5, label="평균 형태")
    for i, c in zip(range(3), ("#c62828", "#e8a33d", "#2f7d32")):
        ax[0].plot(m0 + 2 * sd0[i] * W0[i], GRID, color=c, lw=1.2, ls="--",
                   label=f"PC{i+1} +2σ ({100*ev[i]/ev.sum():.0f}%)")
        ax[0].plot(m0 - 2 * sd0[i] * W0[i], GRID, color=c, lw=1.2, ls=":")
    ax[0].set_xlabel("반지름 (cm)"); ax[0].set_ylabel("높이 (0=굽, 1=아가리)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
    ax[0].set_title(f"(a) 갈래의 형상 사전 — {n}점 PCA", fontsize=11)

    hide, k = 0.30, best[2]
    cut = int((1 - hide) * NB); obs = np.arange(cut); hid = np.arange(cut, NB)
    j = int(np.argsort([np.abs(X[i]).max() for i in range(n)])[n // 2])
    tr = np.delete(X, j, 0)
    mm, WW, ss = pca(tr, k)
    pred = fit_partial(mm, WW, ss, X[j], obs)
    ax[1].plot(X[j], GRID, color="#14406b", lw=2.5, label="실제")
    ax[1].plot(X[j][obs], GRID[obs], color="#2f7d32", lw=4, alpha=.5, label="보여준 부분")
    ax[1].plot(pred[hid], GRID[hid], color="#c62828", lw=2.5, ls="--", label="PCA 예측")
    ax[1].axhline(GRID[cut], color="#888", lw=1, ls=":")
    ax[1].set_xlabel("반지름 (cm)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.25)
    ax[1].set_title(f"(b) {names[j]} — 위 {int(hide*100)}% 가리고 복원", fontsize=11)

    for hv, c in zip((0.20, 0.30, 0.40), ("#2f7d32", "#e8a33d", "#c62828")):
        cut = int((1 - hv) * NB); obs = np.arange(cut); hid = np.arange(cut, NB)
        ks, es = [], []
        for kk in (2, 3, 4, 5, 6, 8, 10):
            if kk >= n - 1:
                continue
            e = []
            for i in range(n):
                mm, WW, ss = pca(np.delete(X, i, 0), kk)
                e.append(np.abs(fit_partial(mm, WW, ss, X[i], obs)[hid] - X[i][hid]))
            ks.append(kk); es.append(np.median(np.concatenate(e)))
        ax[2].plot(ks, es, "o-", color=c, label=f"위 {int(hv*100)}% 가림")
    ax[2].set_xlabel("주성분 개수"); ax[2].set_ylabel("중앙 오차 (cm)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.25)
    ax[2].set_title("(c) leave-one-out 오차", fontsize=11)
    fig.suptitle("3D 프로파일 형상 사전 — 라벨 없이 배우고, 가린 뒤 맞혀서 채점", fontsize=13)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"그림: {a.out}")


if __name__ == "__main__":
    main()
