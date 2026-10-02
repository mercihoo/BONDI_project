"""3D 에서 투창을 센다 — 사진이 못 하던 것.

사진 한 장은 `|θ| ≤ 58°` 만 읽히므로 N=3·4 면 한 단에 창이 평균 하나만 보인다
(`tuchang_from_photo.py` §7.4). **3D 는 360° 를 다 본다.**

그리고 문헌에 개수가 적힌 12점이 **정답표**다. 그래서 이건 추정이 아니라 **채점**이다.

방법
----
1. 회전축을 맞추고 (z, θ) 점유 격자를 만든다
2. 허리(최소 반지름) 아래가 굽다리다
3. 굽다리에서 **θ 방향으로 비어 있는 구간**을 센다 — 그것이 투창이다
4. 높이대로 묶어 단(段)을 가른다

먼저 `--dump` 로 점유도를 글자 그림으로 찍어 **눈으로 보고** 나서 자동 판정을 건다.
(배경 분리에서 문턱을 일곱 번 만지고 배운 것이다 — 보기 전에 맞추지 않는다.)

사용:
python tuchang_from_3d.py --dump 경주_8477 경주_5758
python tuchang_from_3d.py --score




"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from measure_glb import fit_axis, up_axis, vertices

H = Path(__file__).resolve().parent
G3 = H / "trellis3d"
HAN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}


def documented(rec):
    s = rec.get("설명") or ""
    if "透窓" not in s:
        return None
    for pat in (r"각각\s*([0-9一二三四五六])\s*개", r"각\s*([0-9一二三四五六])\s*개",
                r"([0-9一二三四五六])\s*[개個]의?\s*[^,.。]{0,14}?透窓",
                r"透窓이?\s*([0-9一二三四五六])\s*[개個]"):
        m = re.search(pat, s)
        if m:
            g = m.group(1)
            return HAN.get(g, int(g) if g.isdigit() else None)
    return None


def occupancy(path: Path, NZ=80, NT=120, shell_q=0.82):
    """(z, θ) 마다 **바깥 껍질이 있는가**. 그리고 그 유물이 얼마나 회전대칭인가.

    세 번 고쳤다.

    1. 정점 **점유** -> TRELLIS 메시는 닫힌 solid 라 투창 자리에도 절단면·안쪽벽
    정점이 있다. 어디든 '차 있다' 가 된다
    2. 세로축을 `argmax(bbox)` 로 -> 납작한 고배에서 가로축이 뽑힌다
    3. 격자가 성겨 한 칸짜리 구멍이 잔뜩 -> **칸을 키우고 형태학적 닫기**를 건다

    투창은 **바깥 벽이 없는 자리**다. 칸마다 최대 반지름을 재서 그 높이의 바깥
    반지름에 못 미치면 뚫린 것으로 본다.

    `asym` 은 굽다리 높이대의 바깥 반지름이 θ 를 따라 얼마나 흔들리나다.
    이게 크면 **뒷면 전체가 창으로 오인**되므로 채점에서 뺀다.




    """
    V = vertices(path)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    th = np.degrees(np.arctan2(V[:, ax[1]] - c[1], V[:, ax[0]] - c[0])) % 360
    h = (z - z.min()) / (z.max() - z.min())
    iz = np.clip((h * NZ).astype(int), 0, NZ - 1)
    it = np.clip((th / 360 * NT).astype(int), 0, NT - 1)

    rmax = np.zeros(NZ * NT)
    np.maximum.at(rmax, iz * NT + it, r)
    rmax = rmax.reshape(NZ, NT)
    cnt = np.bincount(iz * NT + it, minlength=NZ * NT).reshape(NZ, NT)
    prof = np.array([np.quantile(r[iz == k], 0.97) if (iz == k).sum() > 20 else np.nan
                     for k in range(NZ)])
    with np.errstate(invalid="ignore"):
        shell = (cnt > 0) & (rmax >= shell_q * prof[:, None])

    # 한 칸짜리 잡음을 없앤다 — θ 는 순환이므로 옆으로 이어 붙여 처리한다
    w = np.c_[shell[:, -6:], shell, shell[:, :6]]
    w = ndi.binary_closing(w, np.ones((3, 3)))
    w = ndi.binary_opening(w, np.ones((2, 3)))
    shell = w[:, 6:-6]

    # 회전 대칭성 — 굽다리 높이대에서 바깥 반지름이 θ 따라 얼마나 흔들리나
    band = (h > 0.12) & (h < 0.30)
    if band.sum() > 200:
        jt = np.clip((th[band] / 360 * 36).astype(int), 0, 35)
        ro = np.array([np.quantile(r[band][jt == k], 0.97) if (jt == k).sum() > 15 else np.nan
                       for k in range(36)])
        asym = float((np.nanmax(ro) - np.nanmin(ro)) / max(np.nanmedian(ro), 1e-9))
    else:
        asym = np.nan
    return shell, prof, len(V), asym


def waist(prof):
    """허리 = 중간 구간의 최소 반지름 높이. 그 아래가 굽다리다."""
    n = len(prof)
    lo, hi = int(0.25 * n), int(0.75 * n)
    seg = prof[lo:hi]
    if np.all(np.isnan(seg)):
        return int(0.45 * n)
    return lo + int(np.nanargmin(seg))


def dump(path: Path, name: str, doc=None):
    occ, prof, nv, asym = occupancy(path)
    NZ, NT = occ.shape
    w = waist(prof)
    print(f"\n=== {name} ===  정점 {nv:,} · 허리 h={w/NZ:.2f}"
          + f" · 비대칭 {100*asym:.0f}%" + (f" · 문헌 {doc}개" if doc else ""))
    step = max(1, NT // 120)
    print("      " + "".join("|" if (j * step * 360 // NT) % 30 == 0 else "."
                             for j in range(NT // step)))
    for i in range(w, -1, -1):
        row = occ[i]
        s = "".join("#" if row[j * step:(j + 1) * step].any() else " "
                    for j in range(NT // step))
        mark = " <" if i == w else ""
        print(f"{i/NZ:5.2f} {s}{mark}")


def windows(occ, w, min_deg=8.0, max_deg=70.0):
    """굽다리에서 비어 있는 θ 구간을 단별로 모은다."""
    NZ, NT = occ.shape
    band = occ[:w + 1]
    fill = band.mean(1)
    # 창이 있는 높이 = 채움률이 낮은 곳. 굽 바닥테/돌대는 높다
    openish = fill < (np.nanmax(fill) - 0.06)
    tiers, i = [], 0
    while i <= w:
        if openish[i]:
            j = i
            while j <= w and openish[j]:
                j += 1
            if (j - i) >= max(2, int(0.03 * NZ)):
                tiers.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for a, b in tiers:
        col = band[a:b].mean(0) > 0.5
        m = np.r_[col, col]
        runs, k = [], 0
        while k < NT:
            if not m[k]:
                q = k
                while q < k + NT and not m[q]:
                    q += 1
                d = (q - k) * 360.0 / NT
                if min_deg <= d <= max_deg:
                    runs.append(((k * 360.0 / NT + d / 2) % 360, d))
                k = q
            else:
                k += 1
        out.append(dict(h=(a / NZ, b / NZ), n=len(runs),
                        centers=[round(c, 1) for c, _ in runs],
                        widths=[round(d, 1) for _, d in runs]))
    return out


def nfold_fourier(occ, w, ns=range(2, 9)):
    """기존 파이프라인이 쓰는 방법 — **주기(n-fold) 로 센다.**

    `[수학] 투창-n-fold-판정.md` §2 와 같은 착상이다. 깨진 조각의 θ 중심은 파단이
    갉아먹어 흔들리므로 개수를 직접 세면 흔들린다. 대신 띠 안에서 θ 를 따라
    **'바깥 벽이 있나' 신호**에 푸리에를 걸면, 투창이 n 개일 때 n 차 조화가 서고
    파단은 비주기라 배경으로 깔린다.

    런(run) 세기와 무엇이 다른가 — 런은 **한 칸짜리 잡음 하나에 개수가 바뀐다.**
    푸리에는 전체 신호의 주기성을 보므로 잡음에 둔하다.



    """
    NZ, NT = occ.shape
    band = occ[:w + 1]
    fill = band.mean(1)
    openish = fill < (np.nanmax(fill) - 0.06)
    tiers, i = [], 0
    while i <= w:
        if openish[i]:
            j = i
            while j <= w and openish[j]:
                j += 1
            if (j - i) >= max(2, int(0.03 * NZ)):
                tiers.append((i, j))
            i = j
        else:
            i += 1
    out = []
    for a, b in tiers:
        sig = 1.0 - band[a:b].mean(0)          # 뚫린 정도
        sig = sig - sig.mean()
        if np.allclose(sig, 0):
            continue
        P = np.abs(np.fft.rfft(sig)) ** 2
        tot = P[1:].sum()
        if tot <= 0:
            continue
        best = max(ns, key=lambda n: P[n] if n < len(P) else 0)
        out.append(dict(h=(a / NZ, b / NZ), n=int(best),
                        share=float(P[best] / tot),
                        rank=[(int(n), round(float(P[n] / tot), 3))
                              for n in sorted(ns, key=lambda n: -P[n])[:3]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="*", default=None)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    recs = json.loads((H / "meta/artifacts.json").read_text("utf-8"))
    by = {}
    for r in recs:
        n = (r.get("소장품번호") or "").strip()
        s = re.sub(r"[^0-9A-Za-z가-힣]+", "_", n).strip("_")
        by.setdefault(s, r)

    if a.dump is not None:
        names = a.dump or [p.stem for p in sorted(G3.glob("*.glb"))[:4]]
        for s in names:
            p = G3 / f"{s}.glb"
            if not p.exists():
                print(f"없음: {p.name}"); continue
            dump(p, s, documented(by.get(s, {})))
        return

    VARIANTS = [
        # 바닥선 — 재지 않고 갈래 최빈값을 답한다. 측정이 이걸 못 이기면
        # 그 측정은 정보를 더하지 못한 것이다 (형상 사전의 '단순 연장' 과 같은 자리)
        ("바닥선: 항상 3", "mode"),
        ("런 세기", None),
        ("푸리에 n=2~8 (자유)", range(2, 9)),
        ("푸리에 n=3~8 (2-fold 제외)", range(3, 9)),
        ("푸리에 n∈{3,4} (문헌 사전)", (3, 4)),
    ]
    print("같은 3D · 같은 정답표(문헌 12점) · 후보 집합만 바꾼다\n")
    hdr = f"{'소장품':<14}{'문헌':>5}" + "".join(f"{v[0][:12]:>14}" for v in VARIANTS)
    print(hdr); print("-" * len(hdr))
    hit = {v[0]: 0 for v in VARIANTS}
    tot = 0
    for s_, r in sorted(by.items()):
        d = documented(r)
        p = G3 / f"{s_}.glb"
        if not d or not p.exists():
            continue
        occ, prof, _, asym = occupancy(p)
        w = waist(prof)
        tot += 1
        cells = []
        for lab, ns in VARIANTS:
            if ns == "mode":
                est = 3
            elif ns is None:
                v = [t["n"] for t in windows(occ, w) if t["n"] > 0]
                est = max(set(v), key=v.count) if v else None
            else:
                f = nfold_fourier(occ, w, ns)
                v = [t["n"] for t in f]
                est = max(set(v), key=v.count) if v else None
            hit[lab] += est == d
            cells.append(f"{str(est) + ('O' if est == d else 'X'):>14}")
        print(f"{r['소장품번호']:<14}{d:>5}" + "".join(cells))
    print()
    for lab, _ in VARIANTS:
        bar = "#" * round(20 * hit[lab] / max(tot, 1))
        print(f"{lab:<28}{hit[lab]:>3}/{tot}  {100*hit[lab]/max(tot,1):>3.0f}%  {bar}")


if __name__ == "__main__":
    main()
