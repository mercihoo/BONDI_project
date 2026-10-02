"""사진 -> 실루엣 -> 단면 프로파일 r(z). 1단계.

회전체는 옆에서 보면 **실루엣이 곧 프로파일**이다. 배경이 단색이라 분리가 되고,
`meta/artifacts.csv` 의 `높이` 가 픽셀을 cm 로 바꿔 준다. 그래야 여러 유물을 같은 자에 올린다.

분리 자체는 `_seg.py` 가 한다 (그라데이션 배경 + 그림자 제거). 여기서는 그 뒤를 한다.

조심할 것
---------
1. **글자.** 사진 구석에 `慶州 新收 71489.` 같은 화상 번호가 있다. 테두리에 안 닿는
   가장 큰 덩어리만 남겨 피한다.
2. **축.** 유물이 세워져 있으니 축은 화면에서 수직이다. 다만 아랫부분은 그림자 잔재가
   남을 수 있어 **위쪽 60% 줄**로만 축을 잡는다.
3. **깨진 유물은 좌우가 안 맞는다.** 좌우 반폭 차이를 `asym` 으로 남긴다 —
   PCA 학습에 넣을지 거르는 기준이다.
4. **고배는 아가리가 최대폭이다.** 그보다 넓은 줄이 아래에 나오면 그림자 잔재로 보고
   잘라내되, **몇 줄을 잘랐는지 반드시 보고**한다. 조용히 고치면 다음에 못 찾는다.
5. **원근.** 사진이 살짝 위에서 찍혀 아가리가 타원으로 보인다. 여기서는 재기만 하고
   (`rim_open`) 펴는 건 2단계다.

검증: 온전한 유물은 실루엣 최대폭이 곧 아가리다. **최대폭 vs 기록 입지름**을 맞춰 보면
분리·축척이 한꺼번에 검증된다.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os

# 분리 방식을 바꿔 끼운다 — SEG=rembg 면 학습 모델, 없으면 규칙 기반 v3
if os.environ.get("SEG", "rule") == "rembg":
    from seg_rembg import segment
    TAG = "rembg"
else:
    from _seg import segment
    TAG = "rule"

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        matplotlib.rcParams["font.family"] = f
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
IMG, OUT = HERE / "images", HERE / "work"
MAXDIM, NPROF = 1400, 200


def load(path: Path):
    im = Image.open(path).convert("RGB")
    s = min(1.0, MAXDIM / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def edges(mask):
    xs = np.arange(mask.shape[1])
    rows = np.flatnonzero(mask.any(1))
    L = np.full(len(rows), np.nan); R = np.full(len(rows), np.nan)
    for i, y in enumerate(rows):
        v = xs[mask[y]]
        L[i], R[i] = v[0], v[-1]
    return rows, L, R


def trim_shadow(rows, L, R, axis):
    """그림자를 기하로 걷어낸다 — **그리고 못 걷어낸 곳을 표시한다.**

    고배는 아가리가 최대폭이다 (기록에서도 입지름 18.1 vs 받침지름 11.2).
    그보다 넓은 줄은 유물일 수 없으니 좌/우 반폭을 각각 아가리 폭으로 자른다.

    여기까지가 되는 부분이고, **안 되는 부분이 있다.**

    자르기는 그림자를 *없애지* 않는다. 아가리 폭으로 *줄일* 뿐이다. 그래서 그림자가
    굽 높이에 걸쳐 있으면 굽 자리에 **실제보다 넓은 통짜 기둥**이 남는다.
    받침지름을 아는 2건에서 바닥폭이 -22% 로 어긋난 것이 그 흔적이다.

    시도했다 물린 것 (전부 더 나빴다):
      · 아래에서 위로 넓은 줄 버리기  -> 그림자는 아래로 뾰족해져 첫 줄에서 멈춤. 27장 전부 0줄
      · 넘친 줄부터 아래 전부 버리기  -> 27~45% 를 날려 굽이 사라지고 축척이 +64% 로 뒤집힘
      · 덩어리 내부 Otsu 로 가르기    -> 그림자 d 분포가 유물 어두운 부분과 겹쳐 안 갈림

    그래서 **고치는 대신 표시한다.** 아래쪽 35% 에서 자르기가 얼마나 일어났는지를
    `foot_bad` 로 남긴다. 값이 크면 그 프로파일의 **굽은 믿으면 안 된다** —
    3단계에서 z 범위를 잘라 쓰거나 그 유물을 빼면 된다.
    """
    rl, rr = axis - L, R - axis
    n = len(rows)
    top = max(3, int(0.55 * n))
    rim = float(np.nanmax(np.maximum(rl, rr)[:top]))
    cap = rim * 1.05
    over = (rl > cap) | (rr > cap)
    lo = int(0.65 * n)                       # 아래쪽 35%
    foot_bad = float(over[lo:].mean()) if n - lo > 0 else 0.0
    rl = np.minimum(rl, cap); rr = np.minimum(rr, cap)
    return np.ones(n, bool), rl, rr, rim, int(over.sum()), round(100 * foot_bad, 1)


def elevation_deg(h_px, w_px, H, D):
    """사진이 얼마나 위에서 찍혔나.

    카메라가 각 t 만큼 위에 있으면 실루엣의 세로 길이는 유물 높이보다 **길어진다** —
    아가리가 타원으로 보여 뒤쪽 테두리가 위로 솟기 때문이다.

        겉보기 세로 = H*cos(t) + D*sin(t)
        겉보기 가로 = D                      (가로는 줄어들지 않는다)

    그래서 높이로 축척을 잡으면 폭이 **작게** 나온다. 지금 관측된 것이 정확히 그 방향이다.
    """
    if not (H and D and h_px and w_px):
        return None
    target = (w_px / h_px)                           # = D / (H cos t + D sin t)
    lo, hi = 0.0, 1.2
    for _ in range(60):
        t = (lo + hi) / 2
        if D / (H * np.cos(t) + D * np.sin(t)) > target:
            lo = t
        else:
            hi = t
    return float(np.degrees((lo + hi) / 2))


def main():
    for d in ("silhouettes", "profiles", "figures"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    meta = {r["file"]: r for r in json.loads((HERE / "meta" / "artifacts.json").read_text("utf-8"))}

    out, panels = [], []
    for p in sorted(IMG.glob("*.jpg")):
        m = meta.get(p.name, {})
        a = load(p)
        mask, info = segment(a)
        if info["area"] < 0.004:
            out.append(dict(file=p.name, ok=False, note="분리 실패")); continue

        rows, L, R = edges(mask)
        axis = float(np.nanmedian((L + R) / 2)[()]) if False else float(
            np.nanmedian(((L + R) / 2)[:max(3, int(0.6 * len(rows)))]))
        keep, rl, rr, rim_px, clipped, foot_bad = trim_shadow(rows, L, R, axis)
        rows, rl, rr = rows[keep], rl[keep], rr[keep]
        L, R = axis - rl, axis + rr
        r = (rl + rr) / 2
        asym = float(np.nanmedian(np.abs(rl - rr)) / max(np.nanmedian(r), 1e-6))

        hpx = float(rows[-1] - rows[0] + 1)
        wpx = float(2 * r.max())
        hcm = m.get("height_cm"); dcm = m.get("mouth_cm")
        cpp = (hcm / hpx) if hcm else None
        elev = elevation_deg(hpx, wpx, hcm, dcm)
        z = (rows[-1] - rows) * (cpp or 1.0)
        rc = r * (cpp or 1.0)
        o = np.argsort(z)
        u = np.linspace(0, float(z.max()), NPROF)
        prof = np.interp(u, z[o], rc[o])
        np.savez_compressed(OUT / "profiles" / f"{p.stem}__{TAG}.npz", z=u, r=prof,
                            z_raw=z, r_raw=rc, axis=axis, cm_per_px=cpp or np.nan, asym=asym)

        band = rows[:max(3, len(rows) // 4)]
        holes = sum(1 for y in band
                    if len(v := np.flatnonzero(mask[y])) > 2 and (np.diff(v) > 1).any())
        out.append(dict(file=p.name, ok=True, cls=m.get("class", "?"),
                        num=m.get("소장품번호", ""), h_cm=hcm, h_px=round(hpx, 1),
                        px_per_cm=round(hpx / hcm, 2) if hcm else None,
                        area=round(100 * info["area"], 1), asym=round(100 * asym, 1),
                        wmax_cm=round(2 * float(rc.max()), 2) if cpp else None,
                        mouth_cm=dcm, clipped=clipped, foot_bad=foot_bad,
                        elev=round(elev, 1) if elev else None,
                        cpp_w=round(dcm / wpx, 5) if (dcm and wpx) else None,
                        cpp_h=round(cpp, 5) if cpp else None,
                        rim_open=round(100 * holes / max(len(rows), 1), 1),
                        comps=info["comps"], rule=info.get("rule", ""),
                        shadow=info.get("shadow", ""), cut_frac=info.get("cut_frac", 0)))
        panels.append((m.get("소장품번호", p.stem), a, mask, axis, rows, L, R,
                       m.get("class", "?"), clipped))
        Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "silhouettes" / f"{p.stem}__{TAG}.png")

    cols = []
    with (OUT / "profiles" / f"step1_summary_{TAG}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols[:] = ["file", "ok", "pass", "why", "cls", "num", "h_cm", "h_px", "px_per_cm", "area", "asym",
                "wmax_cm", "mouth_cm", "dropped", "clipped", "elev", "cpp_w", "cpp_h",
                "foot_bad", "ar", "rim_open", "comps", "rule", "shadow", "cut_frac", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(out)

    P = [x for x in panels if x[7] == "plain"]
    nc = 7; nr = -(-len(P) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.1, nr * 2.5))
    for a_, (num, im, mask, axis, rows, L, R, _, tr) in zip(np.ravel(ax), P):
        a_.imshow(im); a_.contour(mask, [0.5], colors="#ff2d2d", linewidths=0.6)
        a_.plot(L, rows, "-", c="#00e5ff", lw=0.8); a_.plot(R, rows, "-", c="#00e5ff", lw=0.8)
        a_.axvline(axis, color="#ffe600", lw=0.6, ls="--")
        a_.set_title(f"{num}" + (f"  [{tr}]" if tr else ""), fontsize=6); a_.axis("off")
    for a_ in np.ravel(ax)[len(P):]:
        a_.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"step1_montage_{TAG}.png", dpi=150)

    ok = [r for r in out if r.get("ok")]
    pl = [r for r in ok if r.get("cls") == "plain"]
    print(f"[{TAG}] 처리 {len(out)}장 · 성공 {len(ok)} · plain {len(pl)}\n")
    print(f"{'소장품':<12}{'높이cm':>7}{'좌우차%':>8}{'최대폭':>8}{'입지름':>8}"
          f"{'오차%':>7}{'굽오염%':>8}{'덩어리':>7}{'앙각도':>7}  {'그림자'}")
    print("-" * 76)
    for r in sorted(pl, key=lambda x: -(x["asym"] or 0)):
        mo, wm = r["mouth_cm"], r["wmax_cm"]
        err = 100 * (wm - mo) / mo if (mo and wm) else float("nan")
        print(f"{r['num']:<12}{r['h_cm'] or 0:>7.1f}{r['asym']:>8.1f}{wm or 0:>8.2f}"
              f"{mo or 0:>8.2f}{err:>7.1f}{r['foot_bad']:>8.1f}"
              f"{r['comps']:>7d}{(r['elev'] if r['elev'] else float('nan')):>7.1f}"
              f"  {r.get('shadow','')}")

    ev = [r["elev"] for r in pl if r.get("elev")]
    if ev:
        print()
        print(f"촬영 앙각 추정 ({len(ev)}건): 중앙 {np.median(ev):.1f}도 · "
              f"사분위 {np.percentile(ev,25):.1f}~{np.percentile(ev,75):.1f}도")
        print("  높이로 축척을 잡으면 폭이 이만큼 작게 나온다 -> 2단계에서 펴야 한다")

    # ── 합격 관문 ─────────────────────────────────────────────
    # 자동 분리가 전부 성공하지는 않는다. 억지로 맞추는 대신 **못 맞힌 것을 표시**한다.
    #   덩어리 <= 5    많으면 마스크가 조각났다는 뜻
    #   좌우차 <= 30%  회전체인데 좌우가 이보다 다르면 분리가 샌 것 (깨진 유물은 따로 본다)
    #   오차 <= 25%    기록 입지름과 이보다 벌어지면 믿을 수 없다 (입지름 없으면 면제)
    for r in pl:
        mo, wm = r["mouth_cm"], r["wmax_cm"]
        e = abs(wm - mo) / mo if (mo and wm) else None
        why = []
        # 덩어리 수는 품질 지표가 아니었다 — 배경의 자잘한 티끌까지 세는 바람에
        # 멀쩡히 뽑힌 박물관 사진들이 무더기로 탈락했다. 고른 덩어리 자체를 봐야 한다.
        # 고배는 폭/높이가 대략 0.8~1.5 다. 그림자 후광을 통째로 삼키면 2를 넘는다.
        ar = (wm / r["h_cm"]) if (wm and r["h_cm"]) else None
        r["ar"] = round(ar, 2) if ar else None
        if ar and ar > 1.7:
            why.append(f"폭/높이{ar:.1f}")
        if r["asym"] > 30:
            why.append(f"좌우{r['asym']:.0f}%")
        if e is not None and e > 0.25:
            why.append(f"오차{100*e:.0f}%")
        r["pass"] = not why
        r["why"] = "·".join(why)
    good = [r for r in pl if r["pass"]]
    print(f"\n관문 통과 {len(good)}/{len(pl)}  (PCA 학습에 쓸 수 있는 것)")
    bad = [r for r in pl if not r["pass"]]
    if bad:
        print("  탈락:", ", ".join(f"{r['num']}({r['why']})" for r in bad))
    with (OUT / "profiles" / f"usable_{TAG}.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(r["file"] for r in good))
    with (OUT / "profiles" / f"step1_summary_{TAG}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(out)

    c = [(r["wmax_cm"], r["mouth_cm"]) for r in good if r["mouth_cm"] and r["wmax_cm"]]
    if c:
        w_, mo = np.array(c).T
        e = np.abs(w_ - mo) / mo
        sg = (w_ - mo) / mo
        print(f"\n검증 — 실루엣 최대폭 vs 기록 입지름 ({len(c)}건)")
        print(f"  상대오차 중앙 {100*np.median(e):.1f}% · p75 {100*np.percentile(e,75):.1f}%"
              f" · 5% 이내 {int((e<0.05).sum())}건 / 10% 이내 {int((e<0.10).sum())}건")
        print(f"  부호 있는 오차 중앙 {100*np.median(sg):+.1f}% "
              f"(음수면 실루엣이 좁다 = 높이 축척이 과소)")
    print("\n몽타주: work/figures/step1_montage.png")


if __name__ == "__main__":
    main()
