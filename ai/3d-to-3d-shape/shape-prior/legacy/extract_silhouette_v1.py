"""사진 -> 실루엣 -> 단면 프로파일 r(z). 1단계.

회전체는 옆에서 보면 **실루엣이 곧 프로파일**이다. 배경이 단색이라 분리는 쉽고,
`meta/artifacts.csv` 의 `높이` 가 픽셀을 cm 로 바꿔 준다. 그래야 여러 유물을 같은 자에 올린다.

조심할 것 네 가지
-----------------
1. **글자.** 사진 구석에 `慶州 新收 71489.` 같은 화상 번호가 찍혀 있다. 배경과 색이 다르니
   전경으로 잡힌다. 그래서 **가장 큰 덩어리 하나만** 남긴다.
2. **축.** 유물이 세워져 있으므로 축은 화면에서 수직이다. 줄마다 좌/우 가장자리의
   가운데를 구해 그 중앙값을 축으로 쓴다.
3. **깨진 유물은 좌우가 안 맞는다.** 71489 가 그렇다. 좌우 반폭의 차이를 재서
   `asym` 으로 남긴다 — PCA 학습에 넣을지 거르는 기준이 된다.
4. **원근.** 사진이 살짝 위에서 찍혀 아가리가 타원으로 보인다. 여기서는 **재기만** 하고
   (`rim_open`), 펴는 건 2단계에서 한다.

검증: 온전한 유물은 실루엣 최대폭이 곧 아가리다. 그래서 **실루엣 최대폭 vs 기록 입지름**을
맞춰 보면 분리·축척이 맞는지 한 번에 드러난다.

출력: work/silhouettes/*.png · work/profiles/*.npz · work/figures/step1_*.png
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
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
IMG, OUT = HERE / "images", HERE / "work"
MAXDIM = 1400
NPROF = 200                      # 프로파일 표본 점 수


def load(path: Path):
    im = Image.open(path).convert("RGB")
    s = min(1.0, MAXDIM / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0, s


def segment(a: np.ndarray):
    """테두리에서 배경색을 추정하고, 거기서 먼 화소를 전경으로."""
    h, w = a.shape[:2]
    m = max(2, round(0.02 * min(h, w)))
    border = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                             a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)])
    bg = np.median(border, 0)
    d = np.linalg.norm(a - bg, axis=2)
    spread = np.median(np.abs(np.linalg.norm(border - bg, axis=1)))
    thr = max(4.0 * spread, 0.055)
    fg = d > thr
    fg = ndi.binary_opening(fg, np.ones((3, 3)), iterations=2)
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n == 0:
        return fg, 0.0, thr
    sizes = ndi.sum(fg, lab, range(1, n + 1))
    big = (lab == (int(np.argmax(sizes)) + 1))
    big = ndi.binary_closing(big, np.ones((5, 5)), iterations=2)
    return ndi.binary_fill_holes(big), float(big.mean()), thr


def profile(mask: np.ndarray):
    """줄마다 좌/우 가장자리 -> 축 -> 반지름."""
    rows = np.flatnonzero(mask.any(1))
    y0, y1 = rows[0], rows[-1]
    xs = np.arange(mask.shape[1])
    L = np.full(mask.shape[0], np.nan)
    R = np.full(mask.shape[0], np.nan)
    for y in rows:
        v = xs[mask[y]]
        L[y], R[y] = v[0], v[-1]
    axis = float(np.nanmedian((L + R) / 2))
    rl, rr = axis - L[rows], R[rows] - axis
    r = (rl + rr) / 2
    asym = float(np.nanmedian(np.abs(rl - rr)) / max(np.nanmedian(r), 1e-6))
    return rows, y0, y1, axis, r, rl, rr, asym


def rim_openness(mask: np.ndarray, rows, y0):
    """아가리가 타원으로 보이는 정도 — 위쪽에서 '속이 보이는' 구간의 세로 길이 / 전체 높이.
    원근을 펴야 할 양의 대략치다 (2단계 입력)."""
    band = rows[:max(3, len(rows) // 4)]
    holes = 0
    for y in band:
        v = np.flatnonzero(mask[y])
        if len(v) > 2 and (np.diff(v) > 1).any():
            holes += 1
    return holes / max(len(rows), 1)


def main():
    for d in ("silhouettes", "profiles", "figures"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    meta = {r["file"]: r for r in json.loads((HERE / "meta" / "artifacts.json").read_text("utf-8"))}

    rows_out, panels = [], []
    for p in sorted(IMG.glob("*.jpg")):
        m = meta.get(p.name, {})
        a, sc = load(p)
        mask, area, thr = segment(a)
        if area < 0.005:
            rows_out.append(dict(file=p.name, ok=False, note="분리 실패"))
            continue
        rr, y0, y1, axis, r, rl, rrr, asym = profile(mask)
        hpx = float(y1 - y0 + 1)
        hcm = m.get("height_cm")
        cm_per_px = (hcm / hpx) if hcm else None

        z = (y1 - rr).astype(np.float64)                  # 아래가 0
        zc = z * cm_per_px if cm_per_px else z
        rc = r * cm_per_px if cm_per_px else r
        u = np.linspace(0, zc[0] if zc[0] > zc[-1] else zc[-1], NPROF)
        order = np.argsort(zc)
        prof = np.interp(u, zc[order], rc[order])
        np.savez_compressed(OUT / "profiles" / f"{p.stem}.npz",
                            z=u, r=prof, z_raw=zc, r_raw=rc, axis=axis,
                            cm_per_px=cm_per_px or np.nan, asym=asym)

        rows_out.append(dict(
            file=p.name, ok=True, cls=m.get("class", "?"), num=m.get("소장품번호", ""),
            h_cm=hcm, h_px=round(hpx, 1), px_per_cm=round(hpx / hcm, 2) if hcm else None,
            area=round(100 * area, 1), asym=round(100 * asym, 1),
            wmax_cm=round(2 * float(rc.max()), 2) if cm_per_px else None,
            mouth_cm=m.get("mouth_cm"), rim_open=round(100 * rim_openness(mask, rr, y0), 1),
            broken=bool(m.get("is_broken")), imgw=a.shape[1]))
        panels.append((p.stem, a, mask, axis, rr, rl, rrr, m.get("class", "?")))

        Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (mask.shape[1] // 2, mask.shape[0] // 2)).save(OUT / "silhouettes" / f"{p.stem}.png")

    with (OUT / "profiles" / "step1_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["file", "ok", "cls", "num", "h_cm", "h_px", "px_per_cm", "area", "asym",
                "wmax_cm", "mouth_cm", "rim_open", "broken", "imgw", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows_out)

    # ── 몽타주 — 분리가 맞았는지는 눈으로 봐야 안다 ──
    P = [x for x in panels if x[7] == "plain"]
    nc = 7; nr = -(-len(P) // nc)
    fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.0, nr * 2.7))
    for a_, (stem, im, mask, axis, rws, rl, rrr, _) in zip(np.ravel(ax), P):
        a_.imshow(im)
        a_.contour(mask, [0.5], colors="r", linewidths=0.7)
        a_.axvline(axis, color="c", lw=0.6)
        a_.set_title(stem[-14:], fontsize=5); a_.axis("off")
    for a_ in np.ravel(ax)[len(P):]:
        a_.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "step1_montage.png", dpi=150)

    ok = [r for r in rows_out if r.get("ok")]
    pl = [r for r in ok if r.get("cls") == "plain"]
    print(f"처리 {len(rows_out)}장 · 성공 {len(ok)} · plain {len(pl)}\n")
    print(f"{'소장품':<12}{'높이cm':>7}{'px/cm':>7}{'면적%':>7}{'좌우차%':>8}{'최대폭cm':>9}{'입지름cm':>9}{'아가리열림%':>11}")
    print("-" * 72)
    for r in sorted(pl, key=lambda x: -(x["asym"] or 0)):
        print(f"{r['num']:<12}{r['h_cm'] or 0:>7.1f}{r['px_per_cm'] or 0:>7.1f}{r['area']:>7.1f}"
              f"{r['asym']:>8.1f}{r['wmax_cm'] or 0:>9.2f}"
              f"{(r['mouth_cm'] if r['mouth_cm'] else 0):>9.2f}{r['rim_open']:>11.1f}")

    cmp_ = [(r["wmax_cm"], r["mouth_cm"]) for r in pl if r["mouth_cm"] and r["wmax_cm"]]
    if cmp_:
        w_, mo = np.array(cmp_).T
        e = w_ - mo
        print(f"\n검증 — 실루엣 최대폭 vs 기록 입지름 ({len(cmp_)}건)")
        print(f"  차이 중앙 {np.median(e):+.2f}cm · 절대 중앙 {np.median(np.abs(e)):.2f}cm "
              f"· p90 {np.percentile(np.abs(e), 90):.2f}cm")
        print(f"  상대오차 중앙 {100*np.median(np.abs(e)/mo):.1f}%")
    print(f"\n몽타주: work/figures/step1_montage.png")


if __name__ == "__main__":
    main()
