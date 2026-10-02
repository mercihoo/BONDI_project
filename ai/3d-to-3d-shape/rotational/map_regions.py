# -*- coding: utf-8 -*-
"""결손 판정 지도 — 어디를 메우고 어디를 남기는지 (y, θ) 전개도로 그린다.

"안 메워야 할 부분을 메운다"를 고치려면 먼저 **어느 영역이 문제인지** 지목해야 한다.
select_regions 의 판정을 영역 번호와 함께 그려서 사진과 대조할 수 있게 한다.

사용:
  python map_regions.py --glb out/probe/A_normal.glb --out out/regions.png
"""
import sys
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

import build_mesh_direct as B
import skeleton_cyl as S

# 콘솔이 cp949 면 em dash(U+2014) 같은 문자에서 print 가 죽는다.
# 인코딩은 그대로 두고(한글이 깨지므로) errors 만 완화한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


# 관측 / 채움 / 투창(남김) / n-fold 로 뚫은 투창 / 미세조각(버림)
COL = {"obs": (206, 202, 192), "fill": (214, 92, 70),
       "openwork": (60, 140, 210), "carved": (40, 190, 175),
       "tiny": (150, 150, 150)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", default="out/regions.png")
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=144)
    ap.add_argument("--min-cells", dest="min_cells", type=int, default=8)
    ap.add_argument("--openwork-max-cells", dest="ow_cells", type=int, default=400)
    ap.add_argument("--openwork-max-deg", dest="ow_deg", type=float, default=40.0)
    ap.add_argument("--px", type=int, default=7, help="칸 하나를 몇 픽셀로")
    ap.add_argument("--nfold", default="off",
                    help="build_mesh_direct 와 같은 값을 주면 카빙까지 지도에 나온다")
    a = ap.parse_args()

    V, F = S.load_glb(a.glb)
    filled, observed, _, _ = B.cyl_map(V, F, a.ny, a.ntheta)
    need, kinds = B.select_regions(~observed, a.min_cells, a.ow_cells, a.ow_deg)
    carved_mask = np.zeros_like(need)
    if a.nfold != "off":
        need2, carved = B.carve_openwork_nfold(need, observed, kinds, a.ntheta,
                                               mode=a.nfold)
        carved_mask = need & ~need2          # 카빙으로 채움에서 빠진 칸
        need = need2

    # 영역별로 다시 라벨링해 번호를 붙인다 (select_regions 와 같은 방식)
    from scipy import ndimage
    miss = ~observed
    big = np.concatenate([miss, miss, miss], axis=1)
    lab, _ = ndimage.label(big, structure=np.ones((3, 3)))
    mid = lab[:, a.ntheta:2 * a.ntheta]

    img = np.zeros((a.ny, a.ntheta, 3), np.uint8)
    img[observed] = COL["obs"]
    rows = []
    for i in np.unique(mid[mid > 0]):
        m = mid == i
        ys, ts = np.nonzero(m)
        area = int(m.sum())
        occ = np.zeros(a.ntheta, bool); occ[ts] = True
        idx = np.nonzero(occ)[0]
        gaps = np.diff(np.concatenate([idx, [idx[0] + a.ntheta]]))
        wdeg = (a.ntheta - gaps.max() + 1) / a.ntheta * 360
        enclosed = bool(ys.min() > 0 and ys.max() < a.ny - 1)
        if area < a.min_cells:
            kind = "tiny"
        elif enclosed and wdeg <= a.ow_deg and area <= a.ow_cells:
            kind = "openwork"
        else:
            kind = "fill"
        img[m] = COL[kind]
        if kind == "fill":                   # 카빙으로 빠진 칸은 따로 칠한다
            img[m & carved_mask] = COL["carved"]
        rows.append(dict(id=int(i), kind=kind, area=area, wdeg=wdeg,
                         enclosed=enclosed, y0=int(ys.min()), y1=int(ys.max()),
                         th=float(ts.mean() / a.ntheta * 360),
                         cy=int(ys.mean()), cx=int(ts.mean())))

    # 위아래를 뒤집어 그린다 — y 가 위로 가도록
    px = a.px
    im = Image.fromarray(img[::-1]).resize(
        (a.ntheta * px, a.ny * px), Image.NEAREST)
    d = ImageDraw.Draw(im)

    # 20도 눈금 + 높이 눈금
    for deg in range(0, 360, 20):
        x = deg / 360 * a.ntheta * px
        d.line([(x, 0), (x, 6)], fill=(90, 90, 90))
        d.text((x + 2, 2), f"{deg}", fill=(60, 60, 60))
    for h in range(0, a.ny, 16):
        y = (a.ny - h) * px
        d.line([(0, y), (6, y)], fill=(90, 90, 90))
        d.text((8, y - 12), f"h{h}", fill=(60, 60, 60))

    for r in rows:
        if r["kind"] == "tiny":
            continue
        x = r["cx"] * px
        y = (a.ny - r["cy"]) * px
        tag = f"#{r['id']}"
        d.text((x - 6, y - 6), tag, fill=(20, 20, 20))
        d.text((x - 6, y + 4), f"{r['area']}", fill=(20, 20, 20))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    im.save(a.out)

    print(f"\n  {'번호':>5} {'판정':>9} {'칸':>6} {'θ폭':>7} {'θ중심':>7} "
          f"{'높이':>10} {'갇힘':>5}")
    print("  " + "-" * 60)
    for r in sorted(rows, key=lambda r: -r["area"]):
        if r["kind"] == "tiny":
            continue
        span = "[%d,%d]" % (r["y0"], r["y1"])
        print(f"  {r['id']:>5} {r['kind']:>9} {r['area']:>6,} {r['wdeg']:>6.1f}도 "
              f"{r['th']:>6.1f}도 {span:>10} "
              f"{'예' if r['enclosed'] else '아니오':>5}")
    ntiny = sum(1 for r in rows if r["kind"] == "tiny")
    print(f"\n  미세조각 {ntiny}개는 생략 (min_cells {a.min_cells} 미만)")
    print(f"  지도 → {a.out}")
    print(f"  색: 회색=관측  빨강=채움  파랑=투창(판정)  청록=투창(n-fold 카빙)  연회색=미세조각\n")


if __name__ == "__main__":
    main()
