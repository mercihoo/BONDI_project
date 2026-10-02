# -*- coding: utf-8 -*-
"""채움면 매끄러움 비교 그림 — 전진복사 vs 이중조화.

(y, θ) 반경 맵을 **음영기복(hillshade)** 으로 그리면 방사 줄무늬와 합류 능선이 눈에 보인다.
그 아래에 |Δr| 을 같은 색눈금으로 깔아 숫자와 그림을 맞춘다.

사용:
  python fig_smoothness.py --glb out/probe/A_normal.glb --out out/fig_smoothness.png
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



def hillshade(r, mask, az=315.0, alt=45.0, z=3.0):
    """반경 맵의 음영기복. 표면의 잔주름이 그대로 드러난다."""
    gy, gx = np.gradient(r * z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    a, A = np.deg2rad(alt), np.deg2rad(360.0 - az + 90.0)
    v = (np.sin(a) * np.sin(slope)
         + np.cos(a) * np.cos(slope) * np.cos(A - asp))
    v = np.clip(v, 0, 1)
    img = np.dstack([v, v, v]) * 255
    img[~mask] = (247, 246, 244)
    return img.astype(np.uint8)


def heat(v, mask, vmax):
    """0=흰 → vmax=진한 빨강. 채움 칸만 칠한다."""
    t = np.clip(v / vmax, 0, 1)[..., None]
    c = (1 - t) * np.array([252, 250, 248]) + t * np.array([150, 20, 15])
    c[~mask] = (232, 230, 226)
    return c.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", default="out/fig_smoothness.png")
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=144)
    ap.add_argument("--px", type=int, default=5)
    a = ap.parse_args()

    V, F = S.load_glb(a.glb)
    ssv = S.load_ssv()
    ssv.N_Y, ssv.N_THETA = a.ny, a.ntheta
    Vm = V * 250.0
    cx, cz = ssv.fit_axis(Vm)
    outer, _, _ = ssv.classify_surfaces(Vm, F, cx, cz)
    y = Vm[:, 1]
    y_edges = np.linspace(y.min(), y.max() + 1e-9, a.ny + 1)
    r, cnt = ssv.radial_grid(Vm, outer, cx, cz, y_edges)
    observed = cnt > 0
    rowmed = np.array([np.nanmedian(r[i]) if np.isfinite(r[i]).any() else np.nan
                       for i in range(a.ny)])
    prof = ssv.fill_profile(rowmed)
    L = B.build_laplacian(a.ny, a.ntheta)
    rb = B.obs_blur(r, 5, ssv)

    old = ssv.fill_grid(r, prof)                       # 전진 복사 (옛 경로)
    new = B.fill_solve(rb, prof, L, order=2)           # 이중조화 + 경계 저역통과
    miss = ~observed

    lap_o = np.abs((L @ old.ravel()).reshape(old.shape))
    lap_n = np.abs((L @ new.ravel()).reshape(new.shape))
    vmax = float(np.percentile(lap_o[miss], 95))

    panels = [
        (hillshade(old, miss), "전진복사 — 채움면 음영기복"),
        (hillshade(new, miss), "이중조화 — 채움면 음영기복"),
        (heat(lap_o, miss, vmax), f"전진복사 — |Dr|  중앙 {np.median(lap_o[miss]):.2f} "
                                  f"p95 {np.percentile(lap_o[miss],95):.1f}mm"),
        (heat(lap_n, miss, vmax), f"이중조화 — |Dr|  중앙 {np.median(lap_n[miss]):.2f} "
                                  f"p95 {np.percentile(lap_n[miss],95):.1f}mm"),
    ]

    px = a.px
    W, H = a.ntheta * px, a.ny * px
    canvas = Image.new("RGB", (W * 2 + 12, (H + 20) * 2 + 8), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, (img, lab) in enumerate(panels):
        col, row = k % 2, k // 2
        x0, y0 = col * (W + 12), row * (H + 28)
        d.text((x0 + 2, y0 + 3), lab, fill=(40, 40, 40))
        canvas.paste(Image.fromarray(img[::-1]).resize((W, H), Image.NEAREST),
                     (x0, y0 + 18))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    canvas.save(a.out)
    print(f"  |Dr| 채움칸 · 전진복사 중앙 {np.median(lap_o[miss]):.3f} "
          f"p95 {np.percentile(lap_o[miss],95):.2f} / 이중조화 중앙 "
          f"{np.median(lap_n[miss]):.3f} p95 {np.percentile(lap_n[miss],95):.2f} mm")
    print(f"  → {a.out}")


if __name__ == "__main__":
    main()
