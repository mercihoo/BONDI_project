# -*- coding: utf-8 -*-
"""
make_flow_figure.py — **점이 어떻게 찍히고 메시가 어떻게 채워지나**를 한 장으로.

발표용이다. 말로 하면 계속 헷갈리는 세 가지를 실제 데이터로 보여준다.

  1. 모델은 **물체 전체**에 점을 찍는다. 대부분(87~92%)은 입력을 다시 그린 것이고,
     새로 만든 것은 8~13% 뿐이다.
  2. **어디를 메울지**는 관측이 빈 칸(A)으로 정하고,
     **어떤 모양으로**는 모델 점으로 정한다. 둘은 다른 마스크다.
  3. 모델 점은 결손 전체를 못 덮는다(A 의 1/3만). 나머지는 **이웃에서 보간**한다.

`(h, θ)` 격자를 그대로 이미지로 보여주는 것이 요점이다 —
이 파이프라인이 3D 문제를 2D 로 바꿔 푼다는 것이 그림으로 드러난다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY make_flow_figure.py --src work/pred_71489_A_normal_v3tta8.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import make_mesh_71489 as M

HERE = Path(__file__).resolve().parent
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORK / "pred_71489_A_normal_v3tta8.npz"))
    ap.add_argument("--nh", type=int, default=144)
    ap.add_argument("--nt", type=int, default=216)
    ap.add_argument("--out", default="images/설명-점과격자.png")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    def font(sz, bold=False):
        f = "C:/Windows/Fonts/malgun%s.ttf" % ("bd" if bold else "")
        return ImageFont.truetype(f, sz) if Path(f).is_file() else ImageFont.load_default()

    d = np.load(args.src, allow_pickle=True)
    P = np.asarray(d["pred"], np.float64)
    new = np.asarray(d["new"], bool)
    dense = np.asarray(d["input_dense"], np.float64)
    up = int(d["up"])

    mg = M.load_measure_glb()
    _, c2 = mg.fit_axis(P, up)
    h, th, r, ax = M.cyl(P, up, c2)
    hd, td, rd, _ = M.cyl(dense, up, c2)

    nh, nt = args.nh, args.nt
    lo, hi = np.percentile(h, [0.5, 99.5])
    he = np.linspace(lo, hi, nh + 1)

    def cells(a, b):
        return (np.clip(np.digitize(a, he) - 1, 0, nh - 1),
                np.clip(((b + np.pi) / (2 * np.pi) * nt).astype(int), 0, nt - 1))

    hbo, tbo = cells(hd, td)
    occ = np.zeros((nh, nt), bool)
    occ[hbo, tbo] = True
    ow, _, _ = M.openwork_cells_nfold(occ, band_h=6)
    A = (~occ) & (~ow)                         # 결손 = 관측이 빈 칸 (투창 제외)

    hbp, tbp = cells(h, th)
    B = np.zeros((nh, nt), bool)
    B[hbp[new], tbp[new]] = True               # 모델이 **새 점**을 찍은 칸

    # A 안에서 모델 점(새것+기존)이 실제로 있는 칸 — 반지름을 직접 정할 수 있는 곳
    havePt = np.zeros((nh, nt), bool)
    havePt[hbp, tbp] = True
    filled_by_model = A & havePt
    filled_by_interp = A & ~havePt             # 이웃에서 보간해야 하는 칸

    # ---------- 그리기 ----------
    CW, CH = 300, 210                          # 격자 패널 (nt:nh ≈ 3:2)
    PADX, PADY, TOP = 26, 74, 108
    NC = 3
    panels = [
        ("① 관측 점유", occ, (60, 60, 70), "흰 = 관측이 있는 칸"),
        ("② 투창 (보호)", ow, (150, 90, 200), "주기 검사로 갈라낸다"),
        ("**A** 결손 = ①의 빈칸 − ②", A, (210, 110, 40), "%d칸 (%.0f%%) — **어디를 메울지**" % (A.sum(), 100 * A.mean())),
        ("**B** 모델이 새 점 찍은 칸", B, (60, 140, 210), "%d칸 (%.0f%%) — A 안에만 있다" % (B.sum(), 100 * B.mean())),
        ("A 중 모델 점이 있는 칸", filled_by_model, (60, 170, 120), "%d칸 — 반지름을 직접 정한다" % filled_by_model.sum()),
        ("A 중 보간으로 채우는 칸", filled_by_interp, (190, 60, 90), "%d칸 (A 의 %.0f%%) — 이웃에서 끌어온다" % (filled_by_interp.sum(), 100 * filled_by_interp.sum() / max(A.sum(), 1))),
    ]
    NR = (len(panels) + NC - 1) // NC
    W = NC * (CW + PADX) + PADX
    H = TOP + NR * (CH + PADY) + 30
    sh = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(sh)

    for i, (title, mask, col, sub) in enumerate(panels):
        cx = PADX + (i % NC) * (CW + PADX)
        cy = TOP + (i // NC) * (CH + PADY)
        img = np.full((nh, nt, 3), 248, np.uint8)
        img[mask] = col
        im = Image.fromarray(img[::-1]).resize((CW, CH), Image.NEAREST)   # 위가 아가리
        sh.paste(im, (cx, cy))
        dr.rectangle([cx, cy, cx + CW, cy + CH], outline=(200, 200, 200))
        dr.text((cx, cy + CH + 8), title.replace("**", ""), font=font(14, True), fill=(25, 25, 25))
        dr.text((cx, cy + CH + 28), sub.replace("**", ""), font=font(12), fill=col)
        dr.text((cx + CW - 44, cy + 6), "아가리", font=font(10), fill=(120, 120, 120))
        dr.text((cx + CW - 30, cy + CH - 18), "굽", font=font(10), fill=(120, 120, 120))

    t = "점은 어떻게 찍히고 메시는 어떻게 채우나 — (h, θ) 격자 %dx%d" % (nh, nt)
    dr.text((W / 2 - dr.textlength(t, font=font(22, True)) / 2, 14), t,
            font=font(22, True), fill=(20, 20, 20))
    u = ("모델 출력 %d점 중 기존 재현 %d (%.0f%%) · 새로 만든 점 %d (%.0f%%)"
         % (len(P), (~new).sum(), 100 * (~new).mean(), new.sum(), 100 * new.mean()))
    dr.text((W / 2 - dr.textlength(u, font=font(14)) / 2, 48), u, font=font(14), fill=(95, 95, 95))
    v = "가로 = 각도 θ 한 바퀴 · 세로 = 높이 h.  3D 회전체를 2D 로 펴서 푼다"
    dr.text((W / 2 - dr.textlength(v, font=font(13)) / 2, 70), v, font=font(13), fill=(140, 140, 140))

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sh.save(out)
    print("저장 → " + str(out))
    print("  A(결손) %d · B(모델 새 점) %d · B⊄A %d · 투창에 새 점 %d"
          % (A.sum(), B.sum(), (B & ~A).sum(), (ow & B).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
