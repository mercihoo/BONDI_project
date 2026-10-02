# -*- coding: utf-8 -*-
"""
render_versions.py — GLB 를 **OpenGL 없이** 그림으로 찍는다.

왜 자체 래스터라이저인가
  `trimesh.Scene.save_image()` 는 pyglet 창을, pyrender 는 EGL 컨텍스트를 연다.
  이 노트북의 TRELLIS venv 에는 둘 다 없고, **그 venv 는 본 파이프라인이 쓰는
  것이라 건드리지 않는다**(`gupdari-71489/실행-구조.md` §5).
  기록용 그림에 필요한 것은 (1) 실제 재질 색 (2) 음영뿐이라
  numpy z-버퍼 평면음영으로 충분하다.

색은 어디서 오나
  프리미티브마다 다르다. 2노드 결과물은
    region_observed — 원본 texture + TEXCOORD_0
    region_ai       — 회전 복사 texture (v4~) 또는 정점색 (v1~v3)
  면 색 = 면 UV 중심의 텍셀, 없으면 정점색 평균, 없으면 baseColorFactor.

사용
  PY=".../venv/Scripts/python.exe"
  $PY render_versions.py --out images/재질-버전비교.png \
      work/restored_adapointr/A_normal_2node_smooth.glb:v1 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------ 면 색

def face_colors(m: trimesh.Trimesh) -> np.ndarray:
    """면마다 RGB(uint8). 텍스처 → 정점색 → 단색 순으로 떨어진다."""
    F = m.faces
    vis = getattr(m, "visual", None)

    uv = getattr(vis, "uv", None)
    img = getattr(getattr(vis, "material", None), "baseColorTexture", None)
    if uv is not None and img is not None and len(uv) == len(m.vertices):
        tex = np.asarray(img.convert("RGB"))
        c = uv[F].mean(axis=1)                       # 면 UV 중심
        h, w = tex.shape[:2]
        x = np.clip((c[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
        y = np.clip((1.0 - (c[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(int)
        return tex[y, x]

    vc = getattr(vis, "vertex_colors", None)
    if vc is not None and len(vc) == len(m.vertices):
        return np.asarray(vc)[F][:, :, :3].mean(axis=1).astype(np.uint8)

    bcf = getattr(getattr(vis, "material", None), "baseColorFactor", None)
    if bcf is not None:
        a = np.asarray(bcf, np.float64)[:3]
        if a.max() <= 1.0001:
            a = a * 255.0
        return np.tile(a.astype(np.uint8), (len(F), 1))
    return np.tile(np.array([180, 175, 165], np.uint8), (len(F), 1))


# ------------------------------------------------------------ 래스터라이저

def render(tris: np.ndarray, cols: np.ndarray, W: int, H: int,
           bg=(250, 249, 246), light=(-0.35, 0.45, 0.82)) -> np.ndarray:
    """정투영 + z-버퍼 + 평면음영. tris (F,3,3) 는 이미 화면 좌표계다."""
    img = np.zeros((H, W, 3), np.float64)
    img[:] = bg
    zb = np.full((H, W), np.inf)

    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    n[keep] /= ln[keep, None]

    L = np.asarray(light, np.float64)
    L /= np.linalg.norm(L)
    # 양면이라 법선 방향을 모른다. |cos| 을 쓰고 환경광을 얹는다.
    lam = 0.30 + 0.70 * np.abs(n @ L)
    shade = np.clip(cols.astype(np.float64) * lam[:, None], 0, 255)

    # 뒤에서 앞으로 — 같은 z 는 나중 것이 이긴다. z-버퍼가 있으니 순서는 안전망.
    order = np.argsort(-tris[:, :, 2].mean(axis=1))
    for f in order:
        if not keep[f]:
            continue
        t = tris[f]
        x0 = max(int(np.floor(t[:, 0].min())), 0)
        x1 = min(int(np.ceil(t[:, 0].max())) + 1, W)
        y0 = max(int(np.floor(t[:, 1].min())), 0)
        y1 = min(int(np.ceil(t[:, 1].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        d = ((t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0])
             + (t[2, 0] - t[1, 0]) * (t[0, 1] - t[2, 1]))
        if abs(d) < 1e-12:
            continue
        a = ((t[1, 1] - t[2, 1]) * (X - t[2, 0])
             + (t[2, 0] - t[1, 0]) * (Y - t[2, 1])) / d
        b = ((t[2, 1] - t[0, 1]) * (X - t[2, 0])
             + (t[0, 0] - t[2, 0]) * (Y - t[2, 1])) / d
        c = 1.0 - a - b
        ins = (a >= 0) & (b >= 0) & (c >= 0)
        if not ins.any():
            continue
        z = a * t[0, 2] + b * t[1, 2] + c * t[2, 2]
        sub = zb[y0:y1, x0:x1]
        win = ins & (z < sub)
        if not win.any():
            continue
        sub[win] = z[win]
        img[y0:y1, x0:x1][win] = shade[f]
    return img.astype(np.uint8)


def to_screen(V: np.ndarray, W: int, H: int, pad: float, box) -> np.ndarray:
    """정투영. y 는 위가 위로 보이게 뒤집는다. 여러 메시가 **같은 자**를 쓴다."""
    lo, hi = box
    s = min(W * (1 - 2 * pad) / max(hi[0] - lo[0], 1e-9),
            H * (1 - 2 * pad) / max(hi[1] - lo[1], 1e-9))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    out = np.empty_like(V)
    out[:, 0] = (V[:, 0] - cx) * s + W / 2
    out[:, 1] = H / 2 - (V[:, 1] - cy) * s
    out[:, 2] = -V[:, 2] * s                      # 카메라는 +z 에서 본다
    return out


def load_parts(path: Path, only=None):
    """(V, F, 면색, 이름) 목록. 씬이면 프리미티브별로 쪼갠다."""
    g = trimesh.load(str(path), process=False)
    geos = g.geometry if isinstance(g, trimesh.Scene) else {"mesh": g}
    out = []
    for name, m in geos.items():
        if only and name not in only:
            continue
        out.append((np.asarray(m.vertices, np.float64), np.asarray(m.faces),
                    face_colors(m), name))
    return out


def yaw_pitch(V, yaw_deg, pitch_deg, up: int):
    """up 축을 화면 세로로 세운 뒤 yaw·pitch 를 준다."""
    ax = [i for i in range(3) if i != up]
    Q = np.stack([V[:, ax[0]], V[:, up], V[:, ax[1]]], axis=1)   # x, 위, 깊이
    ry = np.radians(yaw_deg)
    c, s = np.cos(ry), np.sin(ry)
    Q = np.stack([Q[:, 0] * c + Q[:, 2] * s, Q[:, 1],
                  -Q[:, 0] * s + Q[:, 2] * c], axis=1)
    rp = np.radians(pitch_deg)
    c, s = np.cos(rp), np.sin(rp)
    return np.stack([Q[:, 0], Q[:, 1] * c - Q[:, 2] * s,
                     Q[:, 1] * s + Q[:, 2] * c], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="path.glb:제목  또는  path.glb:제목:프리미티브")
    ap.add_argument("--out", default="images/재질-버전비교.png")
    ap.add_argument("--w", type=int, default=420)
    ap.add_argument("--h", type=int, default=560)
    ap.add_argument("--yaw", type=float, default=25.0)
    ap.add_argument("--pitch", type=float, default=8.0)
    ap.add_argument("--up", type=int, default=1)
    ap.add_argument("--cols", type=int, default=0, help="0 이면 한 줄")
    ap.add_argument("--title", default="")
    ap.add_argument("--sub", default="")
    args = ap.parse_args()

    # 합성은 PIL 로 한다 — trimesh 가 있는 venv 에 matplotlib 이 없다.
    # 그 venv 는 본 파이프라인이 쓰는 것이라 패키지를 추가하지 않는다.
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for f in ("C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"):
            if Path(f).is_file():
                return ImageFont.truetype(f, sz)
        return ImageFont.load_default()

    # 1차 통과 — 모든 결과물을 **같은 자**에 올린다
    loaded, boxes = [], []
    for it in args.items:
        bits = it.split(":")
        path, title = bits[0], bits[1] if len(bits) > 1 else Path(bits[0]).stem
        only = bits[2].split(",") if len(bits) > 2 else None
        parts = load_parts(HERE / path if not Path(path).is_absolute() else Path(path), only)
        if not parts:
            print("  [건너뜀] 프리미티브 없음: " + it)
            continue
        scr = [(yaw_pitch(V, args.yaw, args.pitch, args.up), F, C, n)
               for V, F, C, n in parts]
        allv = np.concatenate([s0[0] for s0 in scr])
        boxes.append((allv.min(0), allv.max(0)))
        loaded.append((title, scr))

    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)

    n = len(loaded)
    nc = args.cols or n
    nr = int(np.ceil(n / nc))
    PAD, CAP, TOP = 10, 30, (40 if args.title else 0)
    BOT = 30 if args.sub else 0
    Wc, Hc = args.w + 2 * PAD, args.h + CAP + PAD
    sheet = Image.new("RGB", (nc * Wc, TOP + nr * Hc + BOT), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)

    for k, (title, scr) in enumerate(loaded):
        tris, cols = [], []
        for V, F, C, _ in scr:
            S = to_screen(V, args.w, args.h, 0.06, (lo, hi))
            tris.append(S[F]); cols.append(C)
        img = render(np.concatenate(tris), np.concatenate(cols), args.w, args.h)
        cx, cy = (k % nc) * Wc, TOP + (k // nc) * Hc
        sheet.paste(Image.fromarray(img), (cx + PAD, cy + CAP))
        tw = dr.textlength(title, font=font(15))
        dr.text((cx + Wc / 2 - tw / 2, cy + 7), title, font=font(15), fill=(30, 30, 30))
        print("  그렸다: %-28s 면 %d" % (title, sum(len(t) for t in tris)), flush=True)

    if args.title:
        f = font(21)
        dr.text((sheet.width / 2 - dr.textlength(args.title, font=f) / 2, 9),
                args.title, font=f, fill=(20, 20, 20))
    if args.sub:
        f = font(13)
        dr.text((sheet.width / 2 - dr.textlength(args.sub, font=f) / 2,
                 sheet.height - BOT + 7), args.sub, font=f, fill=(90, 90, 90))

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(chr(10) + "저장 → " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
