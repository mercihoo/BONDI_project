# -*- coding: utf-8 -*-
"""복원 결과를 **사진과 같은 시점**으로 렌더해 나란히 놓는다.

카메라는 make_render_photo.fit_camera 를 그대로 쓴다 — 손상 메시의 투영 실루엣이
사진 알파와 가장 겹치는 각을 찾고, 그 IoU 가 카메라 추정의 검증치다.

관측(carried)과 채움(filled)을 색으로 갈라 그리므로 **어디를 메웠는지**가 바로 보인다.
투창을 메웠는지 확인하는 용도다.

사용:
  python render_check.py --glb out/v20-cyl+biharm+deg40+nn+up2/restored_direct.glb \\
      --photo out/mirror/orig_rgba.png --damaged out/probe/A_normal.glb --out out/check
"""
import sys
import argparse
import json
import os
import struct

import numpy as np
import trimesh
from PIL import Image

import make_render_photo as MRP

# 콘솔이 cp949 면 em dash(U+2014) 같은 문자에서 print 가 죽는다.
# 인코딩은 그대로 두고(한글이 깨지므로) errors 만 완화한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass


CARRIED = np.array([196, 186, 170], float)     # 잔존 — 흙빛 회색
FILLED = np.array([224, 122, 72], float)       # 채움 — 주황
BG = np.array([248, 247, 245], float)


def log(m):
    print(f"[CHECK] {m}", flush=True)


def glb_color0(path, mesh_name):
    """GLB 에서 특정 메시의 COLOR_0 을 직접 읽는다 (0..255 float, Nx3).

    trimesh 는 머티리얼이 붙은 프리미티브의 COLOR_0 을 visual 로 안 올린다.
    파일 자체는 멀쩡하고 언리얼·three.js 는 잘 읽으므로, 여기서만 우회한다.
    """
    d = open(path, "rb").read()
    off, js, bo = 12, None, 0
    while off < len(d):
        cl, ct = struct.unpack("<II", d[off:off + 8])
        if ct == 0x4E4F534A:
            js = json.loads(d[off + 8:off + 8 + cl].decode())
        else:
            bo = off + 8
        off += 8 + cl + ((4 - cl % 4) % 4 if cl % 4 else 0)
    for m in js.get("meshes", []):
        if m.get("name") != mesh_name:
            continue
        ai = m["primitives"][0]["attributes"].get("COLOR_0")
        if ai is None:
            return None
        a = js["accessors"][ai]
        bv = js["bufferViews"][a["bufferView"]]
        ncomp = {"VEC3": 3, "VEC4": 4}[a["type"]]
        dt, scale = {5126: (np.float32, 255.0), 5121: (np.uint8, 1.0),
                     5123: (np.uint16, 255.0 / 65535.0)}[a["componentType"]]
        st = bv.get("byteStride") or (np.dtype(dt).itemsize * ncomp)
        s = bo + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        raw = np.frombuffer(d[s:s + st * a["count"]], np.uint8).reshape(a["count"], st)
        arr = raw[:, :np.dtype(dt).itemsize * ncomp].copy().view(dt).reshape(-1, ncomp)
        return np.clip(arr[:, :3].astype(float) * scale, 0, 255)
    return None


def rot(V, yaw, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x = V[:, 0] * cy + V[:, 2] * sy
    z = -V[:, 0] * sy + V[:, 2] * cy
    y = V[:, 1] * cp - z * sp
    zz = V[:, 1] * sp + z * cp
    return np.stack([x, y, zz], 1)


def render(V, N, col, cam, shape, splat=1):
    """정사영 + 페인터 알고리즘 점 스플랫. 정점이 조밀해 면처럼 채워진다."""
    yaw, pitch, scale, tx, ty = cam
    P = rot(V, yaw, pitch)
    Nr = rot(N, yaw, pitch)
    u = np.rint(scale * P[:, 0] + tx).astype(np.int64)
    v = np.rint(-scale * P[:, 1] + ty).astype(np.int64)

    # 램버트 + 약한 환경광. 광원은 왼쪽 위 앞.
    L = np.array([-0.45, 0.6, 0.66]); L /= np.linalg.norm(L)
    lam = np.clip(Nr @ L, 0, 1)
    shade = (0.28 + 0.72 * lam)[:, None]
    rgb = np.clip(col * shade, 0, 255)

    img = np.tile(BG, (*shape, 1))
    order = np.argsort(-P[:, 2])                  # 먼 것부터
    uo, vo, co = u[order], v[order], rgb[order]
    h, w = shape
    offs = [(dx, dy) for dy in range(-splat, splat + 1)
            for dx in range(-splat, splat + 1) if (dx, dy) != (0, 0)] + [(0, 0)]
    for dx, dy in offs:                            # 중심을 마지막에 — 정확한 화소가 위로
        uu, vv = uo + dx, vo + dy
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        img[vv[ok], uu[ok]] = co[ok]
    return img.astype(np.uint8)


def strip(images, labels, height=520):
    """가로로 이어 붙인다."""
    out = []
    for im in images:
        p = Image.fromarray(im)
        p = p.resize((int(p.width * height / p.height), height), Image.LANCZOS)
        out.append(p)
    W = sum(p.width for p in out) + 8 * (len(out) - 1)
    canvas = Image.new("RGB", (W, height), (255, 255, 255))
    x = 0
    for p in out:
        canvas.paste(p, (x, 0))
        x += p.width + 8
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True, help="복원 GLB (2노드)")
    ap.add_argument("--photo", required=True, help="배경 제거된 RGBA")
    ap.add_argument("--damaged", required=True, help="손상 GLB — 카메라 맞추기용")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=760, help="렌더 한 변")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arr = np.array(Image.open(args.photo).convert("RGBA"))
    alpha = arr[:, :, 3] > 96
    Vd = MRP.glb_vertices(args.damaged)
    cam, iou = MRP.fit_camera(Vd, alpha)
    log(f"카메라 IoU {iou*100:.1f}% (낮으면 시점이 사진과 안 맞는 것)")

    # 사진 좌표계의 카메라를 렌더 캔버스로 옮긴다
    ys, xs = np.nonzero(alpha)
    S = args.size
    pad = 0.06 * S
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    k = (S - 2 * pad) / span
    yaw, pitch, scale, tx, ty = cam
    cam2 = (yaw, pitch, scale * k,
            S / 2 - k * ((xs.min() + xs.max()) / 2 - tx),
            S / 2 - k * ((ys.min() + ys.max()) / 2 - ty))

    scene = trimesh.load(args.glb)
    parts = []
    for name, c in (("region_carried", CARRIED), ("region_filled", FILLED)):
        m = scene.geometry.get(name)
        if m is None:
            continue
        parts.append((np.asarray(m.vertices), np.asarray(m.vertex_normals),
                      np.tile(c, (len(m.vertices), 1)), name, m))
    V = np.concatenate([p[0] for p in parts])
    N = np.concatenate([p[1] for p in parts])
    C = np.concatenate([p[2] for p in parts])
    log(f"정점 {len(V):,} · " + " · ".join(f"{p[3]} {len(p[0]):,}" for p in parts))

    # 실제 색. carried 는 텍스처, filled 는 COLOR_0 이라 노드마다 경로가 다르고,
    # **trimesh 는 머티리얼이 붙은 프리미티브의 COLOR_0 을 안 넘겨준다.**
    # GLB 는 멀쩡하므로(언리얼에서는 정상) 여기서만 직접 읽는다.
    def vcols(m, name):
        try:
            c = np.asarray(m.visual.to_color().vertex_colors)
            if c.ndim == 2 and c.shape[0] == len(m.vertices) and c.shape[1] >= 3:
                return c[:, :3].astype(float)
        except Exception:
            pass
        c = glb_color0(args.glb, name)
        return c if (c is not None and len(c) == len(m.vertices)) else None

    cols = [vcols(p[4], p[3]) for p in parts]
    if all(c is not None for c in cols):
        Creal = np.concatenate(cols)
    else:
        miss = [p[3] for p, c in zip(parts, cols) if c is None]
        log(f"정점색 사용 불가 ({', '.join(miss)}) — 구분색만 그린다")
        Creal = None

    photo = Image.open(args.photo).convert("RGBA")
    bgm = Image.new("RGBA", photo.size, (248, 247, 245, 255))
    bgm.alpha_composite(photo)
    ph = np.array(bgm.convert("RGB"))
    ys2, xs2 = np.nonzero(alpha)
    ph = ph[max(0, ys2.min() - 20):ys2.max() + 20, max(0, xs2.min() - 20):xs2.max() + 20]

    imgs = [ph, render(V, N, C, cam2, (S, S))]
    labs = ["사진", "복원 (회색=잔존, 주황=채움)"]
    if Creal is not None:
        imgs.append(render(V, N, Creal, cam2, (S, S)))
        labs.append("복원 (실제 색)")
    strip(imgs, labs).save(os.path.join(args.out, "photo_vs_restored.png"))
    log(f"→ {os.path.join(args.out, 'photo_vs_restored.png')}")

    # 네 방향 — 투창을 다 보려면 돌려 봐야 한다
    views = []
    for d in (0, 90, 180, 270):
        c = (cam2[0] + np.deg2rad(d), *cam2[1:])
        views.append(render(V, N, C, c, (S, S)))
    strip(views, [f"+{d}도" for d in (0, 90, 180, 270)]).save(
        os.path.join(args.out, "turntable.png"))
    log(f"→ {os.path.join(args.out, 'turntable.png')} (사진 시점 기준 +0/90/180/270도)")


if __name__ == "__main__":
    main()
