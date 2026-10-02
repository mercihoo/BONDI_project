"""복원 3D를 사진 시점으로 투영해 완형 조건 이미지를 만든다 — 생성 모델 없이.

왜
--
면 연속성(README §0 층 ②)의 남은 용의자가 조건 이미지다. 채움 복셀 자리에서
원본 사진은 "파단면"을 보여주므로 모델이 그 표면을 *조각의 끝*으로 기술한다.
`fresh` 조차 22조각인 것이 근거다.

미러 합성(make_mirror_photo.py)은 구연부 결손이 좌우로 겹쳐 부족했다(IoU 52.6%).
여기서는 **이미 만든 복원 3D**를 사진 시점으로 되돌려 완형 실루엣을 얻는다.

근거
----
완형 형상이 회전대칭에서 나온 것이므로 **조건도 같은 근거**를 쓴다.
생성 모델을 쓰면 manifest 표기가 "AI 추정"으로 떨어진다 (문서 §4.2).
채움 영역의 색은 **같은 행의 가장 가까운 잔존 화소**에서 가져온다 —
이미지 공간에서의 회전대칭이고, 새 정보를 지어내지 않는다.

카메라
------
TRELLIS 는 입력 시점이 정면인 좌표계로 생성한다. 그래도 가정하지 않고
**손상 메시의 투영 실루엣이 사진 알파와 가장 겹치는 카메라를 탐색**한다.
그 IoU 가 곧 카메라 추정의 검증치다. 낮으면 이 경로를 쓰면 안 된다.

사용:
  python make_render_photo.py --photo out/mirror/orig_rgba.png \
      --damaged out/probe/A_normal.glb --restored out/v7-slat+cylskel+fixed/R_fresh.glb --out out/rendercond
"""

import sys
import argparse
import json
import os
import struct

import numpy as np
from PIL import Image
from scipy import ndimage

# 콘솔이 cp949 면 em dash(U+2014) 같은 문자에서 print 가 죽는다.
# 인코딩은 그대로 두고(한글이 깨지므로) errors 만 완화한다.
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass



def log(m):
    print(f"[RCOND] {m}", flush=True)


def glb_vertices(path):
    d = open(path, "rb").read()
    off, js, bo = 12, None, 0
    while off < len(d):
        cl, ct = struct.unpack("<II", d[off:off + 8])
        if ct == 0x4E4F534A:
            js = json.loads(d[off + 8:off + 8 + cl].decode())
        else:
            bo = off + 8
        off += 8 + cl + ((4 - cl % 4) % 4 if cl % 4 else 0)
    pr = js["meshes"][0]["primitives"][0]
    a = js["accessors"][pr["attributes"]["POSITION"]]
    bv = js["bufferViews"][a["bufferView"]]
    s = bo + bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    st = bv.get("byteStride") or 12
    V = np.frombuffer(d[s:s + st * a["count"]], np.uint8).reshape(a["count"], st)[:, :12].copy()
    return V.view(np.float32).reshape(-1, 3).astype(np.float64)


def project(V, yaw, pitch, scale, tx, ty, shape):
    """Y-up 정규화 좌표를 화상으로 투영해 실루엣을 만든다 (정사영 근사).

    박물관 사진은 망원에 가까워 원근이 작다. 정사영으로 두고
    yaw·pitch·scale·이동만 맞춘다.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x = V[:, 0] * cy + V[:, 2] * sy
    z = -V[:, 0] * sy + V[:, 2] * cy
    y = V[:, 1] * cp - z * sp
    u = np.rint(scale * x + tx).astype(np.int64)
    v = np.rint(-scale * y + ty).astype(np.int64)
    h, w = shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sil = np.zeros(shape, bool)
    sil[v[ok], u[ok]] = True
    return sil


def close_holes(sil, r=2):
    st = np.ones((2 * r + 1, 2 * r + 1), bool)
    return ndimage.binary_closing(sil, st)


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def fit_camera(V, alpha, sub=60000, seed=0):
    """손상 메시 실루엣이 사진 알파와 최대로 겹치는 카메라를 거친→고운 순으로 찾는다."""
    rng = np.random.default_rng(seed)
    Vs = V[rng.choice(len(V), min(sub, len(V)), replace=False)]
    shape = alpha.shape
    ys, xs = np.nonzero(alpha)
    tx0, ty0 = xs.mean(), ys.mean()
    # 알파 높이에 물체 높이를 맞춘 초기 스케일
    scale0 = (ys.max() - ys.min()) / max(1e-6, (V[:, 1].max() - V[:, 1].min()))

    best = (-1, None)
    for yaw in np.deg2rad(np.arange(0, 360, 15)):
        for pitch in np.deg2rad([-20, -10, 0, 10, 20]):
            s = iou(close_holes(project(Vs, yaw, pitch, scale0, tx0, ty0, shape)), alpha)
            if s > best[0]:
                best = (s, (yaw, pitch, scale0, tx0, ty0))
    log(f"거친 탐색 IoU {best[0]*100:.1f}% · yaw {np.degrees(best[1][0]):.0f}도 pitch {np.degrees(best[1][1]):.0f}도")

    yaw, pitch, scale, tx, ty = best[1]
    for step_deg, step_s, step_t in [(6, 0.06, 12), (2, 0.02, 4), (1, 0.008, 1.5)]:
        improved = True
        while improved:
            improved = False
            for dy in np.deg2rad([-step_deg, 0, step_deg]):
                for dp in np.deg2rad([-step_deg, 0, step_deg]):
                    for ds in [1 - step_s, 1, 1 + step_s]:
                        for dtx in [-step_t, 0, step_t]:
                            for dty in [-step_t, 0, step_t]:
                                c = (yaw + dy, pitch + dp, scale * ds, tx + dtx, ty + dty)
                                s = iou(close_holes(project(Vs, *c, shape)), alpha)
                                if s > best[0] + 1e-5:
                                    best = (s, c)
                                    improved = True
            yaw, pitch, scale, tx, ty = best[1]
    log(f"정밀 탐색 IoU {best[0]*100:.1f}% · yaw {np.degrees(yaw):.1f}도 pitch {np.degrees(pitch):.1f}도 "
        f"scale {scale:.1f}")
    return best[1], best[0]


def rowwise_fill(rgb, alpha, need):
    """채울 화소의 색을 **같은 행의 가장 가까운 잔존 화소**에서 가져온다.

    이미지 공간의 회전대칭이다. 새 색을 지어내지 않는다.
    """
    out = rgb.copy()
    for row in np.nonzero(need.any(1))[0]:
        src = np.nonzero(alpha[row])[0]
        if len(src) == 0:
            continue
        tgt = np.nonzero(need[row])[0]
        nearest = src[np.abs(tgt[:, None] - src[None, :]).argmin(1)]
        out[row, tgt] = rgb[row, nearest]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True, help="배경 제거된 RGBA (make_mirror_photo.py 산출)")
    ap.add_argument("--damaged", required=True, help="손상 GLB — 카메라 맞추기용")
    ap.add_argument("--restored", required=True, help="복원 GLB — 완형 실루엣용")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.75)
    ap.add_argument("--no-clip-bbox", dest="clip_bbox", action="store_false",
                    help="원본 알파 bbox 밖까지 채운다 (크롭이 달라져 조건-좌표 스케일이 어긋난다)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    im = Image.open(args.photo).convert("RGBA")
    arr = np.array(im)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3] > 96
    log(f"사진 {im.size} · 알파 화소 {int(alpha.sum()):,}")

    Vd, Vr = glb_vertices(args.damaged), glb_vertices(args.restored)
    log(f"손상 메시 {len(Vd):,} · 복원 메시 {len(Vr):,}")

    cam, score = fit_camera(Vd, alpha)
    if score < args.min_iou:
        log(f"경고: 카메라 IoU {score*100:.1f}% < {args.min_iou*100:.0f}%. "
            f"조건 이미지가 사진과 안 맞을 수 있다")

    sil_d = close_holes(project(Vd, *cam, alpha.shape))
    sil_r = close_holes(project(Vr, *cam, alpha.shape))
    log(f"투영 실루엣 · 손상 {int(sil_d.sum()):,} · 복원 {int(sil_r.sum()):,} "
        f"(+{100*(sil_r.sum()-sil_d.sum())/max(1,sil_d.sum()):.1f}%)")

    need = sil_r & (~alpha)

    if args.clip_bbox:
        # preprocess_image 는 알파 bbox 의 max(폭,높이) 로 정사각 크롭한다.
        # 합성으로 bbox 가 커지면 크롭 변이 커지고, 물체가 조건 안에서 상대적으로 작아진다.
        # coords 는 원본 크롭 기준으로 뽑혔으므로 그러면 **조건과 좌표의 스케일이 어긋난다**.
        # v10 이 110조각으로 터진 원인 (크롭 변 407 → 428, +5.2%).
        ys, xs = np.nonzero(alpha)
        keep = np.zeros_like(need)
        keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
        dropped = int((need & ~keep).sum())
        need = need & keep
        log(f"원본 bbox 밖 채움 {dropped:,}화소 제외 → 크롭 변이 원본과 같아진다")

    log(f"채울 화소 {int(need.sum()):,} (사진 대비 +{100*need.sum()/max(1,alpha.sum()):.1f}%)")

    filled_rgb = rowwise_fill(rgb, alpha, need)
    out = np.dstack([filled_rgb, ((alpha | need) * 255).astype(np.uint8)])
    Image.fromarray(out).save(os.path.join(args.out, "composite.png"))

    vis = out.copy()
    vis[need] = (vis[need] * 0.45 + np.array([90, 160, 255, 255]) * 0.55).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(args.out, "composite_marked.png"))
    Image.fromarray((np.dstack([sil_d, sil_r, alpha]) * 255).astype(np.uint8)).save(
        os.path.join(args.out, "silhouettes.png"))     # R=손상투영 G=복원투영 B=사진

    json.dump({"yaw_deg": float(np.degrees(cam[0])), "pitch_deg": float(np.degrees(cam[1])),
               "scale": float(cam[2]), "tx": float(cam[3]), "ty": float(cam[4]),
               "camera_iou": float(score), "filled_px": int(need.sum())},
              open(os.path.join(args.out, "camera.json"), "w"), indent=2)
    log(f"저장 → {args.out}")


if __name__ == "__main__":
    main()
