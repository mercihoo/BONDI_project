"""GLB 를 직접 열어 회전체 비율을 잰다 — 블렌더 없이.

왜 필요한가
-----------
사진에서 실루엣을 뽑으면 **원근에 오염된다.** 카메라가 위에 있으면 아가리가 타원으로
보여 세로가 실제보다 길게 찍히고, 그래서 폭이 좁게 나온다 (측정된 계통 오차 -13~-15%).

3D 메시에는 그 문제가 없다. 투영을 거치지 않았으니 **높이와 지름을 그냥 재면 된다.**
다만 TRELLIS 출력은 정규화돼 있어 **절대 크기는 모른다** — 비율만 준다.
자는 박물관 기록(`높이 15.0cm`)이 댄다.

그래서 이 스크립트가 재는 것은 **비율**이다:

    지름 / 높이        <- 기록의 (입지름 / 높이) 와 맞춰볼 값

사용:
  python measure_glb.py <glb> [<glb> ...] [--height-cm 15.0]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

CT = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: Path):
    b = path.read_bytes()
    if b[:4] != b"glTF":
        raise ValueError("GLB 가 아니다")
    total = struct.unpack_from("<I", b, 8)[0]
    off, js, bin_ = 12, None, None
    while off < total:
        clen, ctype = struct.unpack_from("<II", b, off)
        data = b[off + 8: off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(data.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_ = data
        off += 8 + clen
    return js, bin_


def accessor(js, bin_, i):
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    dt = np.dtype("<" + CT[a["componentType"]])
    n, cnt = NC[a["type"]], a["count"]
    stride = bv.get("byteStride") or dt.itemsize * n
    if stride == dt.itemsize * n:
        return np.frombuffer(bin_, dtype=dt, count=cnt * n, offset=off).reshape(cnt, n)
    raw = np.frombuffer(bin_, dtype=np.uint8, count=stride * cnt, offset=off).reshape(cnt, stride)
    return raw[:, : dt.itemsize * n].copy().view(dt).reshape(cnt, n)


def node_mat(nd):
    if "matrix" in nd:
        return np.array(nd["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    x, y, z, w = nd.get("rotation", [0, 0, 0, 1])
    M[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = M[:3, :3] @ np.diag(nd.get("scale", [1, 1, 1]))
    M[:3, 3] = nd.get("translation", [0, 0, 0])
    return M


def vertices(path: Path):
    js, bin_ = read_glb(path)
    for m in js.get("meshes", []):
        for p in m.get("primitives", []):
            if "KHR_draco_mesh_compression" in p.get("extensions", {}):
                raise ValueError("Draco 압축 — 이 파서로는 못 읽는다")
    out, nodes = [], js.get("nodes", [])

    def walk(i, M):
        nd = nodes[i]
        M = M @ node_mat(nd)
        if "mesh" in nd:
            for p in js["meshes"][nd["mesh"]]["primitives"]:
                if "POSITION" in p["attributes"]:
                    V = accessor(js, bin_, p["attributes"]["POSITION"]).astype(np.float64)
                    out.append(V @ M[:3, :3].T + M[:3, 3])
        for c in nd.get("children", []):
            walk(c, M)

    roots = js["scenes"][js.get("scene", 0)]["nodes"] if js.get("scenes") else range(len(nodes))
    for r in roots:
        walk(r, np.eye(4))
    return np.vstack(out)


def _axis_cost(V, up, bands=32):
    """이 축을 세로로 봤을 때 **반지름이 얼마나 고른가**. 낮을수록 진짜 축이다."""
    ax, c = fit_axis(V, up)
    z = V[:, up]
    span = z.max() - z.min()
    if span <= 0:
        return 1e18
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    b = np.clip(((z - z.min()) / span * bands).astype(int), 0, bands - 1)
    s, n = 0.0, 0
    for k in range(bands):
        m = b == k
        if m.sum() < 40:
            continue
        o = r[m][r[m] >= np.quantile(r[m], 0.60)]      # 바깥 껍질
        s += float(np.percentile(o, 90) - np.percentile(o, 10)) / max(span, 1e-9)
        n += 1
    return s / max(n, 1)


def up_axis(V, sample=40000):
    """회전 대칭축을 고른다.

    두 번 틀렸다.

    1. 처음엔 `argmax(bbox)` 를 썼다. **고배는 높이보다 지름이 큰 것이 흔하다**
       (경주 8477: 높이 7.6cm / 입지름 10.3cm). 가로축이 뽑혔다.
    2. 그 다음엔 '나머지 두 축 길이가 가장 비슷한 축' 으로 바꿨다. 온전한 유물에는
       맞지만 **깨진 유물은 회전 대칭이 아니다.** 경신 71489 에서 Y(0.034) 와
       Z(0.033) 가 사실상 동점이 되어 Z 를 골랐고, 프로파일이 통째로 엉켰다
       (최대지름/높이 1.554 — 실제는 0.98).

    그래서 모양이 아니라 **성질**로 고른다. 회전체의 진짜 축은 **높이대마다 바깥
    반지름이 고르게 나오는 축**이다. 세 축에 각각 `fit_axis` 를 돌려 그 퍼짐을 재고
    가장 작은 것을 쓴다. 깨져 있어도 남은 부분이 여전히 한 원 위에 있으므로 성립한다.
    """
    if len(V) > sample:
        V = V[np.random.default_rng(0).choice(len(V), sample, replace=False)]
    costs = [_axis_cost(V, a) for a in range(3)]
    return int(np.argmin(costs))


def fit_axis(V, up):
    """세로축을 고정하고 (a,b) 중심을 격자 탐색 — 구간마다 반지름이 가장 고른 곳."""
    ax = [i for i in range(3) if i != up]
    z = V[:, up]
    b = np.clip(((z - z.min()) / max(z.max() - z.min(), 1e-9) * 40).astype(int), 0, 39)
    c0 = V[:, ax].mean(0)
    best = tuple(c0)
    span = (z.max() - z.min())
    for step in (span * 0.04, span * 0.008, span * 0.0016):
        cur, bc = None, 1e18
        for i in range(-4, 5):
            for j in range(-4, 5):
                c = (best[0] + i * step, best[1] + j * step)
                r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
                s = 0.0
                for k in range(40):
                    m = b == k
                    if m.sum() < 50:
                        continue
                    rr = r[m]
                    o = rr[rr >= np.quantile(rr, 0.60)]
                    s += float(np.percentile(o, 90) - np.percentile(o, 10))
                if s < bc:
                    bc, cur = s, c
        best = cur
    return ax, np.array(best)


def measure(path: Path, height_cm=None):
    V = vertices(path)
    ext = V.max(0) - V.min(0)
    up = up_axis(V)
    ax, c = fit_axis(V, up)
    z = V[:, up]
    r = np.hypot(V[:, ax[0]] - c[0], V[:, ax[1]] - c[1])
    H = float(z.max() - z.min())
    h = (z - z.min()) / H
    top = h >= 0.92
    D_max = 2 * float(np.quantile(r, 0.999))
    D_rim = 2 * float(np.quantile(r[top], 0.90)) if top.sum() > 50 else float("nan")
    D_foot = 2 * float(np.quantile(r[h <= 0.06], 0.90)) if (h <= 0.06).sum() > 50 else float("nan")
    k = (height_cm / H) if height_cm else None

    print(f"\n=== {path.name} ===")
    print(f"정점 {len(V):,} · 세로축 {'XYZ'[up]} · bbox {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f}")
    print(f"높이 {H:.4f} · 최대지름 {D_max:.4f} · 아가리(위8%) {D_rim:.4f} · 굽(아래6%) {D_foot:.4f}")
    print(f"**최대지름/높이 = {D_max/H:.3f}**   아가리/높이 = {D_rim/H:.3f}")
    if k:
        print(f"기록 높이 {height_cm}cm 로 환산 (1단위={k:.3f}cm):"
              f"  최대지름 {D_max*k:.2f}cm · 아가리 {D_rim*k:.2f}cm · 굽 {D_foot*k:.2f}cm")

    print("\n단면 (왼=축, 오른=바깥)")
    NZ = 24
    ib = np.clip((h * NZ).astype(int), 0, NZ - 1)
    for i in range(NZ - 1, -1, -1):
        m = ib == i
        if not m.any():
            print(f"{i/NZ:5.2f} (없음)"); continue
        rr = 2 * float(np.quantile(r[m], 0.97)) / D_max
        print(f"{i/NZ:5.2f} " + "#" * max(1, int(round(rr * 46))) + f"  {rr:.2f}")
    return dict(H=H, D_max=D_max, D_rim=D_rim, D_foot=D_foot, ratio=D_max / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb", nargs="+")
    ap.add_argument("--height-cm", type=float, default=None)
    a = ap.parse_args()
    res = {}
    for g in a.glb:
        p = Path(g)
        try:
            res[p.name] = measure(p, a.height_cm)
        except Exception as e:
            print(f"\n=== {p.name} ===\n  실패: {e}")
    if len(res) > 1:
        print("\n\n비교 — 비율은 절대 크기와 무관하다")
        print(f"{'파일':<34}{'최대지름/높이':>14}{'아가리/높이':>13}")
        for k, v in res.items():
            print(f"{k:<34}{v['ratio']:>14.3f}{v['D_rim']/v['H']:>13.3f}")


if __name__ == "__main__":
    main()
