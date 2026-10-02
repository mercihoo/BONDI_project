"""LAS 점군 → GLB(POINTS) + 썸네일. local/make_preview.py 의 사본.

Blender 는 LAS 를 못 읽는다. 순수 파이썬으로 좌표를 읽어 glTF POINTS 로 내보내고,
썸네일은 PIL 로 점을 직접 찍는다 — 서버에서도 점군 자료의 미리보기가 나오게 하기 위한 것.
"""
import struct, json, os
from PIL import Image

MAX_POINTS = 300_000


# ───────────────────────── 점군 → GLB (POINTS) ─────────────────────────
def read_las(path):
    """LAS 1.2~1.4, point format 0~10. (x,y,z) 리스트와 (r,g,b) 리스트(없으면 None)."""
    with open(path, "rb") as f:
        h = f.read(375)
        vmaj, vmin = h[24], h[25]
        off = struct.unpack_from("<I", h, 96)[0]
        pfmt = h[104] & 0x3F
        plen = struct.unpack_from("<H", h, 105)[0]
        n = struct.unpack_from("<Q", h, 247)[0] if (vmaj, vmin) >= (1, 4) else struct.unpack_from("<I", h, 107)[0]
        sx, sy, sz, ox, oy, oz = struct.unpack_from("<6d", h, 131)
        f.seek(off)
        data = f.read(n * plen)
    rgb_off = {2: 20, 3: 28, 5: 28, 7: 30, 8: 30, 10: 30}.get(pfmt)
    step = max(1, n // MAX_POINTS)
    xyz, rgb = [], []
    for i in range(0, n, step):
        b = i * plen
        x, y, z = struct.unpack_from("<3i", data, b)
        xyz.append((x * sx + ox, y * sy + oy, z * sz + oz))
        if rgb_off is not None:
            r, g, bl = struct.unpack_from("<3H", data, b + rgb_off)
            rgb.append((r, g, bl))
    if rgb and max(max(c) for c in rgb[:2000]) > 255:
        rgb = [(r >> 8, g >> 8, b >> 8) for r, g, b in rgb]
    # 색 항목은 있는데 전부 0 인 LAS 가 있다(PoinTr 원출력). 그대로 쓰면 썸네일이 새까맣게 나온다 → 색 없음으로 본다
    if rgb and not any(any(c) for c in rgb):
        rgb = None
    return n, xyz, (rgb or None)


def write_glb_points(out, xyz, rgb):
    cx = sum(p[0] for p in xyz) / len(xyz); cy = sum(p[1] for p in xyz) / len(xyz); cz = sum(p[2] for p in xyz) / len(xyz)
    ext = max(max(abs(p[0] - cx), abs(p[1] - cy), abs(p[2] - cz)) for p in xyz) or 1.0
    s = 1.0 / ext                                       # 최대 반경 1 m → 메시와 같은 크기감
    # glTF 는 Y-up. LAS 의 Z 를 Y 로 올린다: (x, z, -y)
    pos = [((p[0] - cx) * s, (p[2] - cz) * s, -(p[1] - cy) * s) for p in xyz]
    pos_bin = struct.pack("<%df" % (3 * len(pos)), *[c for p in pos for c in p])
    mn = [min(p[i] for p in pos) for i in range(3)]; mx = [max(p[i] for p in pos) for i in range(3)]
    col_bin = b""
    if rgb:
        col_bin = struct.pack("<%dB" % (4 * len(rgb)), *[c for r, g, b in rgb for c in (r, g, b, 255)])
    pad = lambda b: b + b"\0" * ((4 - len(b) % 4) % 4)
    pos_bin, col_bin = pad(pos_bin), pad(col_bin)
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bin), "target": 34962}]
    accs = [{"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3", "min": mn, "max": mx}]
    attrs = {"POSITION": 0}
    if rgb:
        views.append({"buffer": 0, "byteOffset": len(pos_bin), "byteLength": len(col_bin), "target": 34962})
        accs.append({"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(rgb), "type": "VEC4"})
        attrs["COLOR_0"] = 1
    gltf = {"asset": {"version": "2.0", "generator": "c201 las_to_glb"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": attrs, "mode": 0}]}],
            "buffers": [{"byteLength": len(pos_bin) + len(col_bin)}], "bufferViews": views, "accessors": accs}
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_ = pos_bin + col_bin
    total = 12 + 8 + len(js) + 8 + len(bin_)
    with open(out, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, total))
        f.write(struct.pack("<I", len(js)) + b"JSON" + js)
        f.write(struct.pack("<I", len(bin_)) + b"BIN\0" + bin_)


def thumb_points(out_png, xyz, rgb, size=512):
    """3/4 시점 직교 투영으로 점을 찍는다."""
    from PIL import Image, ImageDraw
    import math
    cx = sum(p[0] for p in xyz) / len(xyz); cy = sum(p[1] for p in xyz) / len(xyz); cz = sum(p[2] for p in xyz) / len(xyz)
    yaw, pitch = math.radians(45), math.radians(28)
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    pts = []
    for i, (x, y, z) in enumerate(xyz):
        x, y, z = x - cx, y - cy, z - cz
        X = x * cyw + y * syw; Y = -x * syw + y * cyw
        pts.append((X, Y * sp + z * cp, Y * cp - z * sp, i))
    ext = max(max(abs(p[0]), abs(p[1])) for p in pts) or 1.0
    sc = (size * 0.44) / ext
    pts.sort(key=lambda p: -p[2])                       # 먼 점부터
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0)); px = im.load()
    for X, Y, _, i in pts:
        u, v = int(size / 2 + X * sc), int(size / 2 - Y * sc)
        if 0 <= u < size and 0 <= v < size:
            c = rgb[i] if rgb else (170, 170, 170)
            px[u, v] = (c[0], c[1], c[2], 255)
    im.save(out_png, optimize=True)
