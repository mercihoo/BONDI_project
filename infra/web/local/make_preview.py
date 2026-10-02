"""Phase 2 — 3D 카드마다 경량 GLB + 썸네일 PNG 를 만든다 (로컬 GPU/CPU).

catalog.db 의 3D 카드(assets.media_type='3d')를 돌며 대표 파일을 고른다.
  OBJ(모델) > PLY > STL   → Blender (blender_preview.py)
  LAS(점군)                → 순수 파이썬 las_to_glb + PIL 썸네일
출력: <preview_dir>/<artifact_id>/<variant>.glb, .png   (scan.py 가 이 경로를 보고 preview_path 를 채운다)

  python make_preview.py [--db ~/c201-assets/catalog.db] [--only <asset_id 부분문자열>] [--limit N] [--force]
"""
import argparse, json, os, re, sqlite3, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get("BLENDER", r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
TARGET_TRIS = 200_000
MAX_POINTS = 300_000
PICK = {"obj": 0, "glb": 1, "ply": 2, "stl": 3, "las": 4}


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


# ───────────────────────── 드라이버 ─────────────────────────
def pick_file(cx, asset_id):
    rows = cx.execute("SELECT path, kind, filename FROM files WHERE asset_id=? AND kind IN ('obj','glb','ply','stl','las')", (asset_id,)).fetchall()
    if not rows:
        return None
    rows.sort(key=lambda r: (PICK[r[1]], 0 if "model" in r[2].lower() else 1, r[2]))
    return rows[0]


# 위 축 추정이 틀리는 것(바닥 없는 불두·금관, 정육면체 화로 등)은 여기서 지정한다.  {asset_id: {"up": "X|Y|Z", "flip": true|false}}
_ov = os.path.join(HERE, "orientation_overrides.json")
ORIENT = {k: v for k, v in json.load(open(_ov, encoding="utf-8")).items() if not k.startswith("_")} if os.path.exists(_ov) else {}


def run_blender(src, out_glb, out_png, asset_id=None):
    o = ORIENT.get(asset_id or "", {})
    cmd = [BLENDER, "-b", "--python-exit-code", "1", "-P", os.path.join(HERE, "blender_preview.py"), "--",
           src, out_glb, out_png, str(TARGET_TRIS),
           "--up", str(o.get("up", "auto")), "--flip", ("1" if o["flip"] else "0") if "flip" in o else "auto"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    m = re.search(r"PREVIEW_REPORT (\{.*\})", p.stdout)
    if p.returncode != 0 or not m or not os.path.exists(out_glb):
        tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
        raise RuntimeError("blender rc=%d: %s" % (p.returncode, " | ".join(tail)))
    return json.loads(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/c201-assets/catalog.db"))
    ap.add_argument("--root", default=os.path.expanduser("~/c201-assets/staging"))
    ap.add_argument("--preview", default=os.path.expanduser("~/c201-assets/preview"))
    ap.add_argument("--only"); ap.add_argument("--limit", type=int); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cx = sqlite3.connect(a.db)
    assets = cx.execute("SELECT asset_id, artifact_id, variant, title_ko FROM assets WHERE media_type='3d' ORDER BY asset_id").fetchall()
    if a.only:
        assets = [x for x in assets if a.only in x[0]]
    if a.limit:
        assets = assets[:a.limit]
    log = open(os.path.join(a.preview, "_preview_log.jsonl"), "a", encoding="utf-8") if os.path.isdir(a.preview) or os.makedirs(a.preview) or True else None
    ok = skip = fail = 0
    t_all = time.time()
    for k, (aid, art, var, title) in enumerate(assets, 1):
        od = os.path.join(a.preview, art); os.makedirs(od, exist_ok=True)
        out_glb, out_png = os.path.join(od, var + ".glb"), os.path.join(od, var + ".png")
        if not a.force and os.path.exists(out_glb) and os.path.exists(out_png):
            skip += 1; continue
        pf = pick_file(cx, aid)
        if not pf:
            print("[%3d/%d] %-52s 대표 파일 없음" % (k, len(assets), aid), flush=True); fail += 1; continue
        src = os.path.join(a.root, pf[0]); t0 = time.time(); rep = {"asset_id": aid, "source": pf[0]}
        try:
            if pf[1] == "las":
                n, xyz, rgb = read_las(src)
                write_glb_points(out_glb, xyz, rgb); thumb_points(out_png, xyz, rgb)
                rep.update(pt_count=n, preview_pt_count=len(xyz))
            else:
                rep.update(run_blender(src, out_glb, out_png, aid))
            rep.update(glb_mb=round(os.path.getsize(out_glb) / 1048576, 2), sec=round(time.time() - t0, 1)); ok += 1
            orient = (" up=%s%s%s" % (rep["up"], "↓" if rep.get("flip") else "", "*" if rep.get("override") else "")) if "up" in rep else ""
            print("[%3d/%d] %-52s %5.1fs  GLB %6.2f MB  %s%s" % (k, len(assets), aid, rep["sec"], rep["glb_mb"],
                  ("tri %s→%s" % (rep.get("tri_before"), rep.get("tri_after"))) if "tri_after" in rep else "pts %s→%s" % (rep.get("pt_count"), rep.get("preview_pt_count")), orient), flush=True)
        except Exception as e:
            rep["error"] = str(e)[:300]; fail += 1
            print("[%3d/%d] %-52s 실패: %s" % (k, len(assets), aid, str(e)[:120]), flush=True)
            for p in (out_glb, out_png):
                if os.path.exists(p): os.remove(p)
        log.write(json.dumps(rep, ensure_ascii=False) + "\n"); log.flush()
    print("\n완료 %d · 건너뜀 %d · 실패 %d · %.1f분 → %s" % (ok, skip, fail, (time.time() - t_all) / 60, a.preview), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
