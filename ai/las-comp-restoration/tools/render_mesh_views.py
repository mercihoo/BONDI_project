#!/usr/bin/env python3
"""메시를 브라우저 없이 PNG 로 — pyrender(EGL) 오프스크린. 칸 여러 개를 같은 시점 4 방향으로 나란히 붙이고 2×2 시트도 만든다.

    python tools/render_mesh_views.py \
        --pane "입력 (손상 원본)=artifacts/gupdari71489/source_model.glb" \
        --pane "LaS-Comp 출력=results/<라벨>/output_mesh.glb@swap_yz" \
        --pane "합성 · 몸통=results/<라벨>/composite_body.glb" \
        --out results/<라벨>/views [--size 900] [--header "..."]

칸: "<캡션>=<GLB>[@<프레임>]" — 프레임은 composite.py 의 후보 이름(same / swap_yz / …). LaS-Comp 출력은 yz 교환 프레임이다.
카메라는 viewer3.html 과 같다 (정면 az35 el18 · 측면 az125 · 후면 az215 · 위 el75 · 거리 2.4 · fov 40°). 각 칸은 bbox 최대 변 1 로 정규화.
이전 render_views.py 는 점 산포(matplotlib)였다 — "점 말고 유물의 모습을" 이라는 요구로 메시 음영으로 바꿨다.
"""
import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# WSL 의 GPU EGL(mesa d3d12 · "D3D12 (Intel Arc)") 은 오프스크린 색 버퍼 읽기가 전부 0 으로 돌아온다 — 배경색조차 없이 새까만 PNG.
# 소프트웨어 렌더러(llvmpipe)로 강제하면 정상. 느리므로 큰 메시는 아래에서 데시메이션한다.
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from composite import CANDS, apply_frame, frame_is_improper  # noqa: E402

VIEWS = [("front", "정면", 35, 18), ("side", "측면", 125, 18), ("back", "후면", 215, 18), ("top", "위에서", 35, 75)]
DIST, FOV = 2.4, 40.0
MAX_FACES = 250_000        # 질감 없는 메시가 이보다 크면 open3d 쿼드릭 데시메이션 (100 만 → 20 만 면에 13 s). 질감 메시는 UV 가 깨지므로 그대로.


def decimate_if_big(m: trimesh.Trimesh) -> trimesh.Trimesh:
    if len(m.faces) <= MAX_FACES or getattr(m.visual, "kind", None) == "texture":
        return m
    import open3d as o3d          # 렌더링 모듈은 WSL 에서 segfault 나지만 기하 연산은 멀쩡하다 — 지연 임포트
    o = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
                                  o3d.utility.Vector3iVector(np.asarray(m.faces)))
    d = o.simplify_quadric_decimation(MAX_FACES)
    out = trimesh.Trimesh(np.asarray(d.vertices), np.asarray(d.triangles), process=False)
    if getattr(m.visual, "kind", None) == "vertex":
        # 정점 색은 데시메이션에서 사라진다 (처음엔 이걸 놓쳐 모델 예측 색이 하얗게 렌더됐다).
        # 새 정점마다 원래 메시의 가장 가까운 정점 색을 가져온다 — 정점이 촘촘해 최근접이면 충분하다.
        from scipy.spatial import cKDTree
        idx = cKDTree(np.asarray(m.vertices)).query(np.asarray(out.vertices))[1]
        out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=np.asarray(m.visual.vertex_colors)[idx])
    else:
        mat = getattr(m.visual, "material", None)
        if mat is not None:
            out.visual = trimesh.visual.TextureVisuals(material=mat)   # 단색 PBR 재질은 그대로 물려준다 (UV 는 없음)
    print("  데시메이션 %d → %d 면" % (len(m.faces), len(out.faces)))
    return out
BG = (0.11, 0.11, 0.125)
DEFAULT_MAT = dict(baseColorFactor=[0.79, 0.76, 0.71, 1.0], roughnessFactor=0.75, metallicFactor=0.0)


def font(size=22):
    for p in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_pane(path: str, frame: str):
    """GLB 의 지오메트리를 노드 변환 펼친 trimesh 목록으로. 프레임 후보를 적용하고 전체 bbox 로 정규화(viewer 와 같게)."""
    sc = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        g = g.copy()
        g.apply_transform(T)
        if frame != "same":
            cand = CANDS[frame]
            v = apply_frame(np.asarray(g.vertices, dtype=np.float64), cand)
            f = g.faces[:, [0, 2, 1]] if frame_is_improper(cand) else g.faces      # 거울이면 감김을 뒤집어 법선을 지킨다
            g.vertices, g.faces = v, f
        meshes.append(decimate_if_big(g))
    if not meshes:
        sys.exit("메시 없음: %s" % path)
    allv = np.vstack([m.vertices for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max()
    for m in meshes:
        m.apply_translation(-c)
        m.apply_scale(1.0 / s)
    return meshes


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = r, u, -f, eye
    return M


def cam_pose(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = (DIST * math.cos(el) * math.sin(az), DIST * math.sin(el), DIST * math.cos(el) * math.cos(az))
    return look_at(eye)


def to_pyrender(m: trimesh.Trimesh):
    kind = getattr(m.visual, "kind", None)
    if kind == "texture":
        mat = getattr(m.visual, "material", None)
        if getattr(mat, "baseColorTexture", None) is not None or getattr(mat, "image", None) is not None:
            return pyrender.Mesh.from_trimesh(m, smooth=False)                  # 진짜 질감
        rgba = getattr(mat, "baseColorFactor", None)                            # 단색 PBR (합성 채움)
        color = [c / 255.0 for c in rgba] if rgba is not None and max(rgba) > 1 else (list(rgba) if rgba is not None else DEFAULT_MAT["baseColorFactor"])
        return pyrender.Mesh.from_trimesh(m, smooth=True, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color, roughnessFactor=0.85, metallicFactor=0.0))
    mat = None
    if kind != "vertex":
        mat = pyrender.MetallicRoughnessMaterial(**DEFAULT_MAT)
    return pyrender.Mesh.from_trimesh(m, smooth=True, material=mat)


def render_pane(meshes, size, poses):
    # 첫 판(키 3.2 · 주변 0.45)은 단색 메시가 하얗게 날아갔다 — 브라우저 뷰어의 음영에 맞춰 낮춘다
    scene = pyrender.Scene(bg_color=list(BG) + [1.0], ambient_light=[0.28, 0.28, 0.30])
    for m in meshes:
        scene.add(to_pyrender(m))
    cam = pyrender.PerspectiveCamera(yfov=math.radians(FOV), znear=0.01, zfar=100)
    cam_node = scene.add(cam, pose=np.eye(4))
    scene.add(pyrender.DirectionalLight(intensity=1.8), pose=look_at((2, 3, 2)))
    scene.add(pyrender.DirectionalLight(intensity=0.6), pose=look_at((-3, 1, -2)))
    r = pyrender.OffscreenRenderer(size, size)
    out = []
    for pose in poses:
        scene.set_pose(cam_node, pose)
        img, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        out.append(Image.fromarray(img))
    r.delete()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pane", action="append", required=True, help='"<캡션>=<GLB>[@<프레임>]"')
    ap.add_argument("--out", required=True, help="출력 접두 (…_front.png 등)")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--header", default="")
    ap.add_argument("--views", default="", help="그릴 방향만 쉼표로 (front,side,back,top). 비우면 넷 다 — 쓸어보기에 유용.")
    a = ap.parse_args()

    panes = []
    for spec in a.pane:
        cap, rest = spec.split("=", 1)
        path, frame = (rest.split("@", 1) + ["same"])[:2]
        if frame not in CANDS:
            sys.exit("모르는 프레임 %s (가능: %s)" % (frame, ", ".join(CANDS)))
        panes.append((cap, path, frame))

    views = VIEWS
    if a.views:
        want = [w.strip() for w in a.views.split(",") if w.strip()]
        views = [v for v in VIEWS if v[0] in want]
        if not views:
            sys.exit("--views 에 아는 방향이 없다: %s (가능: %s)" % (a.views, ",".join(v[0] for v in VIEWS)))
    poses = [cam_pose(az, el) for _, _, az, el in views]
    rendered = []          # 칸별 4 장
    for cap, path, frame in panes:
        rendered.append(render_pane(load_pane(path, frame), a.size, poses))
        print("렌더:", cap, "←", path, "[%s]" % frame)

    f22, f18 = font(22), font(18)
    GAP, HEAD = 12, 40
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sheets = []
    for vi, (key, ko, _, _) in enumerate(views):
        w = len(panes) * a.size + GAP * (len(panes) - 1)
        im = Image.new("RGB", (w, a.size + HEAD), (28, 28, 32))
        dr = ImageDraw.Draw(im)
        for pi, (cap, _, _) in enumerate(panes):
            x = pi * (a.size + GAP)
            im.paste(rendered[pi][vi], (x, HEAD))
            tw = dr.textlength(cap, font=f18)
            dr.rectangle([x + 8, HEAD + a.size - 36, x + 8 + tw + 16, HEAD + a.size - 8], fill=(0, 0, 0, 160))
            dr.text((x + 16, HEAD + a.size - 32), cap, fill=(235, 235, 235), font=f18)
        head = ko + (" · " + a.header if a.header else "")
        dr.text((10, 8), head, fill=(235, 235, 235), font=f22)
        p = out.parent / (out.name + "_%s.png" % key)
        im.save(p); sheets.append(im); print("저장:", p, im.size)
    w, h = sheets[0].size
    sheet = Image.new("RGB", (2 * w + 10, 2 * h + 10), (18, 18, 20))
    for i, im in enumerate(sheets):
        sheet.paste(im, ((i % 2) * (w + 10), (i // 2) * (h + 10)))
    p = out.parent / (out.name + "_sheet.png")
    sheet.save(p); print("저장:", p, sheet.size)


if __name__ == "__main__":
    main()
