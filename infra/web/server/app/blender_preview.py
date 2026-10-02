"""Blender 안에서 실행. 메시 1개 → 경량 GLB + 썸네일 PNG.

(server/app/ 사본 — 업로드된 복원본의 미리보기를 서버에서 만들 때 쓴다. 원본은 local/blender_preview.py)

  blender -b -P blender_preview.py -- <입력.obj|.ply|.stl> <출력.glb> <썸네일.png> <목표삼각형수> [--up auto|X|Y|Z] [--flip auto|0|1] [--yaw 도]

방향(위 축) 결정 — 박물관 OBJ 는 파일마다 위 축이 제멋대로다 (같은 스캐너 머리말인데 Z-up 도 Y-up 도 있다. 2026-09-08 148건 육안 확인).
그래서 머리말을 믿지 않고 **형상으로 추정**한다:
  · 유물은 평평한 바닥으로 서 있다 → 어느 축의 끝면에 정점이 몰려 있는지(바닥 슬랩 비율)
  · 병·항아리·향로·불상은 위 축 둘레 단면이 둥글다 → 나머지 두 축 길이가 비슷한 축
  표본 25건에서 21건 적중. 못 맞추는 건 바닥이 없는 것(불두·금관)과 정육면체(화로) → orientation_overrides.json 으로 지정.
그 다음: 바운딩박스 중심을 원점, 최대 변 2 m, 목표 삼각형 수로 데시메이션(UV 보존), 텍스처 2048, GLB(JPEG 85), 썸네일 512px.
마지막 줄 PREVIEW_REPORT {...} 를 make_preview.py 가 읽는다.
"""
import bpy, sys, math, pathlib, json
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:]
src, out_glb, out_png, target_tris = argv[0], argv[1], argv[2], int(argv[3])
opt = dict(zip(argv[4::2], argv[5::2]))
want_up, want_flip = opt.get("--up", "auto").upper(), opt.get("--flip", "auto")
# 세로축 둘레 회전(도). up/flip 은 축만 고르므로 "바로 섰지만 옆을 보고 있는" 것은 이걸로 맞춘다.
try:
    yaw_deg = float(opt.get("--yaw", "0"))
except ValueError:
    yaw_deg = 0.0
src_path = pathlib.Path(src)

bpy.ops.wm.read_factory_settings(use_empty=True)

# 축 변환 없이 파일 좌표 그대로 읽는다 (방향은 아래에서 형상으로 정한다)
ext = src_path.suffix.lower()
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath=str(src_path), forward_axis="Y", up_axis="Z")
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=str(src_path))
elif ext == ".stl":
    bpy.ops.wm.stl_import(filepath=str(src_path), forward_axis="Y", up_axis="Z")
elif ext == ".glb":
    # glTF 는 규격이 Y-up 이고 임포터가 블렌더 Z-up 으로 바꿔 준다.
    # 아래 형상 추정은 그 상태에서 돌아 보통 Z 를 고르므로 결과적으로 회전이 없다.
    bpy.ops.import_scene.gltf(filepath=str(src_path))
else:
    raise SystemExit("unsupported: %s" % ext)

meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("nothing imported")
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# ───────────────────────────── 위 축 추정 ─────────────────────────────
def guess_orientation(mesh, max_pts=120_000):
    vs = mesh.vertices
    step = max(1, len(vs) // max_pts)
    pts = [vs[i].co for i in range(0, len(vs), step)]
    n = len(pts)
    mn = [min(p[k] for p in pts) for k in range(3)]
    mx = [max(p[k] for p in pts) for k in range(3)]
    extn = [max(mx[k] - mn[k], 1e-9) for k in range(3)]
    flat = {}
    for k in range(3):
        t = 0.015 * extn[k]
        flat[(k, -1)] = sum(1 for p in pts if p[k] < mn[k] + t) / n      # 아래 끝 슬랩에 든 정점 비율
        flat[(k, +1)] = sum(1 for p in pts if p[k] > mx[k] - t) / n      # 위 끝
    circ = {k: min(extn[(k + 1) % 3], extn[(k + 2) % 3]) / max(extn[(k + 1) % 3], extn[(k + 2) % 3]) for k in range(3)}
    score = {k: max(flat[(k, -1)], flat[(k, +1)]) * (0.5 + circ[k]) for k in range(3)}
    up = max(score, key=score.get)
    flip = flat[(up, +1)] > flat[(up, -1)]          # 정점이 몰린(평평한) 끝이 바닥. 그게 +끝이면 뒤집는다
    return up, flip, {"flat": {"XYZ"[k]: [round(flat[(k, -1)], 3), round(flat[(k, 1)], 3)] for k in range(3)},
                      "circ": {"XYZ"[k]: round(circ[k], 2) for k in range(3)}}


auto_up, auto_flip, evidence = guess_orientation(obj.data)
up = "XYZ".index(want_up) if want_up in ("X", "Y", "Z") else auto_up
flip = (want_flip == "1") if want_flip in ("0", "1") else auto_flip

# 선택한 축을 +Z 로, 바닥이 아래로 가게 회전
if up == 2:
    R = Matrix.Rotation(math.pi, 4, "X") if flip else Matrix.Identity(4)
elif up == 1:                                       # +Y → +Z : X 축 +90°   (뒤집으면 -90°)
    R = Matrix.Rotation(-math.pi / 2 if flip else math.pi / 2, 4, "X")
else:                                               # +X → +Z : Y 축 -90°   (뒤집으면 +90°)
    R = Matrix.Rotation(math.pi / 2 if flip else -math.pi / 2, 4, "Y")
if yaw_deg:
    R = Matrix.Rotation(math.radians(yaw_deg), 4, "Z") @ R     # 세운 뒤 Z 둘레로 돌린다
obj.matrix_world = R @ obj.matrix_world
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ───────────────────────────── 정규화 ─────────────────────────────
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
obj.location = (0, 0, 0)
dims = max(obj.dimensions)
if dims > 0:
    s = 2.0 / dims
    obj.scale = (s, s, s)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# 데시메이션
tri_before = sum(len(p.vertices) - 2 for p in obj.data.polygons)
if tri_before > target_tris > 0:
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = target_tris / tri_before
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)
tri_after = sum(len(p.vertices) - 2 for p in obj.data.polygons)

# 컷아웃 마스크를 색으로 잘못 쓰는 경우 바로잡기
# 금관(bon009435) MTL 은 같은 파일을 map_Kd(색) 와 map_d(투명도) 양쪽에 물려 놨다(Cinema 4D 내보내기).
# 그런데 그 파일은 8192² **완전 무채색**(평균 채도 0.0, 2026-09-21 실측) — 색 사진이 아니라
# 투창 구멍을 뚫는 컷아웃 마스크다. 그대로 두면 색 자리에 흑백 무늬가 찍힌다.
# 규칙: 베이스 컬러와 알파가 같은 이미지이고 그 이미지가 무채색이면 → 색이 아니라 마스크로 보고
#       알파에만 남기고 베이스 컬러는 단색으로 돌린다. (알파는 끊지 않는다 — 구멍은 살려야 한다.)
def _img_of(sock):
    if not sock.is_linked:
        return None
    n = sock.links[0].from_node
    while n and n.type != "TEX_IMAGE":
        ins = [i for i in n.inputs if i.is_linked]
        if not ins:
            return None
        n = ins[0].links[0].from_node
    return getattr(n, "image", None)


def _is_greyscale(img, thr=2.0):
    """작은 사본을 떠서 평균 채도를 잰다 (8192² 원본을 직접 읽으면 느리다)."""
    try:
        c = img.copy(); c.scale(32, 32)
        px = list(c.pixels)
        bpy.data.images.remove(c)
    except Exception:
        return False
    n = len(px) // 4
    if not n:
        return False
    sat = sum(max(px[i*4:i*4+3]) - min(px[i*4:i*4+3]) for i in range(n)) / n
    return sat * 255.0 < thr


mask_fixed = []
for mat in list(obj.data.materials):
    if mat is None or not mat.use_nodes:
        continue
    for node in mat.node_tree.nodes:
        base = node.inputs.get("Base Color") if hasattr(node, "inputs") else None
        alpha = node.inputs.get("Alpha") if hasattr(node, "inputs") else None
        if base is None or alpha is None or not (base.is_linked and alpha.is_linked):
            continue
        bi, ai = _img_of(base), _img_of(alpha)
        if bi is None or bi is not ai or not _is_greyscale(bi):
            continue
        for link in list(base.links):                      # 색에서만 떼어내고 알파는 그대로
            mat.node_tree.links.remove(link)
        base.default_value = (0.8, 0.8, 0.8, 1.0)
        mask_fixed.append("%s:%s" % (mat.name, bi.name))
    if mask_fixed:
        for attr, val in (("blend_method", "CLIP"), ("shadow_method", "CLIP")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except TypeError:
                    pass

# 텍스처 축소
tex_n = 0
for img in bpy.data.images:
    if img.size[0] and (img.size[0] > 2048 or img.size[1] > 2048):
        img.scale(min(img.size[0], 2048), min(img.size[1], 2048))
    if img.size[0]:
        tex_n += 1

# GLB — 익스포터 옵션이 버전마다 달라 없는 옵션은 빼고 내보낸다
has_vcol = bool(obj.data.color_attributes)
opts = dict(filepath=out_glb, export_format="GLB", export_apply=True, export_yup=True,
            export_image_format="JPEG", export_jpeg_quality=85, use_selection=False)
if has_vcol:
    opts["export_vertex_color"] = "ACTIVE"
props = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
bpy.ops.export_scene.gltf(**{k: v for k, v in opts.items() if k in props})

# 썸네일 — Workbench 스튜디오 조명, 3/4 시점
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
has_uv = bool(obj.data.uv_layers) and tex_n > 0
scene.display.shading.color_type = ("MATERIAL" if mask_fixed else
                                   ("TEXTURE" if has_uv else ("VERTEX" if has_vcol else "MATERIAL")))
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 50
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam); scene.camera = cam
cam.location = (2.7, -2.7, 1.9)
cam.rotation_euler = (math.radians(62), 0, math.radians(45))
# 블렌더 5 의 기본 색 변환은 AgX(필름 톤) 라 어두운 유물(청동·나전·철제)이 형체를 알아볼 수 없게 나온다.
# 스캔 텍스처에는 촬영 조명이 이미 구워져 있으므로 톤매핑 없이 그대로 보여주는 Standard 가 맞다.
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.5          # 소장품 사진과 맞춘 값. 백자에서도 흰끝이 날아가지 않는다(실측 0.0%)
scene.render.filepath = out_png
bpy.ops.render.render(write_still=True)

print("PREVIEW_REPORT " + json.dumps({"tri_before": tri_before, "tri_after": tri_after, "textures": tex_n,
                                      "vertex_color": has_vcol, "uv": has_uv, "mask_fixed": mask_fixed,
                                      "up": "XYZ"[up], "flip": bool(flip), "yaw": yaw_deg, "auto_up": "XYZ"[auto_up], "auto_flip": bool(auto_flip),
                                      "override": want_up != "AUTO" or want_flip != "auto" or bool(yaw_deg), "evidence": evidence}))
