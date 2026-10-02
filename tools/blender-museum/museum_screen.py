# Room B folding screen (병풍) — 8 panels, modelled after a real 매화도 screen. Metres, UE axes.
# Origin: centre of the zig-zag chain on the floor; the front (painted) side faces -X (the arrival side).
#
#   SM_RoomB_Screen slots: Frame (dark lacquer rails), Brocade (gold mounting border), Painting (front paper,
#   UV0 = panel i covers u in [i/8, (i+1)/8] of T_RoomB_ScreenPainting), Paper (back board), Brass (hinge plates)
#
# Unlike the other generators this one keeps its own UV0 — the painting must land exactly on the paper —
# and only the lightmap UV1 is auto-unwrapped.
import bpy, bmesh, math, os

OUT = r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
os.makedirs(OUT, exist_ok=True)

PANELS = 8
PW, PH, T = 0.58, 1.80, 0.032         # panel width, height, thickness
RAIL = 0.03                           # frame rail width
BROC = 0.055                          # brocade border (sides)
BROC_TOP, BROC_BOT = 0.20, 0.28       # wider bands above and below the paper, as on real screens
ANGLE = math.radians(13.0)            # zig-zag fold angle
PAPER_U0, PAPER_U1 = RAIL + BROC, PW - RAIL - BROC
PAPER_Z0, PAPER_Z1 = RAIL + BROC_BOT, PH - RAIL - BROC_TOP     # 0.31 .. 1.57 -> 1.26 tall, 0.41 wide (1:3.07)

SLOT_FRAME, SLOT_BROC, SLOT_PAINT, SLOT_PAPER, SLOT_BRASS = range(5)


def chain():
    """Panel origins and (dir, normal) per panel; dir runs along the panel toward +Y, normal toward -X."""
    dirs, norms = [], []
    for i in range(PANELS):
        a = ANGLE if i % 2 == 0 else -ANGLE
        dirs.append((-math.sin(a), math.cos(a)))
        norms.append((-math.cos(a), -math.sin(a)))
    total = [sum(d[0] for d in dirs) * PW, sum(d[1] for d in dirs) * PW]
    p = [-total[0] / 2, -total[1] / 2]
    origins = []
    for d in dirs:
        origins.append(tuple(p))
        p = [p[0] + d[0] * PW, p[1] + d[1] * PW]
    return origins, dirs, norms


def build(bm):
    uv_lay = bm.loops.layers.uv.new("UVMap")
    origins, dirs, norms = chain()

    def W(o, d, n, u, nn, z):
        return (o[0] + d[0]*u + n[0]*nn, o[1] + d[1]*u + n[1]*nn, z)

    def rbox(i, u0, u1, n0, n1, z0, z1, mat, uv=None):
        o, d, n = origins[i], dirs[i], norms[i]
        u0, u1 = sorted((u0, u1)); n0, n1 = sorted((n0, n1)); z0, z1 = sorted((z0, z1))
        c = [W(o, d, n, u, nn, z) for (u, nn, z) in
             [(u0, n0, z0), (u1, n0, z0), (u1, n1, z0), (u0, n1, z0),
              (u0, n0, z1), (u1, n0, z1), (u1, n1, z1), (u0, n1, z1)]]
        vs = [bm.verts.new(p) for p in c]
        # face order: bottom, top, back (n0), side u1, front (n1), side u0
        faces = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
        for k, (a, b, cc, dd) in enumerate(faces):
            f = bm.faces.new((vs[a], vs[b], vs[cc], vs[dd]))
            f.material_index = mat
            for loop in f.loops:
                loop[uv_lay].uv = (0.0, 0.0)
            if uv is not None and k == 4:                       # the front face (max n) carries the painting
                U0, U1 = uv
                for loop in f.loops:
                    idx = vs.index(loop.vert)
                    uu = u1 if idx in (1, 2, 5, 6) else u0
                    zz = z1 if idx >= 4 else z0
                    su = U0 + (U1 - U0) * (uu - PAPER_U0) / (PAPER_U1 - PAPER_U0)
                    sv = (zz - PAPER_Z0) / (PAPER_Z1 - PAPER_Z0)
                    # FBX -> UE mirrors Y, so U runs the other way in the engine; flip here so the
                    # calligraphy lands on the viewer's right (verified 2026-09-16)
                    loop[uv_lay].uv = (1.0 - su, sv)

    for i in range(PANELS):
        # frame rails: full thickness, front at n 0, back at -T
        rbox(i, 0, RAIL, -T, 0, 0, PH, SLOT_FRAME)
        rbox(i, PW - RAIL, PW, -T, 0, 0, PH, SLOT_FRAME)
        rbox(i, RAIL, PW - RAIL, -T, 0, 0, RAIL, SLOT_FRAME)
        rbox(i, RAIL, PW - RAIL, -T, 0, PH - RAIL, PH, SLOT_FRAME)
        # back board (paper back) and brocade mounting on the front
        rbox(i, RAIL, PW - RAIL, -T + 0.002, -0.010, RAIL, PH - RAIL, SLOT_PAPER)
        rbox(i, RAIL, PW - RAIL, -0.010, -0.003, RAIL, PH - RAIL, SLOT_BROC)
        # paper with the painting, 3 mm inset from the brocade face, 1 mm proud of it
        rbox(i, PAPER_U0, PAPER_U1, -0.004, -0.002, PAPER_Z0, PAPER_Z1, SLOT_PAINT,
             uv=(i / PANELS, (i + 1) / PANELS))
        # thin lacquer bead where brocade meets paper
        for (u0, u1, z0, z1) in ((PAPER_U0 - 0.008, PAPER_U0, PAPER_Z0 - 0.008, PAPER_Z1 + 0.008),
                                 (PAPER_U1, PAPER_U1 + 0.008, PAPER_Z0 - 0.008, PAPER_Z1 + 0.008),
                                 (PAPER_U0, PAPER_U1, PAPER_Z0 - 0.008, PAPER_Z0),
                                 (PAPER_U0, PAPER_U1, PAPER_Z1, PAPER_Z1 + 0.008)):
            rbox(i, u0, u1, -0.003, -0.001, z0, z1, SLOT_FRAME)
        # hinge plates on the back face at each joint, two heights
        for z in (0.42, 1.34):
            if i < PANELS - 1:
                rbox(i, PW - 0.045, PW, -T - 0.003, -T, z, z + 0.07, SLOT_BRASS)
            if i > 0:
                rbox(i, 0, 0.045, -T - 0.003, -T, z, z + 0.07, SLOT_BRASS)
        # small foot blocks so the bottom rail reads as standing, not sunk
        rbox(i, 0, 0.08, -T, 0, -0.012, 0.0, SLOT_FRAME)
        rbox(i, PW - 0.08, PW, -T, 0, -0.012, 0.0, SLOT_FRAME)


def finish(name, slots):
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for s in slots:
        ob.data.materials.append(bpy.data.materials.get(s) or bpy.data.materials.new(s))
    bm = bmesh.new()
    build(bm)
    bm.to_mesh(me)
    bm.free()
    me.validate()
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    # lightmap UV1 only — UV0 was written by build()
    me.uv_layers.new(name="Lightmap")
    me.uv_layers.active = me.uv_layers["Lightmap"]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.03)
    bpy.ops.object.mode_set(mode="OBJECT")
    me.uv_layers.active = me.uv_layers["UVMap"]
    bv = ob.modifiers.new("Bevel", "BEVEL")
    bv.width, bv.segments = 0.003, 1
    bv.limit_method, bv.angle_limit = "ANGLE", math.radians(30)
    bpy.ops.object.modifier_apply(modifier=bv.name)
    path = os.path.join(OUT, name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
        use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    return name, len(ob.data.polygons)


print("SCREEN_RESULT", finish("SM_RoomB_Screen", ["Frame", "Brocade", "Painting", "Paper", "Brass"]))
