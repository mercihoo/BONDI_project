# Round 2 props — the pieces still built from primitives.
# Metres, UE axes (X forward, Y right, Z up). See tools/blender-museum/README.md.
#   SM_Chandelier      hall, hangs at (3.30, 0), ceiling 3.20
#   SM_Sofa            hall, two of them at (3.30, +-3.80)
#   SM_FeatureWall     hall, on the west wall face (x = 0), centred z 2.00
#   SM_RoomA_Backdrop  Room A, strata wall at local x 1.63, y +-2.20
#   SM_RoomA_Dais      Room A, excavation dais under the plinth at local (0, 0)
import bpy, bmesh, math, os

OUT = r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- primitives
def box(bm, x0, x1, y0, y1, z0, z1, mat):
    vs = [bm.verts.new(v) for v in [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    for a, b, c, d in [(0,3,2,1), (4,5,6,7), (0,1,5,4),
                       (1,2,6,5), (2,3,7,6), (3,0,4,7)]:
        bm.faces.new((vs[a], vs[b], vs[c], vs[d])).material_index = mat

def cone(bm, cx, cy, r0, r1, z0, z1, mat, n=20):
    """Truncated cone; r1 == 0 gives a point, r0 == r1 a cylinder."""
    def ring_verts(r, z):
        if r <= 1e-6:
            return None
        return [bm.verts.new((cx + r*math.cos(2*math.pi*i/n),
                              cy + r*math.sin(2*math.pi*i/n), z)) for i in range(n)]
    lo, hi = ring_verts(r0, z0), ring_verts(r1, z1)
    if lo is None or hi is None:
        tip = bm.verts.new((cx, cy, z1 if hi is None else z0))
        base = lo if hi is None else hi
        for i in range(n):
            bm.faces.new((base[i], base[(i+1) % n], tip)).material_index = mat
        bm.faces.new(list(reversed(base)) if hi is None else base).material_index = mat
        return
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i])).material_index = mat
    bm.faces.new(list(reversed(lo))).material_index = mat
    bm.faces.new(hi).material_index = mat

def annulus(bm, cx, cy, ri, ro, z0, z1, mat, n=32):
    """A flat ring — hoops, kerbs, canopies."""
    def rv(r, z):
        return [bm.verts.new((cx + r*math.cos(2*math.pi*i/n),
                              cy + r*math.sin(2*math.pi*i/n), z)) for i in range(n)]
    ilo, olo, ihi, ohi = rv(ri, z0), rv(ro, z0), rv(ri, z1), rv(ro, z1)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((olo[i], olo[j], ohi[j], ohi[i])).material_index = mat   # outside
        bm.faces.new((ihi[i], ihi[j], ilo[j], ilo[i])).material_index = mat   # inside
        bm.faces.new((ilo[i], ilo[j], olo[j], olo[i])).material_index = mat   # bottom
        bm.faces.new((ohi[i], ohi[j], ihi[j], ihi[i])).material_index = mat   # top
    return

def jitter(i, lo, hi):
    """Deterministic pseudo-random in [lo, hi) — keeps re-exports identical."""
    t = (math.sin(i * 12.9898) * 43758.5453) % 1.0
    return lo + t * (hi - lo)

# ---------------------------------------------------------------- build/export
def unwrap(ob):
    """UV0 for materials, UV1 for the lightmap. Lightmass needs both — see D-55."""
    me = ob.data
    for layer in list(me.uv_layers):
        me.uv_layers.remove(layer)
    me.uv_layers.new(name="UVMap")
    me.uv_layers.new(name="Lightmap")
    for name, margin in (("UVMap", 0.02), ("Lightmap", 0.05)):
        me.uv_layers.active = me.uv_layers[name]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin)
        bpy.ops.object.mode_set(mode="OBJECT")
    me.uv_layers.active = me.uv_layers["UVMap"]

def finish(name, slots, build, bevel=0.012, segments=2):
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
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    unwrap(ob)
    bv = ob.modifiers.new("Bevel", "BEVEL")
    bv.width, bv.segments = bevel, segments
    bv.limit_method, bv.angle_limit = "ANGLE", math.radians(30)
    bpy.ops.object.modifier_apply(modifier=bv.name)

    path = os.path.join(OUT, name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
        use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    return name, len(ob.data.polygons)

# ================================================================ chandelier
# Origin at floor level so it drops in at the actor's (3.30, 0, 0).
# Slots: 0 metal, 1 candle, 2 glow
def chandelier(bm):
    CEIL, HOOP_R = 3.20, 0.55
    annulus(bm, 0, 0, 0.0, 0.09, CEIL - 0.06, CEIL, 0)          # canopy
    cone(bm, 0, 0, 0.020, 0.020, 2.84, CEIL - 0.06, 0)          # stem
    cone(bm, 0, 0, 0.075, 0.055, 2.74, 2.88, 0)                 # hub
    cone(bm, 0, 0, 0.055, 0.0, 2.62, 2.74, 0)                   # finial
    annulus(bm, 0, 0, HOOP_R - 0.035, HOOP_R, 2.70, 2.74, 0)    # hoop
    for i in range(8):
        a = 2 * math.pi * i / 8
        cx, cy = math.cos(a), math.sin(a)
        # spoke from hub to hoop, as a thin slab rotated into place
        for t in range(6):                                       # 6 links = gentle arc
            f0, f1 = t / 6, (t + 1) / 6
            r0, r1 = 0.07 + f0 * (HOOP_R - 0.07), 0.07 + f1 * (HOOP_R - 0.07)
            z0 = 2.80 - 0.06 * math.sin(math.pi * f0)
            z1 = 2.80 - 0.06 * math.sin(math.pi * f1)
            cone(bm, cx * (r0 + r1) / 2, cy * (r0 + r1) / 2, 0.016, 0.016,
                 min(z0, z1) - 0.008, max(z0, z1) + 0.008, 0, n=8)
        cone(bm, cx * HOOP_R, cy * HOOP_R, 0.040, 0.034, 2.74, 2.80, 0)   # cup
        cone(bm, cx * HOOP_R, cy * HOOP_R, 0.026, 0.023, 2.80, 2.96, 1)   # candle
        cone(bm, cx * HOOP_R, cy * HOOP_R, 0.018, 0.0, 2.96, 3.03, 2)     # flame
    cone(bm, 0, 0, 0.030, 0.026, 2.88, 3.02, 1)                 # centre candle
    cone(bm, 0, 0, 0.020, 0.0, 3.02, 3.09, 2)

# ================================================================ sofa
# 1.94 x 0.88 x 0.78, origin centred on the floor. Slots: 0 fabric, 1 wood
def sofa(bm):
    W, D = 0.97, 0.44                       # half width / half depth
    for sx in (-1, 1):                      # feet
        for sy in (-1, 1):
            box(bm, sx*W - sx*0.16, sx*W - sx*0.06, sy*D - sy*0.14, sy*D - sy*0.06, 0.0, 0.07, 1)
    box(bm, -W + 0.02, W - 0.02, -D + 0.02, D - 0.02, 0.07, 0.17, 1)      # plinth
    for i in range(3):                                                     # seat cushions
        a = -W + 0.20 + i * ((2*W - 0.40) / 3) + 0.008
        b = a + (2*W - 0.40) / 3 - 0.016
        box(bm, a, b, -D + 0.06, D - 0.20, 0.17, 0.36, 0)
    for i in range(3):                                                     # back cushions
        a = -W + 0.20 + i * ((2*W - 0.40) / 3) + 0.008
        b = a + (2*W - 0.40) / 3 - 0.016
        box(bm, a, b, D - 0.30, D - 0.12, 0.34, 0.70, 0)
    box(bm, -W + 0.20, W - 0.20, D - 0.12, D, 0.17, 0.74, 0)               # back panel
    for sx in (-1, 1):                                                     # arms
        box(bm, sx*W - sx*0.20, sx*W, -D, D, 0.17, 0.62, 0)

# ================================================================ feature wall
# A recessed shadow box with a stepped frame, hung on the west wall (x = 0).
# Depth grows along +X into the room. Slots: 0 frame, 1 panel, 2 plate
def feature_wall(bm):
    # Bottom clears the chair rail (top 1.10, proud 0.07); top clears the cove
    # bracket (starts 2.88). The frame is only 0.14 deep, so it must not overlap either.
    HW, Z0, Z1 = 1.72, 1.18, 2.84        # half width, bottom, top
    box(bm, 0.00, 0.05, -HW + 0.12, HW - 0.12, Z0 + 0.12, Z1 - 0.12, 1)   # recessed panel
    for d0, d1, inset in ((0.00, 0.06, 0.12), (0.06, 0.11, 0.06), (0.11, 0.14, 0.00)):
        h = HW - inset
        box(bm, d0, d1, -h, h, Z1 - 0.12 + inset*0.5, Z1 - 0.12 + inset*0.5 + 0.06, 0)  # head
        box(bm, d0, d1, -h, h, Z0 + 0.06 - inset*0.5, Z0 + 0.12 - inset*0.5, 0)         # sill
        for sy in (-1, 1):
            box(bm, d0, d1, sy*h - sy*0.06, sy*h, Z0 + 0.06, Z1 - 0.06, 0)              # jambs
    box(bm, 0.05, 0.07, -0.28, 0.28, Z0 - 0.06, Z0 + 0.04, 2)             # brass label plate

# ================================================================ Room A backdrop
# Excavated strata section, 4.40 wide x 3.00 tall, face pointing -X into the room.
# Slots: 0 strata, 1 coping
def roomA_backdrop(bm):
    HW, T = 2.20, 0.20
    box(bm, 0.0, T, -HW, HW, 0.0, 3.00, 0)                       # backing wall
    layers = [(0.00, 0.52), (0.52, 1.10), (1.10, 1.74), (1.74, 2.28), (2.28, 2.86)]
    SEG = 22
    for li, (lz0, lz1) in enumerate(layers):
        proud = 0.03 + 0.035 * ((li * 7) % 5) / 4.0              # each stratum steps out differently
        for s in range(SEG):
            a = -HW + (2*HW) * s / SEG
            b = -HW + (2*HW) * (s + 1) / SEG
            top = lz1 + jitter(li * 100 + s, -0.035, 0.035)      # ragged bedding plane
            box(bm, -proud, 0.0, a + 0.004, b - 0.004, lz0, top, 0)
    box(bm, -0.10, T + 0.04, -HW - 0.04, HW + 0.04, 2.94, 3.06, 1)   # coping
    box(bm, -0.02, T, -HW, HW, 0.0, 0.05, 1)                          # plinth toe

# ================================================================ Room A dais
# Low excavation dais the plinth stands on: stone kerb ring, sand fill, gravel apron.
# Slots: 0 kerb, 1 sand, 2 gravel
def roomA_dais(bm):
    annulus(bm, 0, 0, 1.72, 1.78, 0.0, 0.015, 2)     # outer gravel edge
    cone(bm, 0, 0, 1.72, 1.72, 0.0, 0.012, 2, n=48)  # gravel apron
    annulus(bm, 0, 0, 1.44, 1.52, 0.0, 0.085, 0)     # stone kerb
    cone(bm, 0, 0, 1.44, 1.44, 0.0, 0.055, 1, n=48)  # sand fill, sitting below the kerb lip
    box(bm, -0.45, 0.45, -0.45, 0.45, 0.055, 0.085, 0)   # slab under the plinth
    box(bm, -0.62, -0.26, -0.34, 0.34, 0.0, 0.04, 0)     # arrival step
    # A label rail (post at x -0.44 carrying a plate at z 0.88~0.91) used to stand here. It sat right
    # in front of the plinth at the same height as the stage-toggle button and hid it, so the visitor
    # could neither see nor reach the button — removed 2026-09-16. The label text is on the title board.

# ================================================================ run
PARTS = [
    ("SM_Chandelier",     ["Metal", "Candle", "Glow"], chandelier, 0.006),
    ("SM_Sofa",           ["Fabric", "Wood"],          sofa,       0.030),
    ("SM_FeatureWall",    ["Frame", "Panel", "Plate"], feature_wall, 0.008),
    ("SM_RoomA_Backdrop", ["Strata", "Coping"],        roomA_backdrop, 0.010),
    ("SM_RoomA_Dais",     ["Kerb", "Sand", "Gravel"],  roomA_dais, 0.010),
]
RESULT = [finish(n, s, f, bevel=b) for n, s, f, b in PARTS]
print("\n".join("%-20s tris~%d" % (n, f * 2) for n, f in RESULT))
