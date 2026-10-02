# Room B — 신라 전각 내부 (Silla hall interior). Metres, UE axes (X forward, Y right, Z up).
# Room B local interior: x -5..5, y -5..5, z 0..4  (level transform 5000, 0, 0)
#
# Concept (2nd pass, 2026-09-16): a bright palace-hall room, not a tomb. Red-lacquered
# post-and-beam frame (주칠 기둥·보) with white plaster panels between, green latticed windows
# glowing with daylight on both side walls, an exposed timber ceiling (창방 → X braces → purlins →
# rafters → plaster), a warm wooden floor, and a six-panel folding screen (병풍) behind the display.
# Vessels sit on wooden furniture: a four-post open stand (사방탁자) for the hero on a low wooden
# platform with a red/blue mat, and two long low tables for the four satellites, plus candlesticks.
#
#   SM_RoomB_Floor      wood planks + dark border
#   SM_RoomB_Walls      plaster panels, red posts/beams (하방·중방·상방·창방), 4 green windows, daylight panes
#   SM_RoomB_Ceiling    X braces on the 창방, purlins, rafters, plaster field
#   SM_RoomB_Screen     six-panel folding screen, zig-zag, at local x 1.80
#   SM_RoomB_Platform   3.2 x 2.6 wooden dais (z 0.12) with a red/blue mat
#   SM_RoomB_Stand      사방탁자 hero stand, top at 1.02 (hero sits at platform 0.12 + 1.02)
#   SM_RoomB_Table      long low table, top 0.80, placed twice (mirrored in UE)
#   SM_RoomB_Candle     brass candlestick with candle and flame (flame = emissive slot)
#   SM_Mock_Silla_*     five placeholder vessels (lathe profiles) — replaced by the real assets
import bpy, bmesh, math, os

OUT = r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- primitives
def box(bm, x0, x1, y0, y1, z0, z1, mat):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
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

def slab_xz(bm, x0, x1, z0, z1, y0, y1, mat):
    """Box given as an XZ rectangle extruded along Y (reads better for wall panels)."""
    box(bm, x0, x1, y0, y1, z0, z1, mat)

def lathe(bm, prof, mat, n=40, cx=0.0, cy=0.0):
    """Revolve an (r, z) profile around a vertical axis at (cx, cy). r == 0 -> centre vertex."""
    rings = []
    for r, z in prof:
        if r <= 1e-5:
            rings.append(bm.verts.new((cx, cy, z)))
        else:
            rings.append([bm.verts.new((cx + r*math.cos(2*math.pi*i/n), cy + r*math.sin(2*math.pi*i/n), z))
                          for i in range(n)])
    for a, b in zip(rings, rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            if isinstance(a, list) and isinstance(b, list):
                bm.faces.new((a[i], a[j], b[j], b[i])).material_index = mat
            elif isinstance(a, list):
                bm.faces.new((a[i], a[j], b)).material_index = mat
            elif isinstance(b, list):
                bm.faces.new((b[j], b[i], a)).material_index = mat

def jitter(i, lo, hi):
    """Deterministic pseudo-random in [lo, hi) — re-exports stay identical."""
    t = (math.sin(i * 12.9898) * 43758.5453) % 1.0
    return lo + t * (hi - lo)

def wall_box(bm, axis, p, s, a, b, d0, d1, z0, z1, mat):
    """Box on a wall face. axis 'x': face plane x = p, span a..b along y. s = out direction.
    d0..d1 is depth from the face (positive = into the room)."""
    q0, q1 = sorted((p + s*d0, p + s*d1))
    if axis == "x":
        box(bm, q0, q1, a, b, z0, z1, mat)
    else:
        box(bm, a, b, q0, q1, z0, z1, mat)

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

def finish(name, slots, build, bevel=0.012, segments=2, smooth=False):
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
    # Welding leaves zero-area faces and zero-length edges where two boxes met. UE's tangent
    # builder then reports "degenerate tangent bases / nearly zero tangents" on import and shades
    # those triangles wrong, so clear them here (2026-09-16).
    bpy.ops.mesh.dissolve_degenerate(threshold=1e-5)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    # Bevel BEFORE unwrapping. The other generators unwrap first, which leaves every bevel strip
    # with an interpolated, collapsed UV triangle — UE then reports "degenerate tangent bases /
    # nearly zero tangents" on import and shades those faces wrong (2026-09-16).
    if bevel > 0:
        bv = ob.modifiers.new("Bevel", "BEVEL")
        bv.width, bv.segments = bevel, segments
        bv.limit_method, bv.angle_limit = "ANGLE", math.radians(30)
        bv.use_clamp_overlap = True      # never let the bevel eat past a neighbouring face
        bpy.ops.object.modifier_apply(modifier=bv.name)
        # the bevel itself can leave slivers where boxes interpenetrate — clear them before unwrapping
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.dissolve_degenerate(threshold=2e-3)   # 2 mm — below the 8 mm chamfer, above the slivers
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode="OBJECT")
    unwrap(ob)
    if smooth:
        for f in me.polygons:
            f.use_smooth = True

    path = os.path.join(OUT, name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
        use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    return name, len(ob.data.polygons)

# ================================================================ Room B shell
RA, RB = -5.0, 5.0          # interior x and y bounds
RH      = 4.0               # wall height
WT      = 0.20
POST_W, POST_D = 0.24, 0.20          # red posts: width along the wall, depth into the room
POSTS = (-5.0, -1.667, 1.667, 5.0)   # post centres along each wall
# horizontal members: (z0, z1, depth) — 하방, 중방, 상방, 창방
HABANG   = (0.00, 0.36, 0.16)
JUNGBANG = (1.18, 1.32, 0.16)
SANGBANG = (2.58, 2.72, 0.16)
CHANGBANG = (3.20, 3.42, 0.20)
PANEL_P = 0.03                       # plaster panel proud of the structural wall
BACK_EPS = 0.004                     # keep applied faces off the wall plane (see roomB_walls)
# windows: on the y = +-5 walls, in the two outer bays, between 중방 and 상방
WIN_Z0, WIN_Z1 = 1.36, 2.54
WIN_MARGIN = 0.30                    # from the post faces
SLAT_W, SLAT_GAP = 0.045, 0.125

def roomB_floor(bm):        # 0 planks, 1 border
    box(bm, RA + 0.45, RB - 0.45, RA + 0.45, RB - 0.45, -0.10, 0.0, 0)
    box(bm, RA, RB, RA, RA + 0.45, -0.10, 0.0, 1)
    box(bm, RA, RB, RB - 0.45, RB, -0.10, 0.0, 1)
    box(bm, RA, RA + 0.45, RA + 0.45, RB - 0.45, -0.10, 0.0, 1)
    box(bm, RB - 0.45, RB, RA + 0.45, RB - 0.45, -0.10, 0.0, 1)

def roomB_walls(bm):        # 0 plaster, 1 lacquer (red), 2 green, 3 daylight
    # The four wall slabs overlap at the corners by BACK_EPS instead of butting: a butt joint leaves
    # two coincident quads that weld together and stack in the lightmap UV.
    box(bm, RA - WT, RA, RA - WT, RB + WT, 0, RH, 0)
    box(bm, RB, RB + WT, RA - WT, RB + WT, 0, RH, 0)
    box(bm, RA - BACK_EPS, RB + BACK_EPS, RA - WT, RA, 0, RH, 0)
    box(bm, RA - BACK_EPS, RB + BACK_EPS, RB, RB + WT, 0, RH, 0)
    faces = [("x", RA, +1), ("x", RB, -1), ("y", RA, +1), ("y", RB, -1)]
    for fi, (axis, p, s) in enumerate(faces):
        W = lambda a, b, d0, d1, z0, z1, mat: wall_box(bm, axis, p, s, a, b, d0, d1, z0, z1, mat)
        # plaster panels between the horizontal members (slightly proud, so the frame reads as a frame).
        # Everything applied to the wall starts at BACK_EPS, not 0: a face laid exactly on the wall
        # plane welds to it in remove_doubles, the two coincident quads end up in one UV island stacked
        # on each other, and Lightmass then reports overlapping lightmap UVs (2026-09-16).
        for z0, z1 in ((HABANG[1], JUNGBANG[0]), (JUNGBANG[1], SANGBANG[0]), (SANGBANG[1], CHANGBANG[0])):
            W(RA - BACK_EPS, RB + BACK_EPS, BACK_EPS, PANEL_P, z0, z1, 0)
        # horizontal red members
        for z0, z1, d in (HABANG, JUNGBANG, SANGBANG, CHANGBANG):
            W(RA - BACK_EPS, RB + BACK_EPS, BACK_EPS, d, z0, z1, 1)
        # posts
        for c in POSTS:
            a, b = max(RA, c - POST_W/2), min(RB, c + POST_W/2)
            W(a, b, BACK_EPS, POST_D, 0.0, CHANGBANG[1], 1)
        # windows on the side walls (y faces), outer bays only — the middle bay stays plaster
        if axis == "y":
            for k in (0, 2):
                a = POSTS[k] + POST_W/2 + WIN_MARGIN
                b = POSTS[k+1] - POST_W/2 - WIN_MARGIN
                W(a, b, PANEL_P + 0.005, PANEL_P + 0.028, WIN_Z0, WIN_Z1, 3)        # daylight pane (>= 2x bevel thick)
                fr = 0.08
                W(a, b, PANEL_P - BACK_EPS, PANEL_P + 0.07, WIN_Z0, WIN_Z0 + fr, 2)   # frame
                W(a, b, PANEL_P - BACK_EPS, PANEL_P + 0.07, WIN_Z1 - fr, WIN_Z1, 2)
                W(a, a + fr, PANEL_P - BACK_EPS, PANEL_P + 0.07, WIN_Z0, WIN_Z1, 2)
                W(b - fr, b, PANEL_P - BACK_EPS, PANEL_P + 0.07, WIN_Z0, WIN_Z1, 2)
                u = a + fr + SLAT_GAP
                while u + SLAT_W < b - fr:
                    W(u, u + SLAT_W, PANEL_P + 0.015, PANEL_P + 0.05, WIN_Z0 + fr, WIN_Z1 - fr, 2)
                    u += SLAT_W + SLAT_GAP
                zm = (WIN_Z0 + WIN_Z1) / 2                                            # one cross rail
                W(a + fr, b - fr, PANEL_P + 0.015, PANEL_P + 0.05, zm - 0.03, zm + 0.03, 2)

def roomB_ceiling(bm):      # 0 plaster, 1 lacquer (red), 2 dark wood
    box(bm, RA - WT, RB + WT, RA - WT, RB + WT, RH, RH + 0.15, 0)               # plaster field
    # X braces (대공) standing on the 창방, one per bay, on all four walls
    zb0, zb1 = CHANGBANG[1], 3.80
    for axis, p, s in (("x", RA, +1), ("x", RB, -1), ("y", RA, +1), ("y", RB, -1)):
        for k in range(len(POSTS) - 1):
            a = POSTS[k] + POST_W/2 + 0.25
            b = POSTS[k+1] - POST_W/2 - 0.25
            c, hw = (a + b) / 2, 0.42
            for sgn in (-1, 1):                                                   # two diagonals as stepped boxes
                for t in range(6):
                    f0, f1 = t/6, (t+1)/6
                    u0 = c + sgn*hw*(1 - 2*f0); u1 = c + sgn*hw*(1 - 2*f1)
                    wall_box(bm, axis, p, s, min(u0, u1) - 0.045, max(u0, u1) + 0.045, 0.02, 0.16,
                             zb0 + (zb1 - zb0)*f0, zb0 + (zb1 - zb0)*f1, 1)
            wall_box(bm, axis, p, s, c - 0.10, c + 0.10, 0.02, 0.16, zb0, zb1, 1)   # centre king post
    # two red purlins along x and one main beam along y
    for c in (-1.667, 1.667):
        box(bm, RA, RB, c - 0.13, c + 0.13, 3.72, 3.96, 1)
    box(bm, -0.16, 0.16, RA, RB, 3.66, 3.96, 1)
    # dark rafters across y, resting on the purlins
    u = RA + 0.25
    while u < RB - 0.1:
        box(bm, u - 0.045, u + 0.045, RA, RB, 3.90, RH, 2)
        u += 0.42

# ================================================================ folding screen (local x 1.80, y 0)
def roomB_screen(bm):       # 0 frame (dark wood), 1 field (gold-cream), 2 painting
    PW, PH, T = 0.70, 1.95, 0.035
    n = 6
    zig = 0.14
    for i in range(n):
        yc = (i - (n - 1) / 2) * (PW * 0.985)
        xo = zig if i % 2 == 0 else 0.0
        box(bm, xo, xo + T, yc - PW/2, yc + PW/2, 0.05, PH, 0)                   # panel core
        box(bm, xo - 0.006, xo, yc - PW/2 + 0.05, yc + PW/2 - 0.05, 0.11, PH - 0.06, 1)   # cream field
        box(bm, xo - 0.010, xo - 0.006, yc - PW/2 + 0.12, yc + PW/2 - 0.12, 0.42, PH - 0.30, 2)  # painting
        box(bm, xo - 0.012, xo + T + 0.012, yc - PW/2, yc + PW/2, 0.0, 0.05, 0)   # foot rail

# ================================================================ platform, stand, table, candle
# 5 cm tall, not 12. The VR pawn's locomotion would not step up the original 12 cm edge, so the
# visitor walked into an invisible wall 1.2 m short of the exhibit. Room A's dais is 5.5 cm and
# walks fine, so match that (2026-09-16).
PLAT_H = 0.05

def roomB_platform(bm):     # 0 dark wood, 1 red mat, 2 blue mat
    box(bm, -1.6, 1.6, -1.3, 1.3, 0.0, PLAT_H, 0)
    box(bm, -1.1, 1.1, -0.9, 0.9, PLAT_H, PLAT_H + 0.015, 2)                     # blue border
    box(bm, -0.95, 0.95, -0.75, 0.75, PLAT_H + 0.015, PLAT_H + 0.025, 1)          # red centre

def roomB_stand(bm):        # 0 dark wood
    H, HW, P = 1.02, 0.30, 0.04
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(bm, sx*HW - P/2, sx*HW + P/2, sy*HW - P/2, sy*HW + P/2, 0.0, H, 0)   # posts
    for z in (0.34, 0.68):                                                            # open shelves
        box(bm, -HW, HW, -HW, HW, z, z + 0.025, 0)
        for sx in (-1, 1):
            box(bm, sx*HW - 0.02, sx*HW + 0.02, -HW, HW, z - 0.06, z, 0)
    box(bm, -HW - 0.03, HW + 0.03, -HW - 0.03, HW + 0.03, H - 0.03, H, 0)           # top
    box(bm, -HW - 0.03, HW + 0.03, -HW - 0.03, HW + 0.03, 0.0, 0.05, 0)              # foot rail

def roomB_table(bm):        # 0 dark wood — origin at the table centre on the floor, length along x
    L, D, H = 1.90, 0.56, 0.80
    box(bm, -L/2, L/2, -D/2, D/2, H - 0.04, H, 0)                                    # top
    box(bm, -L/2 + 0.05, L/2 - 0.05, -D/2 + 0.05, D/2 - 0.05, H - 0.14, H - 0.04, 0) # apron
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx*(L/2 - 0.09), sy*(D/2 - 0.08)
            box(bm, x - 0.035, x + 0.035, y - 0.035, y + 0.035, 0.0, H - 0.04, 0)    # legs
    for sx in (-1, 1):                                                               # stretchers
        x = sx*(L/2 - 0.09)
        box(bm, x - 0.02, x + 0.02, -D/2 + 0.08, D/2 - 0.08, 0.16, 0.20, 0)
    box(bm, -L/2 + 0.09, L/2 - 0.09, -0.02, 0.02, 0.16, 0.20, 0)

def roomB_candle(bm):       # 0 brass, 1 wax, 2 flame — origin at the base centre
    lathe(bm, [(0, 0), (0.075, 0), (0.07, 0.012), (0.03, 0.02), (0.012, 0.04), (0.012, 0.30),
               (0.03, 0.31), (0.045, 0.325), (0.04, 0.335), (0.03, 0.34), (0, 0.34)], 0, n=24)
    lathe(bm, [(0, 0.335), (0.021, 0.335), (0.021, 0.52), (0.018, 0.53), (0, 0.53)], 1, n=20)
    lathe(bm, [(0, 0.53), (0.012, 0.545), (0.009, 0.575), (0.004, 0.60), (0, 0.615)], 2, n=12)

# ================================================================ mock vessels
V_JANGGYEONGHO = [(0, 0), (0.06, 0), (0.12, 0.05), (0.15, 0.15), (0.13, 0.24), (0.075, 0.27),
                  (0.08, 0.33), (0.105, 0.40), (0.095, 0.395), (0.07, 0.33), (0.062, 0.28),
                  (0.05, 0.22), (0, 0.20)]
V_GOBAE = [(0, 0), (0.085, 0), (0.06, 0.02), (0.045, 0.035), (0.04, 0.10), (0.06, 0.12),
           (0.095, 0.17), (0.10, 0.20), (0.09, 0.195), (0.05, 0.135), (0, 0.13),
           (0, 0.13), (0.10, 0.205), (0.08, 0.23), (0.03, 0.245), (0.035, 0.262), (0, 0.268)]
V_GIDAE = [(0, 0), (0.16, 0), (0.15, 0.02), (0.07, 0.22), (0.065, 0.28), (0.14, 0.44),
           (0.15, 0.48), (0.14, 0.475), (0.055, 0.30), (0, 0.29)]
V_DANGYEONGHO = [(0, 0), (0.07, 0), (0.13, 0.05), (0.15, 0.11), (0.12, 0.20), (0.085, 0.225),
                 (0.10, 0.27), (0.09, 0.265), (0.075, 0.225), (0.07, 0.21), (0, 0.19)]
V_GOBAE_TALL = [(0, 0), (0.10, 0), (0.06, 0.02), (0.05, 0.04), (0.045, 0.14), (0.07, 0.16),
                (0.11, 0.24), (0.115, 0.28), (0.105, 0.275), (0.06, 0.20), (0, 0.19)]

def smooth_profile(prof, sub=3):
    """Catmull-Rom through the control points so wheel-thrown curves read as curves."""
    if len(prof) < 3:
        return prof
    P = [prof[0]] + list(prof) + [prof[-1]]
    out = [prof[0]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i-1], P[i], P[i+1], P[i+2]
        for k in range(1, sub + 1):
            t = k / sub
            pt = []
            for a in range(2):
                v = 0.5 * ((2*p1[a]) + (-p0[a] + p2[a]) * t
                           + (2*p0[a] - 5*p1[a] + 4*p2[a] - p3[a]) * t*t
                           + (-p0[a] + 3*p1[a] - 3*p2[a] + p3[a]) * t*t*t)
                pt.append(v)
            if t == 1.0:
                pt = list(p2)
            out.append((max(0.0, pt[0]), pt[1]))
    return out

def vessel(prof):
    def build(bm):
        parts, cur = [], []
        for p in prof:
            if cur and p[0] == 0 and cur[-1][0] == 0:
                parts.append(cur); cur = [p]
            else:
                cur.append(p)
        parts.append(cur)
        for part in parts:
            lathe(bm, smooth_profile(part), 0, n=48)
    return build

# ================================================================ run
PARTS = [
    ("SM_RoomB_Floor",    ["Planks", "Border"],                        roomB_floor,    0.008, False),
    ("SM_RoomB_Walls",    ["Plaster", "Lacquer", "Green", "Daylight"], roomB_walls,    0.008, False),
    ("SM_RoomB_Ceiling",  ["Plaster", "Lacquer", "DarkWood"],          roomB_ceiling,  0.008, False),
    # SM_RoomB_Screen is generated by museum_screen.py (custom UVs for the painting)
    ("SM_RoomB_Platform", ["DarkWood", "MatRed", "MatBlue"],           roomB_platform, 0.008, False),
    ("SM_RoomB_Stand",    ["DarkWood"],                                roomB_stand,    0.005, False),
    ("SM_RoomB_Table",    ["DarkWood"],                                roomB_table,    0.005, False),
    ("SM_RoomB_Candle",   ["Brass", "Wax", "Flame"],                   roomB_candle,   0.0,   True),
    ("SM_Mock_Silla_Janggyeongho", ["Stoneware"], vessel(V_JANGGYEONGHO), 0.0, True),
    ("SM_Mock_Silla_Gobae",        ["Stoneware"], vessel(V_GOBAE),        0.0, True),
    ("SM_Mock_Silla_Gidae",        ["Stoneware"], vessel(V_GIDAE),        0.0, True),
    ("SM_Mock_Silla_Dangyeongho",  ["Stoneware"], vessel(V_DANGYEONGHO),  0.0, True),
    ("SM_Mock_Silla_Gobae_Tall",   ["Stoneware"], vessel(V_GOBAE_TALL),   0.0, True),
]
RESULT = [finish(n, s, f, bevel=b, smooth=sm) for n, s, f, b, sm in PARTS]
print("ROOMB_RESULT", RESULT)
