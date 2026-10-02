# Museum hall shell — generates SM_Hall_Floor / SM_Hall_Walls / SM_Hall_Ceiling
# Modelled in metres, UE axis convention (X forward, Y right, Z up).
# Hall interior: x 0..7.0, y -4.5..4.5, z 0..3.2
import bpy, bmesh, math, os

OUT = os.environ.get("MUSEUM_OUT", r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- dimensions
X0, X1 = 0.0, 7.0          # interior x
Y0, Y1 = -4.5, 4.5         # interior y
Z1     = 3.2               # ceiling height
WT     = 0.20              # wall thickness
FT     = 0.10              # floor slab thickness
BORDER = 0.50              # floor border band width

BASE_H, BASE_P = 0.14, 0.075   # baseboard height / proud
DADO_T, DADO_P = 1.02, 0.040   # wainscot top / proud
RAIL_T, RAIL_P = 1.10, 0.070   # chair-rail top / proud
STILE_W, STILE_P = 0.08, 0.020 # panel stile width / extra proud
STILE_GAP = 1.15               # nominal spacing between stiles

# cove: a bracket, a shelf that hides the strip, and the emissive strip itself.
# The shelf is proud enough that an eye at 1.7 m cannot see the strip directly —
# only the wash it throws on the last 20 cm of wall and the ceiling.
COVE_BRACKET = (2.88, 2.94, 0.06)   # z0, z1, proud
COVE_SHELF   = (2.94, 3.00, 0.14)
COVE_STRIP   = (3.00, 3.035, 0.005, 0.065)  # z0, z1, proud from, proud to
CEIL_T   = 0.15            # ceiling slab thickness
BEAM_D, BEAM_W = 0.12, 0.16
BEAM_X = [1.75, 3.50, 5.25]
BEAM_Y = [-3.0, -1.0, 1.0, 3.0]

BEVEL_W, BEVEL_SEG = 0.012, 2

# ---------------------------------------------------------------- helpers
def box(bm, x0, x1, y0, y1, z0, z1, mat):
    """Add an axis-aligned box to bm, tagging its faces with material index."""
    vs = [bm.verts.new(v) for v in [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    for a, b, c, d in [(0,3,2,1), (4,5,6,7), (0,1,5,4),
                       (1,2,6,5), (2,3,7,6), (3,0,4,7)]:
        f = bm.faces.new((vs[a], vs[b], vs[c], vs[d]))
        f.material_index = mat

def unwrap(ob):
    """Give the mesh a UV0 for materials and a UV1 for the lightmap.

    The surface materials are world-aligned so UV0 is cosmetic, but Lightmass
    cannot bake a mesh with no UVs at all — without this every imported mesh
    samples a garbage lightmap and renders as if only the sky lit it.
    """
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

def finish(name, slots, build):
    """Build one mesh object from a box-emitting callback, bevel it, return it."""
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    for s in slots:
        mat = bpy.data.materials.get(s) or bpy.data.materials.new(s)
        ob.data.materials.append(mat)
    bm = bmesh.new()
    build(bm)
    bm.to_mesh(me)
    bm.free()
    me.validate()

    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob

    # weld coincident verts from abutting boxes, then soften every edge
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    unwrap(ob)
    bv = ob.modifiers.new("Bevel", "BEVEL")
    bv.width = BEVEL_W
    bv.segments = BEVEL_SEG
    bv.limit_method = "ANGLE"
    bv.angle_limit = math.radians(30)
    bv.harden_normals = False
    bpy.ops.object.modifier_apply(modifier=bv.name)
    return ob

def export(ob, name):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    path = os.path.join(OUT, name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
        use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    return path

def stile_positions(a, b):
    """Evenly spaced panel-stile centres between a and b (inclusive ends)."""
    n = max(2, round((b - a) / STILE_GAP))
    return [a + (b - a) * i / n for i in range(n + 1)]

# ---------------------------------------------------------------- floor
# slot 0 = field, slot 1 = border
def build_floor(bm):
    z0, z1 = -FT, 0.0
    ix0, ix1 = X0 + BORDER, X1 - BORDER
    iy0, iy1 = Y0 + BORDER, Y1 - BORDER
    box(bm, ix0, ix1, iy0, iy1, z0, z1, 0)                 # field
    box(bm, X0, X1, Y0, iy0, z0, z1, 1)                    # border S
    box(bm, X0, X1, iy1, Y1, z0, z1, 1)                    # border N
    box(bm, X0, ix0, iy0, iy1, z0, z1, 1)                  # border W
    box(bm, ix1, X1, iy0, iy1, z0, z1, 1)                  # border E

# ---------------------------------------------------------------- walls
# slot 0 = wall field, slot 1 = trim (baseboard / wainscot / rail / cornice)
def build_walls(bm):
    # structural slabs
    box(bm, X0 - WT, X0, Y0, Y1, 0, Z1, 0)                 # W
    box(bm, X1, X1 + WT, Y0, Y1, 0, Z1, 0)                 # E
    box(bm, X0 - WT, X1 + WT, Y0 - WT, Y0, 0, Z1, 0)       # S
    box(bm, X0 - WT, X1 + WT, Y1, Y1 + WT, 0, Z1, 0)       # N

    # each interior face: (axis, plane coord, inward sign, span a..b)
    faces = [("x", X0, +1, Y0, Y1), ("x", X1, -1, Y0, Y1),
             ("y", Y0, +1, X0, X1), ("y", Y1, -1, X0, X1)]

    for axis, p, s, a, b in faces:
        def slab(d0, d1, z0, z1, mat, a=a, b=b, axis=axis, p=p, s=s):
            """Box on this wall face, d0..d1 = depth proud of the face."""
            q0, q1 = sorted((p + s * d0, p + s * d1))
            if axis == "x":
                box(bm, q0, q1, a, b, z0, z1, mat)
            else:
                box(bm, a, b, q0, q1, z0, z1, mat)

        slab(0, BASE_P, 0, BASE_H, 1)                       # baseboard
        slab(0, DADO_P, BASE_H, DADO_T, 1)                  # wainscot field
        slab(0, RAIL_P, DADO_T, RAIL_T, 1)                  # chair rail
        for c in stile_positions(a, b):                     # panel stiles
            lo, hi = max(a, c - STILE_W / 2), min(b, c + STILE_W / 2)
            if hi - lo < STILE_W * 0.4:
                continue
            q0, q1 = sorted((p + s * DADO_P, p + s * (DADO_P + STILE_P)))
            if axis == "x":
                box(bm, q0, q1, lo, hi, BASE_H, DADO_T, 1)
            else:
                box(bm, lo, hi, q0, q1, BASE_H, DADO_T, 1)
        for z0, z1, pr in (COVE_BRACKET, COVE_SHELF):       # cove bracket + shelf
            slab(0, pr, z0, z1, 1)
        z0, z1, p0, p1 = COVE_STRIP                          # emissive strip
        slab(p0, p1, z0, z1, 2)

# ---------------------------------------------------------------- ceiling
# slot 0 = ceiling field, slot 1 = beams
def build_ceiling(bm):
    box(bm, X0 - WT, X1 + WT, Y0 - WT, Y1 + WT, Z1, Z1 + CEIL_T, 0)
    zb = Z1 - BEAM_D
    for c in BEAM_X:
        box(bm, c - BEAM_W / 2, c + BEAM_W / 2, Y0, Y1, zb, Z1, 1)
    for c in BEAM_Y:
        box(bm, X0, X1, c - BEAM_W / 2, c + BEAM_W / 2, zb, Z1, 1)

# ---------------------------------------------------------------- run
def run():
    made = []
    for name, slots, fn in [
            ("SM_Hall_Floor",   ["Floor_Field", "Floor_Border"], build_floor),
            ("SM_Hall_Walls",   ["Wall_Field",  "Trim", "Cove"], build_walls),
            ("SM_Hall_Ceiling", ["Ceil_Field",  "Beam"],         build_ceiling)]:
        old = bpy.data.objects.get(name)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
        ob = finish(name, slots, fn)
        made.append((name, len(ob.data.polygons), export(ob, name)))
    return made

RESULT = run()
print("\n".join("%s  tris~%d  -> %s" % (n, f * 2, p) for n, f, p in RESULT))
