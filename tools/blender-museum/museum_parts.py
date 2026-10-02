# Room A shell + shared props. Metres, UE axes (X forward, Y right, Z up).
# Room A local interior: x -5..5, y -5..5, z 0..4  (level transform 5000,-6000,0)
import bpy, bmesh, math, os

OUT = r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
os.makedirs(OUT, exist_ok=True)
BEVEL_W, BEVEL_SEG = 0.012, 2

# ---------------------------------------------------------------- primitives
def box(bm, x0, x1, y0, y1, z0, z1, mat):
    vs = [bm.verts.new(v) for v in [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]
    for a, b, c, d in [(0,3,2,1), (4,5,6,7), (0,1,5,4),
                       (1,2,6,5), (2,3,7,6), (3,0,4,7)]:
        bm.faces.new((vs[a], vs[b], vs[c], vs[d])).material_index = mat

def cyl(bm, cx, cy, r, z0, z1, mat, n=24):
    ring = lambda z: [bm.verts.new((cx + r*math.cos(2*math.pi*i/n),
                                    cy + r*math.sin(2*math.pi*i/n), z))
                      for i in range(n)]
    lo, hi = ring(z0), ring(z1)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i])).material_index = mat
    bm.faces.new(list(reversed(lo))).material_index = mat
    bm.faces.new(hi).material_index = mat

def arch_ring(bm, outer, inner, y0, y1, mat):
    """Extrude the band between two matched XZ boundaries along Y."""
    V = {}
    def v(p, y):
        k = (round(p[0], 5), round(y, 5), round(p[1], 5))
        if k not in V:
            V[k] = bm.verts.new(k)
        return V[k]
    def quad(a, b, c, d):
        try:
            bm.faces.new((a, b, c, d)).material_index = mat
        except ValueError:
            pass                                   # duplicate face, skip
    for i in range(len(outer) - 1):
        o0, o1, i0, i1 = outer[i], outer[i+1], inner[i], inner[i+1]
        quad(v(o0, y0), v(o1, y0), v(i1, y0), v(i0, y0))          # front band
        quad(v(i0, y1), v(i1, y1), v(o1, y1), v(o0, y1))          # back band
        quad(v(o0, y1), v(o1, y1), v(o1, y0), v(o0, y0))          # outside
        quad(v(i0, y0), v(i1, y0), v(i1, y1), v(i0, y1))          # soffit
    for p, q in ((outer[0], inner[0]), (inner[-1], outer[-1])):    # foot caps
        quad(v(p, y0), v(q, y0), v(q, y1), v(p, y1))

# ---------------------------------------------------------------- build/export
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
    bv.width, bv.segments = BEVEL_W, BEVEL_SEG
    bv.limit_method, bv.angle_limit = "ANGLE", math.radians(30)
    bpy.ops.object.modifier_apply(modifier=bv.name)

    path = os.path.join(OUT, name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_NONE", mesh_smooth_type="FACE",
        use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    return name, len(ob.data.polygons)

# ================================================================ Room A
RA, RB = -5.0, 5.0          # interior x and y bounds
RH      = 4.0               # wall height
WT      = 0.20
GAP_H, FACE_P = 0.08, 0.03  # floor shadow gap height / facing panel proud
COVE_SHELF_A = (3.70, 3.76, 0.14)
COVE_STRIP_A = (3.76, 3.795, 0.005, 0.065)
BORDER_A = 0.60
CEIL_T_A = 0.15
DOWN = 3.0                  # half-size of the raised central ceiling panel
DOWN_W, DOWN_D = 0.25, 0.15 # downstand frame width / drop

def roomA_floor(bm):        # 0 field, 1 border
    z0, z1 = -0.10, 0.0
    a, b = RA + BORDER_A, RB - BORDER_A
    box(bm, a, b, a, b, z0, z1, 0)
    box(bm, RA, RB, RA, a, z0, z1, 1)
    box(bm, RA, RB, b, RB, z0, z1, 1)
    box(bm, RA, a, a, b, z0, z1, 1)
    box(bm, b, RB, a, b, z0, z1, 1)

def roomA_walls(bm):        # 0 wall, 1 shelf, 2 cove strip
    box(bm, RA - WT, RA, RA, RB, 0, RH, 0)
    box(bm, RB, RB + WT, RA, RB, 0, RH, 0)
    box(bm, RA - WT, RB + WT, RA - WT, RA, 0, RH, 0)
    box(bm, RA - WT, RB + WT, RB, RB + WT, 0, RH, 0)
    for axis, p, s, a, b in [("x", RA, +1, RA, RB), ("x", RB, -1, RA, RB),
                             ("y", RA, +1, RA, RB), ("y", RB, -1, RA, RB)]:
        def slab(d0, d1, z0, z1, mat, a=a, b=b, axis=axis, p=p, s=s):
            q0, q1 = sorted((p + s*d0, p + s*d1))
            if axis == "x":
                box(bm, q0, q1, a, b, z0, z1, mat)
            else:
                box(bm, a, b, q0, q1, z0, z1, mat)
        slab(0, FACE_P, GAP_H, COVE_SHELF_A[0], 0)        # floating facing panel
        slab(0, COVE_SHELF_A[2], COVE_SHELF_A[0], COVE_SHELF_A[1], 1)
        z0, z1, p0, p1 = COVE_STRIP_A
        slab(p0, p1, z0, z1, 2)

def roomA_ceiling(bm):      # 0 field, 1 downstand
    box(bm, RA - WT, RB + WT, RA - WT, RB + WT, RH, RH + CEIL_T_A, 0)
    z0, z1 = RH - DOWN_D, RH
    for a, b, c, d in [(-DOWN, DOWN, -DOWN, -DOWN + DOWN_W),
                       (-DOWN, DOWN, DOWN - DOWN_W, DOWN),
                       (-DOWN, -DOWN + DOWN_W, -DOWN + DOWN_W, DOWN - DOWN_W),
                       (DOWN - DOWN_W, DOWN, -DOWN + DOWN_W, DOWN - DOWN_W)]:
        box(bm, a, b, c, d, z0, z1, 1)

# ================================================================ props
def plinth(bm):             # 0.60 x 0.60 x 1.00, recessed foot
    box(bm, -0.26, 0.26, -0.26, 0.26, 0.00, 0.06, 0)
    box(bm, -0.30, 0.30, -0.30, 0.30, 0.06, 1.00, 0)

def bench(bm):              # 1.8 x 0.45 x 0.42, slab seat on two blade legs
    box(bm, -0.90, 0.90, -0.225, 0.225, 0.36, 0.42, 0)
    for x0, x1 in ((-0.80, -0.62), (0.62, 0.80)):
        box(bm, x0, x1, -0.18, 0.18, 0.0, 0.36, 1)

def column(bm):             # base, shaft, capital — reaches a 3.2 m ceiling
    box(bm, -0.25, 0.25, -0.25, 0.25, 0.00, 0.10, 0)
    cyl(bm, 0, 0, 0.22, 0.10, 0.18, 0)
    cyl(bm, 0, 0, 0.18, 0.18, 2.98, 1)
    cyl(bm, 0, 0, 0.24, 2.98, 3.06, 0)
    box(bm, -0.28, 0.28, -0.28, 0.28, 3.06, 3.20, 0)

def portal_arch(bm):
    # The portal gate is a 140 x 220 cm plane spanning z 0.10..2.30. Keep the
    # opening a little *narrower* and *lower* than that (1.30 x 2.05) so the
    # shimmer fills the doorway edge to edge and its straight sides and corners
    # disappear behind the stone.
    W, T, D, SPRING = 0.65, 0.30, 0.50, 1.40      # half-width, band, depth, springline
    N = 20
    out, inn = [], []
    out.append((-(W + T), 0.0)); inn.append((-W, 0.0))
    out.append((-(W + T), SPRING)); inn.append((-W, SPRING))
    for i in range(N + 1):
        t = math.pi * (1 - i / N)
        out.append(((W + T) * math.cos(t), SPRING + (W + T) * math.sin(t)))
        inn.append((W * math.cos(t), SPRING + W * math.sin(t)))
    out.append((W + T, 0.0)); inn.append((W, 0.0))
    arch_ring(bm, out, inn, -D / 2, D / 2, 0)

# ================================================================ run
PARTS = [
    ("SM_RoomA_Floor",   ["Floor_Field", "Floor_Border"], roomA_floor),
    ("SM_RoomA_Walls",   ["Wall_Field", "Shelf", "Cove"], roomA_walls),
    ("SM_RoomA_Ceiling", ["Ceil_Field", "Downstand"],     roomA_ceiling),
    ("SM_Plinth",        ["Stone"],                       plinth),
    ("SM_Bench",         ["Top", "Leg"],                  bench),
    ("SM_Column",        ["Stone", "Shaft"],              column),
    ("SM_PortalArch",    ["Stone"],                       portal_arch),
]
RESULT = [finish(n, s, f) for n, s, f in PARTS]
print("\n".join("%-20s tris~%d" % (n, f * 2) for n, f in RESULT))
