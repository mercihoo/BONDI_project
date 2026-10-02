# Guide-panel pictograms. Metres, UE axes (X forward, Y right, Z up).
# Two material slots throughout: "Joint" is the body, "Spark" is the one thing
# the visitor is meant to look at (the pinch point, the direction, the button,
# the gate). In UE they carry MI_HoloIcon_Joint / MI_HoloIcon_Spark.
#
# Every icon is recentred on its bounding box by finish(), because BP_HoloGuide
# spins them in place — an off-centre origin swings the mesh through the panel.
import bpy, bmesh, math, os

OUT = r"C:/Users/<USER>/AppData/Local/Temp/museum_fbx"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- primitives
def sphere(bm, c, r, mat, seg=10, rings=6, scale=(1.0, 1.0, 1.0)):
    def at(x, y, z):
        return (c[0] + x * scale[0], c[1] + y * scale[1], c[2] + z * scale[2])
    grid = []
    for i in range(1, rings):
        lat = math.pi * i / rings
        row = [bm.verts.new(at(r*math.sin(lat)*math.cos(2*math.pi*j/seg),
                               r*math.sin(lat)*math.sin(2*math.pi*j/seg),
                               r*math.cos(lat))) for j in range(seg)]
        grid.append(row)
    top = bm.verts.new(at(0, 0, r))
    bot = bm.verts.new(at(0, 0, -r))
    for j in range(seg):
        k = (j + 1) % seg
        bm.faces.new((top, grid[0][k], grid[0][j])).material_index = mat
        bm.faces.new((bot, grid[-1][j], grid[-1][k])).material_index = mat
    for i in range(len(grid) - 1):
        for j in range(seg):
            k = (j + 1) % seg
            bm.faces.new((grid[i][j], grid[i][k], grid[i+1][k], grid[i+1][j])).material_index = mat

def tube(bm, p0, p1, r, mat, n=6):
    """Bone between two points — a prism along an arbitrary axis."""
    d = [p1[i] - p0[i] for i in range(3)]
    L = math.sqrt(sum(v*v for v in d))
    if L < 1e-6:
        return
    z = [v / L for v in d]
    up = (0.0, 0.0, 1.0) if abs(z[2]) < 0.9 else (1.0, 0.0, 0.0)
    x = [up[1]*z[2] - up[2]*z[1], up[2]*z[0] - up[0]*z[2], up[0]*z[1] - up[1]*z[0]]
    xl = math.sqrt(sum(v*v for v in x)); x = [v / xl for v in x]
    y = [z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2], z[0]*x[1] - z[1]*x[0]]
    def ring(p):
        return [bm.verts.new(tuple(p[i] + r*math.cos(2*math.pi*k/n)*x[i]
                                        + r*math.sin(2*math.pi*k/n)*y[i] for i in range(3)))
                for k in range(n)]
    a, b = ring(p0), ring(p1)
    for k in range(n):
        m = (k + 1) % n
        bm.faces.new((a[k], a[m], b[m], b[k])).material_index = mat
    bm.faces.new(list(reversed(a))).material_index = mat
    bm.faces.new(b).material_index = mat

def prism(bm, corners, z0, z1, mat):
    """Convex polygon given as XY corners in order, extruded along Z."""
    bot = [bm.verts.new((c[0], c[1], z0)) for c in corners]
    top = [bm.verts.new((c[0], c[1], z1)) for c in corners]
    bm.faces.new(list(reversed(bot))).material_index = mat
    bm.faces.new(top).material_index = mat
    n = len(corners)
    for k in range(n):
        m = (k + 1) % n
        bm.faces.new((bot[k], bot[m], top[m], top[k])).material_index = mat

def box(bm, lo, hi, mat):
    prism(bm, [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])], lo[2], hi[2], mat)

def cyl(bm, c, r, h, mat, n=16):
    """Vertical cylinder centred on c, spanning +-h/2."""
    corners = [(c[0] + r*math.cos(2*math.pi*k/n), c[1] + r*math.sin(2*math.pi*k/n)) for k in range(n)]
    prism(bm, corners, c[2] - h/2, c[2] + h/2, mat)

# ---------------------------------------------------------------- build/export
def unwrap(ob):
    me = ob.data
    for layer in list(me.uv_layers):
        me.uv_layers.remove(layer)
    me.uv_layers.new(name="UVMap")
    me.uv_layers.new(name="Lightmap")
    for name, margin in (("UVMap", 0.02), ("Lightmap", 0.06)):
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

    # spin in place: put the origin at the bounding-box centre, not the wrist
    lo = [min(v.co[i] for v in me.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in me.vertices) for i in range(3)]
    mid = [(lo[i] + hi[i]) / 2 for i in range(3)]
    for v in me.vertices:
        v.co = (v.co[0] - mid[0], v.co[1] - mid[1], v.co[2] - mid[2])

    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    unwrap(ob)
    bpy.ops.object.shade_smooth()
    bpy.ops.export_scene.fbx(
        filepath=os.path.join(OUT, name + ".fbx"),
        use_selection=True, object_types={"MESH"}, apply_scale_options="FBX_SCALE_NONE",
        mesh_smooth_type="FACE", use_mesh_modifiers=True, bake_space_transform=False,
        axis_forward="X", axis_up="Z", add_leaf_bones=False)
    radius = max(math.hypot(v.co[0], v.co[1]) for v in me.vertices)
    return name, len(me.polygons), radius

# ============================================================ 3. pinch hand
# Right hand, fingers along +X, thumb toward -Y, frozen just before the pinch:
# index and thumb tips stop ~2.7 cm apart with the spark filling the gap.
# Solid volumes read at two metres where a constellation of joints does not, so
# the bones are fat capsules that merge into one shape — but the joints stay a
# little fatter, keeping the knuckle bulges of the hand the game actually draws.
R_LEAD, R_REST = 0.0100, 0.0085
B_LEAD, B_REST = 0.0078, 0.0066
SPARK_R = 0.0105

PALM = [(0.014, -0.019), (0.084, -0.032), (0.074, 0.036), (0.012, 0.021)]
PALM_Z = (-0.010, 0.010)
W_IN = (0.010, -0.026, 0.000)

LEAD = {
    "index": [(0.080, -0.028,  0.006), (0.116, -0.030,  0.004), (0.133, -0.031, -0.010), (0.137, -0.031, -0.024)],
    "thumb": [(0.022, -0.040, -0.014), (0.058, -0.060, -0.058), (0.100, -0.052, -0.060), (0.128, -0.038, -0.048)],
}
REST = {  # curl back UNDER the palm, clear of the pinch ring
    "middle": [(0.078, -0.008, -0.002), (0.097, -0.009, -0.015), (0.087, -0.009, -0.032), (0.063, -0.009, -0.033)],
    "ring":   [(0.076,  0.013, -0.004), (0.094,  0.014, -0.017), (0.084,  0.014, -0.033), (0.061,  0.014, -0.034)],
    "pinky":  [(0.072,  0.031, -0.006), (0.088,  0.032, -0.018), (0.078,  0.032, -0.032), (0.059,  0.032, -0.032)],
}

def finger(bm, pts, jr, br, root=None):
    if root is not None:
        tube(bm, root, pts[0], br, 0, n=8)
    for i, p in enumerate(pts):
        sphere(bm, p, jr * (0.88 if i == len(pts) - 1 else 1.0), 0)
        if i:
            tube(bm, pts[i-1], p, br, 0, n=8)

def pinch_hand(bm):
    prism(bm, PALM, PALM_Z[0], PALM_Z[1], 0)
    sphere(bm, (0.014, 0.001, 0.000), 0.0155, 0)   # rounds off the wrist cut
    finger(bm, LEAD["index"], R_LEAD, B_LEAD)
    finger(bm, LEAD["thumb"], R_LEAD, B_LEAD, root=W_IN)
    for pts in REST.values():
        finger(bm, pts, R_REST, B_REST)

    it, tt = LEAD["index"][3], LEAD["thumb"][3]
    mid = tuple((it[i] + tt[i]) / 2 for i in range(3))
    sphere(bm, mid, SPARK_R, 1)
    for dx, dz in ((0.030, 0.014), (0.020, -0.030), (-0.022, 0.026)):
        sphere(bm, (mid[0] + dx, mid[1], mid[2] + dz), SPARK_R * 0.30, 1)

# ============================================================ 2. thumbstick
# A stick pushed forward with two chevrons ahead of it: tilt it and you walk
# that way. No teleport arc — the project deliberately does not have one.
def thumbstick(bm):
    cyl(bm, (0.0, 0.0, 0.000), 0.046, 0.016, 0)         # base plate
    cyl(bm, (0.0, 0.0, 0.011), 0.032, 0.010, 0)         # recess collar
    tilt = math.radians(32)
    root = (0.0, 0.0, 0.012)
    tip = (root[0] + 0.052*math.sin(tilt), 0.0, root[2] + 0.052*math.cos(tilt))
    tube(bm, root, tip, 0.010, 0, n=12)
    sphere(bm, tip, 0.019, 0, scale=(1.0, 1.0, 0.72))   # thumb cap

    # one arrow, floating clear of the base. Two chevrons merged into a blob at
    # two metres; a single shaft-and-head does not.
    box(bm, (0.072, -0.015, 0.010), (0.108, 0.015, 0.030), 1)
    prism(bm, [(0.158, 0.0), (0.104, 0.040), (0.104, -0.040)], 0.010, 0.030, 1)

# ============================================================ 4. plinth
# A pot on a plinth with the button lit. The one thing to do at an artifact is
# press the button in front of it and watch the restoration swap in.
def plinth(bm):
    box(bm, (-0.034, -0.034, -0.078), (0.034, 0.034, 0.006), 0)   # column
    box(bm, (-0.042, -0.042,  0.006), (0.042, 0.042, 0.018), 0)   # cap slab
    box(bm, (-0.042, -0.042, -0.086), (0.042, 0.042, -0.078), 0)  # foot

    sphere(bm, (0.0, 0.0, 0.049), 0.030, 0, scale=(1.0, 1.0, 0.92))  # pot body
    cyl(bm, (0.0, 0.0, 0.076), 0.012, 0.020, 0)                      # neck
    cyl(bm, (0.0, 0.0, 0.089), 0.018, 0.007, 0)                      # lip

    # the button, on the face the visitor stands at (-X)
    sphere(bm, (-0.033, 0.0, -0.030), 0.023, 0, scale=(0.40, 1.0, 1.0))   # bezel
    sphere(bm, (-0.044, 0.0, -0.030), 0.017, 1, scale=(0.70, 1.0, 1.0))   # lit cap

# ============================================================ 5. portal
# An arch with the gate lit. The hall has three of these in three colours, so
# the shape alone has to carry it.
def portal(bm):
    R, ZS, Z0, T = 0.048, -0.012, -0.082, 0.010
    for s in (-1, 1):                                   # jambs
        tube(bm, (0.0, s*R, Z0), (0.0, s*R, ZS), T, 0, n=10)
    seg = 10                                            # semicircular head
    arc = [(0.0, R*math.cos(math.pi*i/seg), ZS + R*math.sin(math.pi*i/seg))
           for i in range(seg + 1)]
    for a, b in zip(arc, arc[1:]):
        tube(bm, a, b, T, 0, n=10)
    for p in arc[::2]:
        sphere(bm, p, T, 0, seg=8, rings=5)             # smooths the joins

    # the gate itself — a thin slab inset inside the opening
    box(bm, (-0.004, -R + 0.006, Z0 + 0.004), (0.004, R - 0.006, ZS + R * 0.70), 1)

PARTS = [
    ("SM_Icon_PinchHand",  ["Joint", "Spark"], pinch_hand),
    ("SM_Icon_Thumbstick", ["Joint", "Spark"], thumbstick),
    ("SM_Icon_Plinth",     ["Joint", "Spark"], plinth),
    ("SM_Icon_Portal",     ["Joint", "Spark"], portal),
]
RESULT = [finish(n, s, f) for n, s, f in PARTS]
print("\n".join("%-22s tris~%-6d spin_r=%.3f m" % (n, f * 2, r) for n, f, r in RESULT))
