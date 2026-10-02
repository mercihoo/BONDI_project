import bpy, sys, os, math, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
obj_in, tex_dir, out_dir = argv[0], argv[1], argv[2]
os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj_in, forward_axis='Y', up_axis='Z')
ob = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
print("mesh:", ob.name, "polys", len(ob.data.polygons), "dims", tuple(round(d,3) for d in ob.dimensions))

# ---- PBR material from the scan-derived tile -------------------------------
mat = bpy.data.materials.new("M_RopeScan")
mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

def img(name, non_color):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(os.path.join(tex_dir, name))
    if non_color:
        n.image.colorspace_settings.name = 'Non-Color'
    n.extension = 'REPEAT'
    return n

base = img("T_Rope_BaseColor.png", False); base.location = (-300, 250)
nrm  = img("T_Rope_Normal.png", True);     nrm.location  = (-300, -50)
orm  = img("T_Rope_ORM.png", True);        orm.location  = (-300, -350)

nt.links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
nmap = nt.nodes.new("ShaderNodeNormalMap"); nmap.location = (0, -50); nmap.inputs["Strength"].default_value = 1.0
nt.links.new(nrm.outputs["Color"], nmap.inputs["Color"])
nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (0, -350)
nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])     # ORM: G = roughness
bsdf.inputs["Metallic"].default_value = 0.0

ob.data.materials.clear()
ob.data.materials.append(mat)

# ---- daylight, roughly like the courtyard ---------------------------------
world = bpy.data.worlds.new("W"); bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.35, 0.45, 0.62, 1.0)
bg.inputs[1].default_value = 1.6

sun_d = bpy.data.lights.new("Sun", type='SUN'); sun_d.energy = 4.0; sun_d.angle = math.radians(2.0)
sun = bpy.data.objects.new("Sun", sun_d); bpy.context.collection.objects.link(sun)
sun.rotation_euler = (math.radians(52), 0, math.radians(35))

scene = bpy.context.scene
try:
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError:
    scene.render.engine = 'BLENDER_EEVEE'
print("engine:", scene.render.engine)
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.view_settings.view_transform = 'AgX' if 'AgX' in [v.name for v in scene.view_settings.bl_rna.properties['view_transform'].enum_items] else 'Filmic'

cam_d = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_d)
bpy.context.collection.objects.link(cam); scene.camera = cam

def shoot(name, loc, look, lens, w=760, h=1000):
    cam_d.lens = lens
    cam.location = mathutils.Vector(loc)
    d = mathutils.Vector(look) - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.filepath = os.path.join(out_dir, name)
    bpy.ops.render.render(write_still=True)
    print("rendered", name)

# the fray sits between z = -0.10 and 0.20
shoot("v1_fray.png",  (0.42, -0.34, 0.26), (0.0, 0.0, 0.06), 55)
shoot("v2_tip.png",   (0.19, -0.15, 0.02), (0.0, 0.0, -0.03), 50)
shoot("v3_hand.png",  (0.55, -0.45, 0.95), (0.0, 0.0, 0.40), 42)
