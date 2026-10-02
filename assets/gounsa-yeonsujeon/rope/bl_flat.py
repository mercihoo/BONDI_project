import bpy, sys, os, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
obj_in, tex_dir, out_dir, mode = argv[0], argv[1], argv[2], argv[3]
os.makedirs(out_dir, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj_in, forward_axis='Y', up_axis='Z')
ob = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]

mat = bpy.data.materials.new("M"); mat.use_nodes = True
nt = mat.node_tree
for n in list(nt.nodes): nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)

if mode == "emit":                      # pure texture colour, no lighting at all
    sh = nt.nodes.new("ShaderNodeEmission")
else:
    sh = nt.nodes.new("ShaderNodeBsdfPrincipled")
    sh.inputs["Roughness"].default_value = 0.9
    sh.inputs["Metallic"].default_value = 0.0
sh.location = (300, 0)
nt.links.new(sh.outputs[0], out.inputs["Surface"])

tx = nt.nodes.new("ShaderNodeTexImage")
tx.image = bpy.data.images.load(os.path.join(tex_dir, "T_Rope_BaseColor.png"))
tx.extension = 'REPEAT'; tx.location = (-200, 0)
nt.links.new(tx.outputs["Color"], sh.inputs[0])
ob.data.materials.clear(); ob.data.materials.append(mat)

world = bpy.data.worlds.new("W"); bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.35, 0.45, 0.62, 1.0); bg.inputs[1].default_value = 1.6
sd = bpy.data.lights.new("S", type='SUN'); sd.energy = 4.0
su = bpy.data.objects.new("S", sd); bpy.context.collection.objects.link(su)
su.rotation_euler = (math.radians(52), 0, math.radians(35))

sc = bpy.context.scene
try: sc.render.engine = 'BLENDER_EEVEE_NEXT'
except TypeError: sc.render.engine = 'BLENDER_EEVEE'
sc.render.image_settings.file_format = 'PNG'
cd = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cd)
bpy.context.collection.objects.link(cam); sc.camera = cam
cd.lens = 55
cam.location = mathutils.Vector((0.42, -0.34, 0.26))
d = mathutils.Vector((0.0, 0.0, 0.06)) - cam.location
cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
sc.render.resolution_x, sc.render.resolution_y = 760, 1000
sc.render.filepath = os.path.join(out_dir, f"flat_{mode}.png")
bpy.ops.render.render(write_still=True)
print("rendered", mode)
