import bpy, sys, os
argv = sys.argv[sys.argv.index("--") + 1:]
obj_in, fbx_out = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=obj_in, forward_axis='Y', up_axis='Z')
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
ob = obs[0]
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
me = ob.data
print("mesh:", ob.name, "verts", len(me.vertices), "polys", len(me.polygons),
      "uv", len(me.uv_layers), "dims(m)", tuple(round(d, 3) for d in ob.dimensions))
me.calc_tangents()          # mikktspace; fails loudly if UVs or triangulation are bad
print("tangents ok")
bpy.ops.export_scene.fbx(
    filepath=fbx_out, use_selection=True, object_types={'MESH'},
    use_tspace=True, mesh_smooth_type='FACE', use_mesh_modifiers=True,
    add_leaf_bones=False, bake_space_transform=False,
    axis_forward='-Z', axis_up='Y', global_scale=1.0,
    apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE', path_mode='STRIP')
print("fbx written:", os.path.getsize(fbx_out), "bytes")
