"""Report triangles whose 3D area or UV area is near zero — the source of UE's
"degenerate tangent bases / nearly zero tangents" import warnings."""
import bpy, sys, os
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
for fbx in [a for a in argv if a.lower().endswith(".fbx")]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    for ob in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        me = ob.data
        me.calc_loop_triangles()
        vs = np.array([v.co[:] for v in me.vertices])
        bad3 = baduv = 0
        worst = []
        for t in me.loop_triangles:
            a, b, c = vs[t.vertices[0]], vs[t.vertices[1]], vs[t.vertices[2]]
            ar3 = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            if ar3 < 1e-8:
                bad3 += 1
            for li, lay in enumerate(me.uv_layers):
                p = [lay.data[l].uv for l in t.loops]
                aruv = 0.5 * abs((p[1][0]-p[0][0])*(p[2][1]-p[0][1]) - (p[2][0]-p[0][0])*(p[1][1]-p[0][1]))
                if aruv < 1e-12:
                    baduv += 1
                    if len(worst) < 6:
                        worst.append((li, tuple(round(x, 4) for x in a), round(ar3, 8)))
        print("DEGEN %-28s tris=%d  zeroArea3D=%d  zeroAreaUV=%d  uvchans=%d"
              % (os.path.basename(fbx), len(me.loop_triangles), bad3, baduv, len(me.uv_layers)))
        for w in worst:
            print("        uv%d near %s  area3d=%s" % w)
