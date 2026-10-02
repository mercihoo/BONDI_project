"""Measure lightmap-UV overlap the way UE's CheckLightMapUVs does: rasterise the chosen UV
channel at the target lightmap resolution and count texels covered by more than one triangle.

    blender -b -P uv_overlap.py -- <fbx> [uv_index] [resolution]
"""
import bpy, sys, os
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
paths = [a for a in argv if a.lower().endswith(".fbx")]
uv_index = int(argv[len(paths)]) if len(argv) > len(paths) else 1
res = int(argv[len(paths) + 1]) if len(argv) > len(paths) + 1 else 1024


def overlap_pct(ob, uv_index, res):
    me = ob.data
    if uv_index >= len(me.uv_layers):
        return None, "no uv channel %d (has %d)" % (uv_index, len(me.uv_layers))
    me.calc_loop_triangles()
    uvs = me.uv_layers[uv_index].data
    count = np.zeros((res, res), dtype=np.uint16)
    for tri in me.loop_triangles:
        p = [uvs[li].uv for li in tri.loops]
        xs = [v[0] * res for v in p]
        ys = [v[1] * res for v in p]
        x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
        y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, res), min(y1, res)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        ax, ay = xs[0], ys[0]
        bx, by = xs[1], ys[1]
        cx, cy = xs[2], ys[2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
        w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
        w2 = 1.0 - w0 - w1
        inside = (w0 > 1e-9) & (w1 > 1e-9) & (w2 > 1e-9)   # strict: a texel exactly on a shared edge must not count twice
        if inside.any():
            block = count[y0:y1, x0:x1]
            block[inside] += 1
            count[y0:y1, x0:x1] = block
    used = int((count > 0).sum())
    over = int((count > 1).sum())
    return (100.0 * over / used if used else 0.0), "%d/%d texels" % (over, used)


for fbx in paths:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    for ob in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        pct, note = overlap_pct(ob, uv_index, res)
        name = os.path.basename(fbx)
        if pct is None:
            print("OVERLAP %-30s uv%d @%d  -> %s" % (name, uv_index, res, note))
        else:
            print("OVERLAP %-30s uv%d @%d  -> %.2f%%  (%s, uvchans=%d, tris=%d)"
                  % (name, uv_index, res, pct, note, len(ob.data.uv_layers), len(ob.data.loop_triangles)))
