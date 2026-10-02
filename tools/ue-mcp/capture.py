"""capture.py <out.png> x y z pitch yaw  -- move the editor camera, wait, capture, restore."""
import sys, time, base64, json
from uemcp import *

out = sys.argv[1]
x, y, z, pitch, yaw = [float(v) for v in sys.argv[2:7]]
prev = call(APP, "GetCameraTransform", {})
pose = xform((x, y, z), (pitch, yaw, 0))
call(APP, "SetCameraTransform", {"transform": pose})
time.sleep(2.5)
r = call(APP, "CaptureViewport", {"captureTransform": pose,
                                   "annotations": {"gridSpacing": 0, "gridExtent": 0, "gridHeight": 0},
                                   "bShowUI": False})
data = None
if isinstance(r, dict):
    img = r.get("image", r)
    data = img.get("data") if isinstance(img, dict) else None
if not data:
    print("no image in", str(r)[:400]); sys.exit(1)
open(out, "wb").write(base64.b64decode(data))
print("saved", out, len(data) // 1024, "KB")
if isinstance(prev, dict) and "location" in prev:
    call(APP, "SetCameraTransform", {"transform": prev})
