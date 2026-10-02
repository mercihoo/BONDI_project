"""capture2.py <out.png> x y z pitch yaw -- set camera, settle, capture twice (2nd is occlusion-correct)."""
import sys, time, base64
from uemcp import *
out = sys.argv[1]
x, y, z, pitch, yaw = [float(v) for v in sys.argv[2:7]]
prev = call(APP, "GetCameraTransform", {})
pose = xform((x, y, z), (pitch, yaw, 0))
ann = {"gridSpacing": 0, "gridExtent": 0, "gridHeight": 0}
call(APP, "SetCameraTransform", {"transform": pose})
time.sleep(3)
for i in range(2):
    r = call(APP, "CaptureViewport", {"captureTransform": pose, "annotations": ann, "bShowUI": False})
    time.sleep(1.5)
img = r.get("image", r) if isinstance(r, dict) else {}
data = img.get("data") if isinstance(img, dict) else None
if not data:
    print("no image", str(r)[:200]); sys.exit(1)
open(out, "wb").write(base64.b64decode(data))
print("saved", out, len(data)//1024, "KB")
if isinstance(prev, dict) and "location" in prev:
    call(APP, "SetCameraTransform", {"transform": prev})
