#!/usr/bin/env python3
"""유물 GLB → LaS-Comp 입력 점군(PLY) 로 바꾼다. 조건 이미지도 옆에 복사한다.

    python tools/prep_artifact.py artifacts/gupdari71489/source_model.glb artifacts/gupdari71489/photo_01.jpg gupdari71489

산출: samples/artifacts/<name>/partial.ply (표면 균일 표본, 기본 100,000 점, **원래 좌표 그대로**)
      samples/artifacts/<name>/image.png
좌표를 건드리지 않는 이유: 프레임 변환은 LaS-Comp 의 --yz-flip / --normalize-partial 이 담당하고
결과를 원좌표로 되돌려 주므로(denormalize), 여기서 바꾸면 되돌리기가 어긋난다.
"""
import sys, shutil
from pathlib import Path
import numpy as np, trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_completion import load_geometry

glb, img, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000

m = load_geometry(glb)
if m is None:
    sys.exit("메시가 아니다: %s" % glb)
pts, face_idx = trimesh.sample.sample_surface(m, n)
pts = np.asarray(pts, dtype=np.float32)

out = Path("samples/artifacts") / name
out.mkdir(parents=True, exist_ok=True)
pc = trimesh.PointCloud(pts)
pc.export(str(out / "partial.ply"))
Image.open(img).convert("RGB").save(out / "image.png")

ext = m.bounds[1] - m.bounds[0]
print("입력 메시   : 정점 %d · 면 %d · bbox %s" % (len(m.vertices), len(m.faces), np.round(ext, 3)))
print("점군 저장   : %s (%d 점, 원좌표)" % (out / "partial.ply", len(pts)))
print("이미지 저장 : %s (%s)" % (out / "image.png", Image.open(out / "image.png").size))
print("권장 인자   : --dataset custom %s --normalize-partial" % ("--yz-flip" if int(np.argmax(ext)) == 1 else "--no-yz-flip"))
