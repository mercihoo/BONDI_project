#!/usr/bin/env bash
# 안 쓰이는 디코더 두 개를 pipeline.json 에서 뺀다.
#
# 근거 — trellis/pipelines/trellis_image_to_3d.py 267~271:
#     ret['mesh'] = self.models['slat_decoder_mesh'](slat)
#     #  ret['gaussian'] = self.models['slat_decoder_gs'](slat)        ← 주석
#     #  ret['radiance_field'] = self.models['slat_decoder_rf'](slat)  ← 주석
# 저자가 가우시안·radiance field 경로를 꺼 두었다. 그런데 from_pretrained 는
# pipeline.json 에 적힌 모델을 전부 올리므로, 쓰지도 않는 343 MB 가 GPU 에 상주한다.
#
# 원본은 pipeline.json.orig 에 있다. 되돌리려면 그걸 덮어쓰면 된다.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd ~/work/LaS-Comp/ckpt/image-large

python - <<'PY'
import json, os

p = "pipeline.json"
d = json.load(open(p, encoding="utf-8"))
models = d["args"]["models"]

DROP = ["slat_decoder_gs", "slat_decoder_rf"]
freed = 0
removed = []
for k in DROP:
    if k in models:
        f = models[k] + ".safetensors"
        if os.path.exists(f):
            freed += os.path.getsize(f)
        del models[k]
        removed.append(k)

json.dump(d, open(p, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
print("제거:", removed or "(없음)")
print("아낀 가중치: %.0f MiB" % (freed / 1024**2))
print("남은 모델:")
for k, v in models.items():
    print("  ", k)
PY
