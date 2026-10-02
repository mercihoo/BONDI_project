#!/usr/bin/env bash
# pipeline.json 에 sparse_structure_encoder 를 추가한다.
#
# 왜 빠져 있었나: 원본 TRELLIS 의 image-to-3D 는 **무에서 생성**하므로 인코더가 필요 없다.
# LaS-Comp 은 관측된 부분 형상을 잠재공간에 **넣어야** 하므로(ERS 의 핵심) 인코더가 필요하다.
# 체크포인트 파일(ss_enc_conv3d_16l8_fp16)은 TRELLIS 저장소에 이미 들어 있어 받아 두었다.
#
# 원본은 pipeline.json.orig 로 남긴다.
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd ~/work/LaS-Comp/ckpt/image-large

[ -f pipeline.json.orig ] || cp pipeline.json pipeline.json.orig

python - <<'PY'
import json

p = "pipeline.json"
d = json.load(open(p, encoding="utf-8"))
models = d["args"]["models"]

added = []
if "sparse_structure_encoder" not in models:
    models["sparse_structure_encoder"] = "ckpts/ss_enc_conv3d_16l8_fp16"
    added.append("sparse_structure_encoder")

json.dump(d, open(p, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

print("추가:", added or "(없음)")
print("현재 키:")
for k, v in models.items():
    print("  ", k, "->", v)
PY

echo
echo "=== 파일 존재 확인 ==="
ls -la ckpts/ss_enc_conv3d_16l8_fp16.safetensors ckpts/ss_enc_conv3d_16l8_fp16.json
