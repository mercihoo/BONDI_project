#!/usr/bin/env bash
# TRELLIS image-large 체크포인트를 LaS-Comp 의 ckpt/image-large 로 받는다.
# huggingface_hub 없이 curl 만 쓴다 (1단계 conda 환경과 무관하게 병렬로 돌리기 위해).
set -euo pipefail
REPO=microsoft/TRELLIS-image-large
DST=~/work/LaS-Comp/ckpt/image-large
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DST/ckpts"
FILES=(
  pipeline.json
  ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json
  ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors
  ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json
  ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
  ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.json
  ckpts/slat_dec_rf_swin8_B_64l8r16_fp16.safetensors
  ckpts/slat_enc_swin8_B_64l8_fp16.json
  ckpts/slat_enc_swin8_B_64l8_fp16.safetensors
  ckpts/slat_flow_img_dit_L_64l8p2_fp16.json
  ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors
  ckpts/ss_dec_conv3d_16l8_fp16.json
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors
  ckpts/ss_enc_conv3d_16l8_fp16.json
  ckpts/ss_enc_conv3d_16l8_fp16.safetensors
  ckpts/ss_flow_img_dit_L_16l8_fp16.json
  ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors
)

for f in "${FILES[@]}"; do
  out="$DST/$f"
  if [ -s "$out" ]; then
    echo "이미 있음: $f"
    continue
  fi
  echo "받는 중: $f"
  curl -fL --retry 5 --retry-all-errors --retry-delay 3 -C - \
       -o "$out" "$BASE/$f" --progress-bar 2>&1 | tail -1
done

echo "=== 결과 ==="
du -sh "$DST"
find "$DST" -name '*.safetensors' -printf '%8.0f MB  %p\n' -size +1k | sed 's#'"$HOME"'##' | sort -k3
echo "=== 끝 ==="
