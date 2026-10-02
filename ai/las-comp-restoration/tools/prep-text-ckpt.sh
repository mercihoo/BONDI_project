#!/usr/bin/env bash
# text-xlarge pipeline.json 을 우리 배치에 맞게 고친다 (원본은 pipeline.json.orig):
#   · 디코더·인코더 경로를 HF 저장소 경로("JeffreyXiang/TRELLIS-image-large/ckpts/…")에서 로컬 ckpts/ 로 (image-large 파일에 링크)
#   · 안 쓰는 slat_decoder_gs / rf 제거 (-343 MiB) — decode_slat 은 오프로드 훅이 메시만 디코드하게 막는다
#   · ERS 가 쓰는 sparse_structure_encoder 추가 (원본 pipeline.json 에 없다 — image-large 도 같았다)
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd "$HOME/work/LaS-Comp/ckpt/text-xlarge"
[ -f pipeline.json.orig ] || cp pipeline.json pipeline.json.orig
python - <<'EOF'
import json, pathlib
p = pathlib.Path("pipeline.json"); d = json.loads(pathlib.Path("pipeline.json.orig").read_text())
d["args"]["models"] = {
    "sparse_structure_decoder":    "ckpts/ss_dec_conv3d_16l8_fp16",
    "sparse_structure_flow_model": "ckpts/ss_flow_txt_dit_XL_16l8_fp16",
    "slat_decoder_mesh":           "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16",
    "slat_flow_model":             "ckpts/slat_flow_txt_dit_XL_64l8p2_fp16",
    "sparse_structure_encoder":    "ckpts/ss_enc_conv3d_16l8_fp16",
}
if pathlib.Path("../clip/model.safetensors").exists():
    d["args"]["text_cond_model"] = str(pathlib.Path("../clip").resolve())   # 병렬 다운로더로 받아 둔 로컬 CLIP (HF 캐시 대신)
p.write_text(json.dumps(d, indent=4))
for v in d["args"]["models"].values():
    for ext in (".json", ".safetensors"):
        f = pathlib.Path(v + ext); print(("  OK  " if f.exists() else "  없음 ") + str(f))
print("text_cond_model:", d["args"]["text_cond_model"])
EOF
