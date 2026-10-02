#!/usr/bin/env bash
# 2단계 — LaS-Comp 실행에 실제로 필요한 의존성만 설치한다.
#
# 코드를 읽어 확정한 필수 3종:
#   spconv-cu121  : trellis/modules/sparse/conv/conv_spconv.py (희소 합성곱, 파이프라인 핵심)
#   xformers      : 실행 스크립트가 ATTN_BACKEND=xformers 로 고정 (flash-attn 은 대안일 뿐)
#   kaolin        : pipelines/samplers/flow_euler.py 최상단에서 즉시 import
#
# 건너뛰는 것 — renderers/__init__.py 가 지연 import(__getattr__) 라
# 렌더러를 실제로 쓸 때만 로드된다. 형상 완성 경로에서는 안 쓴다:
#   nvdiffrast · diff_gaussian_rasterization · diffoctreerast
# 전부 소스 빌드가 필요한 것들이라 여기서 뺀 이득이 크다. 필요해지면 그때 붙인다.
set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lascomp

log "=== torch 확인 (2.4.0+cu121 이어야 한다) ==="
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

log "=== spconv ==="
pip install -q spconv-cu121==2.3.8

log "=== xformers (torch 2.4.0 짝) ==="
pip install -q xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

log "=== kaolin (사전 빌드 휠) ==="
pip install -q kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

log "=== 나머지 파이썬 패키지 ==="
pip install -q \
  numpy==1.26.4 scipy trimesh plyfile open3d \
  easydict tqdm imageio imageio-ffmpeg safetensors huggingface_hub \
  einops transformers==4.44.2 pillow opencv-python-headless

log "=== 임포트 점검 ==="
python - <<'PY'
import importlib, sys
mods = ["torch", "spconv.pytorch", "xformers.ops", "kaolin",
        "numpy", "scipy", "trimesh", "plyfile", "open3d", "easydict", "safetensors"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print("  OK   ", m)
    except Exception as e:
        bad.append((m, str(e)[:90])); print("  실패 ", m, "→", str(e)[:90])
print()
print("실패 개수:", len(bad))
sys.exit(1 if bad else 0)
PY

log "=== torch ↔ GPU 연결 확인 ==="
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print("device :", p.name)
print("총 VRAM: %.0f MiB" % (p.total_memory/1024**2))
print("sm     : %d.%d" % (p.major, p.minor))
free, total = torch.cuda.mem_get_info()
print("가용/전체: %.0f / %.0f MiB" % (free/1024**2, total/1024**2))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
print("행렬곱 결과 shape:", (x @ x).shape)
PY

log "=== 2단계 끝 ==="
