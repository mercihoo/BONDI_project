#!/usr/bin/env bash
# 5단계 — numpy ABI 되돌리기.
#
# 증상: kaolin 의 Cython 확장이 "numpy.dtype size changed, Expected 96 ... got 88" 로 터진다.
# 원인: 4단계 일괄 설치가 numpy 를 2.x 로 올렸다. kaolin 휠은 numpy 1.x 로 빌드돼 있다.
#       (2단계에서 numpy==1.26.4 로 고정했을 때는 임포트가 통과했었다.)
# 대책: numpy 1.26.4 로 되돌리고, 같이 깨지는 것들(opencv·scipy)도 짝을 맞춘다.
set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lascomp
cd ~/work/LaS-Comp

log "=== 지금 상태 ==="
python -c "import numpy; print('  numpy', numpy.__version__)"

log "=== numpy 1.26.4 로 고정 ==="
pip install -q "numpy==1.26.4"
python -c "import numpy; print('  numpy', numpy.__version__)"

log "=== numpy 2.x 로 빌드된 것들 재설치 (ABI 재정렬) ==="
pip install -q --force-reinstall --no-deps "scipy==1.11.4" "opencv-python-headless==4.10.0.84" || true

log "=== 핵심 패키지 버전 ==="
pip list 2>/dev/null | grep -iE "^(numpy|scipy|opencv|torch|torchvision|kaolin|spconv|xformers|pytorch3d|rembg|onnxruntime|utils3d) " || true

log "=== 임포트 점검 ==="
python - <<'PY'
import importlib, traceback
mods = ["numpy", "scipy", "torch", "kaolin", "spconv.pytorch", "xformers.ops",
        "pytorch3d", "utils3d", "rembg", "open3d", "trimesh", "plyfile"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print("  OK   ", m)
    except Exception as e:
        bad.append(m); print("  실패 ", m, "→", str(e)[:100])
print("  실패 개수:", len(bad))
PY

log "=== trellis 임포트 실검증 ==="
ATTN_BACKEND=xformers python - <<'PY'
import sys, traceback
sys.path.insert(0, ".")
try:
    from trellis.pipelines import TrellisImageTo3DPipeline, samplers
    print("  OK: trellis.pipelines 임포트 성공")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
PY

log "=== 5단계 끝 ==="
