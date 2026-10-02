#!/usr/bin/env bash
# 3단계 — 첫 실행에서 드러난 나머지 의존성.
#
#   utils3d    : trellis/representations/gaussian/gaussian_model.py 가 최상단에서 쓴다.
#                저자 requirements 의 핀 커밋을 그대로 따른다 (API 가 자주 바뀌는 저장소다).
#   pytorch3d  : trellis/representations/mesh/point_mesh_dist.py 가 `from pytorch3d import _C`.
#                컴파일된 확장이 필요하지만 py310+cu121+pyt240 사전 빌드 휠이 있어 빌드는 피한다.
#
# 여전히 건너뛰는 것: nvdiffrast · diff_gaussian_rasterization · diffoctreerast (지연 import),
#                    chamfer_3D · emd (벤치마크 평가 스크립트 전용)
set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lascomp

log "=== utils3d (핀 커밋) ==="
pip install -q "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

log "=== pytorch3d 의존성 먼저 (PyPI) ==="
# --no-index 로 휠 저장소만 보게 하면 iopath 같은 의존성을 PyPI 에서 못 가져온다.
# 의존성을 먼저 깔고, 휠은 --no-deps 로 넣는다.
pip install -q iopath fvcore

log "=== pytorch3d 0.7.8 (사전 빌드 휠, --no-deps) ==="
pip install -q --no-deps \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html \
  pytorch3d==0.7.8

log "=== trellis 임포트 실검증 ==="
cd ~/work/LaS-Comp
python - <<'PY'
import sys, traceback
sys.path.insert(0, ".")
try:
    from trellis.pipelines import TrellisImageTo3DPipeline, samplers
    print("  OK: trellis.pipelines 임포트 성공")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
PY

log "=== 3단계 끝 ==="
