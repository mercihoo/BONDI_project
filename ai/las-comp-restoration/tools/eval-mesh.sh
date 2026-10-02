#!/usr/bin/env bash
# 결과 하나를 정확한 점→표면 거리로 평가한다. rtree 가 없으면 설치한다.
#   bash eval-mesh.sh <라벨> [obj]
# (PowerShell → wsl 로 인라인 파이썬을 넘기면 따옴표 중첩에서 깨진다 — 그래서 파일이다)
set -uo pipefail
LABEL=${1:?라벨}
OBJ=${2:-bimba}
SP="${EVAL_SCRATCHPAD:-/tmp/heritage-vr-eval}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd "$HOME/work/LaS-Comp"

python -c "import rtree" 2>/dev/null || { echo "rtree 설치 중"; pip install -q rtree; }
python - <<'PY'
import rtree, trimesh
print("RTREE_OK", rtree.__version__, "| trimesh", trimesh.__version__)
PY

cp "$SP/eval_completion.py" tools/ && python -m py_compile tools/eval_completion.py && echo EVAL_SYNC_OK
time python tools/eval_completion.py \
  --result "results/$LABEL" \
  --gt "samples/CompC_datasets/plyobj/gtdata/$OBJ.ply" \
  --partial "samples/CompC_datasets/plyobj/indata/$OBJ.ply" \
  --label "$LABEL" 2>&1 | grep -vE 'FutureWarning|warnings.warn'
