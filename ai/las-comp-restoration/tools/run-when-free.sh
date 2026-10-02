#!/usr/bin/env bash
# GPU 가 비면(사용 < 600 MiB, LaS-Comp 프로세스 없음) 설정 하나를 돌린다. 최대 15 분 기다린다.
#   bash run-when-free.sh <obj> <quant> <ias> <offload> <meshcpu>
# (PowerShell 인라인은 $(...) 와 따옴표에서 깨진다 — 파일로 둔다)
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd "$HOME/work/LaS-Comp"

OBJ=${1:?}; QUANT=${2:?}; IAS=${3:?}; OFF=${4:-1}; MCPU=${5:-1}
LABEL="lean-${QUANT}-ias${IAS}-off${OFF}-mcpu${MCPU}-${OBJ}"

for i in $(seq 1 45); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  busy=$(pgrep -fc run_lascomp_image_condition_single || true)
  if [ "${used:-9999}" -lt 600 ] && [ "${busy:-0}" = "0" ]; then break; fi
  echo "  GPU 사용 중 (${used} MiB, 프로세스 ${busy}) — 20초 대기 ($i/45)"
  sleep 20
done
echo "GPU_FREE_AT=$(date +%H:%M:%S)"
bash tools/run-lean.sh "$OBJ" "$QUANT" "$IAS" "$OFF" "$MCPU" > "logs/$LABEL.log" 2>&1
rc=$?
echo "RUN_DONE label=$LABEL rc=$rc"
grep -aE '상태 |걸린 시간|판정용 피크|RuntimeError' "logs/$LABEL.log" | head -5
