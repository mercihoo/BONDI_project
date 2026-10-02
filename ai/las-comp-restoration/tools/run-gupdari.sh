#!/usr/bin/env bash
# 굽다리바리 71489 투입 — 채택 설정(fp8 1 1 1) 으로. 인자로 바꿀 수 있다: bash run-gupdari.sh [quant] [ias] [seed]
#   입력: samples/artifacts/gupdari71489/partial.ply (원본 GLB 표면 10 만 점, 원좌표, Y-up, 반폭 0.501)
#   조건: samples/artifacts/gupdari71489/image.png (소장품 사진)
#   inspect_glb 가 낸 권장 인자: --dataset custom --yz-flip --normalize-partial
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd "$HOME/work/LaS-Comp"

QUANT=${1:-fp8}; IAS=${2:-1}; SEED=${3:-1}
NAME=${4:-gupdari71489}                 # samples/artifacts/<NAME>/ — stage-artifact.sh 가 만든 폴더
UPFLAG=${UPFLAG:---yz-flip}             # inspect_glb 가 Y-up 이라 하면 --yz-flip, Z-up 이면 --no-yz-flip
export PARTIAL="samples/artifacts/$NAME/partial.ply"
# 조건 이미지 — 기본은 소장품 사진. **깨진 유물 사진을 넣으면 모델이 깨진 모습을 그대로 따라간다** (첫 실행에서 확인).
# 온전한 모습의 참조(예: 팀 기하 복원본 렌더)를 IMAGE_OVERRIDE 로 넘긴다. 라벨에 태그(REFTAG)를 붙여 구분한다.
export IMAGE="${IMAGE_OVERRIDE:-samples/artifacts/$NAME/image.png}"
REFTAG=${REFTAG:-}
export DATASET=custom
export EXTRA="$UPFLAG --normalize-partial --seed $SEED"
OBJ="${NAME}${REFTAG:+-$REFTAG}-s${SEED}"

# 텍스트 조건: MODE=text PROMPT="..." — 텍스트 스크립트엔 --normalize-partial 이 없어서 미리 정규화한 점군(반폭 0.49)을 넣고,
# 끝나면 결과를 원좌표로 되돌린다 (denorm_result.py). 라벨에 -txt 가 붙는다.
MODE=${MODE:-image}; export MODE PROMPT
if [ "$MODE" = "text" ]; then
  export PARTIAL="samples/artifacts/$NAME/partial_norm.ply"
  export EXTRA="$UPFLAG --seed $SEED"
  [ -f "$PARTIAL" ] || python tools/make_partial_norm.py "samples/artifacts/$NAME/partial.ply" "$PARTIAL"
  OBJ="${NAME}-txt${REFTAG:+-$REFTAG}-s${SEED}"
fi

# GPU 가 비어 있어야 한다
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-9999}" -lt 600 ] && [ "$(pgrep -fc run_lascomp_image_condition_single || true)" = "0" ] && break
  echo "  GPU 사용 중 (${used} MiB) — 20초 대기 ($i/30)"; sleep 20
done

LABEL="lean-${QUANT}-ias${IAS}-off1-mcpu1-${OBJ}"
mkdir -p logs
bash tools/run-lean.sh "$OBJ" "$QUANT" "$IAS" 1 1 > "logs/$LABEL.log" 2>&1
rc=$?
echo "RUN_DONE label=$LABEL rc=$rc"
grep -aE '상태 |걸린 시간|판정용 피크|Warning: only|RuntimeError' "logs/$LABEL.log" | head -6
ls -la "results/$LABEL" 2>/dev/null | grep -v '^total'
if [ "$MODE" = "text" ] && [ -f "results/$LABEL/output_mesh.glb" ]; then
  python tools/denorm_result.py "results/$LABEL" "samples/artifacts/$NAME/partial_norm.json"
fi
# 자동 후처리 — 합성(전체·몸통) · Q4 · 4 방향 메시 렌더. POST=0 이면 건너뜀.
if [ "${POST:-1}" = "1" ] && [ -f "results/$LABEL/output_mesh.glb" ]; then bash tools/postprocess.sh "$LABEL" "$NAME"; fi
