#!/usr/bin/env bash
# 절감책을 한꺼번에 건 실행.
#
# 기준선(양자화 없음)은 VRAM 97% 에 붙은 채 25분을 넘겼고 그 도중 기계가 비정상 종료됐다.
# 한 칸씩 올리며 효과를 분리할 여유가 없으므로, 안전한 것부터 전부 걸고 시작한다.
# 여유가 확인되면 하나씩 되돌리며 개별 효과를 잰다.
#
#   1) 안 쓰이는 디코더 2개 제거   약 343 MB  (코드에서 호출이 주석 처리돼 있다 — 손실 없음)
#   2) FP8 양자화 (DiT 만)         약 1.15 GB (디코더·FFN down-proj 는 제외)
#   3) IAS 최적화 0 회             역전파 활성값 제거 (양자화로는 안 줄어드는 부분)
#   4) expandable_segments         한계 근처 단편화 완화
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd ~/work/LaS-Comp

OBJ=${1:-bimba}
QUANT=${2:-fp8}
IAS=${3:-0}
OFFLOAD=${4:-1}        # 디코딩 직전 DiT·DINOv2·인코더를 CPU 로 내린다. FP8 실행이 메시 디코더에서 OOM 난 뒤 추가
MESHCPU=${5:-1}        # FlexiCubes 256³ 격자·메시 추출을 CPU 로. 오프로드 뒤에도 FlexiCubes 에서 OOM 난 뒤 추가
DECLEAN=${6:-${DECLEAN:-1}}  # 메시 디코더 GroupNorm/Subdivide 블록의 복사·수명 정리(정확 재구성). 47.7k 복셀에서 OOM 난 뒤 추가
LABEL="lean-${QUANT}-ias${IAS}-off${OFFLOAD}-mcpu${MESHCPU}-${OBJ}"

export LASCOMP_QUANT="$QUANT"
export LASCOMP_OFFLOAD="$OFFLOAD"
export LASCOMP_MESH_CPU="$MESHCPU"
export LASCOMP_DECODE_LEAN="$DECLEAN"
# 모델이 예측한 정점 색을 남긴다 (저자 저장 코드는 vertices·faces 만 써서 색을 버린다). COLOR=0 이면 끔.
[ "${COLOR:-1}" = "1" ] && export LASCOMP_COLOR_OUT="results/$LABEL/output_mesh_colors.npz"
export ATTN_BACKEND=xformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== 설정 ==="
echo "  대상        : $OBJ"
echo "  양자화      : $QUANT"
echo "  IAS 스텝    : $IAS"
echo "  오프로드    : $OFFLOAD"
echo "  메시추출CPU : $MESHCPU"
echo "  디코더절약  : $DECLEAN"
echo "  할당자      : $PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 입력 경로 — 기본은 저자 샘플(plyobj). 유물처럼 우리 데이터를 넣을 때는 환경변수로 덮어쓴다:
#   PARTIAL=samples/artifacts/gupdari71489/partial.ply IMAGE=samples/artifacts/gupdari71489/image.png \
#   DATASET=custom EXTRA="--yz-flip --normalize-partial" bash tools/run-lean.sh gupdari71489 fp8 1 1 1
PARTIAL=${PARTIAL:-samples/CompC_datasets/plyobj/indata/$OBJ.ply}
IMAGE=${IMAGE:-samples/image_from_compc/plyobj/${OBJ}_color.png}
DATASET=${DATASET:-plyobj}
EXTRA=${EXTRA:-}
echo "  입력        : $PARTIAL"
echo "  조건 이미지 : $IMAGE"
echo "  dataset     : $DATASET $EXTRA"

MODE=${MODE:-image}          # image | text — text 는 run_lascomp_text_condition_single.py + PROMPT + ckpt/text-xlarge
if [ "$MODE" = "text" ]; then
  echo "  모드        : text"
  echo "  프롬프트    : ${PROMPT:?PROMPT 가 비었다}"
  # shellcheck disable=SC2086
  python tools/measure_vram.py --label "$LABEL" -- \
    run_lascomp_text_condition_single.py \
      --partial-path "$PARTIAL" \
      --prompt "$PROMPT" \
      --ckpt-path ckpt/text-xlarge \
      --dataset "$DATASET" $EXTRA \
      --optimization-step "$IAS" \
      --output-dir   "results/$LABEL"
else
  # shellcheck disable=SC2086
  python tools/measure_vram.py --label "$LABEL" -- \
    run_lascomp_image_condition_single.py \
      --partial-path "$PARTIAL" \
      --image-path   "$IMAGE" \
      --dataset "$DATASET" $EXTRA \
      --optimization-step "$IAS" \
      --output-dir   "results/$LABEL"
fi
rc=$?

echo "=== 산출물 ==="
ls -la "results/$LABEL" 2>/dev/null || echo "(없음)"
echo "=== 종료 코드: $rc ==="
exit $rc          # 마지막 ls 가 0 을 내서 실패가 성공으로 보이던 것을 막는다
