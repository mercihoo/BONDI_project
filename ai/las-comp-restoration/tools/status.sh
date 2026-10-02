#!/usr/bin/env bash
# 실행 상태 한 장 — 라벨 하나를 받아 훅 로그·진행률·GPU·산출물을 보여준다.
#   bash status.sh lean-fp8-ias0-off1-mcpu1-bimba
# (Git Bash 에서 wsl 로 인라인 스크립트를 넘기면 ~ 와 $ 가 MSYS 경로 변환에 먹혀 빈 값이 된다 — 그래서 파일로 둔다)
LABEL=${1:?라벨}
L="$HOME/work/LaS-Comp/logs/$LABEL.log"
R="$HOME/work/LaS-Comp/results/$LABEL"

echo "=== $LABEL ==="
if [ ! -f "$L" ]; then echo "(로그 없음: $L)"; else
  echo "--- 훅/설정 ---"
  grep -aE "=== 설정|대상 |양자화 |IAS|오프로드|메시추출CPU|입력 |조건 이미지|dataset|훅 설치|\[양자화\]|\[mesh-cpu\]|\[offload\]|합계 절감" "$L" | head -16
  echo "--- 진행 ---"
  tr '\r' '\n' < "$L" | grep -aoE "Sampling: +[0-9]+%\|[^|]*\| [0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1
  grep -aE "num of predicted|디코딩 후|상태 |걸린 시간|프로세스 피크|torch 예약 피크|판정용 피크|남은 여유|RuntimeError|Error:|종료 코드" "$L" | tail -10
fi
echo "--- GPU / 프로세스 / RAM ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "python 프로세스: $(pgrep -fc 'python.*run_lascomp_image_condition_single')개"
free -m | awk '/Mem:/ {printf "RAM 사용 %.1f / %.1f GB\n", $3/1024, $2/1024}'
echo "--- 산출물 ---"
ls -la "$R" 2>/dev/null | grep -v '^total' | grep -v ' \.\.\?$' || echo "(없음)"
date "+지금 %H:%M:%S"
