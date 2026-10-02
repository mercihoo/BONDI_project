#!/usr/bin/env bash
# 저자 샘플 — 부분 입력 · 정답지 · 조건 이미지 삼종을 받는다.
# Q1(도는가)과 Q3(양자화 전후 Chamfer 비교)에 둘 다 쓴다. 정답지가 있어야 수치가 나온다.
set -euo pipefail
REPO=DavidYan2001/Omni-Comp3D
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"
DST=~/work/LaS-Comp/samples

# 형태가 다른 셋 — 하나만 보면 우연일 수 있다
OBJS=(bimba cow armadillo)

mkdir -p "$DST/CompC_datasets/plyobj/indata" \
         "$DST/CompC_datasets/plyobj/gtdata" \
         "$DST/image_from_compc/plyobj"

get() {  # get <원격경로> <로컬경로>
  if [ -s "$2" ]; then echo "  이미 있음: $(basename "$2")"; return; fi
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 2 -o "$2" "$BASE/$1" \
    && echo "  받음: $(basename "$2") ($(du -h "$2" | cut -f1))" \
    || echo "  실패: $1"
}

for o in "${OBJS[@]}"; do
  echo "[$o]"
  get "samples/CompC_datasets/plyobj/indata/$o.ply"  "$DST/CompC_datasets/plyobj/indata/$o.ply"
  get "samples/CompC_datasets/plyobj/gtdata/$o.ply"  "$DST/CompC_datasets/plyobj/gtdata/$o.ply"
  get "samples/image_from_compc/plyobj/${o}_color.png" "$DST/image_from_compc/plyobj/${o}_color.png"
done

echo
echo "=== 결과 ==="
find "$DST" -type f \( -name '*.ply' -o -name '*.png' \) -printf '%10s  %p\n' | sed "s#$HOME/work/LaS-Comp/##"
echo "=== 끝 ==="
