#!/usr/bin/env bash
# 모델 실행 뒤 자동 후처리 — 사람 손 없이: 모델 예측 색 적용 → Q4 평가 → 4 방향 메시 렌더.
#   bash tools/postprocess.sh <라벨> [유물이름=gupdari71489]
# run-gupdari.sh 가 끝에 부른다(POST=0 이면 건너뜀). 이미 있는 결과에 따로 돌릴 수도 있다.
#
# 모델(LaS-Comp/TRELLIS)은 표면을 통째로 다시 만든다 — 원본에 덧붙이는 구조가 아니다. 그래서 원본의 실측 외관을
# 결과에 남기려면 **합치는 단계가 반드시 필요하다**. 그것을 사람 손이 아니라 이 파이프라인의 고정 단계로 둔다
# (사용자 지시 2026-09-16: "원본에 추가되길 원해, 네가 합성하는 게 아니라" → 수작업이 아닌 자동 단계로).
#   ADD=0 이면 이 단계를 건너뛴다.
#
# 산출: results/<라벨>/
#   output_mesh.glb            모델 복원본 (기하만, 저자 저장 형식)
#   output_mesh_color.glb/.obj 모델 복원본 + 모델 예측 색
#   restored_with_original.glb 원본(실측, 질감 그대로) + 복원 추가분(모델 색) — 2 노드, 원본 정점 불변
#   restored_tex.glb           같은 것 + 채움부에 원본 실측 질감 (기본 blend — 톤은 높이대, 결은 실측)
#   views_*.png · eval_reports/<라벨>.json
set -uo pipefail
L=${1:?라벨}; NAME=${2:-gupdari71489}
cd "$(dirname "$0")/.."
R="results/$L"; SRC="artifacts/$NAME/source_model.glb"; P="samples/artifacts/$NAME/partial.ply"
[ -f "$R/output_mesh.glb" ] || { echo "[post] $R/output_mesh.glb 없음 — 후처리 건너뜀"; exit 1; }

# 결과 GLB 는 LaS-Comp 내부 프레임(yz 교환·거울)이다. 렌더에 쓸 프레임을 원본 표면 거리로 고른다.
FRAME=$(python tools/pick_frame.py "$SRC" "$R/output_mesh.glb" 2>/dev/null || echo swap_yz)

if [ -f "$R/output_mesh_colors.npz" ]; then
  echo "=== [post] 모델 예측 색 적용 ==="
  python tools/apply_colors.py "$R" 2>&1 | grep -vE "[Ww]arn"
fi
MESH="$R/output_mesh.glb"; [ -f "$R/output_mesh_color.glb" ] && MESH="$R/output_mesh_color.glb"
if [ "${ADD:-1}" = "1" ] && [ -f "$SRC" ]; then
  echo "=== [post] 원본에 복원 추가분 얹기 ==="
  # TAU·MINAREA 로 기준값을 바꿀 수 있다 (기본은 composite.py 의 τ 1.0 복셀 · 섬 0.3%)
  python tools/composite.py --source "$SRC" --result "$MESH" --out "$R/restored_with_original.glb" \
    --tau-voxels "${TAU:-1.0}" --min-area "${MINAREA:-0.003}" 2>&1 \
    | grep -vE "[Ww]arn" | grep -E "채택|LaS 면|채움|저장"
fi
# 채움부 질감 — 모델 예측색은 조건 이미지가 회색 렌더면 무채색 한 톤이 된다. 원본 스캔의 실측 텍스처를
# 같은 높이대에서 이어 와 채운다. TEX=0 이면 건너뛴다. TEXMODE 로 방식(blend|band|mirror),
# TEXK·TEXGAIN 으로 결의 무늬 크기·짙기를 바꾼다 (기본 12 / 0.8 — 2026-09-17 확정).
if [ "${TEX:-1}" = "1" ] && [ -f "$R/restored_with_original.glb" ]; then
  echo "=== [post] 채움부에 원본 실측 질감 입히기 ==="
  python tools/texture_fill.py --source "$SRC" --composite "$R/restored_with_original.glb" \
    --mode "${TEXMODE:-blend}" --detail-k "${TEXK:-12}" --detail-gain "${TEXGAIN:-0.8}" \
    --out "$R/restored_tex.glb" 2>&1 \
    | grep -vE "[Ww]arn" | grep -E "회전축|거울|높이대|blend|채움 색|저장"
fi
echo "=== [post] Q4 관측부 보존 ==="
python tools/eval_completion.py --result "$R" --gt "$P" --partial "$P" --label "$L" 2>&1 | grep -E "메시|관측부|복셀"

echo "=== [post] 렌더 (프레임 $FRAME) ==="
PANES=(--pane "원본 (손상 스캔)=$SRC" --pane "복원본 · 모델 출력=$MESH@$FRAME")
[ -f "$R/restored_with_original.glb" ] && PANES+=(--pane "원본 + 복원 추가분=$R/restored_with_original.glb")
[ -f "$R/restored_tex.glb" ] && PANES+=(--pane "원본 + 복원 추가분 (실측 질감)=$R/restored_tex.glb")
python tools/render_mesh_views.py "${PANES[@]}" --out "$R/views" --header "$L" 2>&1 | grep -vE "[Ww]arn"
echo "[post] 끝: $R/views_sheet.png"
