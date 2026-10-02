#!/usr/bin/env bash
# 절감책을 하나씩 되돌리며 개별 효과를 잰다 — 계획 §4 2→3단계의 "여유가 확인되면 하나씩 되돌린다".
#
#   bash tools/ablate.sh bimba "fp8 0 1" "off 0 1" "fp8 1 1" "int8 0 1"
#
# 인자: 대상 이름, 그리고 "양자화 IAS스텝 오프로드" 삼중항들. 순서대로 **하나씩** 돈다(GPU 는 하나다).
# 각 실행은 로그를 파일에 직접 쓴다 — 파이프로 묶으면 끝날 때까지 버퍼에 갇혀 진행을 못 본다(당했다).
# 산출물(메시)이 나온 실행은 곧바로 eval_completion.py 로 Chamfer·관측부 보존을 잰다.
# 끝나면 표 하나로 모아 ablation_summary.md 에 남긴다.
set -uo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate lascomp
cd ~/work/LaS-Comp

OBJ=${1:?대상 이름}; shift
[ $# -gt 0 ] || { echo "삼중항을 하나 이상 줘야 한다: \"fp8 0 1\""; exit 2; }

GT="samples/CompC_datasets/plyobj/gtdata/$OBJ.ply"
PART="samples/CompC_datasets/plyobj/indata/$OBJ.ply"
mkdir -p logs eval_reports
SUMMARY="ablation_summary.md"

for tuple in "$@"; do
  # "양자화 IAS 오프로드 메시CPU" — 마지막 둘은 생략하면 1
  read -r QUANT IAS OFF MCPU <<<"$tuple"
  OFF=${OFF:-1}; MCPU=${MCPU:-1}
  LABEL="lean-${QUANT}-ias${IAS}-off${OFF}-mcpu${MCPU}-${OBJ}"
  LOG="logs/$LABEL.log"
  echo "════════ $LABEL ════════"
  # GPU 가 비어 있는지 확인 — 앞 실행이 남아 있으면 기다린다
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 600 ] && break
    echo "  GPU 사용 중 (${used} MiB) — 20초 대기 ($i/30)"; sleep 20
  done
  t0=$(date +%s)
  bash tools/run-lean.sh "$OBJ" "$QUANT" "$IAS" "$OFF" "$MCPU" > "$LOG" 2>&1
  rc=$?
  echo "  종료 코드 $rc · $(( $(date +%s) - t0 ))초 · 로그 $LOG"
  grep -E "프로세스 피크|상태 |\[offload\]|RuntimeError" "$LOG" | head -5 | sed 's/^/    /'

  if ls "results/$LABEL"/*.obj "results/$LABEL"/*.ply >/dev/null 2>&1 \
     && ls "results/$LABEL"/ | grep -vq '^input_'; then
    python tools/eval_completion.py --result "results/$LABEL" --gt "$GT" --partial "$PART" --label "$LABEL" \
      2>&1 | grep -E "Chamfer|관측부|복셀|메시" | sed 's/^/    /'
  else
    echo "    (산출물 없음 — 평가 생략)"
  fi
done

# ── 요약표 ──
python - "$OBJ" "$@" <<'PY' > "$SUMMARY"
import json, sys, os
obj, triples = sys.argv[1], sys.argv[2:]
print("# 절감책 소거 실험 — %s\n" % obj)
print("| 라벨 | 양자화 | IAS | 오프로드 | 메시CPU | 상태 | 판정 피크 MiB | 시간 s | Chamfer(정규화) | 관측부 p95 | 복셀 밖 % |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for t in triples:
    parts = t.split()
    q, ias = parts[0], parts[1]
    off = parts[2] if len(parts) > 2 else "1"
    mcpu = parts[3] if len(parts) > 3 else "1"
    lab = "lean-%s-ias%s-off%s-mcpu%s-%s" % (q, ias, off, mcpu, obj)
    v = json.load(open("vram_reports/%s.json" % lab)) if os.path.exists("vram_reports/%s.json" % lab) else {}
    e = json.load(open("eval_reports/%s.json" % lab)) if os.path.exists("eval_reports/%s.json" % lab) else {}
    peak = v.get("effective_peak_mib") or v.get("peak_process_mib", "-")
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        lab, q, ias, off, mcpu, v.get("status", "-"), peak, v.get("elapsed_sec", "-"),
        e.get("chamfer_l2_normalized", "-"), e.get("observed_keep_p95", "-"),
        ("%.1f" % (100 * e["observed_outside_one_voxel_ratio"])) if "observed_outside_one_voxel_ratio" in e else "-"))
PY
echo; echo "요약: $SUMMARY"; cat "$SUMMARY"
