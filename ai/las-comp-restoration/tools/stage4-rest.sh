#!/usr/bin/env bash
# 4단계 — 저자 requirements 중 '그냥 설치되는 것' 을 한 번에 넣는다.
#
# 한 번에 하는 이유: rembg → 또 뭐 → 또 뭐 하는 식으로 한 겹씩 벗기면 실행마다 몇 분씩 날린다.
# 걸러내는 것:
#   file:///  → 저자 로컬 빌드 경로. 우리 환경엔 없다
#   git+      → utils3d 는 이미 깔았고, chamfer_3D·emd·pytorch3d 는 따로 처리했다
#   torch 계열 → 이미 정확한 버전으로 깔려 있다. 덮어쓰면 안 된다
set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate lascomp
cd ~/work/LaS-Comp

log "=== 설치 대상 추리기 ==="
python - <<'PY' > /tmp/req-clean.txt
import re
skip_prefix = (
    "torch", "torchvision", "torchaudio", "xformers", "kaolin",
    "nvidia-", "triton", "spconv", "cumm",
)
out = []
for line in open("requirements.txt", encoding="utf-8"):
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if "@ file://" in s or "@ git+" in s:      # 저자 로컬 빌드 / 별도 처리
        continue
    name = re.split(r"[=<>!\[ ]", s, 1)[0].lower()
    if name.startswith(skip_prefix):
        continue
    if name in ("unknown", "pkg-resources"):
        continue
    out.append(s)
print("\n".join(out))
PY
wc -l < /tmp/req-clean.txt | xargs echo "  대상 줄 수:"

log "=== 설치 (버전 충돌은 무시하고 넘어간다) ==="
# 한 줄이 막아서 전부 실패하는 일이 없게, 실패한 것만 따로 모은다.
if ! pip install -q -r /tmp/req-clean.txt 2>/tmp/pip-err.txt; then
  echo "  일괄 설치 실패 — 한 줄씩 다시 시도한다"
  tail -5 /tmp/pip-err.txt
  fail=0
  while read -r pkg; do
    [ -z "$pkg" ] && continue
    pip install -q "$pkg" >/dev/null 2>&1 || { echo "    실패: $pkg"; fail=$((fail+1)); }
  done < /tmp/req-clean.txt
  echo "  개별 실패 수: $fail"
fi

log "=== torch 가 안 밀렸는지 확인 ==="
python -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available())"

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

log "=== 4단계 끝 ==="
