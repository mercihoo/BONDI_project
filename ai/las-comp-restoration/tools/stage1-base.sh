#!/usr/bin/env bash
# 1단계 — 시스템 패키지 · miniforge · conda 환경 · torch 2.4.0+cu121
# LaS-Comp 이 python 3.10 / cuda 12.1 / torch 2.4.0 으로 고정돼 있어 그대로 맞춘다.
set -euo pipefail
log() { echo "[$(date +%H:%M:%S)] $*"; }

export DEBIAN_FRONTEND=noninteractive

log "=== apt 패키지 ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential cmake ninja-build git git-lfs curl wget ca-certificates \
  pkg-config libgl1 libglib2.0-0 libx11-dev libxrandr-dev libxinerama-dev \
  libxcursor-dev libxi-dev unzip
log "apt 완료"

log "=== miniforge ==="
if [ ! -d "$HOME/miniforge3" ]; then
  wget -q -O /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm -f /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda config --set always_yes true
conda config --set channel_priority flexible
log "miniforge: $(conda --version)"

log "=== conda 환경 lascomp (python 3.10) ==="
if ! conda env list | grep -q "^lascomp "; then
  conda create -n lascomp python=3.10 -q
fi
conda activate lascomp
log "python: $(python --version)"

log "=== torch 2.4.0 + cu121 ==="
pip install -q --upgrade pip
pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
log "torch 설치 완료"

log "=== 확인 ==="
python - <<'PY'
import torch
print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("available  :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device     :", p.name)
    print("총 VRAM    : %.0f MiB" % (p.total_memory / 1024**2))
    print("capability : sm_%d%d" % (p.major, p.minor))
PY

log "=== 1단계 끝 ==="
