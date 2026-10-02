#!/usr/bin/env bash
# WSL Ubuntu 초기 설정 — 사용자 계정과 기본 설정. root 로 실행된다.
set -euo pipefail

USER_NAME=developer

if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "$USER_NAME"
  echo "사용자 생성: $USER_NAME"
else
  echo "사용자 이미 있음: $USER_NAME"
fi

# 이전 설정에서 부여됐을 수 있는 sudo 권한도 정리한다.
if id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx sudo; then
  gpasswd -d "$USER_NAME" sudo
fi
rm -f "/etc/sudoers.d/90-$USER_NAME"

# 기본 로그인 사용자 + systemd 활성화(서비스·빌드 도구가 기대하는 환경)
cat > /etc/wsl.conf <<EOF
[boot]
systemd=true

[user]
default=$USER_NAME

[interop]
appendWindowsPath=true
EOF

echo "--- /etc/wsl.conf ---"
cat /etc/wsl.conf
echo "--- 확인 ---"
id "$USER_NAME"
