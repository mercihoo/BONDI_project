# 데이터 카탈로그 웹 도구

3D 자산의 메타데이터와 프리뷰를 로컬 환경에서 확인하기 위한 보조 도구입니다.

```text
infra/web/server/   FastAPI API와 Three.js 화면
infra/web/local/    GLB·썸네일 생성 도구
infra/web/deploy/   일반화된 nginx·systemd 배포 예시
infra/web/agent/    자산 방향 보정 도구
```

실제 서버 주소, 계정, 인증 정보, CI/CD 연결과 운영 절차는 공개본에 포함하지 않습니다. `.env.example`을 복사해 로컬 환경에 맞는 값을 직접 설정해야 합니다.
