# vr — VR 클라이언트

전시 공간, 이동·상호작용, VR 내부 UI를 담당한다.
담당 직군: **VR** (앱·상호작용·성능) · **FE** (UI/UX)

> **엔진 미확정.** Unreal Engine 5가 1순위이며 W1 게이트 G1에서 확정한다.
> 실패하면 Unity로 전환한다. 폴더 이름을 엔진에 종속되지 않게 `vr`로 둔 이유다.

## 여기에 넣는 것

- 엔진 프로젝트 (`.uproject`, `Config/`, `Content/` 또는 Unity 대응 구조)
- 전시 씬, 전시대 배치
- 이동·잡기·회전 상호작용
- 정보 패널, 원본/복원 토글 UI

## 여기에 넣지 않는 것

| 대상 | 어디로 |
| --- | --- |
| 유물 자산 원본 | `assets/<artifact-id>/` |
| 빌드 산출물 (APK) | `.gitignore` 처리됨. 공유 스토리지 또는 릴리스로 배포 |
| `Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/` | `.gitignore` 처리됨 |

## 관련 문서

- 포트폴리오 개요: 저장소 루트의 `README.md`
- 아날로그 이동 구현 기록: `docs/VR_LOCOMOTION_TROUBLESHOOTING.md`
- 프로젝트 실행 안내: `testproj/README.md`
- 기술 스택: 24.2 엔진·XR
- 성능: NFR-PERF-001 ~ 003 *(수치 미확정 — W1 실기 측정)*

## 반드시 지킬 것

- **컨트롤러가 기본 입력이다.** 손 추적은 SHOULD다. (D-07)
- **스탠드얼론 실행이다.** PC VR 스트리밍과 외부 서버 통신은 비범위다. (D-41)
- 헤드셋은 **검증된 읽기 전용 자산만 렌더링**한다. 앱 안에서 AI 추론을 하지 않는다.
- **바이너리 에셋을 동시에 편집하지 않는다.** 편집 전 팀 채널에 공지한다. (CONTRIBUTING 10절)
