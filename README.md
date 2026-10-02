# 본디 (BONDI)

> 훼손된 문화유산을 AI로 복원하고, 복원 전후를 직접 비교하는 VR 박물관

![Unreal Engine](https://img.shields.io/badge/Unreal_Engine_5.8-0E1128?style=flat-square&logo=unrealengine&logoColor=white)
![OpenXR](https://img.shields.io/badge/OpenXR-3E4E88?style=flat-square&logoColor=white)
![Blender](https://img.shields.io/badge/Blender-F5792A?style=flat-square&logo=blender&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Three.js](https://img.shields.io/badge/Three.js-000000?style=flat-square&logo=threedotjs&logoColor=white)
![Claude Blender MCP](https://img.shields.io/badge/Claude-Blender_MCP-D97757?style=flat-square&logo=claude&logoColor=white)

| 구분 | 내용 |
| --- | --- |
| 프로젝트 | SSAFY 15기 특화 프로젝트 · AI(영상) |
| 개발 기간 | 5주 |
| 팀 구성 | 7명 |
| 시연 기기 | Meta Quest 2 (PC 무선 스트리밍) |
| 담당 | 김신정 · VR 썸스틱 이동, 설명 패널 UI |

## 프로젝트 소개

본디는 사진과 3D 스캔을 바탕으로 문화유산의 손상된 형상과 색상을 복원하고, 그 결과를 Unreal Engine 기반 VR 박물관에서 체험하도록 만든 프로젝트입니다.

완성된 모델 하나만 보여주는 대신 손상 상태와 복원 상태를 같은 자리에서 전환해 보여줍니다. 또한 실제 관측 자료와 AI가 추정한 영역을 구분해 복원 근거와 한계를 함께 전달하는 것을 목표로 했습니다.

![본디 VR 박물관의 안내 패널과 포털](docs/images/bondi-vr-museum-guide.png)

> VR 박물관 내 이동·조작·유물 관람 안내 패널과 전시 공간

## 주요 기능

> 아래 내용은 팀 전체 구현 범위입니다. 김신정의 개인 기여는 다음 섹션에 별도로 정리했습니다.

- 단일 이미지 기반 3D 생성과 모델별 결과 비교
- 회전 대칭 유물의 결손 형상 및 색상 복원
- 관측 영역과 AI 추정 영역을 구분한 결과 관리
- 메인 홀과 테마 전시관으로 구성된 VR 박물관
- 유물 잡기·회전 및 손상/복원 상태 전환
- 전시 정보 패널과 포털 기반 공간 이동
- FastAPI·Three.js 기반 3D 자산 카탈로그

## 담당 역할 — 김신정

### 1. 아날로그 썸스틱 연속 이동

- `IA_MoveStick`의 입력 키를 Quest 2용 2D 썸스틱 키로 수정
- SteamVR 컨트롤러 호환용 2D 썸스틱 키도 함께 설정
- 프로젝트별 SteamVR/OpenXR 바인딩을 갱신해 연속 이동 입력 확인

### 2. 설명 패널 UI

- 유물 설명창과 전시실 안내 패널의 라운드 프레임 디자인 통일
- VR 공간에서 패널의 형태와 가독성이 일관되도록 UI 스타일 조정

## Troubleshooting

### 썸스틱 입력값이 `(0, 0)`으로 들어오는 문제

```text
Quest 2 썸스틱
  → ALVR
  → SteamVR / OpenXR
  → TESTPROJ 컨트롤러 바인딩
  → IMC_Move
  → IA_MoveStick
```

원인은 이동 액션이 X/Y 개별 입력 키로 설정되어 SteamVR/OpenXR가 2D 썸스틱으로 해석하지 못한 것이었습니다. `OculusTouch_Right_Thumbstick_2D`와 호환용 2D 키를 설정하고 `TESTPROJ` 전용 바인딩 캐시를 초기화한 뒤, ALVR·SteamVR·Unreal을 다시 실행해 해결했습니다.

자세한 분석과 검증 과정은 [VR 썸스틱 이동 트러블슈팅](docs/VR_LOCOMOTION_TROUBLESHOOTING.md)에 정리했습니다.

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| VR | Unreal Engine 5.8, Blueprint, OpenXR, Enhanced Input |
| AI·3D | Python, PyTorch, Open3D, trimesh, Blender |
| 자산 카탈로그 | FastAPI, Three.js |
| HMD·스트리밍 | Meta Quest 2, SteamVR, ALVR |
| 협업 | Git, Git LFS, Claude, Blender MCP |

## 저장소 구조

```text
├── ai/       2D→3D, 형상·색상 복원 및 평가
├── assets/   전시 자산과 출처 기록
├── docs/     기술 문서와 트러블슈팅
├── infra/    3D 자산 카탈로그
├── tools/    Blender·Unreal 보조 스크립트
└── vr/       Unreal Engine VR 프로젝트
```

## 실행 방법

1. Git LFS가 설치된 환경에서 저장소를 복제합니다.
2. `vr/testproj/testproj.uproject`를 Unreal Engine 5.8로 엽니다.
3. PC에서 OpenXR 런타임을 SteamVR로 사용하고 ALVR 대시보드를 실행합니다. SteamVR가 실행되는지 확인합니다.
4. Meta Quest 2를 PC와 같은 무선 네트워크에 연결하고 헤드셋에서 ALVR 앱을 실행합니다. ALVR 대시보드에서 헤드셋 연결을 확인합니다.
5. Unreal Editor에서 플레이 모드를 **VR 프리뷰**로 선택해 실행합니다. PC의 Unreal Engine 화면이 SteamVR·ALVR을 거쳐 Quest 2로 무선 스트리밍됩니다.

네트워크와 ALVR 권장 설정은 [VR 스트리밍 설정](docs/vr-streaming-setup.md)을 참고합니다.

상용 또는 재배포 조건이 있는 일부 외부 에셋은 포함하지 않았으므로 Unreal Editor에서 일부 참조가 누락될 수 있습니다. 자세한 내용은 [외부 자산 안내](THIRD_PARTY_NOTICES.md)를 확인해 주세요.

## License

이 저장소는 팀 프로젝트의 포트폴리오 열람을 목적으로 공개합니다. 세부 이용 조건은 [LICENSE.md](LICENSE.md)를 확인해 주세요.
