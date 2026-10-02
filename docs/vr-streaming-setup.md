# VR 스트리밍 성능 세팅 — 다른 PC·다른 헤드셋에서 재현하기

2026-09-20 최적화 결과를 다른 노트북과 다른 Meta Quest 2에서 똑같이 재현하기 위한 문서.

측정된 효과 (같은 씬, 같은 헤드셋):

| 지표 | 최적화 전 | 최적화 후 |
| --- | --- | --- |
| Client FPS | 11 | **71** (72Hz 상한) |
| Total latency | 467 ms | **59 ms** |
| Packet loss | 다수 | **0** |
| 프레임 간격 20ms 초과 비율 | — | 1.4% |

> 프레임 간격은 ALVR `session_log.txt`의 프레임 이벤트 타임스탬프 69,779건(16분 18초)을
> 직접 집계한 값이고, Client FPS·latency는 ALVR 대시보드 Statistics 탭 값이다. 두 방법이
> 71.4 / 71 로 일치했다.

---

## 0. 무엇이 레포에 있고 무엇이 없는가

**레포에 있음 — 클론하면 자동 적용. 할 일 없음.**

| 항목 | 위치 |
| --- | --- |
| 그림자·드로우디스턴스·폴리지 설정 | `vr/testproj/Config/DefaultEngine.ini` → `[SystemSettings]` |
| 나무 LOD (124k → ~30k tri), 컬링 거리 | `SM_Mobile_Trees`, `L_Room_C`의 나무 64그루 |
| 텍스처 2048 제한, 지형 동적 그림자 끔 | `T_Gounsa_Platform`, `T_Gounsa_OutBare`, `L_Room_C` |
| 미러 창 넓게 보기 (`vr.SpectatorScreenMode 1`) | `BP_SpawnFacing` 시작 시 자동 실행 |
| 로프 상호작용, 스폰 위치·방향 | `BP_RopePull`, `BP_SpawnFacing`, `L_Museum` |

**레포에 없음 — PC마다 수동 적용 필요. 이게 이 문서의 본론.**

| 항목 | 왜 없나 |
| --- | --- |
| **ALVR 설정 전부** | ALVR 설치 폴더의 `session.json`에만 존재 |
| 에디터 플레이 모드 설정 | `vr/testproj/Saved/`는 `.gitignore` 7번 줄에서 제외 |

성능 개선의 **가장 큰 몫이 ALVR 설정에서 나왔다.** 이걸 안 하면 다른 PC에서 다시 11fps를 본다.

> **렌더 해상도는 언리얼이 아니라 ALVR이 정한다.**
> `DefaultEngine.ini`에 `xr.SecondaryScreenPercentage.HMDRenderTarget=75`를 넣어두었지만,
> PIE가 시작될 때마다 100으로 되돌아간다 (에디터에서 값을 직접 조회해 확인).
> 눈당 해상도가 2144×2336 → 1376×1536으로 떨어진 것은 전적으로 ALVR의
> `emulated_headset_view_resolution = 1400` 때문이다. 이 ini 항목은 현재 아무 효과가 없으므로
> 다른 PC에서 해상도를 조절할 때는 **2절의 ALVR 설정만 보면 된다.**

---

## 1. 전제 조건

- **ALVR 20.14.0** (스트리머 + 헤드셋 클라이언트, 버전이 서로 같아야 함)
- SteamVR 설치, OpenXR 런타임이 SteamVR로 지정되어 있을 것
- Unreal Engine 5.8
- **5GHz 무선 연결**. 2.4GHz로는 이 수치가 안 나온다.
  - 검증에 쓴 환경: 5GHz 스마트폰 핫스팟, 802.11ac, 링크 약 866 Mbps
  - 핫스팟을 쓸 경우 **핫스팟 절전 모드를 끌 것** (휴대폰 자체 절전은 켜둬도 무관)
  - PC와 헤드셋이 **같은 AP**에 붙어 있어야 한다

---

## 2. ALVR 설정 (핵심)

### 2-a. 스크립트로 한 번에 적용 — 권장

```bash
python tools/alvr/apply_alvr_settings.py "<ALVR 설치 폴더>"
```

**반드시 ALVR 대시보드를 완전히 종료한 뒤 실행할 것.** 대시보드는 종료할 때 자기 메모리의
설정으로 `session.json`을 덮어쓴다. 실제로 이 작업 중 두 번 설정이 되돌아갔다.

순서:

1. ALVR 대시보드 종료 (트레이 아이콘까지)
2. 스크립트 실행 (원본은 `session.json.bak-<날짜시각>`으로 백업됨)
3. 대시보드 재실행
4. 스크립트를 `--verify`로 다시 실행해 값이 살아있는지 확인

### 2-b. 대시보드 GUI로 직접 넣기

Settings 탭에서:

| 위치 | 항목 | 값 | 이유 |
| --- | --- | --- | --- |
| Video | Preferred codec | **HEVC** | H.264 대비 같은 화질을 절반 비트레이트로. 대역폭 병목 해소 |
| Video | Bitrate mode | **Constant 15 Mbps** | Adaptive가 요동치며 큐를 쌓았음 |
| Video | Encoder → Entropy coding | **CABAC** | CAVLC보다 압축률 높음 |
| Video | Transcoding view resolution | **Absolute, width 1400** | 원본 2752는 이 GPU로 인코딩이 못 따라감 |
| Video | Emulated headset view resolution | **Absolute, width 1400** | 위와 같은 값으로 맞출 것 |
| Video | Preferred framerate | **72** | Quest 2 기본 주사율. 헤드셋 쪽에서 바꿀 건 없음 |
| Video | Foveated encoding | **켬**, center 0.40 × 0.35, shift 0.40 × 0.10, edge 4.0 × 5.0 | 시야 주변부 비트 절약 |
| Headset | Position recentering | **Local, view height 1.6 m** | `Local floor`면 공중에 뜬다 |
| Headset | Rotation recentering | **Yaw** | |

### 2-c. 아직 적용하지 않은 개선 항목

측정 결과 남은 지연 59.66ms의 내역:

```
Frame Buffering   21.94ms  ███████████████ 37%   ← ALVR 큐 대기
Client System     19.81ms  █████████████   33%   ← 퀘스트 내부 컴포지터 (손댈 수 없음)
Network            7.66ms  █████
Encode             3.75ms  ██
Game Render        2.69ms  █                     ← 언리얼. 병목 아님
Decode             2.66ms  █
```

`max_buffering_frames`가 현재 **2.0**이다. 72Hz에서 2프레임 = 27.8ms로 측정값과 일치한다.
**1.0으로 낮추면 60ms → 45ms 내외**가 기대된다. 너무 낮추면 순간적인 네트워크 끊김에
프레임을 놓쳐 화면이 튀므로, 튀면 1.5로 되돌린다. (미검증 — 다음 세션에서 확인 예정)

---

## 3. 에디터 설정 (PC마다 1회)

`vr/testproj/Saved/`는 git에서 제외되므로 클론 직후 한 번 지정해야 한다.

툴바의 ▶ 옆 **⋮** 메뉴에서:

- 모드: **VR 프리뷰**
- 다음 위치에 플레이어 스폰: **디폴트 플레이어 스타트**

> 두 번째 항목이 "현재 카메라 위치"로 되어 있으면 `PlayerStart`가 무시되고 에디터 뷰포트
> 카메라 위치에서 시작한다. 이 값은 저절로 되돌아간 적이 있으니 이상하면 여기부터 확인할 것.

---

## 4. 실행 순서

1. 스마트폰 핫스팟(5GHz) 켜기 — 절전 모드 끔
2. PC를 그 핫스팟에 연결
3. ALVR 대시보드 실행 → SteamVR 자동 기동 확인
4. 헤드셋을 같은 핫스팟에 연결 → ALVR 앱 실행 → 대시보드에 기기가 뜨는지 확인
5. 언리얼 에디터에서 ▶ (VR 프리뷰)

---

## 5. 검증 — 이 숫자가 나와야 성공

ALVR 대시보드 **Statistics** 탭 하단:

| 항목 | 기대값 |
| --- | --- |
| Client FPS | **70~72** |
| Streamer FPS | 72~76 |
| Total latency | **60 ms 이하** |
| Total packets lost | **0** |
| Bitrate | 3~5 Mbps (HEVC라 낮은 게 정상) |

`Latency` 그래프에서 커서를 **오른쪽 끝(안정 구간)**에 올리면 항목별 내역이 뜬다.
`Game Render`가 13.9ms를 넘으면 그때는 언리얼 쪽 최적화가 필요하다는 뜻이다.
(현재 2.69ms이므로 언리얼은 여유가 많다.)

---

## 6. 겪었던 문제와 해결

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| ALVR 설정이 저절로 되돌아감 | 대시보드가 종료 시 `session.json`을 덮어씀 | 대시보드를 끈 상태에서만 수정하고, 재시작 후 재확인 |
| VR에서 공중에 떠 있음 | Position recentering이 `Local floor` | `Local`, view height 1.6 m |
| ALVR "Launch SteamVR"이 반응 없음 | `vrserver`는 살아있고 `vrcompositor`/`vrmonitor`만 죽은 좀비 상태 | 세 프로세스를 모두 종료 후 `vrstartup.exe` 실행 |
| VR 프리뷰 종료 시 에디터 크래시 | 헤드셋 연결이 끊긴 상태에서 PIE 종료 (`vrclient_x64` assert) | 헤드셋 연결을 유지한 채 PIE를 먼저 종료 |
| 미러 창이 확대돼 보임 | 기본 `SingleEyeCroppedToFill`이 한쪽 눈을 창 비율에 맞춰 잘라냄 | `vr.SpectatorScreenMode 1` — `BP_SpawnFacing`에 넣어뒀으므로 자동 |
| ping 손실률로 네트워크를 판단 | Quest가 스트리밍 중 ICMP를 후순위로 처리 (ping 83~92% 손실인데 ALVR 패킷 손실 0) | **ping을 믿지 말고 ALVR Statistics를 볼 것** |

---

## 7. 다른 헤드셋을 쓸 때

Quest 쪽은 건드린 것이 없다. 72Hz도 헤드셋 설정이 아니라 **ALVR이 협상하는 값**이라
(`session_log.txt`에 `Refresh Rate: 72`로 기록됨), 같은 버전의 ALVR 클라이언트만 설치하면
다른 Quest 2에서도 동일하게 동작한다. 헤드셋에서 확인할 것은 두 가지뿐이다.

- ALVR 클라이언트 버전이 스트리머와 같은 **20.14.0**인가
- **5GHz** 네트워크에 붙어 있는가 (2.4GHz면 이 수치가 안 나온다)

---

## 8. 측정 이력

최적화 전 기준값은 프로젝트 실기 테스트에서 측정한 결과를 사용했다. 개인별 작업 기록은 공개본에서 제외했다.

| 구간 | 09-17 | 09-20 |
| --- | ---: | ---: |
| `game_time` (언리얼 렌더) | 41.3 ms | **2.69 ms** |
| Total latency | 157.4 ms (최대 414.5) | 59 ms |
| Network | 48.1 ms (최대 273.5) | 7.66 ms |
| Encode | 14.3 ms | 3.75 ms |
| Decode | 6.0 ms | 2.66 ms |
| Client FPS | 57.6 (최저 7) | 71 |
| `vsync_queue` / Frame Buffering | 36.6 ms | 21.94 ms |
| 눈당 해상도 | 2144 × 2336 | 1376 × 1536 |
| 코덱 / 실측 비트레이트 | H.264 / 21.6 Mbps | HEVC / 3.5 Mbps |

그 문서가 제시한 개선 순서 5개 중 1(렌더 시간), 2(주사율·해상도 조합), 3(5GHz 전용 AP)이
해결되었고, 4(`vsync_queue`)는 2-c절의 `max_buffering_frames`로 남아 있다.
5(FOV margin 확대)는 **지연이 157 ms → 59 ms로 줄어 재투영이 벗어날 거리가 작아졌으므로
더 이상 필요하지 않을 수 있다.** 고개를 빠르게 돌릴 때 외곽이 검게 보이는지 먼저 확인할 것.

> **해상도를 다시 올릴 여유가 있다.** 72 Hz의 프레임 예산은 13.9 ms인데 렌더가 2.69 ms만
> 쓰고 있다. 프레임을 지키려고 1400까지 내렸지만 지금은 과하게 깎은 상태이므로,
> 1800 전후로 올려 화질을 되찾는 편이 낫다. 다만 인코딩·네트워크 부담이 함께 늘고 이는
> 현장 네트워크에 좌우되므로 **시연 환경에서 측정한 뒤 정할 것.**
