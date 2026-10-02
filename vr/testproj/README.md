# testproj — UE5 VR 클라이언트

빗살무늬토기(`ssu022891`) 전시 프로토타입. UE **5.8.2** · VR Template 기반.

| 항목 | 값 |
| --- | --- |
| 시작 맵 | `/Game/XRFramework/Levels/L_Museum` |
| 게임모드 · Pawn | `BP_XRGameMode` · `BP_XRPawn` |
| 대상 기기 | Oculus Quest 2 (`D-44`) |

## 처음 clone 했다면

```bash
git lfs install      # 한 번만. 안 하면 에셋이 포인터 텍스트로 내려와 열리지 않는다
```

`.uasset`과 `.umap`은 `lockable`이라 **읽기 전용으로 내려온다.** 편집하려면 먼저 잠근다.

```bash
git lfs lock   vr/testproj/Content/Museum/Blueprints/BP_StageToggle.uasset
git lfs unlock vr/testproj/Content/Museum/Blueprints/BP_StageToggle.uasset
```

## 저장소에 넣지 않은 것

| 대상 | 이유 | 어디에 있나 |
| --- | --- | --- |
| `Saved/` `Intermediate/` `DerivedDataCache/` | 생성물 (1.2 GB) | `.gitignore` |
| `Content/` 안의 `.obj` `.glb` `.jpg` 원본 | `FR-OPS-004` — 엔진 프로젝트 안은 원본 보관소가 아니다 | **`assets/<artifact-id>/`** 로 옮겼다 |
| 미배치 유물 `bon004740` · `bon009435` 의 `.uasset` | 레벨에 배치되지 않았고, 노멀맵이 `TC_Default`+sRGB로 잘못 설정돼 재임포트가 필요하다 | 원본은 `assets/bon*/source/` |
| `MalgunGothic.ttf` | 아래 "한글 폰트" 참조 | `C:\Windows\Fonts\malgun.ttf` |

임포트된 `.uasset` 메시·폰트는 그대로 들어 있으므로 **원본 없이도 열리고 동작한다.**
다시 임포트해야 할 때만 원본이 필요하다.

> `2-model-restored.glb`(복원 ②단계)는 아직 임포트하지 않았다. 3단계 토글(`D-47`)을
> 완성할 때 `assets/ssu022891/master/` 에서 가져와 임포트한다.

## 한글 폰트

`Content/Museum/Fonts/MalgunGothic.uasset` (13.4 MB) 에 **글리프가 임베드돼 있어
clone 만 하면 한글이 정상 표시된다.** 별도 설치가 필요 없다.

폰트를 **다시 임포트해야 할 때만** 원본이 필요하다. 맑은 고딕은 Windows 기본 폰트이므로
다운로드가 아니라 시스템에서 가져온다.

```text
C:\Windows\Fonts\malgun.ttf      (Bold 는 malgunbd.ttf)
```

> ⚠️ **배포 전 교체 대상.** 맑은 고딕은 Windows 시스템 폰트이고 재배포 권한이 없다.
> `.ttf` 를 저장소에서 뺐지만 `.uasset` 에 글리프가 들어 있어 **실질적으로는 이미 재배포 상태**다.
> 내부 개발 중에는 그대로 쓰되, 외부 배포·공개 시연 전에 라이선스가 명확한 한글 폰트로 바꾼다.
> 후보: 본고딕 / Noto Sans KR (SIL OFL), 프리텐다드 (SIL OFL).
> 교체 시 `WBP_ArtifactInfo` 의 `Text_*` 블록 5개와 `MalgunGothic_Font` 참조를 함께 바꾼다.

## VR 프레임 예산

에디터 VR 프리뷰는 원래 느리다 — 에디터 오버헤드 + 패키징 최적화 없음 + ALVR 영상 인코딩이 겹친다.
성능 판단은 패키징 빌드로 해야 한다. 그 위에 실제로 비쌌던 것 둘:

- **안내 패널 렌더 타깃 (2026-09-11 수정)** — `WidgetComponent.RedrawTime = 0` 은 "매 프레임 다시 그린다"는
  뜻이다. 700×900 짜리 슬레이트 렌더 타깃 5장을 매 프레임 갱신하고 있었다. 글자는 바뀌지 않으므로
  `RedrawTime = 1.0` (초당 1회)로 낮췄다. **페이드는 렌더 타깃이 아니라 머티리얼 파라미터**
  (`TintColorAndOpacity`)라서 그대로 매끄럽다.
- **`BP_HoloGuide` 틱 (2026-09-11 수정)** — 알파가 바뀔 때만 패널 틴트 5개·아이콘 스칼라 4개를 쓴다.
  관람자가 서 있거나 완전히 드러난 상태면 그 둘과 `GetAllActorsWithTag` 한 번이 통째로 빠진다.
  픽토그램 회전만 매 프레임 돈다.

**전역 조명은 Movable 로 둔다 — 의도된 선택이다(`D-61`).** `L_XRTemplate_Lighting` 의
`DirectionalLight` 와 `SkyLight` 가 Movable 이라 베이크와 별개로 매 프레임 그림자를 다시
계산한다(VR 이라 눈마다). 템플릿 기본값이다. Stationary 로 내리면 큰 폭으로 싸지지만
**라이팅을 다시 구워야 하고** 안 구우면 어두워진다 — 실기에서 렉이 없으므로 그 위험을 지지 않는다.
**최적화 후보로 다시 꺼내지 말 것.** 프레임이 실제로 모자랄 때가 오면 그때 여기부터 본다.
방 B·C 의 `Fill_*`·`Spot_*` 도 Movable 인데 A 방은 Static 이다 — B·C 는 아직 비계라 같이 정리한다.

## 아웃라이너 폴더

레벨 다섯 개(`L_Museum` · `L_Room_A/B/C` · `L_XRTemplate_Lighting`)가 **같은 폴더 이름**을 쓴다. 번호를 붙인 이유는 아웃라이너가 이름순으로 정렬하기 때문이다 —
껍데기 → 출입구 → 읽을 것 → 장식 → 빛 → 배관 순으로, 처음 열어 본 사람이 찾는 순서대로 놓인다.

| 폴더 | 들어가는 것 |
| --- | --- |
| `01_Structure` | 바닥 · 벽 · 천장 · 기둥 (방은 배경벽 포함) |
| `02_Portals` | `BP_Portal` 과 그 아치 |
| `03_Guide` (홀) | 안내 패널 5장 · 픽토그램 4개 · `Hall_HoloGuide` |
| `03_Exhibit` (방) | 유물 · 좌대 · 단 · 버튼 · 설명판 · 제목판 |
| `04_Props` | 가구 · 식물 · 러그 · 피처월 — **보이는 것** |
| `05_Lighting` | 라이트 · 코브 · 리플렉션 캡처 · 포스트프로세스 — **안 보이는 것** |
| `06_System` | PlayerStart · 스펙테이터 · 내비메시 볼륨 |

규칙은 하나다 — **형상으로 렌더되면 Structure/Props, 빛이나 볼륨이면 Lighting.**
그래서 샹들리에 **메시**는 `04_Props`, 그 안의 **라이트**는 `05_Lighting` 으로 갈린다.
둘은 서로 영향을 준다(기구 메시가 자기 빛을 가려 천장이 검게 나온 적이 있다, `D-55`) — 조명을 만질 때는 두 폴더를 같이 본다.

`Brush0` · `WorldSettings` · 내비메시 데이터 같은 엔진 내부 액터는 폴더에 넣지 않았다(넣을 수 없거나 의미가 없다).

## 알려진 상태 (2026-09-09 저녁)

- 실기(Quest 2 · SteamVR OpenXR) 검증 **완료** — 손끝 버튼 누름, 핀치 집기·이동·회전, 관절 구체 손 표시 (MR !16).
  손은 관절 구체 26개로 그린다. 템플릿 장갑 메시는 항상 숨김 — Live Link 리타게팅은 스위즐 축 보정 단계에서 보류
  (`LLRA_HandLeft/Right`, `LLP_HandTracking` 커밋됨).
- 라이팅은 **베이크 대상**으로 전환했다(22.2.1 갱신 2026-09-10). 홀·테마관 A의 구조·가구는 Static, 키 라이트는 Stationary, 필은 Static이고
  라이트맵 해상도(바닥·천장 1024, 벽 512, 가구 128~256)와 리플렉션 캡처(홀 3, A방 2)가 들어가 있다.
  레벨을 고친 뒤에는 **Build → Build Lighting Only**(CPU Lightmass)를 눌러야 하며, 빌드 전에는 `LIGHTING NEEDS TO BE REBUILT`가 뜬다.
  베이크 결과 `L_*_BuiltData.uasset`은 저장소에 넣는다(clone 직후 빌드 없이 열어도 조명이 맞게). Preview 품질로 구워져 있고, 외부 시연 전에 Production 으로 다시 굽는다.
  포스트프로세스 `PP_Gallery`는 무한 범위·블룸 0.35·모션블러/비네트/그레인 0으로 전 레벨에 적용된다.
  노출은 **프로젝트 고정값**을 쓰고 `AutoExposureBias`만 0으로 둔다 — 포스트프로세스에서 `AEM_Manual`로 바꾸면 물리 카메라 EV가 적용돼 실내가 새까맣게 나온다.
- 현재 최종본에는 김신정이 구현한 **아날로그 스틱 연속 이동**이 반영되어 있다.
  입력 매핑과 충돌 처리 과정은 `docs/VR_LOCOMOTION_TROUBLESHOOTING.md`에 정리했다.
- 복원 단계가 2단계다. 3단계는 ②단계 에셋 임포트 후.
- **공간 구조 1차 구현** (`D-54`, 명세 21.1.1) — `L_Museum` = 메인 홀(7×9×3.2 m, 시작점→포털 4 m, 안내 패널 5장·먼 벽에 포털 3개),
  안내 패널은 `WBP_HallGuide_1_Welcome`~`5_Portal`(환영·이동·손 조작·관람·포털)을 시작점 좌우 ±2.5 m에 두 줄로 번갈아 세우고(1·3 왼쪽, 2·4 오른쭉), 5번은 포털 A·B 사이 벽 앞에 둔 튜토리얼 동선.
  문구는 위젯의 `Text_*` 블록에 직접 들어 있다. AI 복원 고지는 모든 패널 하단.
  `Content/Museum/Levels/L_Room_A/B/C` = 테마관. **스트리밍 서브레벨**(`LevelStreamingDynamic`, Initially Loaded/Visible, 레벨 트랜스폼 (5000, ∓6000/0, 0))로 홀에 붙어 있다.
  처음엔 Level Instance였지만 인스턴스는 임시 복사본으로 로드돼 베이크 라이팅이 유지되지 않아(빌드 데이터 GUID 충돌) 2026-09-10에 서브레벨로 바꿨다.
  A는 기존 방, B·C는 A의 복제(비계).
  방 사이 이동은 `BP_Portal` 텔레포트 — 세워진 2.2×1.4 m 게이트에 `M_Portal`(Unlit·Translucent, 시간 노이즈 이미시브 + 가장자리 발광)을 입혀
  일렁이는 차원문으로 표시하고, 인스턴스마다 `PortalColor`로 색을 구분한다(홀→A 파랑·B 보라·C 초록, 방→홀 호박색). 머리가 **걸어서** 반경 70 cm 안에 들어온 순간 발동(한 틱 3 m 이상 점프로 들어온 경우는 무시 — 도착 직후 되튐 방지).
  홀→테마관 도착은 **짝 포털을 이미 빠져나온 자리** — 머리(HMD)가 짝 포털 면에서 방 안쪽으로 1 m 나온 지점에, 방 안쪽을 보는 방향으로 놓인다(HMD 오프셋·시선 보정, `RotateVectorAroundAxis`).
  테마관→홀 도착은 **(280, 0), 포털 쪽을 봄** 이다. 페이드 0.08/0.12 s. 원래 (350,0)이었는데, 그때는 홀에 안내 패널이 없었다 — 패널 호가 생기고 나서 정면 170 cm 에 3번 패널이 놓여 규칙을 깨고 있었다(`D-57`).
  처음엔 게이트 30 cm **뒤**에 두어 한 걸음 걸어 나오게 했지만, **메타 버튼으로 뷰를 리센터하면** 트래킹 원점이 초기화돼 관람자가 폰 원점으로 튀고 그 자리가 게이트 안이나 벽이 됐다(2026-09-10 수정). 게이트 앞 1 m는 발동 반경 70 cm 밖이라 되튀지도 않는다.
  **도착 지점은 사방으로 2 m 이상 비어 있어야 한다** — 리센터가 관람자를 폰 원점으로 보내는데, 그 거리는 플레이 공간에서 얼마나 걸어 다녔는지에 달려 있어 예측할 수 없다. 후보 지점마다 `SceneTools.trace_world` 로 사방 거리를 실측해서 고른다(벽뿐 아니라 소파·기둥·안내 패널까지 잡힌다).
  그래서 2026-09-11에 **입장 동선 전체를 +70 cm 앞으로** 옮겼다 — 시작점 150→220, 안내 패널 5장, 포털·아치 550→620. 시작점→포털 4 m는 그대로 두면서 시작점 뒤 여유를 145→215 cm로 늘렸다. 아치 뒷면(645)과 동쪽 벽(700) 사이는 55 cm 남는다.
  테마관의 방→홀 포털은 유물 정면 3.8 m(로컬 x −380)에 서 있고 도착 지점은 유물 정면 2.8 m다 — 발굴 단(반경 1.78 m) 앞에 여유를 두어야 리센터로 폰이 밀려도 단이나 좌대 안에 서지 않는다. 실기 미검증.
- **리센터는 제자리에 둔다 (2026-09-11, `D-57`)** — 포털 왕복 뒤 메타 버튼을 누르면 엉뚱한 곳에 가 있었다.
  트래킹 원점이 `LocalFloor` 라 리센터하면 HMD 오프셋이 0 이 되고 카메라가 폰 원점으로 튄다.
  `BP_Portal` 은 카메라를 목표점에 맞추려고 폰을 `목표 − 회전된 오프셋` 에 놓으므로, 왕복하면
  폰 원점이 머리에서 2~3 m 벌어져 있다. 두 군데를 고쳤다:
  ① 세 방의 홀 복귀 지점 `TargetLoc` 을 (350,0,0) → **(280,0,0)**. 기존에는 정면 170 cm 에 3번 패널이
  있었다(위젯 컴포넌트도 트레이스에 걸린다). 지금은 사방 최소 240 cm.
  ② `BP_XRPawn` 의 `VRNotifications` 에 `HMDRecenteredDelegate`·`XRTrackingOriginChangedDelegate` 를
  물려 `MarkRecenter` → 틱의 `KeepPlaceOnRecenter` 가 `AddActorWorldOffset(직전 카메라 XY − 현재 XY)` 로
  되돌린다. yaw 는 건드리지 않는다 — 정면 재정렬이 리센터를 누른 이유다.
  2틱 미루는 것은 델리게이트와 카메라 포즈 갱신 순서가 런타임마다 다르기 때문이다.
  **`BP_XRPawn` 의 EventGraph 는 `write_graph_dsl` 로 다시 쓰면 안 된다** — 노드가 158개인데 DSL 은
  입력 이벤트 본문을 비워서 낸다. `add_event`·`add_component_bound_event`·`create_node`·`connect_pins` 로 얹었다.
  `BP_StageToggle`은 BeginPlay에서 가장 가까운 유물·패널을 자기 방 것으로 고정한다(태그 전역 검색 제거).
  각 방 `.umap`이 LFS 락 단위 — 방을 맡으면 그 파일만 잠근다. 홀은 공용이라 변경 전 공유.
- **꾸미기 1차 (2026-09-10)** — 홀은 호텔 로비(대리석 바닥·월넛 웨인스코트·황동 레일·코브 조명·샹들리에·문틀·러그·소파), 테마관 A는 신석기 토기 전시실(어두운 석재·황토 모래 링·지층 배경벽·라벨 레일·코브·제목판).
  형태는 프리미티브 컴포넌트(`PrimitiveTools`)로만 만들었다. 재질은 **Fab Quixel Megascans 표면 8장**(`Content/Fab/Megascans/Surfaces/` — Ziarat White Marble, Walnut Veneer, Rough Cement Plaster, Damaged Wall Plaster, Furniture Fabric, Slate Floor Tiles, Stone Floor, Dry Cracked Mud, 2K·Fab 표준 라이선스)을
  월드 정렬 트라이플래너 마스터 `M_Museum_Scan`(BaseColorTex·NormalTex·TileSize·Tint·Roughness, 엔진 `WorldAlignedTexture/Normal` 함수)에 물려 쓴다. 프리미티브는 UV가 면마다 0~1이라 UV 타일링으로는 결이 늘어지므로 월드 좌표 투영이 필수다.
  인스턴스 `MI_Lobby_*`·`MI_RoomA_*`가 그 자식이고(홀 벽·천장·바닥 = 대리석, 웨인스코트 = 월넛, A방 벽 = 어두운 시멘트 석고, 바닥 = 슬레이트, 모래·배경 = 마른 진흙, 좌대 = 석재), 황동·러그·식물·발광은 절차 재질 `M_Museum_Surface`·`M_Museum_Emissive` 그대로.
- **꾸미기 2차 — 건축을 모델링 메시로 교체 (2026-09-10, `D-55`)** — 홀·A방의 벽·바닥·천장과 기둥·문틀·좌대·벤치를 Blender에서 만들어 `Content/Museum/Meshes/`에 임포트했다(10종, 셸 합계 약 9,000 삼각형).
  VR 템플릿이 주던 1 m 큐브 벽(`1M_Cube*`·`TemplateFloor`)은 A방에서 전부 제거했다.
  홀은 고전 어법(걸레받이·웨인스코트 판넬·챙목 레일·코브 홈·격자 천장·반원 아치 3개), A방은 현대 미술관 어법(바닥 8 cm 그림자 홈·천장 단차·코브 홈). 모든 모서리에 1.2 cm 챔퍼.
  아치 개구부는 130×205 cm로 포털 게이트(140×220 cm)보다 좁아 일렁임이 문간을 가득 채우고 가장자리는 석재 뒤로 숨는다.
  > ⚠️ **메시를 다시 뽑을 때 UV를 빠뜨리지 말 것.** 표면 재질이 월드 정렬이라 재질용 UV는 필요 없지만 **라이트맵은 UV가 없으면 굽히지 않는다.**
  > 생성 스크립트(`hall_shell.py`·`museum_parts.py`)의 `unwrap()`이 UV0·UV1을 만든다. 임포트 후 `LightMapCoordinateIndex`가 0이면 잘못된 것이다.
- **안내 패널 = 홀로그램, 걸으면 하나씩 (2026-09-11)** — 위젯 재질을 `M_HoloPanel`로 덮어써 판때기를 지우고 글자만 빛으로 띄운다.
  원리: 엔진 위젯 재질이 노출하는 `SlateUI`(Texture) 파라미터를 내 재질에 같은 이름으로 두면 UE가 위젯 렌더타겟을 넣어 준다.
  불투명도를 알파가 아니라 **휘도**(`dot(rgb, .3/.59/.11)`)에서 뽑아 짙은 배경을 지우고, `FieldColor`로 옅은 어두운 유리면을 깔아 흰 대리석 위에서도 읽히게 했다.
  공개는 `BP_HoloGuide`(홀에 1개)가 매 틱 `HoloPanel` 태그 액터를 돌며
  `alpha = MapRangeClamped(카메라x, 340→230, 0→1)` 를 설정한다 —
  **상태도 보간 변수도 액터 간 통신도 없다.** 패널 5장은 시작점(220,0)에서 반지름 300 의 호에 정면으로 세워
  (405,∓236) · (490,∓132) · (520,0), 스폰하자마자 다섯 장이 한눈에 들어오고 앞으로 걸으면 같이 짙어진다.
  재질 파라미터로 조절: `FieldOpacity` `FieldColor` `Glow` `HoloTint` `ScanDensity` `ScanSpeed`.
- **줄글 대신 그림 — 3D 픽토그램 (2026-09-11, `D-56`)** — 3번(손 조작) 패널 앞 40 cm 에
  `SM_Icon_PinchHand` 를 띄웠다. 게임이 손을 관절 구체로 그리므로 같은 모양의 손이 핀치하는 장면 하나가
  "관절마다 구체" 와 "엄지·검지를 맞대면 집기" 를 동시에 설명한다. 접점의 밝은 구슬은 두 번째 머티리얼
  슬롯(`MI_HoloIcon_Spark`)이다. 본문은 7줄 → 3줄.
  **아이콘은 초당 35° 씩 돈다** — 정지시키면 관람자가 서는 위치에 따라 "손"으로도 "덩어리"로도 보인다.
  회전은 `BP_HoloGuide` 틱이 `HoloIcon` 태그 액터에 `AddActorWorldRotation` 을 걸고, 같은 알파를
  `SetScalarParameterValueOnMaterials("Intensity", alpha*2.5)` 로 넘겨 패널과 함께 떠오르게 한다.
  패널을 뚫지 않도록 메시 원점을 바운딩 박스 중심으로 옮겨 회전 반경을 8.6 cm 로 줄였다(배율 3.6 → 31 cm).
  **아이콘 컴포넌트는 `Mobility: Movable` 이어야 한다** — StaticMeshActor 기본값 Static 이면 런타임
  회전이 조용히 무시된다. PIE 에서 yaw 가 0 에 붙어 있으면 이것부터 본다.
- **픽토그램 5종과 음영 (2026-09-11)** — 2번 `SM_Icon_Thumbstick`(조이스틱 + 진행 화살표) ·
  3번 `SM_Icon_PinchHand` · 4번 `SM_Icon_Plinth`(받침대 + 유물 + 켜진 버튼) · 5번 `SM_Icon_Portal`(빛나는 아치).
  1번은 텍스트만 둔다 — 입장 인사와 AI 고지가 들어가 있어 그림을 더하면 오히려 붐빈다.
  각 패널 정면 45 cm 앞, z 124.
  **처음엔 다 덩어리로 보였다.** `M_Museum_Emissive` 가 Unlit 이라 면 방향이 결과에 안 들어가서
  실루엣만 남았기 때문이다. `M_HoloIcon`(= `M_Museum_Emissive` 복제 + 프레넬 림 + 방향 음영)으로 바꿨다:
  `Emissive = EmissiveColor * Intensity * (Lerp(1, N·L*0.5+0.5, ShadeAmount) + Fresnel * RimBoost)`.
  본체는 `ShadeAmount 0.85 · RimBoost 2.6`, 스파크는 둘 다 0 이라 납작하게 밝다.
  `Intensity` 는 2.5 → 1.2 로 낮췄다 — 2.5 에서는 톤매핑에 날아가 음영이 안 보였다.
  **`MaterialTools.recompile` 의 인자는 `material_or_function` 이다.** `material` 로 부르면 조용히
  실패하고 셰이더가 옛 그래프 그대로 남아 "그래프는 맞는데 화면만 이상한" 상태가 된다.
  **AI 복원 고지는 1·5번과 테마관 유물 패널에만** 남기고 2·3·4번에서는 위젯을 제거했다 —
  `Visibility: Collapsed` 로는 애셋이 `Collapsed` 로 읽히는데도 레벨에서 계속 그려져서
  `UMGToolSet.RemoveWidget` 으로 뺐다.
- **Fab 소품** — `museum_showcase_stand`(A방 진열장 2대, 8K 텍스처 3장은 우리 석재 재질로 덮어써서 삭제 — 193 MB 절약), `snake_plant_scan_lowpoly`, `free_livistona_chinensis_fan_palm`(홀 화분).
- **꾸미기 3차 — 남은 소품도 모델링 메시로 (2026-09-10)** — `museum_props.py` 로 5종 추가.
  홀 `SM_Chandelier`(황동 후프에 아치형 팔 8개·초·불꽃, 조명은 초 높이 z 288로 올림) · `SM_Sofa` ×2(굽도리·좌방석 3·등방석 3·팔걸이) · `SM_FeatureWall`(3단 몰딩 액자에 6 cm 파인 패널과 황동 명판).
  A방 `SM_RoomA_Backdrop`(지층 5겹, 겹마다 돌출을 달리하고 22칸으로 나눠 층리면을 흐트러뜨림) · `SM_RoomA_Dais`(석재 연석 링 + 낮은 모래 + 자갈 에이프런 + 라벨 강대). 좌대·유물은 단 위 z 8.5로 올렸다.
  소파의 사이드 테이블·황동 램프는 프리미티브 그대로 남겨 `Lobby_SideTables_S/N` 로 분리했다.
  미사용 Fab 다운로드(`NationalMaritimeMuseum` 4.9 GB, `StylizedHomePlantsFree` 52 MB)는 프로젝트에서 제거했다.
- **테마관 B — 신라 전각 내부 1차 (2026-09-16)** — A의 복제(비계)였던 `L_Room_B` 를 신라 시대 토기 전시실로 바꿨다. 첫 시안은 적석목곽분 안쪽(돌·어둠)이었는데 **A방과 어법이 같아 구분이 안 된다**는 피드백으로 폐기하고, **밝은 전각 내부**로 다시 지었다:
  주칠(붉은) 기둥과 하방·중방·상방·창방이 흰 회벽 판을 나누고, 양쪽 옆벽 바깥 칸에는 **초록 창살 창** 4개가 빛을 낸다(`MI_RoomB_Daylight` 이미시브 + 창마다 Static 스팟 320 cd 5600 K). 천장은 창방 위 **X자 대공**, 붉은 도리 2·대들보 1, 어두운 서까래, 흰 앙토를 드러냈다. 바닥은 나무 마루.
  유물은 **5점**: 가운데 사방탁자(`SM_RoomB_Stand`, 상판 1.02 m, 낮은 마루단 위 홍·청 자리) 위 1점(`BP_Artifact` 유지 — 버튼·집기 그대로) + 좌우 긴 탁자(`SM_RoomB_Table`, 상판 0.80 m) 위 4점. 뒤에는 8폭 **병풍**(`SM_RoomB_Screen`), 탁자 앞머리에 황동 촛대 2(불꽃 이미시브 + 1900 K 포인트 30 cd).
  **병풍은 실물 매화도 병풍을 본떠 따로 만든다** (`tools/blender-museum/museum_screen.py`): 폭 0.58 × 1.80 m 8폭을 13° 지그재그로 세우고, 폭마다 검은 칠 테(`MI_RoomB_FrameLacquer`) → 금빛 비단 테두리(`MI_RoomB_Brocade`, 위 0.20·아래 0.28 m 넓은 띠) → 종이 화면 순으로 겹친다. 뒷면 황동 힌지판, 발굽 포함.
  그림은 팀이 고른 **신라 기마 전투 삽화**를 `make_screen_tex.py` 가 8폭 4096×1536 텍스처(`T_RoomB_ScreenPainting`)로 편집한다 — 삽화를 높이에 맞춰 가운데 5.5폭에 놓고, 양옆은 삽화의 배경 톤으로 종이를 이어 붙이고 원경 군단 띠를 좌우 반전 복제해 이어 붙였다. 오른쪽 폭에 제발 `新羅騎兵圖 · 慶州` 와 붉은 낙관. 원본은 `tools/blender-museum/assets/roomB_screen_source.png`.
  종이 면은 **메시 UV0 에 폭 i → [i/8, (i+1)/8]** 로 직접 박아 두었고(월드 정렬 스캔 재질로는 그림을 붙일 수 없다), 이를 위해 UV 샘플 마스터 `M_Museum_UVTex`(BaseColorTex·Roughness) 를 새로 만들었다. **FBX → UE 에서 Y 가 뒤집혀 U 도 뒤집힌다** — 스크립트가 `1 − u` 로 보정해 제발이 관람자 오른쪽에 온다.
  메시는 `tools/blender-museum/museum_roomB.py` 로 만들고(`SM_RoomB_*` 8종, 셸 합계 약 31k 삼각형), UE 반입·재질(`MI_RoomB_*` 12종 + `MI_Artifact_Stoneware`)·배치·조명은 `tools/ue-mcp/build_roomB.py` 가 `import materials level lights cleanup save` 단계로 재현한다. 주칠·초록은 `M_Museum_Surface` 절차 재질(`MI_Lobby_Brass` 복제, Metallic 0), 회벽은 `Damaged_Wall_Plaster` 스캔을 타일 320 으로 키워 벽돌 결을 죽였다(`Rough_Cement_Plaster` 는 원색이 어두워 흰 벽이 안 된다).
  조명은 키(Stationary 450 cd 4800 K)·림(200 cd 5600 K)·필 4(170 cd 5000 K)·위성 스팟 4(150 cd 4300 K)·병풍 워시 1·창 스팟 4·촛불 2, 리플렉션 캡처 2. Stationary 는 2개라 겹침 한도 안이다.
  > **유물은 전부 목업이다.** `Content/Museum/Artifacts/Mock/SM_Mock_Silla_*`(장경호·유개고배·기대·단경호·대각고배, 회전 프로파일, 회청색 경질토기 톤) — 팀이 실제 에셋을 주면 액터의 `StaticMesh` 만 바꾼다. 가운데 유물의 `BP_StageToggle`·정보 패널(`Panel_Damaged/Restored`)은 아직 빗살무늬토기 것이라 함께 교체해야 한다.
  > **남은 손작업 2개.** ① `Build → Build Lighting Only` 후 `L_Room_B_BuiltData` 저장(MCP로 굽지 못한다). ② 제목판: `WBP_RoomB_Title` 은 만들어 두었으나 위젯 컴포넌트를 MCP로 레벨에 놓을 수 없다 — `L_Room_A` 의 `RoomA_TitleBoard` 를 복사해 `L_Room_B` 에 붙이고 `Panel.WidgetClass` 를 `WBP_RoomB_Title` 로, 위치를 로컬 (−190, 300, 150) yaw −115 로 둔다.
- **A·B방 상호작용 수리 (2026-09-16)** — 실기에서 드러난 네 가지를 고쳤다.
  - **버튼이 좌대 속에 묻혀 있었다.** `StageToggleButton` 이 로컬 x −23.8 인데 `SM_Plinth` 의 앞면은 x −30 이다 — 디스크가 돌기둥 **안쪽 6 cm** 에 있어 보이지도, 눌리지도 않았다(`ConsiderTip` 은 `RestLoc` 기준 깊이 0.9 cm 를 요구하므로 손을 돌 속으로 밀어 넣어야 했다). x −31.75 로 빼서 뒷면을 좌대 앞면에 붙이고, 지름 10 → **18 cm**(스케일 0.18, 두께 3.5 cm)로 키우고 `LateralLimit` 5.5 → 9 로 맞췄다. `MI_Button` 색을 밝은 호박색으로, `Spot_Button` 을 120 → 260 cd 로 올렸다. B방 버튼도 같은 크기로 사방탁자 앞(x −33.75)에 낸다.
  - **좌대 앞 라벨 탁자가 버튼을 가렸다.** `SM_RoomA_Dais` 에 들어 있던 라벨 강대(x −0.44 기둥 + z 0.88~0.91 판)가 버튼과 같은 높이에서 정면을 막고 있었다. `museum_props.py` 에서 제거하고 다시 임포트했다(단 높이 91 → 8.5 cm). 라벨 문구는 제목판이 맡는다.
  - **A방 제목판 기둥이 공중에 떠 있었다.** `RoomA_TitleBoard` 액터가 z 150 에 놓였는데 `Stand`·`StandBase` 는 바닥 기준으로 만들어져 기둥이 z 150~250 에 떠 있었다. 패널은 그대로 두고 기둥을 상대 z −110(스케일 z 0.8 → 월드 0~80), 받침을 상대 z −148.5(월드 0~3)로 내렸다.
  - **B방 유물 5점을 모두 집을 수 있게 했다.** `BP_HandGrab` 은 `Artifact` 태그 + `BP_GrabComponent` 를 가진 액터만 집는다. 위성 4점에는 둘 다 없었다 — 태그를 달고 `GrabComponent`(Free, 각 그릇 중심 높이)를 붙였으며 루트 `StaticMeshComponent0` 을 Static → **Movable** 로 바꿨다(Static 은 런타임에 움직이지 않는다). 놓았을 때 제자리로 돌아오도록 `BP_StageToggle` 에 `RoomArtifacts`·`HomeLocs`·`HomeRots` 배열을 추가해, BeginPlay 에서 버튼 반경 7 m 안의 `Artifact` 를 모두 기억하고 `ReturnArtifact` 가 전부 되돌린다(기존에는 가장 가까운 1점만).
- **A·B방 전시 단위 정리 (2026-09-16, 2차)** — 버튼을 실제로 누를 수 있는 자리로 옮기고, B방 유물 5점을 각각 하나의 전시 단위로 만들었다.
  - **버튼은 이제 좌대·탁자 위에 눕혀 위에서 누른다.** 수직 디스크를 앞면에 붙이는 방식은 손이 돌을 통과해야 눌리는 자리였다. 지름 **4.5 cm · 두께 1.8 cm** 원판을 상판에 올리고 `PressAxis` 를 (0,0,−1) 로, `LateralLimit` 을 3 cm 로 바꿨다. `MI_Button` 은 **빨강**(0.72, 0.06, 0.05). A방은 좌대 상판 (−25, 0, 109.4), B방 중앙은 사방탁자 상판 (−25, 0, 114.9), 위성 4점은 각 그릇에서 통로 쪽 24 cm 지점(상판 z 80.9)이다. 버튼과 유물 집기 반경(`GrabRadius` 25 cm)이 겹치지 않도록 27 cm 이상 띄웠다 — 누르다가 그릇이 집히면 안 된다.
  - **유물마다 정보 패널.** `WBP_RoomB_S1~S4`(유개고배·기대·단경호·대각고배)를 만들고, 그릇 뒤 32 cm·높이 112 cm 에 통로를 향해 세웠다. 패널은 **액터 2개 한 쌍**(`Stage_Damaged`/`Stage_Restored`)으로 두어 버튼 상태와 무관하게 글이 떠 있게 했다 — 목업에는 손상·복원 구분이 없기 때문이다. 중앙 유물은 요청대로 기존 측면 패널을 그대로 둔다.
  - **위성 유물마다 `BP_StageToggle` 1개**(`Btn_S1~S4`). 각 토글이 `PickNearest` 로 자기 그릇과 자기 패널 쌍을 잡는 것을 확인했다(유물 24 cm, 패널 64 cm). 목업에는 교체할 짝이 없으므로 루트 메시의 `ArtifactMesh` 컴포넌트 태그를 비워 두었다 — 태그가 있으면 `ToggleStage` 가 빗살무늬토기 메시를 끼워 넣는다. 실제 신라 유물이 오면 태그를 되살리고 `ToggleStage` 의 메시 경로를 인스턴스 변수로 빼야 한다.
  - 위성 패널은 `ShowRadius` 110 / `HideRadius` 170 — 다가간 그릇의 글만 뜬다.
  - **위젯 컴포넌트를 스크립트로 붙일 수 있다.** `ActorTools.add_component {owner, component_type, name}` 로 빈 `Actor` 에 `/Script/UMG.WidgetComponent` 를 붙이면 패널 액터가 완성된다. "MCP 로는 위젯을 레벨에 놓을 수 없다"는 앞선 메모는 틀렸다 — 제목판도 이 방법으로 만들 수 있다.
- **B방 그릇이 라이팅 빌드 뒤에도 어두웠던 이유 (2026-09-16)** — **Static 광원은 Movable 메시에 직접광을 주지 않는다.** 그릇 5점은 집을 수 있어야 하므로 `Movable` 이고, 그릇을 비추라고 놓은 `Spot_S1~S4` 는 내가 `Static` 으로 만들어 두었다. 그래서 그릇에 닿는 직접광이 **하나도 없었고**, 볼류메트릭 라이트맵의 약한 간접광만 받아 알베도가 그대로 보였다 — 굽든 안 굽든 어둡다.
  - `Spot_S1~S4` 를 **Stationary** 로 바꿨다. 다만 원래 위치(통로 쪽으로 물러난 곳)에서 반경을 키우면 **중앙 유물까지 닿아 Stationary 겹침이 7개**가 됐다(한도 4). 각 스팟을 자기 그릇 위 90 cm 뒤·높이 230 으로 옮기고 반경을 **175** 로 좁혔다 — 이제 어떤 스팟도 중앙 유물에 닿지 않는다. 입사각은 약 57°로 사광을 유지한다(21.1.2).
  - `Spot_Exhibit_02` 600 → 260, `Spot_Rim` 500 → 200, `Spot_Button` 320 → 180 으로 줄여 중앙 조명이 측면 탁자로 번지지 않게 했다.
  - 실측 확인: 그릇 5점 **모두 최소 1개의 Stationary 스팟이 조준**하고, Stationary 겹침은 지점별 최대 **3**(한도 4 이내).
  - 재질도 밝혔다. `MI_Artifact_Stoneware` 알베도 0.30 → **0.44**(약간 푸른 회색), 러프니스 0.72 → 0.55. 0.30 은 박물관 스팟 아래에서 남색으로 읽힌다.
  > ⚠️ **광원 이동성·반경을 바꿨으므로 라이팅을 한 번 더 구워야 한다.** 굽는 대상이 달라지는 변경이라 피할 수 없었다. 이후로는 조명에 영향을 주는 변경을 더 넣지 않는다.
  > **규칙으로 남긴다 — 집을 수 있는 유물(Movable)을 비추는 광원은 반드시 Stationary 다.** Static 은 벽·바닥·가구 같은 Static 메시만 비춘다.

- **위성 버튼 크기 (2026-09-16)** — B방 위성 버튼 4개가 메인보다 훨씬 작았다. 액터 스케일은 다섯 개가 같았지만 **`ButtonMesh` 컴포넌트의 상대 스케일**이 달랐다 — 손으로 만든 메인 버튼은 (1, 1, 1) 로 덮어써져 있었고, `add_to_scene_from_class` 로 새로 스폰한 것은 CDO 기본값 (0.12, 0.12, 0.04) 을 받았다. 실제 크기가 **지름 5 mm · 두께 0.7 mm** 라 사실상 보이지 않았다. 네 개를 (1, 1, 1) 로 맞춰 다섯 개 모두 **지름 4.5 cm · 두께 1.8 cm** 가 됐고, 상판에 정확히 얹힌다(위성 80.0~81.8, 메인 107.0~108.8).
  > ⚠️ **`ObjectTools.set_properties` 로 `RelativeScale3D` 를 쓰면 한 번에 한 축만 들어간다.** `{x,y,z}` 를 한꺼번에 주면 x 만 적용되고 조용히 끝난다. 축마다 따로 호출하고 되읽어 확인한다.
  > **`BP_StageToggle` 을 새로 스폰할 때는 `ButtonMesh` 의 상대 스케일을 확인한다** — CDO 기본값이 작다.

- **되돌림 · 틱 비용 · 프레임 진단 (2026-09-16, 3차)**
  - **복원/손상 교체를 되돌렸다.** 실기 재확인 결과 방은 **손상(원본 스캔)으로 시작**해야 하고 버튼이 복원을 보여 주는 것이 맞다. `ToggleStage` 를 원래대로(`true` → `Pot_3_Pristine` 0.001/roll 0, `false` → `Pot_Damaged` 0.1/roll 90) 돌리고 A·C방 레벨 초기 메시도 `Pot_Damaged` 로 맞췄다. 앞서 넣은 `Pot_3_Pristine` LOD 4단은 그대로 둔다 — 버튼을 누르면 이 메시가 나오므로 여전히 필요하다.
    **이 네 값은 한 묶음이다.** 메시만 바꾸면 스케일이 100배 틀어져 화면에서 사라지거나 방을 가득 채운다.
  - **B방 `BP_StageToggle` 5개가 유물 5점을 각각 전부 검사하고 있었다.** 2026-09-16 에 넣은 `RoomArtifacts`/`HomeLocs`/`HomeRots` 배열을 **제거하고 원래의 "토글 1개가 자기 유물 1개를 되돌린다" 설계로 복귀**했다 — 지금은 유물마다 토글이 하나씩 있으므로 5점 모두 덮이고, 프레임당 유물 검사가 **25 → 5** 로 줄었다. 유물을 든 동안 다섯 액터가 같은 유물을 매 프레임 거리 검사하던 것도 사라진다.
  - **B방 중앙 유물의 `ArtifactMesh` 태그를 비웠다.** 태그가 남아 있어 B방 버튼을 누르면 신라 목업이 **빗살무늬토기로 바뀌었다.** 위성 4점과 같은 처리다. A·C방 태그는 그대로 둔다.
  - **프레임 제한은 걸려 있지 않다 (실측).** `t.MaxFPS=0`, `r.VSync=0`, `r.ScreenPercentage=100`, 동적 해상도 비활성. 렌더 설정도 이미 VR 에 맞다 — `r.ForwardShading=True`, `vr.InstancedStereo=True`, `vr.MobileMultiView=True`, Lumen GI·반사 모두 0(끔), `r.AntiAliasingMethod=3`(MSAA). **즉 낮은 프레임은 설정이 아니라 내용 비용이다.**
  - **아직 남은 비용, 큰 것부터.**
    ① **라이팅이 구워지지 않았다.** 위 조명 정리(Movable 17 → 3)와 라이트맵 UV 수정은 굽기 전에는 효과가 없고 화면도 틀리다. 굽기 전 `L_XRTemplate_Lighting` 서브레벨을 로드해야 한다(로그에 매번 경고).
    ② **C방이 항상 로드된다.** 액터 25개, 37만 삼각형 토기, 조명 8개를 아무도 방문하지 않는 방이 계속 물고 있다. 내부 빌드로 갈 때 포털 진입 로드로 바꾸면 통째로 빠진다(21.1.1 이 이미 그렇게 적혀 있다).
    ③ **유물 두 점이 배치 삼각형의 80 %** (98만 중 78만). 두 메시 모두 LOD 4단이 붙었으니 이제 거리에 따라 떨어지지만, 관람 거리 2 m 에서는 LOD0~1 이다. Nanite 는 꺼져 있다.
    ④ 포비티드 VRS 가 꺼져 있다(`xr.VRS.FoveationLevel=0`). MSAA 와 함께 쓸 때 제약이 있어 **적용하지 않고 후보로만 남긴다** — 켤 때는 실측으로 확인한다.
  - ⚠️ **에디터 VR 프리뷰 + ALVR 로는 프레임을 판정하지 말 것.** 이 문서 위쪽 "VR 프레임 예산" 에 적힌 그대로다. 숫자는 패키징 빌드에서 본다.

- **프레임 수리 · 모델 교체 · 스펙테이터 제거 (2026-09-16)**
  - **복원/손상 메시를 서로 바꿨다.** 실기에서 두 모델이 뒤바뀌어 보인다는 판정이 나왔다. `ToggleStage` 의 두 분기를 **메시·스케일·회전 묶음으로** 교환했다 — `ShowRestored` true → `Pot_Damaged`(0.1, roll 90), false → `Pot_3_Pristine`(0.001, roll 0). 레벨의 초기 상태(A·C방)도 `Pot_3_Pristine` 로 맞췄다. 되돌릴 때는 이 네 값을 함께 뒤집는다 — 메시만 바꾸면 스케일이 100배 틀어진다.
  - **위젯 패널 15장이 매 프레임 다시 그려지고 있었다.** 홀 안내 패널은 2026-09-11 에 `RedrawTime 1.0` 으로 고쳤지만 **방 안 패널은 고친 적이 없었고**, 내가 8장을 더 얹었다. 700×560 렌더 타깃 15장 × VR 양쪽 눈. 전부 `RedrawTime = 1.0` 으로 바꿨다 — 글자는 바뀌지 않는다. **새 `WidgetComponent` 를 만들 때마다 이걸 확인한다.**
  - **B·C방 조명이 아직 Movable 이었다.** A방만 베이크 대상으로 정리돼 있었다(README 위쪽에 적어 둔 그대로). B방은 Movable 9개 중 **2개가 동적 그림자**를 켜고 있었다. A방과 같은 배치로 맞췄다 — Fill 4 Static, 키·림 Stationary, 버튼 Static. C방도 같이 정리했다.
    방 세 개의 Movable 합계 **17 → 3**(각 방에 남은 1개는 `BP_Portal` 이 컨스트럭션 스크립트로 만드는 포털 라이트다).
  - **`Pot_3_Pristine` 에 LOD 가 없었다.** 37만 삼각형 단일 LOD 인데, 위 교체로 이 메시가 두 방의 **기본 표시물**이 됐다. `Pot_Damaged` 와 같은 비율(30 / 10 / 3 %)로 LOD 4단을 만들었다 — 374,212 / 112,264 / 37,422 / 11,226.
    참고로 배치된 전체 삼각형 약 98만 중 토기 두 점이 78만이다. **장면 예산의 80 %가 유물 두 점**이다.
  - **`BP_VRSpectator` 를 홀에서 지웠다.** UE VR 템플릿이 넣어 주는 스펙테이터 캠코더(`SceneCaptureComponent2D` + 카메라 메시)로, 미러링 화면을 HMD 시점 대신 그 카메라 시점으로 보내 촬영하는 용도다. 우리 손은 구체 손이고 집기는 `Artifact` 태그만 대상이라 집히지 않았고, `SceneCapture` 는 켜져 있으면 장면을 한 번 더 그린다. 다시 필요하면 템플릿에서 가져온다.
  - **남은 프레임 비용(우리가 안 건드린 것).** `L_XRTemplate_Lighting` 의 Movable 광원 3개(`D-61` 로 감수한 선택), C방은 25개 액터와 37만 삼각형 토기를 **항상 로드**하면서 아무도 방문하지 않는다(내부 빌드 전환 때 포털 진입 로드로 바꾸면 통째로 빠진다), 홀 야자수 6.2만 삼각형.
  - ⚠️ **위 조명·LOD 변경은 라이팅을 다시 굽지 않으면 효과가 없고 화면도 틀리다.** `Build → Build Lighting Only` 전에 `L_XRTemplate_Lighting` 서브레벨을 로드해야 한다(로그에 매번 경고가 뜬다).

- **VR 실기에서 드러난 B방 3중 결함 (2026-09-16, 긴급 수리)** — 시연에서 "직진이 안 되고, 집으면 화면이 검게 된다"가 나왔다. 원인 셋이 겹쳐 있었다.
  - **보이지 않는 1 m 충돌 상자 (내가 만든 회귀).** MapCheck 의 `NULL StaticMesh` 경고를 지우려고 중앙 유물 루트에 `Cube` 를 물렸는데, **충돌이 `QueryAndPhysics` 로 살아 있었다.** 방 한가운데 1 m 상자가 생겨 이동을 막고, 그것을 집으면 머리와 겹쳐 화면이 검게 됐다. 루트 메시를 다시 `None` 으로 돌렸다 — **MapCheck 경고는 의도적으로 남긴다.** 컨테이너 액터에 가짜 메시를 물리는 대가가 이것이다.
  - **중앙 유물 액터의 roll −90.** 액터가 roll −90, `ArtifactMesh` 가 roll +90 이라 그릇은 똑바로 보였지만 **집기 충돌구(반지름 20)와 집기 지점이 옆으로 19 cm 돌아가** 통로 가슴 높이에 떠 있었다. A방은 좌대 안에 묻혀 있어 드러나지 않았던 문제다. 둘 다 0 으로 맞췄다.
  - **마루단 12 cm 단차.** 폰의 `VRLocomotion` 은 그 높이를 오르지 못해 전시대 1.2 m 앞에서 멈췄다. A방 발굴 단은 5.5 cm 이고 잘 걸어진다 — `PLAT_H` 를 0.05 로 낮추고 사방탁자(z 5)·유물(z 107)·버튼(z 107.9)을 7 cm 함께 내렸다.
    실측: 도착점→전시대 바닥 단차 12 → **5 cm**, 가슴·허리 높이 통로 **전 구간 통과**.
  - **유물 충돌 제거.** 집은 유물은 모션 컨트롤러에 붙으므로 그 콜라이더가 플레이어 캡슐을 밀어 월드 밖으로 내보낼 수 있다. B방 유물 5점의 `StaticMeshComponent0`·`ArtifactMesh`·`GrabCollision` 을 모두 `NoCollision` 로 바꿨다 — 물리 시뮬레이션을 쓰지 않고(`bSimulateOnDrop` false) `ReturnArtifact` 가 제자리로 돌리므로 충돌이 필요 없다. 집기 판정은 `BP_HandGrab` 의 **거리** 기반이라 영향이 없다. A방은 실기 검증을 통과한 상태라 그대로 두었다 — A방에서도 같은 증상이 나오면 같은 스위치를 쓴다.
  - 시작 위치는 문제가 없었다. 홀 `PlayerStart` 는 (220, 0, 0) 로 문서값 그대로이고 머리 높이에서 사방 3 m 가 비어 있다.
  - **`BP_VRSpectator` (250, −380, 187).** UE VR 템플릿이 넣어 주는 **스펙테이터 캠코더**다. `SceneCaptureComponent2D` + 카메라 메시로 된 손에 드는 캠코더이고, 미러링 화면을 HMD 시점 대신 이 카메라 시점으로 보내 VR 세션을 촬영하는 용도다. 우리 손은 구체 손이고 집기는 `Artifact` 태그만 대상으로 하므로 **이 캠코더는 집히지 않는다** — 홀 머리 높이에 떠 있는 장식일 뿐이다. 미러링 화면이 엉뚱한 곳을 보여 준다면 이 카메라가 활성화된 것이니, 쓰지 않으면 지우는 편이 낫다.

- **메시지 로그 정리 (2026-09-16)** — 로그에 남아 있던 경고를 하나씩 갈랐다. **우리 잘못 2건은 고쳤고, 나머지는 환경·엔진 소음이다.**
  - **접선 퇴화 (우리 잘못).** `SM_RoomB_Walls` 임포트 때 "degenerate tangent bases … will result in incorrect shading" 이 떴다. 원인은 `finish()` 가 **언랩을 먼저 하고 베벨을 나중에** 적용한 것이다 — 베벨로 새로 생긴 띠 면들이 보간된 0 면적 UV 삼각형을 받는다. 순서를 **베벨 → 언랩** 으로 뒤집고, 베벨 뒤에 `dissolve_degenerate` 를 한 번 더 걸었다. 7 mm 였던 창호 발광면도 베벨(8 mm)보다 두껍게 2.8 cm 로 키웠고, 하방·중방·상방·창방과 회벽 판의 끝을 측벽 안으로 `BACK_EPS` 만큼 묻어 맞댐면을 없앴다.
    0 면적 UV 삼각형 **1,835 → 28**, 0 면적 3D 삼각형 **152 → 16**(13,072 중), 라이트맵 겹침 0.10 → **0.07 %**. "incorrect shading" 경고는 사라졌고 "nearly zero tangents" 만 남는다 — 남은 16개는 1 mm 이하라 눈에 보이지 않는다.
    **다른 생성기(`museum_parts.py`·`museum_props.py`·`hall_shell.py`)도 같은 순서 문제를 갖고 있다.** 지금 경고가 나지 않으니 건드리지 않았지만, 그 메시를 다시 뽑을 때는 순서를 함께 뒤집는다.
  - **MapCheck: `StaticMeshActor_7` has NULL StaticMesh (세 방 모두).** 세 방의 중앙 유물은 **컨테이너**다 — 보이는 메시는 `ArtifactMesh` 자식에 있고 루트는 비어 있어 MapCheck 가 매번 경고했다. 루트에 보이지 않는 `Cube` 를 물려 경고를 없앴다.
    > ⚠️ 처음에 루트 스케일을 0.001 로 줄였다가 **유물이 사라졌다.** 자식 컴포넌트는 루트 스케일을 곱해서 받는다 — `ArtifactMesh` 가 0.1 × 0.001 이 되어 보이지 않게 됐다. 루트 스케일은 1 로 두고 `bVisible`·`bHiddenInGame` 만 끈다.
  - **남은 것은 우리 문제가 아니다.** `LogHMD` 의 `XR_BD_controller_interaction` 없음·10 bit 스왑체인 폴백·visibility mask 실패는 OpenXR 런타임 쪽이고, `LogAudioMixerWasapi`/`LogAudioMixer` 는 헤드셋을 꽂고 뺄 때 오디오 장치가 바뀌며 나는 것이다. `LogHttp`/`LogEOSSDK`/`LogJson` 은 Epic 온라인 서비스 소음(전체 경고의 절반)이다. `LogCrowdFollowing: Unable to find RecastNavMesh` 는 방에 `NavMeshBoundsVolume` 이 없어서인데 이동이 텔레포트라 지금은 무해하다. `LogEditorServer: WorldSettings_1 contains streaming level 'L_XRTemplate_Lighting' which isn't loaded` 는 홀이 템플릿 조명 서브레벨을 참조하는데 에디터에서 로드되지 않은 상태라 뜬다 — **라이팅을 구울 때는 이 서브레벨을 로드해야 한다.**
  - `LogScript` 경고 다수와 `LogSavePackage: Cannot remove … as it is read only` 는 이 세션에서 MCP 로 작업하며 낸 것이다(실패한 속성 읽기, 잠그기 전 저장 시도). 남는 문제가 아니다.

- **라이트맵 UV 겹침 수리 (2026-09-16)** — `SM_RoomB_Walls` 3.8 %, `SM_RoomB_Ceiling` 2.4 % 경고가 났다. 원인 두 가지였다.
  ① 벽면에 **정확히 겹쳐 붙인 면**(판·기둥·보를 깊이 0 에서 시작) 이 `remove_doubles` 로 용접되면서 두 면이 한 UV 아일랜드에 겹쳐 쌓였다. `BACK_EPS` 4 mm 를 띄웠다. ② 네 벽 슬래브가 모서리에서 **맞대어** 같은 문제를 만들었다 — 이제 `BACK_EPS` 만큼 서로 파고든다.
  UE 가 자동 생성하는 채널 2 대신 **Blender 가 만든 채널 1** 을 쓰도록 `LightMapCoordinateIndex` 를 1 로 바꿨다. 측정 결과 1024 기준 벽 3.8 % → **0.10 %**, 천장 2.4 % → **0.00 %**.
  검증 도구는 `tools/ue-mcp/uv_overlap.py` — UE 의 `CheckLightMapUVs` 와 같은 방식으로 FBX 의 UV 를 목표 해상도에 래스터화해 겹친 텍셀 비율을 낸다.

  > ⚠️ `write_graph_dsl` 로 `EventGraph` 를 다시 쓰면 **Name 핀의 문자열이 None 으로 날아간다.** `PickNearest("Artifact"/"Stage_Damaged"/"Stage_Restored")` 세 개가 그렇게 비었고 `set_pin_value` 로 되살렸다. 그래프를 다시 쓴 뒤에는 Name·Enum 핀을 반드시 되읽어 확인한다.

## 관련 문서

- 포트폴리오 개요: 저장소 루트의 `README.md`
- 이동 구현 기록: `docs/VR_LOCOMOTION_TROUBLESHOOTING.md`
- 렌더링 경로: Lumen·RayTracing·Substrate는 꺼둔 상태를 유지한다
