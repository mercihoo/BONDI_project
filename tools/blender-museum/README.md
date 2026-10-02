# blender-museum — 전시 공간 건축 메시 생성기

홀과 테마관 A의 **벽·바닥·천장과 기둥·아치·좌대·벤치**를 Blender에서 만들어
`vr/testproj/Content/Museum/Meshes/` 로 임포트할 FBX를 뽑는다. 박물관 레벨 설계에 맞춘 제작 도구다.

| 스크립트 | 만드는 것 |
| --- | --- |
| `hall_shell.py` | `SM_Hall_Floor` · `SM_Hall_Walls` · `SM_Hall_Ceiling` |
| `museum_parts.py` | `SM_RoomA_Floor/Walls/Ceiling` · `SM_Plinth` · `SM_Bench` · `SM_Column` · `SM_PortalArch` |
| `museum_props.py` | `SM_Chandelier` · `SM_Sofa` · `SM_FeatureWall` · `SM_RoomA_Backdrop` · `SM_RoomA_Dais` |
| `museum_icons.py` | 안내 패널용 3D 픽토그램 — `SM_Icon_PinchHand` · `SM_Icon_Thumbstick` · `SM_Icon_Plinth` · `SM_Icon_Portal` |
| `museum_roomB.py` | 테마관 B(신라 전각 내부) — `SM_RoomB_Floor/Walls/Ceiling`(주칠 기둥·보, 흰 회벽, 초록 창살 창, X자 대공·서까래) · `SM_RoomB_Platform`(마루단 + 홍·청 자리) · `SM_RoomB_Stand`(사방탁자) · `SM_RoomB_Table` · `SM_RoomB_Candle` · 목업 토기 5종 `SM_Mock_Silla_*` |
| `museum_screen.py` | `SM_RoomB_Screen` — 8폭 병풍(검은 칠 테·비단 테두리·종이·힌지). **그림용 UV0 를 직접 박는다**(폭 i → u [i/8, (i+1)/8], FBX Y 반전 보정으로 `1−u`), 라이트맵 UV1 만 자동 |
| `make_screen_tex.py` | 병풍 텍스처 — `assets/roomB_screen_source.png`(신라 기마 삽화)를 8폭 4096×1536 `T_RoomB_ScreenPainting.png` 로 편집(양옆 종이 이어 붙임 + 제발·낙관) · 타일 비단 `T_RoomB_Brocade.png`. `py`(Pillow 필요), Blender 불필요 |

`museum_roomB.py` 는 Blender 를 열지 않고도 돈다 — `"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b --factory-startup -P tools/blender-museum/museum_roomB.py`.
`SM_Mock_Silla_*` 는 (반지름, 높이) 프로파일을 회전시킨 **자리표시용** 메시다. 실제 유물 에셋이 오면 액터의 `StaticMesh` 만 바꾸고 `Content/Museum/Artifacts/Mock/` 은 지운다.
UE 쪽 임포트·재질·배치·조명은 `tools/ue-mcp/build_roomB.py` 가 단계별로 재현한다.

## 쓰는 법

Blender를 열고 스크립팅 탭에서 파일을 열어 실행하거나, MCP 애드온이 붙어 있으면:

```python
exec(compile(open(r"tools/blender-museum/hall_shell.py", encoding="utf-8").read(), "hall_shell.py", "exec"))
```

FBX는 `C:/Users/<USER>/AppData/Local/Temp/museum_fbx/` 에 떨어진다(`OUT` 상수로 바꾼다).
UE 쪽 임포트는 `StaticMeshTools.import_file {folder_path, asset_name, source_file}`.

## 규칙

- **미터 단위, UE 축 그대로** 모델링한다 — Blender X/Y/Z가 UE X/Y/Z로 1:1 들어간다.
  홀 내부는 `x 0..7`, `y -4.5..4.5`, `z 0..3.2`. A방은 로컬 `x,y -5..5`, `z 0..4`.
- **UV를 빼먹지 않는다.** `unwrap()` 이 UV0(재질용)·UV1(라이트맵용)을 만든다.
  표면 재질이 월드 정렬이라 재질용 UV는 안 쓰지만 **Lightmass는 UV 없이 굽지 못한다** —
  UV0이 없으면 UE의 라이트맵 UV 생성이 조용히 실패하고 그 메시만 스카이라이트만 받아 남색이 된다.
  임포트 후 `LightMapCoordinateIndex` 가 0이면 잘못된 것이다.
- **모든 모서리에 1.2 cm 챔퍼**(Bevel 모디파이어)를 넣는다. CG 느낌을 가장 크게 줄이는 한 가지다.
- **임포트 직후 콜리전을 켠다.** FBX 임포트는 콜리전을 만들어 주지 않는다 — `BodySetup` 은 생기지만
  심플 프리미티브가 0개이고 `CollisionTraceFlag` 가 `CTF_UseDefault` 라 **관람자가 벽을 그냥 통과한다.**
  `StaticMeshTools.generate_convex_collisions {hull_count:24, max_hull_verts:24, hull_precision:100000}`
  로 볼록 껍질을 씌운다. 껍질 24개면 속이 빈 방 셸도 안쪽이 막히지 않는다.
  **`CTF_UseComplexAsSimple` 을 쓰면 안 된다** — 심플 콜리전이 사라지는데 캐릭터 캡슐 스윕은 심플만
  조회하므로(`bTraceComplexOnMove` 기본 false) 관람자가 바닥을 뚫고 떨어지고 VR에는 검은 화면만 나온다.
  확인은 `SceneTools.trace_world {start, end}` 로 벽을 향해 쏴 보면 된다 (거리를 돌려주고, null이면 안 맞은 것).
- **픽토그램은 슬롯 두 개로 만든다.** `Joint` 가 본체, `Spark` 가 관람자가 봐야 할 한 곳
  (핀치 접점 · 방향 화살표 · 버튼 · 포털 면)이다. UE 에서 `MI_HoloIcon_Joint` / `_Spark` 를 문다.
- **실루엣만으로 읽히는지 먼저 본다.** 두 발짝 떨어지면 내부 디테일은 사라지고 윤곽만 남는다.
  갈매기 화살표 두 개는 한 덩어리로 뭉쳐서 화살표 하나(자루 + 머리)로 바꿨다.
- **회전하는 아이콘은 원점을 바운딩 박스 중심으로 옮긴다.** 손 픽토그램은 손목이 원점이라 제자리
  회전이 아니라 크게 휘둘러져 패널을 뚫고 나갔다. `museum_icons.py` 의 `finish()` 가 정점을 박스
  중심 기준으로 옮겨 회전 반경을 절반 이하로 줄인다.
- Blender의 **머티리얼 슬롯 이름이 UE 슬롯 이름으로 그대로 넘어간다.**
  `StaticMeshTools.set_material {mesh, slot_name, material}` 로 `MI_Lobby_*`·`MI_RoomA_*` 를 물린다.

## 다시 임포트할 때

`import_file` 은 **덮어쓰지 않는다.** 기존 애셋을 `AssetTools.delete` 하기 전에
그 메시를 쓰는 액터의 `StaticMesh` 를 임시 메시로 돌려놓고, 임포트 후 다시 연결한다.
액터를 지웠다 다시 놓으면 트랜스폼·라벨을 잃는다.
