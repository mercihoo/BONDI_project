# ai/3d-to-3d-color — 산출물 설명

세 스크립트가 만드는 파일이 무엇인지, 어디에 쓰는지, 저장소에 넣는지(`.gitignore`) 정리한다.
원칙(CONTRIBUTING 10절): **스크립트·기록 JSON·비교 그림은 커밋, 메시(GLB)·텍스처 PNG·마스크 PNG·원본 사진은 커밋하지 않고 공유 스토리지**. 기록 JSON에 적힌 설정으로 전부 재생성할 수 있다.

## 1. `color_match.py` — 사진 색 맞춤

출력 폴더 예: `out/`(굽다리바리), `out_crown/`(금관)

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `<모델>_colormatched_<방식>.glb` | base color 텍스처만 사진 색 분포로 바꾼 GLB. 메시·UV·다른 텍스처 동일 | 다음 단계(요철, 복원) 입력. VR "현재 상태색" | ✗ |
| `basecolor_before.png` | 원본 GLB의 base color 텍스처 | 전후 비교, 다른 툴에서 재사용 | ✗ |
| `basecolor_after_<방식>.png` | 색 맞춤 후 텍스처 (GLB 안 텍스처와 동일) | 같음 | ✗ |
| `color_match_compare_<방식>.png` | 사진(배경 제거) · 전 4방향 · 후 4방향 점 렌더 | 검수, 발표 | ✓ |
| `color_match_hist_<방식>.png` | 사진·전·후 Lab 채널 분포 히스토그램 | 분포가 겹치는지 확인 | ✓ |
| `color_match_<방식>.json` | 입력 파일, 방식, 강도, 마스크 침식·배경색 제거 값, 사진 전경 픽셀 수, Lab 평균·표준편차(사진/전/후), 평균색 ΔE 전후, 출력 경로, 소요 시간 | **FR-AI-005 기록**, manifest 출처 | ✓ |

`<방식>` = `reinhard`(무광 기본) / `hist` / `chroma` / `hue`(금속).

## 2. `relief_maps.py` — 높이맵 · 노멀맵

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `height_<tag>_16bit.png` | 색 텍스처 밝기 대역(1~16 px)에서 추정한 높이맵, 16비트 **마스터** | 노멀맵 재생성, 나중에 디스플레이스먼트 | ✗ |
| `height_<tag>.png` | 같은 높이맵 8비트 미리보기 | 눈으로 확인 | ✗ |
| `normal_<tag>.png` | 높이맵 기울기로 만든 탄젠트 공간 노멀맵 (glTF/OpenGL 규약) | GLB `normalTexture` | ✗ |
| `uvmask_<tag>.png` | 메시가 실제로 쓰는 텍셀(UV 삼각형 래스터) | 필터 범위, 디버그 | ✗ |
| `<모델>_normal.glb` | 노멀맵을 끼운 GLB. 메시·색 텍스처 동일 | 뷰어·UE5 | ✗ |
| `relief_<tag>_preview.png` | 텍스처 일부의 색 · 높이 · 노멀 · 조명 미리보기 | 검수 | ✓ |
| `relief_<tag>.json` | 필터 대역, 정규화 분위수, 기울기 계수 k, 유효 텍셀 비율, `ai_inferred` 명시 | 기록 | ✓ |

## 3. `restore_original_colour.py` — 제작 당시 색 복원 v1.3

출력 폴더 예: `out_restore/sword/`, `out_restore/gupdari/`, `out_restore/crown/`, `out_restore/crown_gilt/`

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `<모델>_restored_conserved_<재질>.glb` | 손상부 색도(a, b)만 목표색으로 옮기고 밝기(L)는 보존한 GLB. 형상 안정화(Taubin, 0.07 % 클램프) 적용 | **기본 산출.** VR "제작 당시(가설)" | ✗ |
| `<모델>_restored_pristine_<재질>.glb` | conserved + 손상부 밝기를 건전부 분포로 맞춘 "새것" 버전. 건전부 3 % 미만이면 만들지 않음(JSON `pristine_skipped`에 이유) | 발표 비교용 옵션 | ✗ |
| `texture_restored_<mode>_<재질>.png` | 위 GLB에 들어간 base color 텍스처 | 다른 툴 재사용 | ✗ |
| `mask_valid_<재질>.png` | 메시가 쓰는 텍셀 (흰색) | 모든 통계의 범위 | ✗ |
| `mask_healthy_<재질>.png` | 건전부(잔존 원표면)로 판정된 텍셀 | 목표색 근거, 검수 | ✗ |
| `mask_damage_<재질>.png` | 손상도 d를 0~255로 저장한 **연속값** 마스크. 값이 클수록 많이 바뀐 곳 | **FR-UI-008 복원 영역 표시 레이어**, VR 블렌드 가중 | ✗ |
| `mask_protect_<재질>.png` | 색을 건드리지 않은 텍셀(투각 알파, 도자기 문양 군집·근흑색) | 장식 보호 검수 | ✗ |
| `mask_incised_<재질>.png` | 높이맵 하위 10 %(음각으로 추정). 색 이행을 30 %로 감쇠한 곳 | 검수 | ✗ |
| `mask_coherence_<재질>.png` | 구조 텐서 일관성 0~255. 밝을수록 선·무늬 같은 방향성 세부(재부착·노멀맵에 쓰임) | 무늬 보존 검수 | ✗ |
| `mask_geom_support_<재질>.png` (v1.8) | 메시 잔차 3 mm 대역의 국소 에너지(0~255). 흰색 = 이 자리에 실제 요철이 있다 | 얼룩/도구 자국 판정 검수 | ✗ |
| `relief_mesh_<재질>.png` (v1.7, 재양각·`--geom-cracks` 시) | 메시를 스무딩한 사본까지의 부호 있는 거리(128 = 0, ±1.5 mm) | 실제 요철 확인, 재양각 입력 | ✗ |
| `mask_geom_crack_<재질>.png` (`--geom-cracks` 시) | 메시 요철에서 찾은 길고 넓은 홈 | **검수용**: 장식띠가 섞였는지 반드시 확인 | ✗ |
| JSON `pattern_mirror` (v1.7) | 띠 폭(mm), 유효 텍셀 중 띠 비율, 대칭 원본을 찾은 비율 | 무늬 되살리기 검수 | ✓ (JSON) |
| JSON `crack_geometry.reemboss` (v1.7) | 재양각 세기·상한·평균 변위(mm) | 기하 무늬 검수 | ✓ (JSON) |
| JSON `crack_geometry` (균열 제거 시) | v1.6: `method` biharmonic/taubin, `rings`, 풀린 정점 비율, 이동 평균·p95·최대(mm), 클램프 비율 | 홈 메우기 검수 | ✓ (JSON) |
| JSON `uv_seams.border_band` (균열 제거 시) | v1.6: 경계 띠 폭, 유효 텍셀 중 띠 비율, 띠 안 평균 ΔL | 경계 밝기 보정 검수 | ✓ (JSON) |
| `mask_groove_<재질>.png` (균열 제거 시) | 메시 오목 곡률(홈)을 텍셀로 래스터한 것. 상위 5 %가 접합선 후보 | 접합선 검출 검수 | ✗ |
| `mask_convex_<재질>.png` (도자기) | 메시 볼록 곡률을 텍셀로 래스터한 것. 상위 10 %가 마모 후보 | 마모/그림 분리 검수 | ✗ |
| `normal_pattern_<재질>.png` | 원본 텍스처의 일관성 높은 요철만 남긴 노멀맵(균열 띠 제외). GLB `normalTexture`에 들어감 | 무늬 요철 | ✗ |
| `restore_compare_<재질>.png` | 윗줄: 원본 텍스처 · 건전도 점수 · 건전부 · 손상도 · 보호 · 복원 텍스처 / 아랫줄: 전후 렌더 3방향 | 검수, 발표 | ✓ |
| `restore_<재질>.json` | 재질 프리셋(목표 a·b, PBR 값), 임계값 전부, 목표색과 **출처**(건전부 중앙값 또는 프리셋과 그 이유), 건전부·손상·보호 비율, L-SSIM, 손상부 ΔE 전후, 정점 이동 통계(평균·95 %·상한·클램프 비율·보호 정점 비율), 소요 시간, `ai_inferred` 명시 | **FR-AI-005 기록**, 227 완료 조건 지표 | ✓ |

### JSON에서 먼저 볼 값

| 키 | 뜻 | 기대 범위 |
| --- | --- | --- |
| `target_source` | 목표색을 어디서 가져왔나. `healthy surface median`이면 그 유물 실측, `preset …`이면 프리셋과 이유 | — |
| `metrics.L_ssim_valid` | 밝기 채널 보존도 | 도자기 ≥ 0.95. 금속은 탈색이 얼룩을 걷어내므로 0.5~0.7로 떨어지는 것이 정상 |
| `chroma_strength`, `tone`, `destain` | v1.2~1.3 재마감 설정과 실측(톤 목표 L·표면 평균 L 전후·gain, 탈색 평균 상승량) | 재질군 기본값과 같은지 |
| `metrics.damage_dE_ab_to_target_before/after` | 손상부가 목표색까지 남은 거리 | after ≤ 5 |
| `metrics.healthy_dE_mean` | 건전부가 바뀐 정도 | ≤ 1 |
| `metrics.protect_fraction`, `protect_parts_fraction` | 보호된 비율과 그 내역 | 유물별 기준선 |
| `smoothing.clamped_fraction` | 상한에 걸린 정점 비율. 높으면 메시가 거칠다는 뜻 | — |

## 3a-1. `synth_corrosion.py` — 합성 부식 정답 (설계 8절 F)

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `C:/ai/poc_synth/<키>/<stem>.{obj,mtl,jpg}` | 부식을 합성해 입힌 입력 한 벌. OBJ·MTL은 원본 복사 | 복원 입력 | ✗ (저장소 밖) |
| `out_restore/synth_<키>/mask_truth.png` | 정답 구간(합성 부식을 입힌 건전 텍셀) | `eval_synth.py` 입력 | ✗ |
| `out_restore/synth_<키>/mask_severity.png` | 부식 강도 0~255 | 검수 | ✗ |
| `out_restore/synth_<키>/synth.json` | 설정, 실제 손상부에서 가져온 목표값, 입힌 ΔL·ΔE | 재현·보고 | ✓ |

## 3b. `segment_masks.py` — SAM 2 반자동 마스크 (설계 8절 A)

출력 폴더 예: `out_restore/sam_ssu022891/`, `out_restore/sam_don000498_001/`

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `mask_sam_seam.png`, `mask_sam_wear.png` | SAM 2가 만든 8K 마스크(흰색). 유효 텍셀 안, 차트를 통째로 잡은 결과는 제거 | `restore_original_colour.py --crack-mask / --wear-mask` 입력 | ✗ |
| `mask_seam_semiauto.png` | SAM 조각 + 사람이 그은 선을 합친 접합선 마스크 | 같음 | ✗ |
| `sam_overlay.jpg` | 2K 아틀라스 위에 마스크(빨강 접합선, 파랑 마모)와 프롬프트 박스를 그린 검수 그림 | 프롬프트가 맞았는지 확인 | ✓ |
| `prompts_used.json`, `prompts_manual.json` | 쓴 프롬프트(2K 좌표), 가중치, 마스크 비율·성분 수 | 재현 기록 | ✓ |

## 3c. `idealize_revolution.py` — 이상화 회전체 (제작 당시 형상)

출력 폴더 예: `out_restore/poc_ssu022891_ideal/` (`<tag>` = `earthenware_ideal`)

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `<stem>_restored_conserved_<tag>.glb` | 새 회전체 메시(바깥·안쪽·구연부) + 8K 원통 아틀라스 + 잔차 노멀맵, 스캔과 같은 좌표계 | 뷰어·VR | ✗ |
| `texture_restored_conserved_<tag>.png` | 아틀라스(위 절반 바깥 벽 θ×z, 아래 절반 안쪽 벽) | 검수 | ✗ |
| `normal_relief_<tag>.png`, `relief_<tag>.png` | 반지름 잔차(±2.5 mm)로 만든 노멀맵과 잔차 자체(128 = 0) | 새김 깊이 검수 | ✗ |
| `mask_coverage_<tag>.png`, `mask_crack_<tag>.png` | 스캔이 덮은 텍셀(접합선·균열 제외), 검출된 균열·구멍 | 채움 범위 확인 | ✗ |
| `profile_<tag>.png` | 스캔 반지름 점 + 적합 프로필 + 안쪽 프로필, 바깥 아틀라스 축소본 | 축·프로필 검수 | ✓ |
| `raster_<tag>.npz` | 래스터 캐시(스캔 UV·반지름·커버리지) | `--reuse-raster` | ✗ |
| `restore_<tag>.json` | 축·원점·z 범위·바닥 z·벽 두께·커버리지·검출 균열 수·옵션 | 기록, 뷰어 라벨(`method`) | ✓ |

## 3d. 뷰어 마스크 붓 (`serve_viewer.py` + `viewer.html`)

| 파일 | 내용 | 용도 | 커밋 |
| --- | --- | --- | --- |
| `out_restore/mask_paint/<유물키>_paint.png` | 뷰어에서 손으로 칠한 마스크(4096², 흰색 = 대상). 유물 텍스처 UV 좌표계 | `restore_original_colour.py --crack-mask` / `--wear-mask` 입력, 뷰어에서 다시 불러오기 | ✗ |

## 4. 뷰어 · 보조 파일

| 파일 | 내용 | 커밋 |
| --- | --- | --- |
| `out_restore/viewer.html` | 복원 전후 비교 뷰어. 유물·복원 버전 선택, 좌 원본·우 복원 카메라 동기, 텍스처 원색/조명·재질 표시, roughness·metallic 덮어쓰기 슬라이더, 정면 버튼 | ✓ |
| `out_restore/viewer/manifest.json` | 뷰어가 읽는 유물·버전 목록과 각 버전의 수치(목표색 출처, ΔE, 보호 비율, 톤·탈색·채도 설정, PBR) | ✓ |
| `out_restore/viewer/<유물>_original.glb`, `<유물>_restored_<mode>_<태그>.glb`, `<유물>_poc0827_*.glb` | 2K 텍스처 뷰어 복사본(`make_viewer_copies.py`). 위치 기준 부드러운 노멀·프리셋 PBR 포함. 8/27 PoC 결과(`C:\ai\poc_results\`)도 같은 형식으로 | ✗ (약 1분에 재생성) |
| `make_viewer_copies.py` | 위 복사본 생성. `--only <유물>`로 일부만 다시 만들면 manifest에 합쳐진다 | ✓ |
| `out/viewer.html` | three.js 색 맞춤·노멀맵 뷰어 (굽다리바리) | ✓ |
| `out/photo_small.jpg` | 뷰어용 참조 사진 축소판 | ✗ |
| `crown.jpg`, `굽다리바리_….jpg`, `crown.glb`, `sample_….glb` | 입력 원본 사진·GLB | ✗ (공유 스토리지, 경로·출처만 기록) |

## 5. `.gitignore` 규칙 요약

```text
ai/3d-to-3d-color/**/*.glb        메시 전부
ai/3d-to-3d-color/*.jpg           입력 사진
**/basecolor_*.png, height_*.png, normal_*.png, uvmask_*.png   텍스처·맵
**/mask_*.png, texture_restored_*.png                          복원 마스크·텍스처
**/photo_small.jpg, **/*.mp4
```

JSON과 `*_compare_*.png`, `*_hist_*.png`, `*_preview.png`는 규칙에 걸리지 않아 커밋된다.
