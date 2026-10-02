# 2D→3D 생성 모델 3종 vs 실측 스캔 비교 (2026-09-03 ~ 04)

입력 사진 1장(빗살무늬토기)을 **TripoSR, Stable Fast 3D(SF3D), SPAR3D**로 각각 3D 생성하고, 같은 유형 유물의
실측 3D 스캔(`ssu022891`)을 정답지로 두어 형상 오차를 측정했다. 명세 FR-AI-005(모델명·버전·설정·지표 보존)와
G3 게이트(3D 생성 모델 선정) 판단 근거로 쓴다.

> 주의. 정답지 스캔이 사진의 유물과 **동일 개체인지는 확인되지 않았다.** 형태 유형(첨저 빗살무늬토기)과 비율은
> 일치하지만, 개체가 다르면 아래 오차에는 모델 오차 외에 개체 차이도 섞여 있다. 같은 유물의 두 GT 파일(PLY, OBJ)
> 사이 Chamfer는 1.06 mm로, 측정 방법 자체의 오차는 결과의 1/10 이하다.

## 1. 결과 요약

GT 유물 높이 383 mm. 생성 메시를 GT 좌표계에 회전·이동·배율(bbox 추정치 ±20 % 안에서) 정렬한 뒤 계산. 단위 mm.

| 모델 | Chamfer | 정확도 중앙값 | 완전성 중앙값 | F@5.4 mm | F@10.7 mm | F@26.8 mm | Hausdorff | 정점 / 면 | 생성 시간 | 피크 VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **TripoSR** | **10.1** | **5.0** | **7.2** | **0.44** | **0.71** | **0.93** | **113** | 79,389 / 158,748 | 약 104 s | 미계측 (렌더 중 2.7 GB 관측) |
| SF3D | 46.6 | 10.7 | 63.9 | 0.12 | 0.27 | 0.48 | 236 | 9,728 / 17,772 | 22 s (추론 6 s) | 7.9 GB (텍스처 2048), 6.2 GB (1024) |
| SPAR3D | 46.8 | 10.7 | 70.6 | 0.15 | 0.27 | 0.46 | 207 | 12,961 / 22,920 | 약 60 s 추정* | 8.3 GB (저 VRAM 모드) |

\* SPAR3D 첫 실행은 236 s였으나 그중 약 3분이 AlphaCLIP 가중치(891 MB) 다운로드였다. 순수 추론 시간은 분리 계측하지 못했다.

**이 사진에서는 TripoSR이 압도적으로 낫다.** SF3D와 SPAR3D는 토기를 **앞면만 있는 얇은 부조**로 만들었다.
생성 메시의 깊이/폭 비율이 TripoSR 0.94, SF3D 0.35, SPAR3D 0.36이다(실물은 1.0). SF3D는 foreground ratio를
0.75, 0.85, 1.0으로 바꿔도 0.49, 0.35, 0.39로 같은 현상이 반복됐다. 배경 제거 결과(`inputs/sf3d_input_bg_removed.png`,
`inputs/spar3d_input_bg_removed.png`)는 깨끗하므로 전처리 문제는 아니다.

## 2. 시각적 소견

그림: `figures/multi_compare.png`(3모델 4방향 + 단면), `figures/multi_compare_metrics.png`, `figures/side_by_side.png`, `figures/cross_section.png`

### TripoSR

1. **외곽 실루엣은 잘 맞는다.** 바닥에서 높이 약 300 mm까지 외벽 프로파일 오차 중앙값 5 mm 안팎. 높이/폭 비율도 일치
   (생성 376 × 260 mm vs GT 383 × 265 mm, 정렬 후).
2. **입구가 막혀 있다.** 구연부 아래 약 55 mm 지점(z ≈ 325 mm)에 얕은 접시 모양 뚜껑을 만들었다. Hausdorff 113 mm의 원인.
3. **속이 비어 있지 않다.** GT는 두께 15~20 mm의 이중 표면이지만 TripoSR은 단일 폐곡면 덩어리다.
4. **바닥이 뭉개졌다.** GT는 뾰족한 첨저, TripoSR은 폭 약 50 mm의 평평한 바닥.
5. **뒷면은 추정이고 19° 기울어져 나온다.** 앞면 평균 오차 10.6 mm, 뒷면 12.4 mm. 형상보다 무늬(텍스처)가 더 번진다.
6. **크기 정보가 없다.** 정렬 배율 1 unit = 364 mm.

### SF3D

1. **앞에서 보면 토기, 옆에서 보면 렌즈.** 단면이 얇은 타원 고리이고 두께가 폭의 1/3이다. 뒷면이 없는 것이 아니라
   앞면 바로 뒤에 닫혀 있다.
2. **완전성이 매우 나쁘다.** GT 점의 중앙값 64 mm가 생성 표면과 떨어져 있다. 정확도 중앙값 10.7 mm는 "앞면 껍질이
   GT 앞면에 대충 붙어 있다"는 뜻일 뿐이다.
3. **텍스처는 조명 제거(delighting)로 평탄해져 빗살 무늬가 거의 사라졌다.** 2048² base color와 normal map, PBR 재질
   (metallic 0, roughness 0.48)은 정상 출력됐다.
4. 정렬 결과의 기울기 96°는 GLB가 Y-up이라서 예상되는 값이다. 다만 납작한 물체를 둥근 GT에 맞추는 정렬은 원래
   잘 정의되지 않으므로 SF3D·SPAR3D 수치는 "토기에서 얼마나 먼가"로 읽어야 한다.

### SPAR3D

1. **SF3D와 같은 부조 형태.** 단면이 한쪽으로 접힌 얇은 고리. 깊이/폭 0.36.
2. **텍스처는 셋 중 가장 좋다.** 빗살 무늬와 균열이 그대로 보인다. 형상이 맞았다면 전시용으로 가장 나은 표면이었을 것이다.
3. **포인트 클라우드 조건(512점, `meshes/spar3d/points.ply`)이 있어도 깊이를 못 살렸다.** 뒷면 개선을 노린 모델이지만
   이 입력에서는 앞면 깊이 자체가 무너졐다.
4. 저 VRAM 모드로 8 GB에서 돌아가긴 했으나 피크 8.3 GB로 물리 VRAM을 넘겨 공유 메모리를 썼다. 실행은 성공했지만 여유가 없다.

## 3. 대칭 규칙 실험 (TripoSR 결과 기반)

`../tools/symmetry_test.py`, `figures/symmetry_test.png`, `metrics/symmetry_test.json`

| 모델 | Chamfer | F@10.7 mm | 최대 오차 |
| --- | --- | --- | --- |
| A. TripoSR 원본 | 10.1 mm | 0.71 | 113 mm |
| B. 앞면 실루엣 회전, 속 찬 덩어리 | 11.4 mm | 0.60 | 119 mm |
| C. B + 입구 개방 + 벽 두께 15 mm | **5.1 mm** | **0.88** | 25 mm |
| D. GT 자체를 완전대칭으로 회전 | 3.6 mm | 1.00 | 12 mm |

- 대칭만 걸면(B) 원본보다 나빠진다. 큰 오차(막힌 입구, 속 찬 내부, 평평한 바닥)가 앞뒤 대칭인 오류라서다.
- 대칭 + 그릇 구조 규칙(C)은 학습 없이 오차를 절반으로 줄인다. 벽 두께 15 mm는 GT를 보고 정한 값이며, 실제로는 검수자가 정하는 파라미터다.
- 실물의 비대칭은 평균 3.6 mm(D). 어떤 대칭 모델도 이 아래로는 못 내려간다.
- 규칙 C는 Blender Screw 모디파이어 하나로 구현되므로 ③ MCP 복원의 첫 작업 항목으로 적합하다. 회전대칭 토기에만 통한다.

## 4. 명세에 대한 시사점

- **D-18의 2D→3D 모델은 TripoSR 유지가 맞다.** SF3D·SPAR3D는 이 데이터에서 대체 후보가 못 된다. 앞선 검토에서 SF3D를
  "TripoSR의 직계 후속이라 바로 갈아탈 수 있는 1순위"로 추천했는데, 실측으로는 틀렸다. 사진 1장이라 단정은 못 하지만
  세 가지 foreground ratio에서 재현됐다.
- **G3 통과 기준을 수치로 둘 수 있다.** TripoSR 기준선 Chamfer 10.1 mm, F@2 % 0.71. 다른 후보는 같은 스크립트로 이 값을
  넘어야 교체를 논의한다.
- **막힌 입구와 속이 찬 형상은 2D→3D 단발 모델 공통 한계다.** 세 모델 모두 속이 빈 그릇을 만들지 못했다. ③ Blender MCP
  단계에서 입구 개방과 벽 두께 부여를 복원 작업 항목으로 넣는 것이 현실적이다(3장 규칙 C 참고).
- **스캔이 있는 유물은 스캔이 ①이어야 한다.** 명세 11.2, G3 폴백 경로.
- **`compare.py` / `multi_compare.py`가 FR-AI-005의 지표 도구가 된다.** 노이즈 바닥 1 mm로 GT 안정성도 확인됐다.
- **VRAM 실측이 공식 표기와 다르다.** SF3D는 "6 GB"라지만 텍스처 2048에서 7.9 GB, SPAR3D 저 VRAM 모드는 8.3 GB였다.
  기준 PC 8 GB에서 두 모델은 여유가 없다.

## 5. 실행 기록 (재현용)

공통: Windows 11, RTX 4070 Laptop 8 GB, 드라이버 591.44, Python 3.11.9, uv 0.12.5, torch 2.11.0+cu128. CUDA Toolkit 없음.
입력 `ai/2d-to-3d/test-images/comb-patterned pottery1.jpg`, 388 × 515 RGB,
SHA-256 `b5cf90113bf68b2e9129e8cce526b05da8fd31a03c2e806a35b23678ef2291fa`.
정답지 `ssu022891 (1).zip` 70,054,554 bytes, SHA-256 `b63fc9c7de4fe30371264cd07b83e4e232c22ae2ab2d78a3767f11e8ece7b3da` (저장소 밖 보관).

### 5.1 TripoSR

| 항목 | 값 |
| --- | --- |
| 코드 | 저장소 커밋 `107cefdc244c39106fa830359024f6a2f1c78871` (2026-06-04), MIT |
| 가중치 | HF `stabilityai/TripoSR` 스냅숏 `5b521936b01fbe1890f6f9baed0254ab6351c04a`; `model.ckpt` 1,677,246,742 bytes SHA-256 `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee`; `config.yaml` SHA-256 `74ca708ce086bf68e97709ea6b3d91f14717921c04691e84043f0eb8fcc68e62` |
| 옵션 | `run.py <img> --mc-resolution 256 --model-save-format obj` (rembg 배경 제거, foreground ratio 0.85) |
| 패키지 | transformers 4.35.0, trimesh 5.1.0, rembg 2.0.83, scikit-image 0.26.0 (`requirements/triposr-requirements-frozen.txt`) |
| 패치 | `tsr/models/isosurface.py`의 torchmcubes 호출을 scikit-image `marching_cubes`(CPU)로 대체. torchmcubes 빌드 실패(CMake/pybind11) |
| 시간 | 모델 초기화 11.3 s, 배경 제거 66.5 s, 추론 6.1 s, 메시 추출 6.7 s, 내보내기 0.9 s |
| 실패 | `--bake-texture`: moderngl WGL 컨텍스트 생성 불가(헤드리스 셸). 정점색만 출력 |

### 5.2 Stable Fast 3D

| 항목 | 값 |
| --- | --- |
| 코드 | 저장소 커밋 `ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2` (2025-01-22) |
| 가중치 | HF `stabilityai/stable-fast-3d` 스냅숏 `f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778`; `model.safetensors` 4,024,289,892 bytes SHA-256 `a3416e1cf654e7d4f5e75f116cec2c3f0a14501a77d30c2f6068bbda178de388`; `config.yaml` SHA-256 `5e880cfec3dec28d4e20afe07bfa1799c8a561f56f0532ebf1b4022ae4ddc8e4` |
| 라이선스 | Stability AI Community License. 게이트 저장소, 2026-09-04 계정 주인이 동의. "Powered by Stability AI" 표기 의무 |
| 옵션 | `run.py <img> --texture-resolution 2048` (rembg 배경 제거, foreground ratio 0.85). 변형: `--texture-resolution 1024 --foreground-ratio 0.75 / 1.0` |
| 패키지 | transformers 4.42.3, trimesh 4.4.1, rembg 2.0.57, numpy 1.26.4, open-clip-torch 2.24.0 (`requirements/sf3d-requirements-frozen.txt`) |
| 빌드 | `uv_unwrapper`, `texture_baker` C++ 확장은 VS 2026 개발자 환경(`vcvarsall.bat x64`, `DISTUTILS_USE_SDK=1`, `USE_CUDA=0`) 안에서 `--no-build-isolation`으로만 빌드됨. 일반 setuptools는 VS를 못 찾음 |
| 패치 | CPU 전용으로 빌드된 `texture_baker`가 CUDA 텐서를 받으면 실패 → 설치된 `texture_baker/baker.py`에서 입력을 `.cpu()`로 옮기고 결과를 원래 device로 복귀 |
| 시간 | 전체 22 s (추론 6.0 s/it) |
| 출력 | GLB, 2048² base color + normal map, PBR(metallic 0, roughness 0.48). bbox [0.654, 0.918, 0.229] |

### 5.3 SPAR3D

| 항목 | 값 |
| --- | --- |
| 코드 | 저장소 커밋 `fdc311b16809e6a8adc2f5a3407ebb3db1a95bd1` (2025-05-05) |
| 가중치 | HF `stabilityai/stable-point-aware-3d` 스냅숏 `5699918cb34f55cd7d828493d2725f3038313761`; `model.safetensors` 7,326,949,440 bytes SHA-256 `62673b63fd9dad425e74b213ffe8501262d9621d5174310ac33017747be31f58`; `config.yaml` SHA-256 `2795e11a7a6cb381a442e07abfdb5a3c0cc20e711c6e7e349ba73bf343c2ebdd`. 첫 실행 시 AlphaCLIP 가중치 891 MB 추가 다운로드 |
| 라이선스 | Stability AI Community License. 2026-09-04 동의 |
| 옵션 | `run.py <img> --low-vram-mode --texture-resolution 1024` (transparent-background 배경 제거, foreground ratio 1.3 기본) |
| 패키지 | transformers 4.42.3, trimesh 4.4.1, transparent-background 1.3.3, CLIP, AlphaCLIP (`requirements/spar3d-requirements-frozen.txt`) |
| 빌드 | 5.2와 같은 방식. CLIP/AlphaCLIP git 설치는 `--no-build-isolation` 필요(pkg_resources) |
| 패치 | ① `texture_baker` CPU 패치(5.2와 동일) ② `transparent_background/__init__.py`의 flet GUI import를 try/except로 감쌈(최신 flet과 비호환) ③ `run.py`가 리메셔 미설치 시 존재하지 않는 `args.reduction_count_type`을 참조하는 버그 → getattr 기본값 |
| 시간 | 전체 236 s (AlphaCLIP 다운로드 포함), 피크 VRAM 8,320 MB |
| 출력 | GLB, 1024² base color + normal map; `points.ply` 512점. bbox [0.722, 1.028, 0.259] |

### 정답지 파일 메모

- OBJ(Cinema 4D 출력): z-up, mm, 바닥 z ≈ 0. 정점 197,289 / 면 381,464. **mtl의 텍스처 파일명이 `su022891-...jpg`로 앞의 `s`가 빠져 있다.**
- PLY/STL(ZBrush "Step3"): y-up, PLY 정점색은 전부 흰색. Open3D는 한글 경로에서 열지 못하므로 ASCII 경로로 복사해 사용.
- OBJ와 PLY 사이 배율 차 0.06 %, Chamfer 1.06 mm.

## 6. 비교 방법

`../tools/compare.py` — 두 메시 표면에서 각 200,000점 샘플 → 24개 축 정렬 초기값에서 각각 유사변환 ICP(트림 90 %) → 최적 정렬을
60,000점으로 재정련(트림 95 %) → GT 좌표계(mm)에서 최근접 거리 기반 지표. 배율은 bbox 대각선 비율로 초기화하고
**±20 % 안에서만** 허용한다. 처음에 배율을 자유롭게 두었을 때 SF3D의 납작한 메시가 GT 표면 일부에 축소·밀착하는
퇴화 해(Chamfer 69 mm, 정확도 2 mm, 완전성 136 mm)가 나와 제한을 추가했다. TripoSR 수치는 제한 전후가 같다.
`../tools/multi_compare.py`는 여러 모델을 같은 절차로 정렬해 한 그림에 그린다. 텍스처는 UV 무게중심 보간으로 점마다 샘플링한다.

```bash
cd ai/2d-to-3d/eval/tools
python multi_compare.py --gt <GT>/ssu022891-001-30000.obj --out ../2026-09-03_pottery1_vs_ssu022891/metrics \n  --model TripoSR=../2026-09-03_pottery1_vs_ssu022891/meshes/triposr/mesh.obj \n  --model SF3D=../2026-09-03_pottery1_vs_ssu022891/meshes/sf3d/mesh.glb \n  --model SPAR3D=../2026-09-03_pottery1_vs_ssu022891/meshes/spar3d/mesh.glb
```

## 7. 파일

| 경로 | 내용 |
| --- | --- |
| `report.md` | 이 문서 |
| `metrics/multi_compare.json`, `multi_compare_table.md` | 3모델 지표, 정렬 변환, 메시 통계 |
| `metrics/triposr_metrics.json`, `triposr_gen_to_gt_4x4.json` | TripoSR 단독 지표와 GT 좌표계 변환 행렬 |
| `metrics/noise_floor_ply_vs_obj.json` | GT 두 파일(PLY, OBJ) 사이 오차 = 측정 바닥 |
| `metrics/symmetry_test.json` | 대칭 규칙 실험 지표 |
| `figures/multi_compare.png`, `multi_compare_metrics.png` | 3모델 4방향 렌더 + 단면, 지표 막대 |
| `figures/side_by_side.png`, `cross_section.png`, `views.png`, `overlay_profile_hist.png` | TripoSR vs GT 상세 그림 |
| `figures/gt_views.png`, `triposr_turntable.png`, `symmetry_test.png` | 정답지 4방향, TripoSR 자체 렌더, 대칭 실험 |
| `inputs/*_input_bg_removed.png` | 각 모델에 실제로 들어간 배경 제거 이미지 |
| `requirements/*-requirements-frozen.txt` | 환경별 패키지 고정 목록 (TripoSR, SF3D, SPAR3D) |
| `meshes/triposr/mesh.obj`, `turntable.mp4` | TripoSR 원본 출력(정점색), 자체 턴테이블 렌더 |
| `meshes/sf3d/mesh.glb`, `mesh_fr0.75.glb`, `mesh_fr1.0.glb` | SF3D 원본 출력 3종 (foreground ratio 0.85 / 0.75 / 1.0) |
| `meshes/spar3d/mesh.glb`, `points.ply` | SPAR3D 원본 출력과 조건 포인트 클라우드 |
| `meshes/aligned/*_aligned_to_gt.*` | GT 좌표계(mm)로 정렬한 각 모델 출력 |
| `meshes/symmetry/rule_B_*.obj`, `rule_C_*.obj` | 대칭 규칙 B(속 찬 덩어리), C(개방 + 벽 두께) 메시 |
| `../tools/compare.py`, `multi_compare.py`, `symmetry_test.py` | 비교 스크립트 (실험 폴더 밖, 재사용) |

메시 파일은 공개본에서 Git LFS로 관리한다(약 28 MB).
이 폴더의 `.obj`, `.glb`, `.ply`, `.mp4`가 LFS 대상이며 정답지 스캔은 저장소에 넣지 않았다.
