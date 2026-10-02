# 3D → 3D 형상 복원 — 회전대칭 원통 전개

손상 3D 메시의 결손부를 **생성 모델 없이** 채운다.
근거는 검증 가능한 가정 둘 — **회전대칭**과 **n-fold 투창 주기**다.

- 담당: 기여자
- 대상 검증 유물: 굽다리바리 경주 신수 71489
- 관련 요구사항: `FR-AI-003` · `NFR-ETH-001` ~ `NFR-ETH-004`

---

## 1. 파이프라인에서의 위치

명세 §3 의 3단계 중 **② 모델 복원** 자리다.

```text
사진 1장
  └─[A] TRELLIS.2 ──→ 손상 GLB          ① 손상 상태 3D
          restore_run.py                    │
                                            ▼
       [B] 원통 전개 → 결손 판정 → 채움   ② 이 모듈
          build_mesh_direct.py              │
                                            ▼
                                  GLB 2노드 + manifest.json
                                            │
       [C] 채움면에 재질 ─────────────────  ▼
          texture_from_mesh.py        → MATERIAL.md (형상 무개입)
```

> **주의 — 이 모듈은 ② 자리에 있으나 학습 모델을 쓰지 않는다.**
> 명세 §3.1 의 분류가 `observed` / `ai_inferred` 두 칸뿐이라 현재 manifest 는
> 채움부를 `ai_inferred` 로 적고 있는데, **실제로는 AI 가 관여하지 않은 기하 추정**이다.
> 분류 한 칸을 더 둘지는 팀 결정 사항이다 (MR 참조).

## 2. 왜 생성 모델을 안 쓰나

SLAT 주입(생성 모델이 결손을 채우게 하는 경로)은 메시 조각 수가 **22 아래로 안 내려갔다**
(손상본은 2). 원인은 표현 자체였다 — 복셀은 한 덩어리인데 메시가 갈라진다.

`(y, θ)` 격자에서 메시를 직접 만들면 **인접 칸이 정점을 공유하므로 면 연속성이 정의상 보장**된다.
이 실패 양상이 구조적으로 존재하지 않는다.

## 3. 실행

### 3.0 처음 받았다면

**필요한 것 세 가지.** 형상만 돌릴 거면 ①②로 끝난다.

**① 파이썬 패키지**

```bash
pip install numpy scipy trimesh Pillow
pip install opencv-python            # compare_material.py 만 쓴다
```

CUDA 도 TRELLIS 도 **형상 복원에는 필요 없다.** 순수 기하 계산이다.

**② 손상 GLB (`A_normal.glb`) — 기여자에게 받는다**

이 저장소에는 없다. `out/.gitignore` 가 `*.glb` 를 빼기 때문이고,
A단계(사진 → 손상 3D)는 `ai/2d-to-3d` 영역이라 여기서 다시 만들지 않는다.

> **AI 직군 공유 스토리지 또는 담당자에게 `A_normal.glb` 를 받아 아무 데나 둔다.**
> 굽다리바리 71489 기준 정점 94,877 · 면 149,150 · bbox `[0.904, 1.002, 0.843]` 이다.
> 받은 것이 맞는지는 `manifest.json` 의 `carried` 와 대조하면 된다.

**③ TRELLIS.2 설치본 — 재질까지 갈 때만**

```powershell
$env:TRELLIS2_HOME = "<trellis2-stableprojectorz 의 code 폴더>"
```

형상(`build_mesh_direct.py`)에는 필요 없고, 재질(`texture_from_mesh.py`)과
A단계(`restore_run.py`)만 이것과 CUDA 8GB 를 요구한다.

### 3.1 형상 복원

```bash
python build_mesh_direct.py --glb <손상GLB> --out out/v24-cyl+biharm+n6+nn+up2 \
    --ny 96 --ntheta 144 --openwork-max-deg 50 --nfold 6
```

굽다리바리 71489 채택 실행에서 **기본값이 아닌 인자는 둘뿐**이다.

| 인자 | 값 | 기본값 | 뜻 |
| --- | --- | --- | --- |
| `--openwork-max-deg` | **50** | 40 | 투창 판정 θ 폭 상한 |
| `--nfold` | **6** | `off` | n-fold 주기로 투창 위치를 추정해 다시 뚫는다 |
| `--ny` / `--ntheta` | 96 / 144 | 같음 | 격자. 명시적으로 넘겼다 |

나머지(두께 0.008 · 경계 스냅 10mm · 이중조화 order 2 · obs-blur 5 · mesh-upsample 2)는 전부 기본값이다.

**여기까지가 이 모듈의 본체다.** 나온 GLB 는 노드 2개(`region_carried` / `region_filled`)를 갖는다(§4).

### 3.2 재질 — 채움면에만

형상이 끝난 메시에 재질을 입히는 것은 **[MATERIAL.md](MATERIAL.md)** 에 있다.
요약하면 세 줄이다.

```bash
python texture_from_mesh.py encode          # 메시 → shape SLAT
python texture_from_mesh.py texture         # → 전 면에 새 재질 (v27)
python compose_material.py --filled-from v27-… --color-match labshift --out v32-… \
    --glb-name restored_labshift.glb        # → 관측부는 원본, 채움만 새 재질
```

**세 번째 줄을 빼면 안 된다.** 두 번째까지는 관측부 재질까지 재생성되어
`NFR-ETH-003` 이 재질 축에서 깨진다.

### 3.3 A단계를 직접 돌리려면

사진에서 손상 3D 를 만드는 단계다. **`A_normal.glb` 를 받았다면 건너뛴다.**

```bash
python restore_run.py --image <사진> --out <출력폴더>
```

> `restore_run.py` 는 SLAT 주입 실험까지 포함한 실행기이고,
> **채택 경로에서 쓰는 것은 `load_pipeline`·`get_cond` 뿐**이다(재질 단계가 import 한다).
> `A_normal.glb` 자체를 만든 스크립트(`probe_coords.py`)는 이 저장소에 넣지 않았다 —
> A단계 산출물은 팀에서 이미 공유되고 있고, 이 모듈은 그것을 **입력으로 받는** 위치다.

> **A단계는 `ai/2d-to-3d` 영역과 겹친다.** 이 모듈을 단독으로 재현할 수 있게 같이 뒀다.
> 배치를 옮길지는 팀 협의 사항이다.

## 4. 출력

GLB 의 **노드 2개**가 관측과 추정을 나눈다. 이것이 `NFR-ETH-003` 을 만족시키는 방식이다.

| 노드 | 뜻 |
| --- | --- |
| `region_carried` | 입력 메시 그대로. **한 정점도 수정하지 않는다** |
| `region_filled` | 채운 부분 |

옆에 `manifest.json` 을 같이 쓴다. 주요 필드는 아래와 같다.

| 필드 | 뜻 |
| --- | --- |
| `method` | 폴더 이름과 같은 방법 태그 |
| `grid` · `axis_xz` · `thickness` · `snap_mm` | 재현에 필요한 설정 |
| `fill` | 채움 방식. `scheme`(이중조화) · `solved_on`(회전대칭 프로파일과의 잔차) 등 |
| `observed_cells_frac` | 격자 칸 중 관측이 있던 비율 |
| `regions` | 빈칸 덩어리 분류 — 채울 것 / 투창 / 무시할 잡티 |
| `openwork_kept_open` | **판정해서** 안 막은 투창. 면적·θ폭·높이·방위각 |
| `openwork_nfold_carved` | **추정해서** 다시 뚫은 투창. `nearest_known_deg` 로 근거를 추적한다 |
| `carried` / `filled` | 노드별 정점·면 수 |
| `provenance` | **면별 근거 문자열.** 명세 §3.1 에 따라 UI 가 이 값을 그대로 따른다 |

> `carried.modified` 는 **코드에 박힌 상수 0** 이다 — 관측 메시를 건드리지 않는 구조라 설계상 참이지만,
> manifest 가 그것을 측정한 값은 아니다. 실측 검증은 재질 경로에서 따로 했다(정점 거리 중앙·p95 `0.00e+00`).

## 5. 71489 실측치

| | |
| --- | ---: |
| 관측 칸 비율 | **64.5%** |
| 채움 영역 / 보존 투창 / 무시한 잡티 | 5 / **10** (판정 6 + n-fold 추정 4) / 97 |
| `region_carried` 정점·면 | 94,877 / 149,150 |
| `region_filled` 정점·면 | 33,864 / 62,188 |
| 채움면 이면각 중앙 | **0.948°** (전진복사 4.905° → 이중조화 1.029° → n-fold 0.948°) |
| 사진 시점 실루엣 IoU | **85.5%** |

### 5.1 근거 그림

복원본 GLB는 저장소에 넣지 않고 별도 작업 스토리지에 둔다.
재현에 필요한 것은 `out/<버전>/manifest.json` 이고, 아래 그림이 위 수치를 뒷받침한다.

**결손·투창·잡티 판정** — `(y, θ)` 격자 위의 판정 결과. manifest 의 `regions` 가 이 그림이다.

![결손·투창 판정 지도](out/figures/region-map.png)

**`--nfold 6` 을 고른 근거** — n=4 와 n=6 을 나란히 렌더해 사진과 대조했다.
푸리에 1위는 다른 값이었으나 렌더를 보고 6 으로 갔다.

![n-fold 4 대 6](out/figures/nfold-4-vs-6.png)

**전진 복사 대 이중조화** — 채움면 이면각 중앙 4.905° → 1.029°.
`Δ²u = 0` 을 한 번에 푸는 것으로 바꾼 결과다.

![채움면 매끄러움 비교](out/figures/fill-smoothness.png)

**입력 사진 시점 대조** — 사진과 같은 카메라로 렌더했다. 실루엣 IoU 85.5%.
**이 값은 사진 시점에서만 잰 것이고 반대편은 검증된 바 없다** (§6).

![사진 대 복원](out/figures/photo-vs-restored.png)

재질 단계의 그림은 [MATERIAL.md](MATERIAL.md) §3·§4 에 있다.

## 6. 알려진 한계

`NFR-ETH-002`(알려진 한계 표시) 대상이다.

1. **입력 사진이 1장이다.** 뒷면·내부는 A단계가 만든 추정인데 현재 `region_carried` 에 섞여 있다.
   IoU 85.5% 는 **사진 시점에서만** 잰 값이고 반대편은 검증된 바 없다.
2. **구연부.** 투창 판정 조건이 "위아래 끝에 안 닿는 구멍"이라 구연부 결손은 원리적으로 못 가린다.
   상단 채움에 "없던 구연 연장"이 섞였을 수 있다.
3. **회전대칭은 가정이다.** 측정해서 확인한 값이 아니다. 회전체가 아닌 기종에는 그대로 못 쓴다.

## 7. 파일과 의존

### 실행 경로 (형상)

```text
build_mesh_direct.py                    형상 — 채움·투창 카빙·2노드 GLB
  └ skeleton_cyl.py                     원통 전개 판. GLB 로드와 (y,θ) 격자
      └ restore_pottery_self_supervised.py
            fit_axis · classify_surfaces · radial_grid · fill_profile · fill_grid

texture_from_mesh.py                    재질 — shape SLAT → tex SLAT → PBR 베이크
  └ restore_run.py                      load_pipeline · get_cond 만 쓴다
compose_material.py                     합성 — 관측부는 원본, 채움만 갈아끼운다 (의존 없음)
```

세 파일이 한 묶음이다. `skeleton_cyl.py` 는 같은 폴더에서
`restore_pottery_self_supervised.py` 를 찾는다 (`SSV_DIR` 환경변수로 덮어쓸 수 있다).

`restore_run.py` 는 A단계 실행기라 TRELLIS.2 설치본과 CUDA 를 따로 요구한다.
**채택 경로에서는 `load_pipeline`·`get_cond` 만 쓴다** — 재질 단계가 import 한다.
`A_normal.glb` 를 만든 `probe_coords.py` 는 이 저장소에 없다 (§3.0 ②).

### 그림 재현

| 스크립트 | 만드는 그림 |
| --- | --- |
| `map_regions.py` | `region-map.png` (`out/regions.png`) |
| `fig_smoothness.py` | `fill-smoothness.png` (`out/fig_smoothness.png`) |
| `render_check.py` (+ `make_render_photo.py`) | `photo-vs-restored.png` |
| `compare_material.py` | `material-compare-v28ref.png` |

> `nfold-4-vs-6.png` 와 `material-512-vs-1024.png` 는 **일회성 비교라 생성 스크립트가 없다.**
> 다시 만들려면 n=4·n=6 을 따로 돌려 `render_check.py` 로 렌더하고 붙이면 된다.

### 선택 경로

| 모듈 | 언제 | 상태 |
| --- | --- | --- |
| `lama.py` | `--color-inpaint` — 채움 색을 LaMa 로 인페인팅 | **없다.** 채택 실행은 최근접 복사를 쓴다 |
| `samplers_masked.py` | `restore_run.py --masked` — RePaint 식 마스킹 샘플링 | **없다.** SLAT 주입 경로는 종결됐다 |
| `unet_relief.py` | `--relief` — 채움면에 요철을 얹는다 (층 ③) | **없다.** 미착수 |

> **이 저장소에는 실제로 쓴 것만 둔다.** 위 세 플래그는 `--help` 에 남아 있으나
> 모듈이 없어 쓰면 `ImportError` 가 난다. 쓸 일이 생기면 그때 해당 이슈로 올린다.

### 패키지

`numpy` · `scipy` · `trimesh` · `Pillow`.
`compare_material.py` 는 `opencv-python` 을, A단계와 재질은 TRELLIS.2 설치본과 CUDA 를 더 요구한다.
