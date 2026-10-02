# 재질 — 완형 메시에 TRELLIS.2 텍스처

형상이 끝난 메시의 **채움면에만** 재질을 생성한다. **형상에는 일절 개입하지 않는다.**

- 담당: 기여자
- 관련 요구사항: `FR-AI-003` · `NFR-ETH-001` ~ `NFR-ETH-004`
- 형상 단계: [README.md](README.md)

---

## 1. 형상과 왜 분리하나

v24 까지의 채움 색은 **최근접 관측 정점 복사**라 채움면에 얼룩이 남는다.
그렇다고 TRELLIS 에 형상을 다시 맡기면 메시가 조각난다(조각 수 22~168).

`Trellis2TexturingPipeline` 은 **shape flow model 이 없다** — 형상을 만들지 않고 **인코딩만** 한다.
그래서 완성된 메시를 주고 **속성 볼륨만 받아** 원래 정점에 입힐 수 있다.

```text
v24 완형 메시 ──인코딩──→ shape SLAT ──→ tex SLAT ──디코딩──→ 속성 볼륨
                                                                  │
                            v24 정점·면 + 새 속성 ────────────────┘
                            (디코딩된 메시는 버린다)
```

**형상 보존 실측 — v24 대비 정점 거리 중앙·p95 가 `0.00e+00`** (두 노드 모두).

## 2. 실행

**세 단계다. 마지막을 빼면 안 된다.**

```bash
# ① 형상 메시 → shape SLAT      (out/probe/slat_from_mesh.npz)
python texture_from_mesh.py encode

# ② + 사진 → 전 면에 새 재질     (out/v27-cyl+biharm+n6+trellistex+up2/)
python texture_from_mesh.py texture

# ③ 관측부는 원본, 채움만 새 재질 (out/v28-… 채택본)
python compose_material.py     --filled-from v27-cyl+biharm+n6+trellistex+up2     --out v28-cyl+biharm+n6+trellisfill+up2     --glb-name restored_trellisfill.glb
```

②까지만 하면 **관측부 재질까지 재생성되어 `NFR-ETH-003` 이 재질 축에서 깨진다**(§5).
③은 `to_glb` 를 노드마다 따로 돌려둔 덕에 가능하다 — UV 마스크 합성이 필요 없다.

색까지 맞춘 v32 는 ③의 입력을 v28 로 바꾸고 `--color-match` 를 준다.

```bash
python compose_material.py     --filled-from v28-cyl+biharm+n6+trellisfill+up2 --color-match labshift     --out v32-cyl+biharm+n6+trellisfill+labshift+up2     --glb-name restored_labshift.glb
```

| `--color-match` | 무엇 | 쓸 것인가 |
| --- | --- | --- |
| `none` | 그대로 (v28) | 채택본 |
| `labshift` | **Lab 에서 평균만 평행이동** (v32) | **권장.** 분산이 정의상 보존된다 |
| `rgbgain` | RGB 채널별 이득 (v31) | **쓰지 말 것.** 질감 6.7배 악화 (§4) |

인자를 안 넘기면 기본값이 채택 실행 그대로다 — `--resolution 512` · `--texsize 2048` · `--seed 1`.
①②는 TRELLIS.2 설치 경로와 CUDA 를 요구한다. ③은 `trimesh` 만 있으면 된다.

```powershell
$env:TRELLIS2_HOME = "<trellis2-stableprojectorz 의 code 폴더>"
```

재질 품질 비교는 아래로 잰다. 폴더마다 `material_report.json` 이 남는다.

```bash
python compare_material.py --ref v28-cyl+biharm+n6+trellisfill+up2
```

> **1024 를 쓰려면 인코딩부터 다시 한다.** latent 가 64³ 이 되므로
> `encode --resolution 1024 --slat <다른경로>` 로 뽑고 `texture --resolution 1024` 를 같은 `--slat` 으로 준다.

## 3. 판본

| 판본 | 구성 | 판정 |
| --- | --- | --- |
| v27 | 전 면 TRELLIS 512 | **관측부까지 재생성 — provenance 깨짐** |
| **v28** | v24 관측부 + TRELLIS 512 채움 | **채택** (3D 눈 판정) |
| v29·v30 | 1024 볼륨 | 질감은 1.65배인데 관측부와 대역이 어긋난다 |
| v31 | v30 + RGB 채널 이득 톤 보정 | **반증.** 색은 잡히고 질감이 6.7배 나빠졌다 |
| **v32** | v28 + Lab 평균 평행이동 | **현재 최적.** 색·질감 둘 다. **3D 눈 확인 미실행** |

**노드가 분리돼 있어 관측부 재질을 그대로 두고 채움만 갈아끼울 수 있다.** UV 마스크 합성이 필요 없다.

![재질 512 대 1024 대 톤보정](out/figures/material-512-vs-1024.png)

> 왼쪽부터 관측부(기준) · 512 · 1024 · 1024+톤보정. 위는 UV 아틀라스 전체, 아래는 중앙 확대.
> **1024 에서 태토 알갱이가 보이기 시작하고 톤 보정이 색을 관측부에 맞춘다.**
> 그런데 3D 로 보면 판단이 달라진다 — §4.

## 4. 지표 — 눈 판정을 숫자로

처음 쓴 **단일 고주파 값은 틀린 지표였다.** 절대 디테일 양만 재서 1024 가 512 를 이기는데,
3D 로 보면 512 가 자연스러웠다. 재는 것이 *"얼마나 세밀한가"* 가 아니라
***"관측부와 같은 종류의 질감인가"*** 였기 때문이다.

관측부를 기준으로 **파워 스펙트럼 대역 거리**와 **Lab 색차 ΔE** 를 함께 본다. 축은 둘이다.

| 축 | 무엇 | 왜 필요 |
| --- | --- | --- |
| **A 내부 이질감** | 같은 GLB 안 관측부 ↔ 채움부 | "보기에 자연스러운가" |
| **B 관측부 충실도** | 관측부가 원본과 같은가 | A 만 보면 **전 면 재생성이 유리해진다** |

| 판본 | 내부 ΔE | 내부 질감 | 관측부 충실도 |
| --- | ---: | ---: | --- |
| v28 | 7.288 | 0.00641 | **완벽** |
| v31 RGB 이득 | 1.116 | 0.04293 (6.7배 악화) | 완벽 |
| **v32 Lab 이동** | **1.483** | **0.00594** | **완벽** |

> **색 평균을 맞추는 연산은 질감을 건드린다.** RGB 채널 이득은 평균만이 아니라 **편차까지 같은 배수로**
> 늘려 색얼룩 구조를 증폭한다. Lab 에서 평균만 더하면 분산이 정의상 보존된다.

**표본이 타일 60장이라 지표가 v27 대 v28 을 가르지는 못했다. 3D 눈 대조를 없애지 말고 1차 필터로 쓸 것.**

![재질 비교 v28 기준](out/figures/material-compare-v28ref.png)

> 축 A(내부 이질감)와 축 B(관측부 충실도)를 함께 본 결과. **v28 만이 "내부 질감 최선급 + 관측부 완벽"이다.**
> v27 은 내부 지표가 사실상 동률인데 관측부를 재생성해 ΔE 5.05 만큼 바꿔버렸다.

## 5. provenance 가 축별로 갈라진다

| | 근거 |
| --- | --- |
| 형상 | `"회전대칭 가정 + n-fold 투창 가정"` |
| **재질** | `"TRELLIS.2 텍스처링 512 (AI 추정)"` |

v24 에서는 재질도 관측 색의 복사라 근거가 뒤섞여 있었다.
manifest `material_source` 가 **노드별로 어느 실행에서 재질을 가져왔는지** 적는다.
v32 의 Lab 이동값은 **사람이 건 보정**이라 그 사실과 값을 같이 적는다.

## 6. 알려진 한계

1. **사진 한 장으로 만든 재질이라 뒷면 재질에 근거가 없다.** 형상 §6 의 1번과 같은 문제다.
2. **v32 는 3D 눈 확인을 안 했다.** 지표만 최선이다. 전시에 넣기 전에 봐야 한다.
3. **톤 보정을 쓸지 자체가 미정이다.** 파단면이 실제로 다른 색이라면 맞추지 않는 것이 옳은데,
   실물 근거가 없다.

## 7. 의존

| 단계 | 요구 |
| --- | --- |
| ①② `texture_from_mesh.py` | `numpy` · `trimesh` · `Pillow` + **TRELLIS.2 설치본 · CUDA** |
| ③ `compose_material.py` | `numpy` · `trimesh` · `Pillow` (+ `opencv-python` — `--color-match` 쓸 때) |
| 검증 `compare_material.py` | 위 + `opencv-python` · `scipy` |

8GB VRAM 에서 512·1024 볼륨 모두 OOM 없이 돈다 — 1024 는 복셀 1,200만 · tex 122.6s · decode 49.2s.
