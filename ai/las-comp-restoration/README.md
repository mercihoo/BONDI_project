# LaS-Comp 형상 복원 (8GB 로컬)

손상 유물 3D 스캔을 **원본 + 복원 추가분** 형태로 완성한다.
LaS-Comp(CVPR 2026, TRELLIS 기반 제로샷 3D 완성)을 RTX 4070 Laptop 8GB 에서 돌리고,
결과를 원본 위에 얹어 실측 외관·질감을 남긴다. 담당: 기여자

## 문서

| 문서 | 내용 |
| --- | --- |
| [복원구현 단계.md](docs/복원구현%20단계.md) | 0~8단계 전체 절차 — 각 단계의 세부 단계·실측값·함정 |
| [양자화내용 수치로 비교.md](docs/양자화내용%20수치로%20비교.md) | 양자화 방법(torchao weight-only) + 메모리 수단 4가지의 실측 비교 |
| [사용모델 + 파라미터설명.md](docs/사용모델%20+%20파라미터설명.md) | TRELLIS·DINOv2·LaS-Comp 구성 + τ·섬 임계·질감 k/세기 설명 |
| [docs/lascomp-8gb-quantization-plan.md](docs/lascomp-8gb-quantization-plan.md) | 초기 계획서 (진행하며 갱신) |

측정 원자료: `docs/vram_reports/` (VRAM) · `docs/eval_reports/` (합성·질감·Q4)
예시 사진은 아직 넣지 않았다 — 추후 추가 예정.

## 코드 (`tools/`)

저자 코드(LaS-Comp/TRELLIS)는 한 줄도 고치지 않는다 — 전부 **런타임 훅** 또는 후처리다.

| 구분 | 파일 |
| --- | --- |
| 메모리·훅 | `quantize_trellis.py` (fp8/int8 양자화 · 오프로드 · 메시 CPU · 디코딩 절약 · 색 저장 · 다중 이미지) |
| 실행 | `run-gupdari.sh` → `run-lean.sh` → `postprocess.sh` (자동 후처리) |
| 후처리 | `composite.py` (원본+채움) · `texture_fill.py` (실측 질감) · `bake_fill_texture.py` (UV 굽기) · `hybrid_fill.py` (높이로 갈라 합치기) · `apply_colors.py` · `denorm_result.py` |
| 평가·렌더 | `eval_completion.py` (Q4) · `render_mesh_views.py` · `tune_tau.py` · `pick_frame.py` |
| 준비·진단 | `prep_artifact.py` · `inspect_glb.py` · `make_partial_norm.py` · `diag_*.py` · `measure_vram.py` · `pdl.py` (병렬 내려받기) |
| 환경 구축 | `wsl-setup-user.sh` · `stage1~5-*.sh` · `fetch-ckpt.sh` · `prep-text-ckpt.sh` |

## 실행 한 줄

```bash
IMAGE_OVERRIDE=samples/artifacts/<이름>/image_guide.png REFTAG=<태그> \
TAU=0.6 MINAREA=0.002 TEXMODE=blend TEXK=12 TEXGAIN=0.8 \
  bash tools/run-gupdari.sh fp8 1 1 <이름>
```

채택 설정: fp8 · IAS 1 · 오프로드 · 메시 CPU · 디코딩 절약 훅 → 피크 4.5~4.9 GB / 실행 8~9분.

## 여기에 넣지 않는 것

| 대상 | 어디로 |
| --- | --- |
| 모델 가중치 (`ckpt/`) | 저장소 밖 (`fetch-ckpt.sh` 로 내려받음) |
| 실행 결과 GLB·렌더 PNG | 로컬 `results/` — 확정본만 자료 서버에 등록 |
| LaS-Comp/TRELLIS 저자 코드 | 로컬 체크아웃 (`~/work/LaS-Comp`) |
