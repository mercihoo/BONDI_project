# lascomp_tools — LaS-Comp 을 8 GB 노트북에서 돌리기 위한 도구

저자 코드(`~/work/LaS-Comp`, WSL)는 한 줄도 고치지 않는다. 전부 **런타임 훅**과 실행·계측 스크립트다.
설계·근거·수치는 `docs/lascomp-8gb-quantization-plan.md`, 지표 원본은 `docs/lascomp-eval/`.

| 파일 | 역할 |
| --- | --- |
| `quantize_trellis.py` | `from_pretrained` 훅: DiT 만 FP8/INT8 (`LASCOMP_QUANT`) · FlexiCubes 격자/메시 추출을 CPU 로 (`LASCOMP_MESH_CPU=1`) · `decode_slat` 훅: 디코딩 전 오프로드 (`LASCOMP_OFFLOAD=1`). 저장소 루트에 둔다 |
| `measure_vram.py` | 저자 스크립트를 runpy 로 그대로 돌리며 nvidia-smi·torch 피크 기록 → `vram_reports/<라벨>.json` |
| `run-lean.sh OBJ QUANT IAS OFFLOAD MESHCPU` | 한 설정 실행. 채택 설정 `fp8 1 1 1` |
| `eval_completion.py` | Q3 Chamfer · Q4 관측부 보존(메시는 200 만 점 밀집 표본). **`output_points.ply` 로 Q4 를 재면 무효** — 저자가 입력 점군을 이어붙인다 |
| `ablate.sh OBJ "q ias off mcpu" ...` | 설정을 순서대로 돌리고 평가, `ablation_summary.md` |
| `run-when-free.sh` · `status.sh` · `eval-mesh.sh` | GPU 비면 실행 · 상태 한 장 · 메시 평가 |
| `diag_*.py` | 진단 — 양자화 필터/바이트 계측, .glb 프레임, 관측부 꼬리 |
| `stage*.sh`, `fetch-*.sh`, `fix-pipeline-json.sh`, `trim-decoders.sh`, `wsl-setup-user.sh` | 환경 구축 순서 그대로 (WSL 사용자 → 시스템·torch → 휠 의존성 → utils3d/pytorch3d → 나머지 → numpy 되돌림) |

WSL 배치: `quantize_trellis.py` 는 `~/work/LaS-Comp/`, 나머지는 `~/work/LaS-Comp/tools/`.
호출은 PowerShell → `wsl.exe -d Ubuntu-22.04 -- bash <스크립트>` 로 한다 (Git Bash 인라인은 `~`·`$` 가 MSYS 경로 변환에 먹힌다).
