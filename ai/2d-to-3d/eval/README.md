# ai/2d-to-3d/eval — 2D→3D 생성 결과 평가

생성 모델이 만든 3D 메시를 실측 스캔(정답지)에 정렬해 형상 오차를 재는 도구와, 실험별 결과 기록을 둔다.
명세 FR-AI-005(모델명·버전·설정·지표 보존)와 G3 게이트(3D 생성 모델 선정)의 근거 자료다.

## 구조

```text
eval/
  tools/                      재사용 스크립트 (실험과 무관하게 유지)
    compare.py                메시 1개 vs GT: 유사변환 ICP 정렬 + Chamfer/F-score/Hausdorff + 그림
    multi_compare.py          메시 여러 개 vs GT: 같은 절차로 한 그림·한 표에 비교
    symmetry_test.py          회전대칭 규칙(실루엣 회전, 입구 개방, 벽 두께)이 오차를 줄이는지 검증
  <YYYY-MM-DD>_<입력>_vs_<정답지>/   실험 1건 = 폴더 1개
    report.md                 결과·소견·실행 기록 (재현에 필요한 커밋·해시·옵션·패치 포함)
    metrics/                  지표 JSON, 정렬 변환, 비교 표
    figures/                  렌더 비교, 단면, 오차 히트맵, 지표 막대
    inputs/                   각 모델에 실제로 들어간 배경 제거 이미지
    requirements/             환경별 패키지 고정 목록 (.gitignore가 env/ 를 가상환경으로 제외하므로 이 이름을 씀)
    meshes/                   생성 메시·정렬 메시·영상 (LFS 활성화 전이라 일반 파일로 커밋, 약 28 MB)
```

`meshes/`의 대용량 형식은 현재 공개본에서 Git LFS로 관리한다.
report.md에 기록된 가중치 해시와 옵션으로 같은 결과를 다시 만들 수 있다.

## 사용법

```bash
cd ai/2d-to-3d/eval/tools
python compare.py --gen <생성.obj|glb> --gt <정답지.obj> --out <출력폴더> --gen-name <모델명>
python multi_compare.py --gt <정답지.obj> --out <출력폴더> --model TripoSR=<a.obj> --model SF3D=<b.glb>
python symmetry_test.py --gen-aligned <정렬된_생성.obj> --gt <정답지.obj> --out <출력폴더> --wall 15
```

- 의존성: numpy, scipy, trimesh, matplotlib (TripoSR 환경에 모두 포함). Open3D는 필요하지 않다.
- GT 파일은 저장소에 넣지 않는다. 한글 경로에서 일부 로더가 실패하므로 ASCII 경로에 복사해 쓴다.
- 정렬은 24개 축 초기값에서 유사변환 ICP를 돌려 최적을 고르고, 배율은 bbox 추정치 ±20 % 안에서만 허용한다.
  사진 입력에는 절대 크기가 없어 배율을 풀어야 하지만, 완전히 풀면 납작한 메시가 GT 표면 일부에 축소·밀착하는
  퇴화 해가 나온다.
- 지표는 GT 단위(mm)로 보고한다. F-score 임계값은 GT bbox 대각선의 1 %, 2 %, 5 %다.

## 실험 목록

| 폴더 | 입력 | 정답지 | 모델 | 요약 |
| --- | --- | --- | --- | --- |
| `2026-09-03_pottery1_vs_ssu022891` | `test-images/comb-patterned pottery1.jpg` | 빗살무늬토기 스캔 `ssu022891` (OBJ/PLY/STL) | TripoSR, SF3D, SPAR3D | TripoSR Chamfer 10.1 mm · F@2 % 0.71. SF3D·SPAR3D는 앞면만 있는 부조로 실패(46~47 mm). 대칭+그릇 규칙 후처리로 5.1 mm |
