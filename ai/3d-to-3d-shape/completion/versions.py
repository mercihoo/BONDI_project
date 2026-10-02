# -*- coding: utf-8 -*-
"""
versions.py — 71489 결과물 **버전 한 곳에서 관리**.

왜 스크립트로 두나
  `A_normal_2node_tta8_144x216.glb` 처럼 이름에 설정을 욱여넣으면 금세 한계가 온다 —
  돌대 gain, 평활 sigma, 텍스처 방식까지 다 붙일 수가 없다. 그렇다고 이름을 줄이면
  **뭐가 달라서 좋아졌는지**가 사라진다.

  그래서 여기 표 하나에 (이름, 한 줄 특징, 인자) 를 적고 파일명은 `vN` 으로만 둔다.
  표가 곧 변경 이력이고, `--build` 로 **아무 버전이나 그대로 다시 만들 수 있다.**

  결과물은 `work/versions/vN/` 밑에 `restored.glb` 와 `meta.json`(인자+실측치) 로 간다.

무엇을 재나
  이면각    표면 거칠기. bitsal QA 가 쓰는 잣대다. 관측과 **같아야** 잘된 것이다
  돌대      r(h) 대역통과 진폭. 가로 줄무늬가 결손부로 이어졌나
  색 국소std  9x9 국소 편차. 낮으면 단색, 관측과 같아야 옆과 이어져 보인다

사용
  PY=".../venv/Scripts/python.exe"
  $PY versions.py --list
  $PY versions.py --build v7
  $PY versions.py --measure            # 있는 것 전부 재서 표로
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
VDIR = WORK / "restored_adapointr"      # 결과물은 여기 한 곳에 모은다
OLD = VDIR / "_이전파일"                  # 폴더 체계 이전의 낱개 GLB
SRC = "work/pred_71489_A_normal_tta8.npz"          # v2 가중치 + TTA8
SRC_V3 = "work/pred_71489_A_normal_v3tta8.npz"     # v3 가중치 + TTA8

BASE = ["--nh", "144", "--nt", "216", "--thickness", "--smooth", "20",
        "--edge-rings", "3", "--mask-smooth", "2.5"]

# (버전, 폴더 꼬리표, 한 줄 특징, 추가 인자, 옛 파일명)
#
# 폴더 이름은 `v01_정점색단색` 처럼 **번호 + 특징**이다.
#   번호만 쓰면 목록에서 뭐가 뭔지 안 보이고,
#   특징만 쓰면 순서가 안 보인다. 둘 다 있어야 한다.
#   번호는 두 자리로 채운다 — 안 그러면 v1, v10, v2 순으로 정렬된다.
VERSIONS = [
    ("v1", "정점색단색", "정점색 단색 — 결손부를 한 가지 색으로 칠했다",
     None, "A_normal_2node_smooth.glb"),
    ("v2", "TTA8_144x216", "TTA8 + 144x216 격자 — 점 간격 1.90mm → 0.67mm",
     None, "A_normal_2node_tta8_144x216.glb"),
    ("v3", "회전복사텍스처", "회전 복사 텍스처 도입 — 같은 높이 다른 각도의 무늬를 가져온다",
     None, "A_normal_2node_rottex.glb"),
    ("v4", "돌대이식_거침", "돌대 이식 (f - lowpass) — 줄무늬는 이어졌으나 3배 거칠어졌다",
     None, "A_normal_FINAL.glb"),
    ("v4s", "두께껍질", "두께 있는 껍질 — 안쪽 벽을 만들어 속이 비치지 않게",
     None, "A_normal_SHELL.glb"),
    ("v5", "돌대대역통과", "돌대를 대역통과로 — 거칠기는 잡혔으나 색이 단색(국소 39%)",
     None, "A_normal_v5.glb"),
    ("v6", "거울타일링", "텍스처 거울 타일링 — 무늬는 살았지만 산탄잡음까지 복사(국소 183%)",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.37", "--r-smooth", "2.0",
      "--tex-clean", "0", "--no-tex-outer"], None),
    ("v7", "텍스처직접표집", "원본 텍스처 직접 표집 + 산탄잡음 정리 — 색 국소 82%",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.37", "--r-smooth", "1.5",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--no-tex-outer"], None),
    ("v8", "기본면평활", "기본면을 더 매끄럽게 깎고 돌대만 얹는다 — 정밀도 조정",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--no-tex-outer"], None),
    ("v9", "돌대증가", "v8 에서 돌대만 조금 더 — 관측 진폭에 맞춘다",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.32", "--r-smooth", "1.2",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--no-tex-outer"], None),
    ("v10", "바깥면표본만", "텍스처에 바깥면 표본만 — 안쪽 그늘이 섞여 어둡던 것을 바로잡는다",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--tex-samples", "3000000", "--tex-clean", "0.8"], None),
    ("v11", "바깥면_표본늘림", "v10 에 표본을 되돌려 무늬를 더 살린다 — 현재 최선",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--tex-samples", "1500000", "--tex-clean", "0.8"], None),
    ("v12", "회전체되돌림_이음새tuck",
     "결손부를 회전체로 되돌려 울퉁불퉁함 제거 + 이음새를 관측 밑으로 깔아 틈 막음",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.5", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8"], None),
    ("v13", "밝기이음", "무늬는 복사하고 **밝기만** 이음새에 맞춘다 — 회전대칭 아닌 얼룩 보정",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.5", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v14", "두께교정", "벽 두께를 단면 빈틈으로 다시 재고 프로파일을 중앙값 필터로 — 8.2mm 오측 교정",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.5", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v15", "바깥벽정렬", "칸 중앙값이 벽 한가운데라 복원면이 4mm 파묻혀 있었다 — 바깥벽으로 정렬",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--outer-shift", "1.0", "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v16", "점유가중_바깥벽",
     "관측 theta 점유가 낮은 높이에서는 관측 대신 모델을 목표로 — 파단선 끝이 챙이 되던 것",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v17", "밀도거르기",
     "새 점의 국소 밀도 상위 70%만 쓴다 — 흩뿌려진 후광이 챙을 만들던 것",
     ["--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--density-keep", "0.70", "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),

    # ---- 여기부터 **가중치가 v3** (완형 16 · 패턴 6). 예측 npz 가 바뀐다 ----
    ("v18", "v3가중치", "학습 데이터 확대(완형 16·패턴 6)로 재학습한 v3 가중치",
     ["--src", SRC_V3,
      "--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--density-keep", "0.70", "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--mask-close", "0", "--min-island", "0", "--drop-loose", "0", "--min-piece", "0",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v19", "띠별밀도", "밀도 임계를 높이 띠마다 — 전역 하나면 아가리 연장이 5%만 살아남는다",
     ["--src", SRC_V3,
      "--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--density-keep", "0.70", "--density-bands", "24", "--density-cap", "0.92",
      "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--mask-close", "0", "--min-island", "0", "--drop-loose", "0", "--min-piece", "0",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v20", "마스크닫기", "결손 마스크의 잔구멍을 메우고 작은 섬을 버린다",
     ["--src", SRC_V3,
      "--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--density-keep", "0.70", "--density-bands", "24", "--density-cap", "0.92",
      "--mask-close", "2", "--min-island", "40", "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--drop-loose", "0", "--min-piece", "0",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v21", "조각날리기", "관측에 안 닿거나 300면 미만인 부스러기를 버린다 — **현재 최선**",
     ["--src", SRC_V3,
      "--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--density-keep", "0.70", "--density-bands", "24", "--density-cap", "0.92",
      "--mask-close", "2", "--min-island", "40",
      "--drop-loose", "6", "--min-piece", "300", "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),
    ("v22", "투창교정", "투창 높이 제한 + **깨진 가로살 복원** — 살이 끊겨 위아래 창이 이어진 자리",
     ["--src", SRC_V3,
      "--rot-texture", "--ridge", "--ridge-gain", "0.25", "--r-smooth", "1.2",
      "--openwork-hmax", "0.55", "--strut-ratio", "1.6", "--strut-min", "20",
      "--density-keep", "0.70", "--density-bands", "24", "--density-cap", "0.92",
      "--mask-close", "2", "--min-island", "40",
      "--drop-loose", "6", "--min-piece", "300", "--max-scatter", "0.07",
      "--outer-shift", "1.0", "--outer-cov-lo", "0.30", "--outer-cov-hi", "0.55",
      "--axisym", "0.85", "--axisym-ramp", "6",
      "--dilate", "3", "--seam-tuck", "0.25", "--seam-tuck-ramp", "3",
      "--tex-samples", "1500000", "--tex-clean", "0.8", "--tex-blend", "40"], None),

    # --- v23 — **격자를 안 쓰면 어떻게 되나.** 비교군이지 현행 경로가 아니다.
    #
    # `(h,θ)` 격자는 회전체 가정을 하나 깐다. 그래서 *"가정 없이 AI 예측대로만"* 을
    # 물을 수 있다. `make_mesh_direct.py` 로 점군에서 바로 면을 만든다.
    #
    # 결론부터: **푸아송은 투창과 결손을 못 가른다.** 점군에서 둘 다 "점이 없는 곳"
    # 이라 구별할 정보가 점 자체에 없다. 밀도 절단을 올리면 투창이 열리는 만큼
    # 결손도 안 메워진다. 격자 + n-fold 가 하는 일이 그 정보를 밖에서 넣는 것이다.
    ("v23a", "직접_푸아송_절단없음", "격자 없이 푸아송 — 결손 99.9% 메우나 **투창도 메운다**(열림 5.3%)",
     ["--script", "make_mesh_direct.py", "--src", SRC_V3,
      "--method", "poisson", "--depth", "8", "--trim", "0.0"], None),
    ("v23b", "직접_푸아송_15절단", "밀도 하위 15% 버림 — 투창 68.8% 열리나 결손은 13.3% 만 메운다",
     ["--script", "make_mesh_direct.py", "--src", SRC_V3,
      "--method", "poisson", "--depth", "8", "--trim", "0.15"], None),
    ("v23c", "직접_푸아송_30절단", "밀도 하위 30% 버림 — 투창 82.9% 열리나 결손 5.9%. **상충이 분명하다**",
     ["--script", "make_mesh_direct.py", "--src", SRC_V3,
      "--method", "poisson", "--depth", "8", "--trim", "0.30"], None),
    ("v23d", "직접_볼피벗", "볼피벗 — 구멍은 존중하나 성긴 곳에서 부서진다 (이면각 31.9°)",
     ["--script", "make_mesh_direct.py", "--src", SRC_V3,
      "--method", "bpa", "--radii", "2,4,6"], None),
]


def folder(name: str, slug: str) -> Path:
    """`v01_정점색단색` 처럼 번호를 두 자리로 채우고 꼬리표를 붙인다."""
    num = name[1:]
    suffix = ""
    while num and not num[-1].isdigit():
        suffix = num[-1] + suffix
        num = num[:-1]
    return VDIR / ("v%02d%s_%s" % (int(num), suffix, slug))


def build(tag: str, py: str) -> int:
    row = next((r for r in VERSIONS if r[0] == tag), None)
    if row is None:
        print("[!] 모르는 버전: " + tag)
        return 1
    name, slug, feat, extra, old = row
    d = folder(name, slug)
    d.mkdir(parents=True, exist_ok=True)
    if extra is None:
        if old and (OLD / old).is_file():
            shutil.copy2(OLD / old, d / "restored.glb")
            (d / "meta.json").write_text(json.dumps(
                {"version": name, "특징": feat, "출처": "옛 결과물 " + old,
                 "인자": "기록 없음 — 이 버전은 재현 인자를 남기지 않았다"},
                ensure_ascii=False, indent=2), encoding="utf-8")
            print("  %s ← %s (복사)" % (name, old))
            return 0
        print("  [건너뜀] %s — 재현 인자도 옛 파일도 없다" % name)
        return 1
    src = SRC
    if "--src" in extra:
        i = extra.index("--src"); src = extra[i + 1]
        extra = extra[:i] + extra[i + 2:]
    # **다른 메시화 스크립트도 쓸 수 있다.** v23 계열은 격자를 안 쓰는 비교군이라
    # `make_mesh_direct.py` 로 간다. 그쪽은 격자 인자(BASE)를 안 받으므로 같이 뺀다.
    script, base = "make_mesh_2node.py", BASE
    if "--script" in extra:
        i = extra.index("--script"); script = extra[i + 1]
        extra = extra[:i] + extra[i + 2:]
        base = []
    cmd = [py, script, "--src", src] + base + extra + ["--name", "_tmp_" + name]
    print("  %s 빌드: %s" % (name, " ".join(extra)))
    r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        return r.returncode
    shutil.move(str(VDIR / ("_tmp_%s.glb" % name)), str(d / "restored.glb"))
    # 비교하려면 같은 폴더에 원본이 있어야 한다. 지금까지 손으로 복사하고 있었다.
    if not (d / "original.glb").is_file():
        src_glb = ROOT / "gupdari-71489" / "in" / "A_normal.glb"
        if src_glb.is_file():
            shutil.copy2(src_glb, d / "original.glb")
    (d / "meta.json").write_text(json.dumps(
        {"version": name, "특징": feat, "src": src,
         "인자": base + extra, "스크립트": script, "로그": r.stdout[-4000:]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("     → " + str(d / "restored.glb"))
    return 0


# ------------------------------------------------------------ 실측

def measure(path: Path, ref=None):
    import trimesh
    from scipy.ndimage import gaussian_filter1d, uniform_filter
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mg", ROOT / "gupdari-shape-prior" / "measure_glb.py")
    mg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg)

    s = trimesh.load(str(path), process=False)
    g = s.geometry if hasattr(s, "geometry") else {"mesh": s}
    ai = g.get("region_ai")
    ob = g.get("region_observed")
    if ai is None:
        return None

    def dih(m):
        w = m.copy(); w.merge_vertices()
        fa = w.face_adjacency
        if len(fa) == 0:
            return float("nan")
        n = w.face_normals
        return float(np.degrees(np.arccos(
            np.clip((n[fa[:, 0]] * n[fa[:, 1]]).sum(1), -1, 1))).mean())

    def tex(m):
        im = getattr(getattr(m.visual, "material", None), "baseColorTexture", None)
        if im is None:
            vc = getattr(m.visual, "vertex_colors", None)
            if vc is None:
                return float("nan")
            # 정점색이면 국소 편차를 잴 격자가 없다. 전체 편차를 대신 쓴다.
            return float(np.asarray(vc)[:, :3].astype(np.float32).std())
        a = np.asarray(im.convert("RGB"), np.float32).mean(2)
        return float((a - uniform_filter(a, 9, mode="wrap")).std())

    # --- 이음새 밝기차: 맞닿는 띠에서 관측과 AI 의 평균 밝기가 얼마나 벌어지나.
    #     색 국소std 는 "무늬가 살아 있나"를, 이건 "옆과 톤이 맞나"를 본다.
    #     둘은 따로 논다 — v10 은 국소std 는 그대로인데 이 값만 좋아졌다.
    from scipy.spatial import cKDTree
    import render_versions as RV
    seam = float("nan")
    if ob is not None:
        up0, ax0, c20 = ref["up"], ref["ax"], ref["c2"]

        def outer(m, n=150000):
            """**바깥을 보는 표본만.** 두께 있는 껍질이라 안쪽벽이 같이 잡힌다.
            v12 처럼 패치를 관측 벽 밑으로 밀어 넣으면 3차원 최근접이 안쪽벽을
            짝지어 값이 통째로 튄다 (실측 -1.1 → +18.7). 법선으로 거른다."""
            P, fid = trimesh.sample.sample_surface(m, n, seed=3)
            P = np.asarray(P, float)
            fn = m.face_normals[fid]
            rx, ry = P[:, ax0[0]] - c20[0], P[:, ax0[1]] - c20[1]
            rr = np.hypot(rx, ry)
            ok = rr > 1e-9
            dot = np.zeros(len(P))
            dot[ok] = (fn[ok, ax0[0]] * rx[ok] + fn[ok, ax0[1]] * ry[ok]) / rr[ok]
            k = dot > 0.15
            return P[k], RV.face_colors(m)[fid][k].astype(float)

        Po, Co = outer(ob)
        Pa, Ca = outer(ai)
        if len(Po) > 100 and len(Pa) > 100:
            # **같은 (h, theta) 칸끼리 견준다.**
            #   3차원 최근접은 두 면이 겹치면 못 믿는다 — v12 는 패치를 관측 벽
            #   밑으로 밀어 넣어서 최근접 짝이 엉뚱한 곳을 골랐다.
            #   회전체의 좌표는 (h, theta) 이므로 그 칸에서 비교하는 게 맞다.
            lo0, hi0 = ref["rng"]
            NB, NT = 72, 108

            def cells(P):
                hh = np.clip(((P[:, up0] - lo0) / max(hi0 - lo0, 1e-9) * NB
                              ).astype(int), 0, NB - 1)
                tt = np.arctan2(P[:, ax0[1]] - c20[1], P[:, ax0[0]] - c20[0])
                jj = np.clip(((tt + np.pi) / (2 * np.pi) * NT).astype(int), 0, NT - 1)
                return hh * NT + jj

            def cell_mean(P, C):
                k = cells(P)
                n = np.bincount(k, minlength=NB * NT)
                v = np.stack([np.bincount(k, C[:, i], minlength=NB * NT)
                              for i in range(3)], 1)
                ok = n >= 12
                out = np.full((NB * NT, 3), np.nan)
                out[ok] = v[ok] / n[ok, None]
                return out

            Mo, Ma = cell_mean(Po, Co), cell_mean(Pa, Ca)
            both = np.isfinite(Mo[:, 0]) & np.isfinite(Ma[:, 0])
            if both.sum() >= 20:
                d = Ma[both].mean(1) - Mo[both].mean(1)
                seam = float(np.median(d))

    P = np.asarray(trimesh.sample.sample_surface(ai, 80000, seed=1)[0], float)
    up, ax, c2, rng = ref["up"], ref["ax"], ref["c2"], ref["rng"]

    def prof(Q, nb=140):
        r = np.hypot(Q[:, ax[0]] - c2[0], Q[:, ax[1]] - c2[1])
        b = np.clip(((Q[:, up] - rng[0]) / (rng[1] - rng[0]) * nb).astype(int), 0, nb - 1)
        return np.array([np.percentile(r[b == i], 90) if (b == i).sum() > 25 else np.nan
                         for i in range(nb)])

    def hf(p, sig=2.0):
        ok = np.isfinite(p)
        if ok.sum() < 8:
            return float("nan")
        f = p.copy()
        f[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], p[ok])
        return float(np.std(gaussian_filter1d(f, sig) - gaussian_filter1d(f, sig * 4)))

    # --- 여기부터 **격자 위에서 재는 넷**.
    #
    # 이 다섯(면 어긋남·벽 두께·θ 흔들림·투창 열림·결손 메움)은 §5.8.1 과 §5.12 에서
    # **일회성 스크립트로** 재고 있었다. 그래서 버전마다 자동으로 안 재졌고
    # 파이프라인.md 의 표에 v15 값이 v22 자리에 남아 있었다. 여기로 옮긴다.
    #
    # 격자는 **원본 메시에서** 만든다. 예측 npz 를 안 봐도 되고, 무엇보다
    # **버전끼리 같은 잣대**가 된다 (예측이 바뀌어도 투창 자리는 안 바뀐다).
    import sys as _sys
    _sys.path.insert(0, str(HERE))
    import make_mesh_71489 as MM

    orig_path = path.parent / "original.glb"
    grid = {}
    if orig_path.is_file():
        og = trimesh.load(str(orig_path), process=False)
        om = trimesh.util.concatenate(list(og.geometry.values()))             if hasattr(og, "geometry") else og
        fin = trimesh.util.concatenate([m for m in g.values()])

        NH, NT = 144, 216
        MMPU = 150.0 / (rng[1] - rng[0])        # 높이 15cm 환산 (§5.8.1 과 같은 기준)
        Vo = np.asarray(trimesh.sample.sample_surface(om, 1500000, seed=7)[0], float)
        Vf = np.asarray(trimesh.sample.sample_surface(fin, 1500000, seed=7)[0], float)
        Va = np.asarray(trimesh.sample.sample_surface(ai, 800000, seed=7)[0], float)
        he = np.linspace(rng[0], rng[1], NH + 1)

        def cyl(Q):
            x, y = Q[:, ax[0]] - c2[0], Q[:, ax[1]] - c2[1]
            return (np.clip(np.digitize(Q[:, up], he) - 1, 0, NH - 1),
                    np.clip(((np.arctan2(y, x) + np.pi) / (2 * np.pi) * NT).astype(int),
                            0, NT - 1),
                    np.hypot(x, y))

        hbo, tbo, ro = cyl(Vo)
        hbf, tbf, _ = cyl(Vf)
        hba, tba, ra = cyl(Va)
        occ = np.zeros((NH, NT), bool); occ[hbo, tbo] = True
        fin_occ = np.zeros((NH, NT), bool); fin_occ[hbf, tbf] = True

        # **투창 판정에는 점유 마스크를 넘긴다.** 빈칸 마스크를 넘기면 거의 0칸이
        # 나온다 — 실제로 두 번 틀렸다 (결과.md §5.11 · §5.12). 파이프라인이
        # make_mesh_2node 에서 `openwork_cells_nfold(dense_occ)` 로 부르는 것과 같다.
        ow, _per, _info = MM.openwork_cells_nfold(occ, band_h=10)
        ow[int(NH * 0.55):] = False              # 투창은 굽다리에만 (--openwork-hmax)
        dmg = (~occ) & (~ow)
        grid["투창칸"] = int(ow.sum())
        grid["결손칸"] = int(dmg.sum())
        if ow.sum():
            grid["투창열림"] = round(100 * float((ow & ~fin_occ).sum()) / ow.sum(), 1)
        if dmg.sum():
            grid["결손메움"] = round(100 * float((dmg & fin_occ).sum()) / dmg.sum(), 1)

        # --- **셋 다 바깥벽에서 잰다.**
        #
        # 처음엔 칸 안 반지름을 통째로 썼다가 틀렸다. 이 유물은 두께 있는 껍질이라
        # 한 칸에 **안쪽벽과 바깥벽이 같이** 들어간다. 그대로 변동계수를 내면
        # 울퉁불퉁함이 아니라 **벽 두께**를 재게 된다 (실측 0.079 vs 바깥벽만 쓰면 훨씬 작다).
        #
        # 그래서 칸마다 r 을 정렬해 **가운데 빈 틈**에서 자르고 (make_mesh_2node 의
        # wall_thickness 와 같은 방법), 바깥 무리·안쪽 무리를 따로 들고 셋을 뽑는다.
        NB, NTC = 48, 72

        def split_cells(hb, tb, r, need=60):
            """칸마다 (바깥벽 반지름, 안쪽벽 반지름, 행) 을 낸다."""
            hh = (hb * NB) // NH
            tt = (tb * NTC) // NT
            key = hh * NTC + tt
            o = np.argsort(key)
            k, rr = key[o], r[o]
            ub, st = np.unique(k, return_index=True)
            rows, outer, inner = [], [], []
            for kk, grp in zip(ub, np.split(rr, st[1:])):
                if len(grp) < need:
                    continue
                grp = np.sort(grp)
                lo_, hi_ = int(len(grp) * 0.2), int(len(grp) * 0.8)
                if hi_ - lo_ < 4:
                    continue
                gap = np.diff(grp[lo_:hi_])
                if gap.size == 0 or gap.max() <= 0:
                    continue
                j = lo_ + int(np.argmax(gap))
                rows.append(int(kk) // NTC)
                outer.append(float(np.median(grp[j + 1:])))
                inner.append(float(np.median(grp[:j + 1])))
            return (np.array(rows), np.array(outer), np.array(inner))

        ro_row, ro_out, ro_in = split_cells(hbo, tbo, ro)
        ra_row, ra_out, ra_in = split_cells(hba, tba, ra, need=40)

        # 벽 두께 — 바깥 − 안쪽
        if len(ro_out) >= 20 and len(ra_out) >= 20:
            t_o = float(np.median(ro_out - ro_in))
            t_a = float(np.median(ra_out - ra_in))
            grid["벽두께"] = round(t_a * MMPU, 2)
            grid["벽두께_관측"] = round(t_o * MMPU, 2)
            grid["벽두께%"] = round(100 * t_a / t_o)

        # 면 어긋남 — 같은 행에서 AI 바깥벽 − 관측 바깥벽. 음수면 파묻힘
        d_off = []
        for i in set(ra_row.tolist()) & set(ro_row.tolist()):
            d_off.append(np.median(ra_out[ra_row == i]) - np.median(ro_out[ro_row == i]))
        if len(d_off) >= 10:
            grid["면어긋남"] = round(float(np.median(d_off)) * MMPU, 2)

        # θ 흔들림 — 한 행 안에서 **바깥벽 반지름**이 각도에 따라 얼마나 흔들리나
        def wobble(row, out, need=6):
            v = [float(np.std(out[row == i]) / max(np.mean(out[row == i]), 1e-9))
                 for i in set(row.tolist()) if (row == i).sum() >= need]
            return float(np.median(v)) if len(v) >= 8 else float("nan")
        grid["θ흔들림"] = round(wobble(ra_row, ra_out), 4)
        grid["θ흔들림_관측"] = round(wobble(ro_row, ro_out), 4)

    out = {"이면각": dih(ai), "돌대": hf(prof(P)), "색국소": tex(ai), "이음새": seam,
           "정점": len(ai.vertices), "면": len(ai.faces)}
    out.update(grid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--build", default="", help="버전 이름, 또는 all")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not (args.build or args.measure):
        print("버전   특징")
        for n, sl, f, e, o in VERSIONS:
            mark = "재현가능" if e else "옛파일"
            print("  %-22s [%s] %s" % (folder(n, sl).name, mark, f))
        return 0

    if args.build:
        VDIR.mkdir(parents=True, exist_ok=True)
        tags = [r[0] for r in VERSIONS] if args.build == "all" else args.build.split(",")
        for t in tags:
            build(t, args.py)
        if not args.measure:
            return 0

    # --- 실측
    import trimesh
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mg", ROOT / "gupdari-shape-prior" / "measure_glb.py")
    mg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg)

    any_glb = next((folder(r[0], r[1]) / "restored.glb" for r in VERSIONS
                    if (folder(r[0], r[1]) / "restored.glb").is_file()), None)
    if any_glb is None:
        print("[!] 잰 것이 없다. 먼저 --build all")
        return 1
    s0 = trimesh.load(str(any_glb), process=False)
    ob = s0.geometry["region_observed"]
    O = np.asarray(trimesh.sample.sample_surface(ob, 60000, seed=1)[0], float)
    up = mg.up_axis(O); ax, c2 = mg.fit_axis(O, up)
    ref = {"up": up, "ax": ax, "c2": c2, "rng": (O[:, up].min(), O[:, up].max())}
    tgt = measure(any_glb, ref)          # 관측 기준값을 같은 함수로
    from scipy.ndimage import gaussian_filter1d, uniform_filter

    def dih(m):
        w = m.copy(); w.merge_vertices(); fa = w.face_adjacency
        n = w.face_normals
        return float(np.degrees(np.arccos(
            np.clip((n[fa[:, 0]] * n[fa[:, 1]]).sum(1), -1, 1))).mean())

    def prof(Q, nb=140):
        r = np.hypot(Q[:, ax[0]] - c2[0], Q[:, ax[1]] - c2[1])
        b = np.clip(((Q[:, up] - ref["rng"][0]) / (ref["rng"][1] - ref["rng"][0]) * nb
                     ).astype(int), 0, nb - 1)
        return np.array([np.percentile(r[b == i], 90) if (b == i).sum() > 25 else np.nan
                         for i in range(nb)])

    def hf(p, sig=2.0):
        ok = np.isfinite(p); f = p.copy()
        f[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], p[ok])
        return float(np.std(gaussian_filter1d(f, sig) - gaussian_filter1d(f, sig * 4)))

    oi = getattr(ob.visual.material, "baseColorTexture", None)
    a = np.asarray(oi.convert("RGB"), np.float32).mean(2)
    T = {"이면각": dih(ob), "돌대": hf(prof(O)),
         "색국소": float((a - uniform_filter(a, 9, mode="wrap")).std())}
    print("\n관측(목표)  이면각 %.2f°  ·  돌대 %.5f  ·  색 국소std %.2f"
          % (T["이면각"], T["돌대"], T["색국소"]))
    print("\n%-5s %-34s %7s %13s %13s %8s |%6s %8s %5s %7s"
          % ("버전", "특징", "이면각", "돌대", "색 국소std", "이음새",
             "어긋남", "θ흔들림", "두께%", "투창열림"))
    rows = []
    for n, sl, f, e, o in VERSIONS:
        p = folder(n, sl) / "restored.glb"
        if not p.is_file():
            continue
        m = measure(p, ref)
        if m is None:
            continue
        rows.append((n, f, m))
        cs = ("  텍스처 없음  " if not np.isfinite(m["색국소"])
              else "%7.2f(%3.0f%%)" % (m["색국소"], 100 * m["색국소"] / T["색국소"]))
        sm = ("     -  " if not np.isfinite(m["이음새"]) else "%+7.1f" % m["이음새"])
        gv = lambda k, w=6, d=1: ("%*s" % (w, "-")) if k not in m or not np.isfinite(
            m[k]) else ("%*.*f" % (w, d, m[k]))
        print("%-5s %-34s %6.2f° %6.5f(%3.0f%%) %s %s |%s %s %s %s"
              % (n, f[:34], m["이면각"], m["돌대"], 100 * m["돌대"] / T["돌대"], cs, sm,
                 gv("면어긋남"), gv("θ흔들림", 7, 4), gv("벽두께%", 5, 0), gv("투창열림", 6, 1)))
    import csv
    with (VDIR / "버전표.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.writer(fp)
        NEW = ["면어긋남mm", "θ흔들림", "θ흔들림_관측", "벽두께mm", "벽두께_관측mm",
               "벽두께%", "투창칸", "투창열림%", "결손칸", "결손메움%"]
        w.writerow(["버전", "폴더", "특징", "이면각", "돌대", "돌대%",
                    "색국소", "색국소%", "이음새밝기차"] + NEW)
        w.writerow(["관측", "-", "원본 무수정 (목표)", round(T["이면각"], 2),
                    round(T["돌대"], 5), 100, round(T["색국소"], 2), 100, 0])
        for n, f, m in rows:
            def pct(x, t):
                return "" if not np.isfinite(x) else round(100 * x / t)
            fold = next(folder(r[0], r[1]).name for r in VERSIONS if r[0] == n)
            w.writerow([n, fold, f, round(m["이면각"], 2), round(m["돌대"], 5),
                        pct(m["돌대"], T["돌대"]),
                        "" if not np.isfinite(m["색국소"]) else round(m["색국소"], 2),
                        pct(m["색국소"], T["색국소"]),
                        "" if not np.isfinite(m["이음새"]) else round(m["이음새"], 1)]
                       + [m.get(k.rstrip("mm%").rstrip("_관측") if False else
                                {"면어긋남mm": "면어긋남", "벽두께mm": "벽두께",
                                 "벽두께_관측mm": "벽두께_관측", "투창열림%": "투창열림",
                                 "결손메움%": "결손메움"}.get(k, k), "") for k in NEW])
    print("\n표 → " + str(VDIR / "버전표.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
