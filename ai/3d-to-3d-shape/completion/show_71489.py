# -*- coding: utf-8 -*-
"""
show_71489.py — A_normal.glb 를 AdaPoinTr 로 복원한 결과를 **형상으로** 본다.

지표(보충점 20/8192)만으로는 감이 안 오니 세 가지를 만든다.
  1. 정면·측면·위 3방향 × (입력 / AdaPoinTr / v32 회전대칭) 비교 그림
  2. 출력 점군을 PLY 로 내보내 블렌더에서 열 수 있게
  3. "출력이 입력과 얼마나 다른가" 숫자

먼저 `run_71489.py` 를 돌려 work/pred_71489_A_normal.npz 가 있어야 한다.

    PY2="<restore>/gupdari-shape-prior/.venv/Scripts/python.exe"
    $PY2 show_71489.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _plt():
    """matplotlib 은 shape-prior venv 에만 있다. --dump-v32 는 TRELLIS venv 로
    돌려야 해서(trimesh 가 거기 있다) 최상위에서 import 하면 안 된다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
OUT = WORK / "restored_adapointr"
GRAY, BLUE, GREEN, RED = "#b8b8b8", "#2b6cb0", "#27ae60", "#e23b3b"


def unit(P):
    P = np.asarray(P, np.float64)
    c = P.mean(0)
    s = float(np.linalg.norm(P - c, axis=1).max())
    return (P - c) / s


def axes_of(P, up):
    hor = [i for i in range(3) if i != up]
    sp = [np.ptp(P[:, i]) for i in hor]
    h0 = hor[int(np.argmax(sp))]
    h1 = hor[1 - int(np.argmax(sp))]
    return h0, h1


def write_ply(path: Path, P: np.ndarray, rgb=None):
    P = np.asarray(P, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        head = ("ply\nformat binary_little_endian 1.0\n"
                "element vertex %d\n"
                "property float x\nproperty float y\nproperty float z\n" % len(P))
        if rgb is not None:
            head += ("property uchar red\nproperty uchar green\n"
                     "property uchar blue\n")
        head += "end_header\n"
        f.write(head.encode("ascii"))
        if rgb is None:
            f.write(P.tobytes())
        else:
            rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec["x"], rec["y"], rec["z"] = P[:, 0], P[:, 1], P[:, 2]
            rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            f.write(rec.tobytes())


def dump_v32() -> int:
    """TRELLIS venv 로 돌린다. v32 복원본을 샘플링해 npy 로 떨군다."""
    import trimesh
    p = (ROOT / "gupdari-71489" / "out"
         / "v32-cyl+biharm+n6+trellisfill+labshift+up2" / "restored_labshift.glb")
    if not p.is_file():
        print("[!] 없다: " + str(p))
        return 1
    s = trimesh.load(str(p), process=False)
    m = s.to_geometry() if hasattr(s, "to_geometry") else s
    pts, _ = trimesh.sample.sample_surface(m, 40000, seed=3)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "v32_sample.npy", np.asarray(pts, np.float32))
    print("저장 → " + str(WORK / "v32_sample.npy"))
    return 0


def main() -> int:
    if "--dump-v32" in sys.argv:
        return dump_v32()
    f = WORK / "pred_71489_A_normal.npz"
    if not f.is_file():
        print("[!] 없다: %s  — run_71489.py 를 먼저 돌릴 것" % f)
        return 1
    d = np.load(f, allow_pickle=True)
    inp = np.asarray(d["input_dense"], np.float64)     # 정규화된 입력(조밀)
    pred = np.asarray(d["pred"], np.float64)
    new = np.asarray(d["new"], bool)
    up = int(d["up"])

    # v32 회전대칭 복원본 (비교용) — 자기 좌표계로 정규화해 형상만 본다.
    # trimesh 는 TRELLIS venv 에만, matplotlib 은 이쪽 venv 에만 있다.
    # 그래서 샘플링은 미리 해 두고(`--dump-v32`) 여기선 npy 를 읽기만 한다.
    v32f = WORK / "v32_sample.npy"
    v32 = unit(np.load(v32f)) if v32f.is_file() else None
    if v32 is None:
        print("[안내] v32 비교를 넣으려면 먼저:")
        print("       $PY_TRELLIS show_71489.py --dump-v32")

    # ---- 숫자
    dd, _ = cKDTree(inp).query(pred)
    print("=== AdaPoinTr 출력이 입력과 얼마나 다른가 ===")
    print("  출력 8192점의 입력 표면까지 거리 (정규화 단위, 유물 높이=약 2)")
    for q in (50, 90, 99, 100):
        print("    %3d%%  %.5f" % (q, np.percentile(dd, q)))
    print("  보충점(3×점간격 초과) %d개 / 8192" % int(new.sum()))
    print("  → **출력은 사실상 입력을 8192점으로 다시 뿌린 것이다.**")

    # ---- PLY
    rgb = np.tile(np.array([90, 110, 140], np.uint8), (len(pred), 1))
    rgb[new] = (226, 59, 59)
    write_ply(OUT / "A_normal_adapointr.ply", pred, rgb)
    write_ply(OUT / "A_normal_input.ply", inp[::10])
    print("\nPLY 저장 → %s" % OUT)

    # ---- 그림
    h0, h1 = axes_of(inp, up)
    views = [("정면", h0, up), ("측면", h1, up), ("위에서", h0, h1)]
    cols = [("입력 A_normal.glb", inp, GRAY, None),
            ("AdaPoinTr 복원", pred, BLUE, new),
            ("v32 회전대칭 복원 (참고)", v32, GREEN, None)]
    cols = [c for c in cols if c[1] is not None]

    plt = _plt()
    fig, axes = plt.subplots(len(views), len(cols), figsize=(4.0 * len(cols), 3.7 * len(views)))
    for r, (vn, a, b) in enumerate(views):
        for c, (title, P, col, mask) in enumerate(cols):
            ax = axes[r, c]
            ax.scatter(P[:, a], P[:, b], s=0.6, c=col, lw=0)
            if mask is not None and mask.any():
                ax.scatter(P[mask, a], P[mask, b], s=9, c=RED, lw=0,
                           label="보충점 %d" % int(mask.sum()))
                if r == 0:
                    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11.5)
            if c == 0:
                ax.text(-0.06, 0.5, vn, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=11)
    fig.suptitle("A_normal.glb — AdaPoinTr 복원 결과 (빨강 = 모델이 새로 만든 점)",
                 fontsize=13.5, y=0.998)
    fig.tight_layout(rect=[0.01, 0, 1, 0.99])
    fig.savefig(HERE / "images" / "결과-71489-복원.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("그림 저장 → images/결과-71489-복원.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
