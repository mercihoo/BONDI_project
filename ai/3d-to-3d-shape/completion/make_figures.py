# -*- coding: utf-8 -*-
"""데이터.md 에 넣을 그림을 만든다.

matplotlib 이 TRELLIS venv 에 없다. gupdari-shape-prior 의 venv 로 돌린다
(거기 `make_figures.py` 와 같은 환경·같은 한글 폰트 설정).
그 venv 에는 pandas 도 없으므로 표는 `csv` 모듈로 읽는다 — 이웃 폴더와 같은 방식이다.

    PY="<restore>/gupdari-shape-prior/.venv/Scripts/python.exe"
    $PY make_figures.py
"""
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
FIG = HERE / "images"
FIG.mkdir(parents=True, exist_ok=True)

GRAY = "#b8b8b8"
DARK = "#2f3640"
RED = "#e23b3b"
BLUE = "#2b6cb0"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def slugify(s: str) -> str:
    return re.sub(r"\s+", "_", unicodedata.normalize("NFC", str(s)).strip())


def nospine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ------------------------------------------------------------ 1. 선별 깔때기

def fig_selection():
    steps = [
        ("trellis3d GLB", 42, ""),
        ("plain (굽다리바리)", 27, "handled 15 · dish 1 제외"),
        ("is_broken = False", 16, "깨짐 표시 11 제외"),
        ("입지름(mouth_cm) 있음", 9, "아가리 결손 7 제외"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(steps))[::-1]
    vals = [s[1] for s in steps]
    colors = [GRAY, GRAY, GRAY, BLUE]
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, (label, v, note) in zip(y, steps):
        ax.text(v + 0.6, yi, str(v), va="center", ha="left", fontsize=13,
                fontweight="bold", color=BLUE if v == 9 else DARK)
        if note:
            ax.text(v + 3.6, yi, note, va="center", ha="left",
                    fontsize=9.5, color="#777")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps], fontsize=11)
    ax.set_xlim(0, 50)
    ax.set_xlabel("유물 수")
    ax.set_title("완형 선별 — trellis3d 42점에서 9점만 남는다", fontsize=13, pad=12)
    nospine(ax)
    ax.text(0.5, -0.30,
            "경신 71489(복원 대상)는 마지막 단계에서 빠진다 — 아가리가 둘레 전체 톱니로 깨져 입지름이 없다",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=RED)
    fig.tight_layout()
    fig.savefig(FIG / "데이터-선별-깔때기.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-선별-깔때기.png")


# ------------------------------------------------------------ 2. 결손 패턴

def side_view(P, up):
    """옆에서 본 투영 (가로 = 퍼짐이 큰 수평축, 세로 = 회전축)."""
    hor = [i for i in range(3) if i != up]
    spread = [P[:, i].max() - P[:, i].min() for i in hor]
    return P[:, hor[int(np.argmax(spread))]], P[:, up]


def fig_patterns(source: str = None):
    man = rows(WORK / "pairs_manifest.csv")
    if source is None:
        source = next(r["source"] for r in man if r["pattern"] == "bottom")
    order = ["none", "bottom", "rim", "side"]
    title = {"none": "none — 결손 없음", "bottom": "bottom — 하부 결실",
             "rim": "rim — 구연부 결손", "side": "side — 측면 파단"}

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.4))
    for ax, pat in zip(axes, order):
        hit = [r for r in man if r["source"] == source and r["pattern"] == pat]
        if not hit:
            ax.axis("off")
            continue
        r0 = hit[0]
        d = np.load(WORK / "pairs" / (r0["name"] + ".npz"), allow_pickle=True)
        comp, up = d["complete"], int(d["up_axis"])
        # 생성 때 저장해 둔 정답 마스크를 그대로 쓴다.
        # 거리 문턱으로 역추정하면 partial(2048점)이 성긴 곳까지 결손으로 잡혀
        # 결손 0% 인 `none` 에도 빨간 점이 흩어진다.
        removed = d["complete_removed"].astype(bool)
        cx, cy = side_view(comp, up)
        ax.scatter(cx[~removed], cy[~removed], s=1.2, c=GRAY, linewidths=0)
        if removed.any():
            ax.scatter(cx[removed], cy[removed], s=1.6, c=RED, linewidths=0)

        ax.set_title("%s\n결손 %.1f%%" % (title[pat], 100 * fnum(r0["removed_ratio"])),
                     fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("결손 패턴 4종 — %s · 회색=남은 것 · 빨강=떼어낸 것" % source,
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "데이터-결손-패턴.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-결손-패턴.png  (%s)" % source)


# ------------------------------------------------------------ 3. AI-Hub 판정

def fig_aihub():
    f = WORK / "scan_EA.csv"
    if not f.is_file():
        print("  [건너뜀] scan_EA.csv 없음")
        return
    rr = rows(f)
    cnt = {}
    for r in rr:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    order = [o for o in ["파단의심", "판정불가", "보류", "회전체아님", "완형후보"]
             if o in cnt]
    vals = [cnt[o] for o in order]

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    cols = [RED if o == "완형후보" else GRAY for o in order]
    bars = ax.bar(order, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("유물 수")
    ax.set_title("AI-Hub 211-2 · EA(토기) %d점 전수 판정" % len(rr), fontsize=13, pad=10)
    ax.set_ylim(0, max(vals) * 1.18)
    nospine(ax)
    ax.text(0.5, -0.26,
            "완형후보 3점도 크기 1.0~1.7m·회전축 수평인 오탐 — 쓸 완형이 사실상 없다",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=RED)
    fig.tight_layout()
    fig.savefig(FIG / "데이터-aihub-판정.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-aihub-판정.png")


# ------------------------------------------------------------ 4. 완형 9점 크기

def fig_sizes():
    meta = ROOT / "gupdari-shape-prior" / "meta" / "artifacts.csv"
    if not meta.is_file():
        print("  [건너뜀] artifacts.csv 없음")
        return
    seen, plain = set(), []
    for r in rows(meta):
        s = slugify(r["소장품번호"])
        if r["class"] != "plain" or s in seen:
            continue
        seen.add(s)
        plain.append({"slug": s, "h": fnum(r["height_cm"]),
                      "m": fnum(r["mouth_cm"]), "broken": r["is_broken"] == "True"})

    used = {r["source"] for r in rows(WORK / "pairs_manifest.csv")}
    ours = [p for p in plain if p["slug"] in used]
    miss = [p for p in plain if p["m"] is None]
    other = [p for p in plain if p["slug"] not in used and p["m"] is not None]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter([p["h"] for p in other], [p["m"] for p in other], s=44, c=GRAY,
               label="제외한 plain (%d점)" % len(other), zorder=2)
    ax.scatter([p["h"] for p in ours], [p["m"] for p in ours], s=80, c=BLUE,
               label="완형 %d점 — 이걸 쓴다" % len(ours), zorder=3)
    for p in ours:
        ax.annotate(p["slug"], (p["h"], p["m"]), fontsize=8.5,
                    xytext=(5, 4), textcoords="offset points", color=DARK)
    # 입지름이 없는 7점은 y 값이 아예 없다. 아래쪽 별도 띠에 늘어놓는다.
    strip = 6.3
    ax.axhline(7.4, color="#ddd", lw=1, ls="--", zorder=1)
    ax.scatter([p["h"] for p in miss], np.full(len(miss), strip), s=60, c=RED,
               marker="v", label="입지름 없음 = 아가리 결손 (%d점)" % len(miss), zorder=3)
    ax.text(6.1, strip + 0.42, "↓ 입지름을 박물관도 못 잰 것 = 아가리 결손",
            fontsize=9, color=RED, va="bottom")
    for p in miss:
        if "71489" in p["slug"]:
            ax.annotate("경신 71489\n(복원 대상)", (p["h"], strip), fontsize=9.5,
                        xytext=(0, -30), textcoords="offset points", color=RED,
                        ha="center", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    ax.set_xlabel("높이 (cm)")
    ax.set_ylabel("입지름 (cm)")
    ax.set_title("굽다리바리 plain 27점 — 완형 9점이 어디 있나", fontsize=13, pad=10)
    ax.set_ylim(4.8, 21.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.grid(alpha=0.25)
    nospine(ax)
    fig.tight_layout()
    fig.savefig(FIG / "데이터-완형-크기분포.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-완형-크기분포.png")


# ------------------------------------------------------------ 5. 실물 사진

def _meta_rows():
    """artifacts.csv 를 slug 기준으로. `file` 이 images/ 의 사진 이름이다."""
    meta = ROOT / "gupdari-shape-prior" / "meta" / "artifacts.csv"
    seen, out = set(), []
    for r in rows(meta):
        s = slugify(r["소장품번호"])
        if s in seen:
            continue
        seen.add(s)
        out.append({"slug": s, "file": r["file"], "cls": r["class"],
                    "h": fnum(r["height_cm"]), "m": fnum(r["mouth_cm"]),
                    "broken": r["is_broken"] == "True"})
    return out


def _thumb(ax, img_path: Path, title: str, sub: str, color: str):
    from PIL import Image
    if not img_path.is_file():
        ax.text(0.5, 0.5, "사진 없음", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return
    im = Image.open(img_path)
    im.thumbnail((420, 420))
    ax.imshow(im)
    ax.set_title(title, fontsize=10, color=color, fontweight="bold", pad=3)
    ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#666")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(1.6)


def fig_photos_used():
    """실제로 쓰는 완형의 사진. v2 는 9점, v3 는 16점."""
    used = {r["source"] for r in rows(WORK / "pairs_manifest.csv")}
    items = [m for m in _meta_rows() if m["slug"] in used]
    items.sort(key=lambda m: m["h"] or 0)
    IMG = ROOT / "gupdari-shape-prior" / "images"

    # v3 에서 9점 → 16점으로 늘렸다. 칸 수는 실제 점수에서 정한다.
    import math
    nc = 3 if len(items) <= 9 else 4
    nr = math.ceil(len(items) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.13 * nc, 3.4 * nr))
    for ax, m in zip(np.atleast_1d(axes).ravel(), items):
        # v3 에 넣은 7점 중 일부는 **입지름이 없다** — 아가리가 깨져 있어서다.
        # 완형 판정은 `(h,θ)` 격자 결손률로 따로 했다 (결과.md 5.10).
        cap = ("높이 %.1fcm · 입지름 %.1fcm" % (m["h"], m["m"])) if m["m"]               else ("높이 %.1fcm · 입지름 없음" % m["h"] if m["h"] else "치수 없음")
        _thumb(ax, IMG / m["file"], m["slug"].replace("_", " "), cap, BLUE)
    for ax in np.atleast_1d(axes).ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle("쓰는 데이터 — 완형 굽다리바리 %d점 (e뮤지엄)" % len(items),
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG / "데이터-완형-사진.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-완형-사진.png  (%d점)" % len(items))


def fig_photos_excluded():
    """입지름이 없어 제외한 7점 — 전부 아가리가 깨져 있다. 71489 포함."""
    items = [m for m in _meta_rows() if m["cls"] == "plain" and m["m"] is None]
    items.sort(key=lambda m: m["h"] or 0)
    IMG = ROOT / "gupdari-shape-prior" / "images"

    fig, axes = plt.subplots(1, len(items), figsize=(2.05 * len(items), 3.4))
    for ax, m in zip(np.atleast_1d(axes), items):
        tgt = "71489" in m["slug"]
        _thumb(ax, IMG / m["file"], m["slug"].replace("_", " "),
               "복원 대상" if tgt else "높이 %.1fcm" % m["h"],
               "#c0392b" if tgt else RED)
    fig.suptitle("제외 — 입지름이 기록돼 있지 않다 = 아가리가 깨졌다 (plain 7점)",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "데이터-제외-사진.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  데이터-제외-사진.png  (%d점)" % len(items))


# ------------------------------------------------------------ 6. 모델 구조

def fig_architecture():
    """AdaPoinTr 추론 경로. 결손부 점이 나오는 자리를 표시한다."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    boxes = [
        ("부분 점군\n(2048, 3)", "#eef2f7", DARK),
        ("① FPS + DGCNN\n포인트 프록시", "#eef2f7", DARK),
        ("② geometry-aware\n트랜스포머 인코더", "#eef2f7", DARK),
        ("③ 쿼리 생성\n$Q_I \\cup Q_O$ + scoring", "#ffe9e9", "#c0392b"),
        ("④ 디코더\n결손 프록시", "#eef2f7", DARK),
        ("⑤ FoldingNet 헤드\n$P_i = f(H_i) + c_i$", "#ffe9e9", "#c0392b"),
        ("완성 점군\n(8192, 3)", "#eef2f7", DARK),
    ]
    fig, ax = plt.subplots(figsize=(15.2, 4.6))
    w, h, gap = 1.72, 0.94, 0.42
    for i, (txt, fc, ec) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.045",
                                    fc=fc, ec=ec, lw=1.9))
        ax.text(x + w / 2, h / 2, txt, ha="center", va="center", fontsize=10.2,
                color=ec, linespacing=1.5)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.04, h / 2), (x + w + gap - 0.04, h / 2),
                arrowstyle="-|>", mutation_scale=15, lw=1.5, color="#999"))

    x3 = 3 * (w + gap)
    x5 = 5 * (w + gap)
    ax.annotate("★ AdaPoinTr 가 바꾼 곳 ①\nscoring 모듈이 top-M 을 적응 선택\n"
                "→ 입력을 완형으로 읽으면\n   결손 쿼리를 안 뽑을 수 있다",
                (x3 + w / 2, h + 0.06), ha="center", va="bottom",
                fontsize=9.6, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5", ec="#f0c0c0"))
    ax.annotate("결손부에 점이 나오는 자리는\n$Q_O$ 의 중심좌표 $c_i$ 하나뿐이다.\n"
                "거기 중심이 안 찍히면 점이 생길 자리가 없다\n— 08-28 의 '출력점 0점'",
                (x5 + w / 2, -0.10), ha="center", va="top",
                fontsize=9.6, color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5", ec="#f0c0c0"))
    ax.annotate("★ 바꾼 곳 ② denoising 보조 과제 — 학습 전용이라 추론 경로에 없다 (학습 15배 가속)",
                (x3 * 0.42, -0.92), ha="left", va="center", fontsize=9.6, color="#777")

    ax.set_xlim(-0.35, len(boxes) * (w + gap) + 0.1)
    ax.set_ylim(-1.25, h + 1.15)
    ax.axis("off")
    ax.set_title("AdaPoinTr 추론 경로 — 점은 프록시 중심 주변에만 생긴다",
                 fontsize=13.5, pad=14)
    fig.tight_layout()
    fig.savefig(FIG / "모델-구조.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  모델-구조.png")


if __name__ == "__main__":
    print("그림 →", FIG)
    fig_selection()
    fig_patterns()
    fig_aihub()
    fig_sizes()
    fig_photos_used()
    fig_photos_excluded()
    fig_architecture()
