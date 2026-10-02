# -*- coding: utf-8 -*-
"""
run_damaged_batch.py — **같은 기종의 다른 파손품**에서도 되는지 본다.

왜
  지금까지 실제 유물은 71489 하나뿐이었다. 그래서 *"이 유물에서만 된 것 아닌가"* 와
  *"기종 안에서는 확장되나"* 를 구분할 수 없었다.

  `trellis3d/` 에는 굽다리바리 43점이 있고 그중 **완형 9점만 학습에 썼다.**
  나머지 34점은 파손품이고 **한 번도 안 넣어봤다** (71489 제외하면 33점).
  전부 사진에서 TRELLIS 로 만든 것이라 **71489 와 상류 경로가 같다.**

무엇을 재나
  정답(완형)이 없다. `run_71489.run_one()` 과 같은 잣대를 쓴다.

      보충점        입력 표면에서 τ 보다 멀리 떨어진 출력점 = 모델이 새로 만든 것
      아가리 위     알려진 결손 방향으로 연장했나
      높이 중앙     보충점이 위쪽에 몰렸나

  **대조군이 핵심이다.** 같은 유물에 사전학습과 v2 를 둘 다 돌려 차이를 본다.
  파인튜닝이 기종 안에서 일반화하면 **모든 유물에서** 보충점이 늘어야 한다.
  71489 에서만 늘면 그건 우연이다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY run_damaged_batch.py --limit 4                    # 맛보기
  $PY run_damaged_batch.py --runs 8                     # 전부, TTA 8회
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import run_adapointr as R
import run_71489 as T

HERE = Path(__file__).resolve().parent
# `gupdari-shape-prior` · `gupdari-71489` 는 **이 폴더 밖**에 있다 (원본 사진·완형 3D·v32 출력).
# 저장소 안에서 돌릴 때는 `RESTORE_ROOT` 로 그 폴더들이 모인 곳을 가리킨다.
# 안 걸면 예전처럼 이 스크립트의 부모 폴더를 본다.
ROOT = Path(__import__("os").environ.get("RESTORE_ROOT", HERE.parent))
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PRIOR = ROOT / "gupdari-shape-prior"
GLB = PRIOR / "trellis3d"
OUT = WORK / "damaged_batch"


def damaged_slugs(exclude_target: bool) -> list[str]:
    """완형(학습에 쓴 것)을 뺀 나머지 = 파손품."""
    from make_pairs import complete_slugs
    keep, _ = complete_slugs(PRIOR / "meta" / "artifacts.csv")
    have = {p.stem for p in GLB.glob("*.glb")}
    out = sorted(have - set(keep))
    if exclude_target:
        out = [s for s in out if s != "경신_71489"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpts/AdaPoinTr_gupdari_v2.pth")
    ap.add_argument("--runs", type=int, default=8, help="축 둘레 회전 TTA")
    ap.add_argument("--n-input", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-target", action="store_true",
                    help="71489 도 넣는다 (기준점으로 쓸 때)")
    ap.add_argument("--only-ft", action="store_true", help="사전학습 대조군을 건너뛴다")
    ap.add_argument("--only", default="", metavar="SLUG",
                    help="""유물 하나만 돌린다 (쉼표로 여럿).

    전수 배치는 27점 x 2모델 x TTA8 이라 오래 걸린다. 한 점만 다시 볼 때 쓴다.
    예) --only 경주_583 --ckpt ckpts/AdaPoinTr_gupdari_v3.pth --only-ft""")
    args = ap.parse_args()

    slugs = damaged_slugs(not args.include_target)
    if args.only:
        want = [x.strip() for x in args.only.split(",") if x.strip()]
        miss = [w for w in want if w not in slugs]
        if miss:
            print("[!] 파손품 목록에 없다: %s" % ", ".join(miss))
            print("    (완형은 학습에 썼으므로 여기 안 나온다)")
            return 1
        slugs = want
    if args.limit:
        slugs = slugs[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    print("파손품 %d점 · TTA %d회\n" % (len(slugs), args.runs))

    R.install_shims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **npz 는 유물별 폴더에 쓴다.** run_one 은 WORK 바로 밑에 "pred_71489_<tag>.npz"
    # 로 저장하므로 tag 에 유물명을 넣어 서로 안 덮게 한다.
    # 태그는 **가중치 파일 이름에서** 딴다. 전에는 "v2" 로 박혀 있어서
    # v3 로 돌려도 파일이 `__v2.npz` 로 저장돼 앞 결과를 덮었다.
    ftname = Path(args.ckpt).stem.split("_")[-1] if args.ckpt else "ft"
    models = [(ftname, args.ckpt)] if args.only_ft else [("base", None), (ftname, args.ckpt)]
    rows: dict[str, dict] = {}
    for mname, ck in models:
        print("=" * 60)
        print("모델: %s" % (ck or "ShapeNet-55 사전학습"))
        model = R.build(device, ck)
        for i, s in enumerate(slugs, 1):
            p = GLB / (s + ".glb")
            try:
                r = T.run_one(model, device, p, "%s__%s" % (s, mname),
                              args.n_input, args.runs)
            except Exception as ex:                     # 한 점이 죽어도 계속
                print("  [실패] %s — %s" % (s, ex))
                continue
            rows.setdefault(s, {"slug": s})
            rows[s]["%s_new" % mname] = r["n_new"]
            rows[s]["%s_above" % mname] = r["n_above"]
            rows[s]["%s_hn" % mname] = round(r["new_hn_median"], 3) \
                if np.isfinite(r["new_hn_median"]) else ""
            print("    (%d/%d)" % (i, len(slugs)), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("[!] 아무것도 못 돌렸다.")
        return 1
    # --- **덮어쓰지 말고 합친다.**
    #
    # 전에는 무조건 `open("w")` 였다. 그래서 `--only` 로 한 점만 돌리면
    # 27점짜리 표가 1행으로 날아갔다 (실제로 겪었다. 로그에서 되살렸다).
    # 이미 있는 행·열은 살리고, 이번에 돈 것만 덮어쓴다.
    out_csv = WORK / "damaged_batch.csv"
    old: dict[str, dict] = {}
    old_cols: list[str] = []
    if out_csv.is_file():
        with out_csv.open(encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            old_cols = [c for c in (rd.fieldnames or []) if c != "slug"]
            for r in rd:
                old[r["slug"]] = r
    new_cols = [c for m, _ in models for c in
                ("%s_new" % m, "%s_above" % m, "%s_hn" % m)]
    cols = ["slug"] + old_cols + [c for c in new_cols if c not in old_cols]
    merged = dict(old)
    for s, r in rows.items():
        merged.setdefault(s, {"slug": s}).update(r)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in sorted(merged):
            row = {c: "" for c in cols}
            row.update({k: v for k, v in merged[s].items() if k in cols})
            row["slug"] = s
            w.writerow(row)

    print("\n" + "=" * 60)
    # 열 이름은 가중치에서 딴 태그를 따른다. "v2" 로 박아 두면 v3 를 못 읽는다 —
    # 실제로 그래서 v3 배치 결과가 v2 라는 이름으로 저장됐다 (결과.md §5.13).
    print("%-16s %10s %10s   %s" % ("유물", "사전학습", ftname, "배수"))
    ratios = []
    for s in slugs:
        r = rows.get(s)
        if not r:
            continue
        b, v = r.get("base_new"), r.get("%s_new" % ftname)
        if v is None:
            continue
        if b is None:
            print("%-16s %10s %10d" % (s, "-", v))
            continue
        k = (v / b) if b else float("inf")
        if np.isfinite(k):
            ratios.append(k)
        print("%-16s %10d %10d   %s" % (s, b, v, "%.1f배" % k if np.isfinite(k) else "-"))
    if ratios:
        print("\n보충점 배수 중앙 %.1f배 · %d/%d 점에서 늘었다"
              % (np.median(ratios), sum(1 for k in ratios if k > 1.2), len(ratios)))
    print("\n표 → " + str(out_csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
