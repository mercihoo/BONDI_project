# -*- coding: utf-8 -*-
"""
train_adapointr.py — [3] AdaPoinTr 를 굽다리바리로 파인튜닝한다.

가설
    [1][2] 가 갈라낸 실패 유형은 **둘레 전체가 제각각이라 기준 테두리가 없는 결손**이다.
    좁은 기종(굽다리바리) 안에서는 높이·입지름 비율이 규칙적이므로
    *"굽다리에 비해 바리가 이렇게 얕은 고배는 없다"* 를 **배울 수 있지 않을까.**

    그래서 학습 데이터에 `ragged`(둘레 전체 불규칙 침식) 패턴을 넣었다.
    그게 없으면 가르치려는 것을 안 가르치는 셈이다.

베이스라인 (사전학습 그대로)
    71489        보충점 20 / 8192
    EA_028       결실 하부 출력점 0
    합성 `bottom` 커버리지 순증 0.785 · `rim` 0.961

CUDA 확장 없이 학습하기
    `get_loss` 가 `self.loss_func`(= extensions.chamfer_dist.ChamferDistanceL1)를 쓴다.
    추론에선 안 불려서 더미로 막아뒀지만 **학습에는 실제로 필요하다.**
    그래서 순수 PyTorch 챔퍼로 갈아끼운다. 메모리 때문에 질의를 쪼개서 계산한다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY train_adapointr.py --epochs 30
  $PY train_adapointr.py --epochs 30 --freeze-encoder     # 표본이 적을 때
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import run_adapointr as R

HERE = Path(__file__).resolve().parent
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PAIRS = WORK / "pairs"
CKPT_OUT = HERE / "ckpts" / "AdaPoinTr_gupdari_selfnorm.pth"

# 표본이 9점뿐이라 2점을 검증으로 뺀다. **형상 단위로** 나눠야 의미가 있다
VAL_SOURCES = ("경주_887", "대구대_1834")


# ------------------------------------------------------------ 손실

def chamfer_l1(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """양방향 평균 최근접 거리 (CD-L1).

    `torch.cdist` 를 8192x8192 로 한 번에 만들면 268MB(fp32)에 기울기까지 붙어
    8GB 에서 터진다. 질의를 chunk 로 쪼개 최소값만 남긴다.
    """
    def one_way(x, y):
        outs = []
        for i in range(0, x.shape[1], chunk):
            d = torch.cdist(x[:, i:i + chunk], y)        # B, c, M
            outs.append(d.min(dim=2).values)
        return torch.cat(outs, dim=1).mean()
    return (one_way(a, b) + one_way(b, a)) * 0.5


class ChamferL1(nn.Module):
    def forward(self, a, b):
        return chamfer_l1(a, b)


# ------------------------------------------------------------ 선택 지표

@torch.no_grad()
def coverage_gain(pred, comp, removed, part, k: float = 3.0) -> float:
    """**결손부 커버리지 순증** — `eval_completion.py` 와 같은 정의의 토치판.

    왜 이걸로 고르나
      처음엔 val CD(전체 형상 챔퍼)로 체크포인트를 골랐다. **틀린 지표였다** —
      CD 는 입력을 그대로 복사해도 꽤 좋게 나온다. 우리가 원하는 것은
      *"결손부를 메웠나"* 이고 그건 CD 가 거의 안 본다.
      실제로 지난 학습에서 val CD 는 ep8 이후 정체했는데, 결손부 성능이
      거기서 최선이었는지는 확인할 방법이 없었다 (ep8 만 저장돼 있었다).

    바닥값(floor)을 빼서 0(안 메움)~1(완벽)로 맞춘다.
    """
    outs = []
    for b in range(pred.shape[0]):
        gt = comp[b][removed[b]]
        if gt.numel() == 0:            # none 패턴 — 결손이 없어 잴 것이 없다
            continue
        p = part[b]
        dpp = torch.cdist(p[None], p[None])[0]
        dpp.fill_diagonal_(float("inf"))
        delta = k * dpp.min(1).values.median()
        cover = (torch.cdist(gt[None], pred[b][None])[0].min(1).values <= delta).float().mean()
        floor = (torch.cdist(gt[None], p[None])[0].min(1).values <= delta).float().mean()
        outs.append(float((cover - floor) / (1.0 - floor).clamp(min=1e-9)))
    return float(np.mean(outs)) if outs else float("nan")


# ------------------------------------------------------------ 데이터

class PairSet(torch.utils.data.Dataset):
    """**자기 정규화**로 낸다.

    `make_pairs` 는 partial 을 complete 의 중심·축척으로 정규화해 저장한다.
    그러면 잘려나간 쪽에 빈 공간이 남아 **모델이 원래 크기를 액자에서 읽는다.**
    실제 유물에는 complete 가 없으니 그 단서가 없다 — 실측으로 확인했다:

        같은 쌍, 같은 가중치        complete 정규화 | 자기 정규화
        깎임 0.10~0.20 커버리지 순증      0.700   |   0.102
        깎임 0.40~0.50                   0.521   |   0.019

    그래서 partial 자신의 중심·축척으로 둘 다 옮긴다. 정답도 같은 자로 옮겨야
    손실이 말이 된다. 이러면 **모델이 입력의 경계 밖을 예측해야** 하는,
    실제와 같은 과제가 된다.
    """

    def __init__(self, files, jitter=0.0, self_norm=True):
        self.files = list(files)
        self.jitter = jitter
        self.self_norm = self_norm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        p = np.asarray(d["partial"], np.float64)
        c = np.asarray(d["complete"], np.float64)
        if self.self_norm:
            cp = p.mean(0)
            sp = float(np.linalg.norm(p - cp, axis=1).max()) or 1.0
            p, c = (p - cp) / sp, (c - cp) / sp
        rm = np.asarray(d["complete_removed"], bool)
        p, c = p.astype(np.float32), c.astype(np.float32)
        if self.jitter:
            # 약한 표본 잡음. 형상 다양성은 9개 그대로라 과적합 완화용일 뿐이다
            p = p + np.random.normal(0, self.jitter, p.shape).astype(np.float32)
        return torch.from_numpy(p), torch.from_numpy(c), torch.from_numpy(rm)


def split_files():
    files = sorted(PAIRS.glob("*.npz"))
    if not files:
        raise SystemExit("[!] 쌍이 없다. make_pairs.py --per-pattern 20 을 먼저.")
    tr = [f for f in files if f.stem.split("__")[0] not in VAL_SOURCES]
    va = [f for f in files if f.stem.split("__")[0] in VAL_SOURCES]
    return tr, va


# ------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8, help="기울기 누적 — 실효 배치")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="파인튜닝이라 config(1e-4)보다 낮춘다")
    ap.add_argument("--jitter", type=float, default=0.002)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--leak-norm", action="store_true",
                    help="complete 기준 정규화로 되돌린다 (누출 확인용)")
    ap.add_argument("--patience", type=int, default=6, help="조기 종료")
    ap.add_argument("--out", default=str(CKPT_OUT))
    args = ap.parse_args()

    R.install_shims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("디바이스:", device)
    model = R.build(device)

    # 더미로 막아둔 챔퍼를 진짜로 갈아끼운다 (모듈 docstring 참조)
    model.loss_func = ChamferL1().to(device)
    print("손실 함수 교체: 순수 PyTorch CD-L1")
    print("정규화: %s" % ("complete 기준 (누출)" if args.leak_norm else "**자기 정규화**"))

    if args.freeze_encoder:
        n = 0
        for name, p in model.named_parameters():
            if "base_model" in name and "decoder" not in name:
                p.requires_grad = False
                n += 1
        print("인코더 동결: 텐서 %d개" % n)

    tr_f, va_f = split_files()
    print("학습 %d쌍 (형상 %d) · 검증 %d쌍 (형상 %s)"
          % (len(tr_f), 9 - len(VAL_SOURCES), len(va_f), ", ".join(VAL_SOURCES)))

    tr = torch.utils.data.DataLoader(PairSet(tr_f, args.jitter, not args.leak_norm), batch_size=args.bs,
                                     shuffle=True, num_workers=0, drop_last=True)
    va = torch.utils.data.DataLoader(PairSet(va_f, 0.0, not args.leak_norm), batch_size=args.bs, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    hist, best, since = [], -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (p, c, _rm) in enumerate(tr):
            p, c = p.to(device), c.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ret = model(p)
                l_dn, l_rc = model.get_loss(ret, c)
                loss = (l_dn + l_rc) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += float(l_rc); nb += 1
        sched.step()

        model.eval()
        vt, vn, gains = 0.0, 0, []
        with torch.no_grad():
            for p, c, rm in va:
                p, c, rm = p.to(device), c.to(device), rm.to(device)
                out = model(p)
                fine = out[-1] if isinstance(out, (tuple, list)) else out
                vt += float(chamfer_l1(fine, c)); vn += 1
                g = coverage_gain(fine, c, rm, p)
                if np.isfinite(g):
                    gains.append(g)
        vloss = vt / max(vn, 1)
        vgain = float(np.mean(gains)) if gains else float("nan")
        hist.append({"epoch": ep, "train_recon": tot / max(nb, 1),
                     "val_cd": vloss, "val_gain": vgain})
        print("ep %3d/%d  train %.5f  val CD %.5f  **순증 %.4f**  %5.1fs"
              % (ep, args.epochs, tot / max(nb, 1), vloss, vgain, time.time() - t0),
              flush=True)

        # **결손부 커버리지 순증으로 고른다** (val CD 가 아니라)
        if np.isfinite(vgain) and vgain > best:
            best, since = vgain, 0
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"base_model": model.state_dict(), "epoch": ep,
                        "val_gain": vgain, "val_cd": vloss}, args.out)
            print("      저장 (순증 최고)")
        else:
            since += 1
            if since >= args.patience:
                print("      조기 종료 — %d에폭 동안 순증이 안 올랐다" % args.patience)
                break

    (WORK / "train_history.json").write_text(
        json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n최저 val CD %.5f → %s" % (best, args.out))
    print("다음:  $PY run_adapointr.py --ckpt %s  후  eval_completion.py" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
