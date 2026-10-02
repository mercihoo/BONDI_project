# -*- coding: utf-8 -*-
"""
run_adapointr.py — 쌍 npz 를 AdaPoinTr 에 넣고 출력을 저장한다.

    work/pairs/*.npz  ──[partial 2048점]──▶ AdaPoinTr ──▶ work/pred/*.npy (8192점)

CUDA 확장 없이 돈다. 세 가지를 sys.modules 에 미리 꽂는다.

  1. pointnet2_ops        기여자 `restore_standalone.py` 의 순수 PyTorch 블록을 그대로 쓴다.
                          `misc.fps` 가 이걸 부른다
  2. extensions.chamfer_dist  AdaPoinTr.py 가 **모듈 최상위에서** import 한다.
                          손실 계산용이라 추론에선 안 불리므로 빈 클래스면 된다
  3. gridding · emd 등     다른 모델(GRNet 등)이 쓰는 것. models 패키지가 훑고 지나간다

`knn_cuda` 는 필요 없다 — 저장소가 이미 `knn_point`(topk) 로 대체해 두었다.

체크포인트 키 주의
  `ckpt['base_model']` 로 한 겹 들어가도 **그 안의 키가 또 `base_model.` 로 시작**한다.
  안 떼면 `strict=False` 때문에 전부 missing 이 되고 **난수 가중치로 추론**한다.
  그래서 로드 후 missing 개수를 찍고, 많으면 멈춘다.

사용
  PY=".../venv/Scripts/python.exe"
  $PY run_adapointr.py --limit 4       # 맛보기
  $PY run_adapointr.py                 # 쌍 전부
"""
from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
# 산출물(점군·메시·체크포인트)은 수 GB 라 저장소에 안 넣는다.
# `ADAPOINTR_WORK` 로 저장소 밖을 가리킬 수 있다. 없으면 이 폴더 밑 work/.
WORK = Path(__import__("os").environ.get("ADAPOINTR_WORK", HERE / "work"))
PAIRS = WORK / "pairs"
VENDOR = HERE / "vendor" / "PoinTr"
CKPT = HERE / "ckpts" / "AdaPoinTr_s55.pth"
SHIM_SRC = HERE / "restore_standalone.py"


# ------------------------------------------------------------ 확장 대체

def install_shims() -> None:
    """models 를 import 하기 **전에** 불러야 한다."""
    # 1. 기여자의 pointnet2_ops 대체를 파일에서 떼어 실행한다.
    #    그 파일은 맨 위에서 open3d·laspy 를 import 하는데 둘 다 venv 에 없다.
    #    우리 입력은 npz 라 LAS 경로가 필요 없으므로 그 줄만 빼고 shim 만 쓴다.
    #    (bitsal 에서 UNet 클래스를 AST 로 떼어 쓴 것과 같은 방식이다.)
    src = SHIM_SRC.read_text(encoding="utf-8")
    head = src[: src.index("def load_las")]
    head = re.sub(r"^import\s+(open3d as o3d|laspy)\s*$", "", head, flags=re.M)
    ns: dict = {"__name__": "pointr_shim"}
    exec(compile(head, str(SHIM_SRC), "exec"), ns)

    # 2. extensions.* — CUDA 확장. **추론 경로에서는 하나도 안 불린다.**
    #    AdaPoinTr.py 가 최상위에서 chamfer_dist 를 import 하고,
    #    models/__init__.py 가 GRNet 등을 훑으면서 gridding·emd 까지 끌고 온다.
    #    이름이 무엇이든 '부르면 터지는' 더미를 주는 모듈로 덮는다 —
    #    조용히 0 을 반환하면 틀린 결과가 조용히 나온다.
    def _boom(self, *a, **k):
        raise RuntimeError(
            "CUDA 확장 %s 가 실제로 호출됐다. 추론 경로가 아닌 곳을 돌리고 있다."
            % type(self).__name__)

    class _DummyExt(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (torch.nn.Module,), {"forward": _boom})
            setattr(self, name, cls)
            return cls

    ext = types.ModuleType("extensions")
    ext.__path__ = []                      # 패키지로 인식시킨다. 없으면 하위 import 실패
    sys.modules["extensions"] = ext
    for n in ("chamfer_dist", "cubic_feature_sampling", "emd",
              "gridding", "gridding_loss"):
        m = _DummyExt("extensions." + n)
        setattr(ext, n, m)
        sys.modules["extensions." + n] = m

    # 3. 그 밖에 최상위로 import 되는 것들
    for n in ("gridding", "gridding_distance", "cubic_feature_sampling",
              "emd", "chamfer"):
        sys.modules.setdefault(n, types.ModuleType(n))

    # 4. matplotlib — `utils/misc.py` 가 점군 그림 함수 때문에 최상위에서 부른다.
    #    TRELLIS venv 에 없다. **그 venv 는 본 파이프라인이 쓰는 것이라 건드리지 않는다**
    #    (`gupdari-71489/실행-구조.md` §5 — 별도 환경을 만들지 않는다).
    #    추론에 그림은 필요 없으므로 여기서만 덮는다.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        mpl = types.ModuleType("matplotlib")
        mpl.__path__ = []
        for name, attrs in (("matplotlib.pyplot", ()),
                            ("mpl_toolkits", ()),
                            ("mpl_toolkits.mplot3d", ("Axes3D",))):
            m = _DummyExt(name)
            sys.modules[name] = m
            for a in attrs:
                setattr(m, a, object)
        sys.modules["matplotlib"] = mpl
        sys.modules["mpl_toolkits"].__path__ = []


# ------------------------------------------------------------ 모델

def build(device, ckpt_path=None) -> torch.nn.Module:
    sys.path.insert(0, str(VENDOR))
    import yaml
    from easydict import EasyDict

    cfg_path = VENDOR / "cfgs" / "ShapeNet55_models" / "AdaPoinTr.yaml"
    cfg = EasyDict(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))

    from models.AdaPoinTr import AdaPoinTr
    model = AdaPoinTr(cfg.model).to(device)

    ck = torch.load(ckpt_path or CKPT, map_location="cpu", weights_only=False)
    raw = ck.get("base_model", ck.get("model", ck))
    # `ck['base_model']` 안의 키도 `base_model.` 로 시작하는데, **떼면 안 된다** —
    # AdaPoinTr 클래스가 `self.base_model` 을 갖고 있어 모델 키도 같은 접두어다.
    # 실제로 대조했다: 그대로 335/335 일치, 떼면 15/335.
    # 기여자 스크립트처럼 `module.` 만 벗긴다 (DataParallel 로 저장된 경우 대비).
    sd = {k.replace("module.", "", 1): v for k, v in raw.items()}

    inc = model.load_state_dict(sd, strict=False)
    n_miss, n_unexp = len(inc.missing_keys), len(inc.unexpected_keys)
    n_par = sum(1 for _ in model.state_dict())
    print("체크포인트: 텐서 %d · missing %d · unexpected %d (모델 텐서 %d)"
          % (len(sd), n_miss, n_unexp, n_par))
    if inc.missing_keys:
        print("  missing 예시:", inc.missing_keys[:4])
    if n_miss > n_par * 0.05:
        raise SystemExit(
            "[!] missing 이 너무 많다 (%d/%d). 접두어 처리가 틀렸다 — "
            "이대로 돌리면 난수 가중치로 추론한다." % (n_miss, n_par))

    model.eval()
    if ck.get("metrics"):
        print("  학습 시 기록된 metrics:", ck["metrics"])
    return model


def rot_about(P, up, ang):
    """회전축 둘레로 ang 만큼 돌린다. 회전체라 물체는 그대로다."""
    ax = [i for i in range(3) if i != up]
    c, s = np.cos(ang), np.sin(ang)
    Q = P.copy()
    Q[:, ax[0]] = P[:, ax[0]] * c - P[:, ax[1]] * s
    Q[:, ax[1]] = P[:, ax[0]] * s + P[:, ax[1]] * c
    return Q


@torch.no_grad()
def infer_tta(model, pts: np.ndarray, device, up: int, runs: int) -> np.ndarray:
    """**축 둘레 회전 TTA.** 돌려 넣고 되돌려 합친다.

    왜 되나 — 이 유물은 회전체다. 축 둘레로 돌리면 **물체는 그대로**인데
    모델은 회전 불변이 아니라 다른 답을 낸다. 그 차이가 모델 분산이고,
    여러 각도를 합치면 분산이 줄고 **점 밀도도 runs 배**가 된다.

    출력 8,192점이 학습 시점 고정이라 한 번 추론으로는 못 늘린다 —
    이 우회가 유일하게 싼 방법이다.
    """
    outs = []
    for k in range(runs):
        a = 2 * np.pi * k / runs
        q = rot_about(pts.astype(np.float64), up, a)
        o = infer(model, q.astype(np.float32), device)
        outs.append(rot_about(o.astype(np.float64), up, -a))   # 되돌린다
    return np.concatenate(outs).astype(np.float32)


@torch.no_grad()
def infer(model, pts: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(pts).float().unsqueeze(0).to(device)
    out = model(x)
    # eval 모드 반환은 (coarse, rebuild). 조밀한 쪽을 쓴다.
    if isinstance(out, (tuple, list)):
        out = max(out, key=lambda t: t.shape[-2])
    return out.squeeze(0).cpu().numpy().astype(np.float32)


# ------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIRS))
    ap.add_argument("--out", default=str(WORK / "pred"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--ckpt", default="", help="파인튜닝 가중치로 바꿔 돌릴 때")
    ap.add_argument("--runs", type=int, default=1,
                    help="축 둘레 회전 TTA 횟수. 출력이 8192×runs 점이 된다")
    ap.add_argument("--self-normalize", action="store_true",
                    help="""입력을 **자기 자신** 기준으로 정규화한다.

    make_pairs 는 partial 을 complete 의 중심·축척으로 정규화한다. 그러면 잘려나간
    쪽에 빈 공간이 남아 **모델이 원래 크기를 액자에서 읽을 수 있다.**
    실제 유물에는 complete 가 없으므로 그 정보가 없다 — 71489 를 넣을 때도
    입력 자신으로 정규화했다. 이 옵션이 합성 평가를 실제 조건과 같게 만든다.""")
    args = ap.parse_args()

    if not CKPT.is_file():
        print("[!] 가중치가 없다: " + str(CKPT))
        return 1

    install_shims()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("디바이스:", device)

    model = build(device, args.ckpt or None)

    files = sorted(Path(args.pairs).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n쌍 %d개 추론" % len(files))
    for i, f in enumerate(files, 1):
        d = np.load(f, allow_pickle=True)
        part = np.asarray(d["partial"], np.float64)
        up = int(d["up_axis"]) if "up_axis" in d else 1
        run = (lambda x: infer_tta(model, x, device, up, args.runs)) if args.runs > 1             else (lambda x: infer(model, x, device))
        if args.self_normalize:
            c_p = part.mean(0)
            s_p = float(np.linalg.norm(part - c_p, axis=1).max()) or 1.0
            pred = run(((part - c_p) / s_p).astype(np.float32))
            pred = pred.astype(np.float64) * s_p + c_p      # 원래 액자로 되돌린다
        else:
            pred = run(part.astype(np.float32))
        np.save(out_dir / (f.stem + ".npy"), pred.astype(np.float32))
        if i % 10 == 0 or i == len(files):
            print("  %d/%d  %s  출력 %d점" % (i, len(files), f.stem, len(pred)), flush=True)

    print("\n저장 → " + str(out_dir))
    print("다음:  $PY eval_completion.py --pred " + str(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
