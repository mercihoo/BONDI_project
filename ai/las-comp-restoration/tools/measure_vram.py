#!/usr/bin/env python3
"""LaS-Comp 실행의 VRAM 피크를 잰다.

왜 두 가지로 재나:
  - torch 의 max_memory_allocated 는 **torch 캐싱 할당자만** 센다. spconv·xformers 가
    자체로 잡는 메모리, CUDA 컨텍스트, cuDNN 워크스페이스는 안 잡힌다.
  - nvidia-smi 의 프로세스 사용량이 실제로 8 GB 에 들어가는지를 말해 준다. 이쪽이 진짜 기준.

원본 실행 스크립트를 복제하지 않고 runpy 로 그대로 돌린다 — 저자 코드가 바뀌어도 계측이 따라간다.

  python measure_vram.py --label baseline -- run_lascomp_image_condition_single.py --partial-path ... --image-path ...
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _nvsmi(args):
    try:
        return subprocess.run(["nvidia-smi"] + args, capture_output=True,
                              text=True, timeout=5).stdout
    except Exception:
        return ""


def nvsmi_used_mib(pid):
    """GPU 에서 쓰이는 MiB.

    먼저 이 프로세스 몫을 물어보고, 안 되면 GPU 전체 사용량으로 떨어진다.
    **WSL 에서는 프로세스별 사용량이 `[N/A]` 로 나온다** — 실측으로 확인했다.
    이 장비는 화면이 내장 그래픽에 물려 있어 4070 을 쓰는 게 우리뿐이라,
    전체 사용량이 곧 우리 사용량이다.
    """
    out = _nvsmi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                break          # [N/A] — 아래 전체 사용량으로 간다

    out = _nvsmi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return None


class Sampler(threading.Thread):
    """nvidia-smi 를 주기적으로 찔러 프로세스 피크를 기록한다."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = os.getpid()
        self.peak = 0
        self.samples = []
        # 이름을 _stop 으로 두면 threading.Thread 의 내부 메서드 _stop() 을 덮어써서
        # join() 안에서 'Event' object is not callable 로 터진다. 실제로 당했다.
        self._stop_evt = threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            v = nvsmi_used_mib(self.pid)
            if v is not None:
                self.peak = max(self.peak, v)
                self.samples.append((time.time(), v))
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="결과 파일 이름에 붙는 꼬리표")
    ap.add_argument("--out", default="vram_reports", help="결과 JSON 을 둘 폴더")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- 뒤에 실행할 스크립트와 인자")
    args = ap.parse_args()

    rest = args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        ap.error("-- 뒤에 실행할 스크립트를 적어야 한다")

    script, script_argv = rest[0], rest[1:]

    import torch  # 계측 대상과 같은 프로세스에서 써야 한다
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()

    # runpy 는 `python 스크립트.py` 와 달리 **스크립트 폴더를 sys.path 에 안 넣는다.**
    # 저장소 루트를 직접 넣어 줘야 `import trellis` 가 된다.
    repo_root = str(Path(script).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # LASCOMP_QUANT 가 켜져 있으면 DiT 만 양자화하는 훅을, LASCOMP_OFFLOAD=1 이면 디코딩 전
    # 오프로드 훅을 건다 (둘 다 저자 스크립트는 안 고친다).
    quant_mode = os.environ.get("LASCOMP_QUANT", "off").strip().lower()
    offload_on = os.environ.get("LASCOMP_OFFLOAD", "").strip() in ("1", "true", "yes")
    meshcpu_on = os.environ.get("LASCOMP_MESH_CPU", "").strip() in ("1", "true", "yes")
    if quant_mode not in ("off", "") or offload_on or meshcpu_on:
        import quantize_trellis
        if quant_mode not in ("off", ""):
            quantize_trellis.install_hook()
        if meshcpu_on:
            quantize_trellis.install_mesh_cpu_hook()      # 양자화 훅 뒤에 걸어 그것을 감싼다
        if offload_on:
            quantize_trellis.install_offload_hook()
        quantize_trellis.install_decode_lean_hook()    # 스스로 LASCOMP_DECODE_LEAN 을 본다
        quantize_trellis.install_color_hook()          # LASCOMP_COLOR_OUT 이 있으면 예측 색을 저장
        quantize_trellis.install_ers_empty_hook()      # LASCOMP_KEEP_EMPTY 면 투창 자리를 빈 칸으로 고정

    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory / 1024 ** 2

    sampler = Sampler()
    sampler.start()
    t0 = time.time()
    status = "ok"
    err = None

    sys.argv = [script] + script_argv
    try:
        import runpy
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            status, err = "exit%s" % e.code, None
    except Exception as e:  # 실패해도 그때까지의 피크는 남긴다 — OOM 지점을 알아야 한다
        import traceback
        status, err = "error", "%s: %s" % (type(e).__name__, e)
        print("\n----- 대상 스크립트에서 터진 예외 -----", file=sys.stderr)
        traceback.print_exc()
        print("---------------------------------------\n", file=sys.stderr)
    finally:
        elapsed = time.time() - t0
        sampler.stop()

    # "CUDA driver error: out of memory" 뒤에는 컨텍스트가 죽어 torch.cuda.* 조회도 실패한다.
    # 그래도 nvidia-smi 로 잰 피크는 남아 있으니 보고서는 반드시 만든다.
    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    report = {
        "label": args.label,
        "status": status,
        "error": err,
        "elapsed_sec": round(elapsed, 1),
        "device": props.name,
        "total_vram_mib": round(total_mib),
        "peak_process_mib": sampler.peak or None,       # nvidia-smi 기준 — 진짜 피크
        "peak_torch_alloc_mib": _safe(lambda: round(torch.cuda.max_memory_allocated() / 1024 ** 2)),
        "peak_torch_reserved_mib": _safe(lambda: round(torch.cuda.max_memory_reserved() / 1024 ** 2)),
        "argv": sys.argv,
        "quant_mode": quant_mode,
        "offload": offload_on,
        "mesh_cpu": meshcpu_on,
    }
    try:
        import quantize_trellis
        if quantize_trellis.OFFLOAD_REPORT:
            report["offload_detail"] = quantize_trellis.OFFLOAD_REPORT
        if quantize_trellis.MESH_CPU_REPORT:
            report["mesh_cpu_detail"] = quantize_trellis.MESH_CPU_REPORT
        if quantize_trellis.DECODE_LEAN_REPORT:
            report["decode_lean_detail"] = quantize_trellis.DECODE_LEAN_REPORT
        if quantize_trellis.COLOR_REPORT:
            report["color_detail"] = quantize_trellis.COLOR_REPORT
        if quantize_trellis.KEEP_EMPTY_REPORT:
            report["keep_empty_detail"] = quantize_trellis.KEEP_EMPTY_REPORT
    except Exception:
        pass
    if quant_mode not in ("off", ""):
        try:
            import quantize_trellis
            report["quant_detail"] = quantize_trellis.LAST_REPORT
            report["quant_saved_mib"] = round(
                sum(c["saved_mib"] for c in quantize_trellis.LAST_REPORT), 1)
        except Exception:
            pass
    # nvidia-smi 는 0.25 초마다 찍어서 순간 급등을 놓친다 (실측: 프로세스 피크 6022 vs torch 예약 7054).
    # 8 GB 판정은 둘 중 큰 쪽으로 한다.
    cands = [v for v in (report["peak_process_mib"], report["peak_torch_reserved_mib"]) if v]
    report["effective_peak_mib"] = max(cands) if cands else None
    if report["effective_peak_mib"]:
        report["headroom_mib"] = round(total_mib - report["effective_peak_mib"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ("%s.json" % args.label)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 56)
    print("  VRAM 보고서 —", args.label)
    print("=" * 56)
    print("  상태            :", status, ("(%s)" % err) if err else "")
    print("  걸린 시간       : %.1f 초" % elapsed)
    print("  전체 VRAM       : %d MiB" % report["total_vram_mib"])
    print("  프로세스 피크   : %s MiB   ← 8 GB 판정 기준" %
          (report["peak_process_mib"] if report["peak_process_mib"] else "측정 실패"))
    print("  torch 할당 피크 : %s MiB" % (report["peak_torch_alloc_mib"] if report["peak_torch_alloc_mib"] is not None else "조회 불가(컨텍스트 죽음)"))
    print("  torch 예약 피크 : %s MiB" % (report["peak_torch_reserved_mib"] if report["peak_torch_reserved_mib"] is not None else "조회 불가"))
    if report.get("offload_detail"):
        od = report["offload_detail"]
        print("  디코딩 전 오프로드: %s → GPU 할당 %d → %d MiB" % (", ".join(od["moved_to_cpu"]), od["cuda_alloc_before_mib"], od["cuda_alloc_after_mib"]))
    if report.get("effective_peak_mib"):
        print("  판정용 피크     : %d MiB   (nvidia-smi · torch 예약 중 큰 쪽)" % report["effective_peak_mib"])
    if "headroom_mib" in report:
        print("  남은 여유       : %d MiB" % report["headroom_mib"])
    print("  저장            :", out_file)
    print("=" * 56)

    sys.exit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
