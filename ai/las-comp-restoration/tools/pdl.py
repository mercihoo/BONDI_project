#!/usr/bin/env python3
"""병렬 범위 요청 다운로더 — 이 네트워크는 연결 하나가 ~400 KB/s 로 묶여 있어(HF CDN) 3.8 GB 에 2.5 시간이 걸린다.
연결 16 개로 ~1.1 MB/s, 그 이상은 완만하게 는다. huggingface_hub 의 snapshot_download 는 6 분에 3 MB 를 받고 멈춰 있었다.

    python tools/pdl.py <URL> <출력 파일> [연결 수=32]

파일을 미리 크기대로 만들고, 조각을 스레드가 각자 pwrite 로 써 넣는다. 조각은 5 번까지 재시도. 다 받으면 크기를 확인하고 .done 을 남긴다.
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

url, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
CHUNK = 8 << 20                 # 8 MiB 조각 — 연결이 끊겨도 잃는 양이 작게

if os.path.exists(out + ".done"):
    print("이미 있음:", out); sys.exit(0)

head = requests.head(url, allow_redirects=True, timeout=30)
head.raise_for_status()
size = int(head.headers["Content-Length"])
final_url = head.url
print("%s  %.1f MiB  연결 %d" % (os.path.basename(out), size / 2**20, N), flush=True)

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, size)
done = 0
lock = threading.Lock()
t0 = time.time()


def fetch(start):
    end = min(start + CHUNK, size) - 1
    for attempt in range(5):
        try:
            r = requests.get(final_url, headers={"Range": "bytes=%d-%d" % (start, end)}, timeout=60, stream=True)
            r.raise_for_status()
            buf = r.content
            if len(buf) != end - start + 1:
                raise IOError("짧은 조각 %d != %d" % (len(buf), end - start + 1))
            os.pwrite(fd, buf, start)
            global done
            with lock:
                done += len(buf)
            return
        except Exception as e:
            time.sleep(2 * (attempt + 1))
            last = e
    raise SystemExit("조각 %d 실패: %s" % (start, last))


def report():
    while done < size:
        time.sleep(15)
        el = time.time() - t0
        rate = done / el if el else 0
        eta = (size - done) / rate if rate else float("inf")
        print("  %5.1f%%  %6.1f MiB  %.2f MB/s  남은 %.0f 분" % (100 * done / size, done / 2**20, rate / 1e6, eta / 60), flush=True)


threading.Thread(target=report, daemon=True).start()
with ThreadPoolExecutor(N) as ex:
    list(ex.map(fetch, range(0, size, CHUNK)))
os.close(fd)
assert os.path.getsize(out) == size
open(out + ".done", "w").write(str(size))
print("완료 %s  %.0f s  평균 %.2f MB/s" % (out, time.time() - t0, size / (time.time() - t0) / 1e6), flush=True)
