#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ALVR 20.14.0의 session.json에 고운사 VR 시연용 설정을 적용한다.

측정 근거와 각 값의 이유는 docs/vr-streaming-setup.md 참고.
적용 전후: Client FPS 11 -> 71, Total latency 467ms -> 59ms, packet loss 0.

사용법:
    python apply_alvr_settings.py "C:/path/to/alvr_streamer_windows"
    python apply_alvr_settings.py "C:/path/to/alvr_streamer_windows" --verify

주의: ALVR 대시보드는 종료할 때 자기 메모리의 설정으로 session.json을 덮어쓴다.
반드시 대시보드를 완전히 끈 상태에서 실행하고, 재시작한 뒤 --verify로 재확인할 것.
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

# (session.json 안의 경로, 넣을 값, 왜)
SETTINGS = [
    (['session_settings', 'video', 'preferred_codec'],
     {'variant': 'Hevc'},
     'H.264 대비 같은 화질을 절반 비트레이트로 — 대역폭 병목 해소'),

    (['session_settings', 'video', 'bitrate', 'mode', 'variant'],
     'ConstantMbps',
     'Adaptive가 요동치며 프레임 큐를 쌓았음'),
    (['session_settings', 'video', 'bitrate', 'mode', 'ConstantMbps'],
     15,
     '15 Mbps 고정'),

    (['session_settings', 'video', 'transcoding_view_resolution', 'variant'],
     'Absolute', ''),
    (['session_settings', 'video', 'transcoding_view_resolution', 'Absolute', 'width'],
     1400,
     '원본 2752는 이 GPU의 인코더가 못 따라감'),
    (['session_settings', 'video', 'emulated_headset_view_resolution', 'variant'],
     'Absolute', ''),
    (['session_settings', 'video', 'emulated_headset_view_resolution', 'Absolute', 'width'],
     1400,
     'transcoding 쪽과 같은 값이어야 함'),

    (['session_settings', 'video', 'preferred_fps'],
     72.0,
     'Quest 2 기본 주사율 — 헤드셋에서 따로 바꿀 것 없음'),

    (['session_settings', 'video', 'encoder_config', 'entropy_coding'],
     {'variant': 'Cabac'},
     'CAVLC보다 압축률이 높음'),

    (['session_settings', 'video', 'foveated_encoding', 'enabled'],
     True, ''),
    (['session_settings', 'video', 'foveated_encoding', 'content', 'center_size_x'],
     0.4, ''),
    (['session_settings', 'video', 'foveated_encoding', 'content', 'center_size_y'],
     0.35,
     '시야 주변부 비트 절약'),

    (['session_settings', 'headset', 'position_recentering_mode'],
     {'Local': {'view_height': 1.6}, 'variant': 'Local'},
     'Local floor면 플레이어가 공중에 뜬다'),
    (['session_settings', 'headset', 'rotation_recentering_mode'],
     {'variant': 'Yaw'}, ''),

    (['session_settings', 'extra', 'logging', 'log_to_disk'],
     True,
     'session_log.txt로 프레임 간격을 사후 집계하기 위함'),
]

# 아직 검증하지 않은 개선 항목 (--with-buffering 으로만 적용)
EXPERIMENTAL = [
    (['session_settings', 'video', 'max_buffering_frames'],
     1.0,
     '측정된 Frame Buffering 21.9ms(전체 지연의 37%)를 줄이기 위함. '
     '화면이 튀면 1.5로 되돌릴 것'),
]


def dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def put(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    old = obj.get(path[-1])
    obj[path[-1]] = value
    return old


def dashboard_running():
    try:
        out = subprocess.run(['tasklist'], capture_output=True, text=True,
                             timeout=20).stdout.lower()
    except Exception:
        return False
    return 'alvr dashboard.exe' in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('alvr_dir', help='ALVR 스트리머 설치 폴더 (session.json이 있는 곳)')
    ap.add_argument('--verify', action='store_true', help='쓰지 않고 현재 값만 확인')
    ap.add_argument('--with-buffering', action='store_true',
                    help='미검증 항목(max_buffering_frames=1.0)까지 적용')
    ap.add_argument('--force', action='store_true',
                    help='대시보드가 켜져 있어도 강행 (권장하지 않음)')
    args = ap.parse_args()

    path = Path(args.alvr_dir) / 'session.json'
    if not path.is_file():
        sys.exit(f'session.json을 찾을 수 없음: {path}')

    session = json.loads(path.read_text(encoding='utf-8'))
    version = session.get('server_version', '?')
    if not str(version).startswith('20.'):
        print(f'경고: ALVR {version} — 이 스크립트는 20.14.0 기준으로 작성됨', file=sys.stderr)

    wanted = SETTINGS + (EXPERIMENTAL if args.with_buffering else [])

    if args.verify:
        print(f'ALVR {version}  {path}')
        bad = 0
        for keys, value, _ in wanted:
            try:
                cur = dig(session, keys)
            except (KeyError, TypeError):
                print(f'  [없음] {"/".join(keys)}')
                bad += 1
                continue
            match = cur == value
            bad += 0 if match else 1
            print(f'  [{"OK " if match else "다름"}] {"/".join(keys)} = '
                  f'{json.dumps(cur, ensure_ascii=False)}'
                  + ('' if match else f'  (기대: {json.dumps(value, ensure_ascii=False)})'))
        print('\n전부 일치' if bad == 0 else f'\n{bad}개 항목이 기대와 다름')
        return 0 if bad == 0 else 1

    if dashboard_running() and not args.force:
        sys.exit('ALVR 대시보드가 실행 중이다. 종료한 뒤 다시 실행할 것.\n'
                 '(대시보드는 종료할 때 자기 메모리의 설정으로 session.json을 덮어쓴다)')

    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    backup = path.with_name(f'session.json.bak-{stamp}')
    shutil.copy2(path, backup)
    print(f'백업: {backup.name}')

    changed = 0
    for keys, value, why in wanted:
        try:
            old = put(session, keys, value)
        except (KeyError, TypeError) as e:
            print(f'  건너뜀 {"/".join(keys)}: 경로 없음 ({e})')
            continue
        if old != value:
            changed += 1
            note = f'   # {why}' if why else ''
            print(f'  {"/".join(keys[-3:])}: {json.dumps(old, ensure_ascii=False)} '
                  f'-> {json.dumps(value, ensure_ascii=False)}{note}')

    path.write_text(json.dumps(session, indent=4, ensure_ascii=False), encoding='utf-8')
    print(f'\n{changed}개 항목 변경. 대시보드를 켜고 --verify로 재확인할 것.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
