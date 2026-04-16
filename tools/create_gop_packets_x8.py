"""
create_gop_packets_x8.py

Creates video_samples_packets/ for nuscenes-mini-x8 by physically copying
each source GOP directory 8 times with _x{n} naming.

Physical copies ensure the benchmark reflects real I/O behavior
(page cache misses, disk seek patterns) rather than symlink shortcuts.

Source:
  {src}/video_samples_packets/{scene_token}__CAM_FRONT.mp4/
    ├── gop.0.30.bin
    └── ...

Target:
  {dst}/video_samples_packets/{scene_token}_x0__CAM_FRONT.mp4/
    ├── gop.0.30.bin
    └── ...
  {dst}/video_samples_packets/{scene_token}_x1__CAM_FRONT.mp4/
    └── ...

Usage:
    python tools/create_gop_packets_x8.py
    python tools/create_gop_packets_x8.py --src /data/nuscenes-mini --dst /data/nuscenes-mini-x8
    python tools/create_gop_packets_x8.py --multiplier 4 --workers 16
"""
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    parser = argparse.ArgumentParser(
        description='Create x8 GOP packets by physical copy')
    parser.add_argument('--src', default='/data/nuscenes-mini',
                        help='Source nuscenes-mini root')
    parser.add_argument('--dst', default='/data/nuscenes-mini-x8',
                        help='Destination nuscenes-mini-x8 root')
    parser.add_argument('--multiplier', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel copy threads')
    args = parser.parse_args()

    src_packets = os.path.join(args.src, 'video_samples_packets')
    dst_packets = os.path.join(args.dst, 'video_samples_packets')

    if not os.path.isdir(src_packets):
        print(f'ERROR: source not found: {src_packets}')
        return

    os.makedirs(dst_packets, exist_ok=True)

    # Collect source GOP directories
    src_dirs = []
    for entry in sorted(os.listdir(src_packets)):
        full = os.path.join(src_packets, entry)
        if os.path.isdir(full) and '__CAM_' in entry:
            src_dirs.append((entry, full))

    if not src_dirs:
        print(f'ERROR: no GOP directories (*__CAM_*) found in {src_packets}')
        return

    print(f'Found {len(src_dirs)} source GOP directories')
    print(f'Multiplier: {args.multiplier}')
    print(f'Workers: {args.workers}')

    # Build copy tasks
    tasks = []
    skipped = 0
    for entry_name, src_dir in src_dirs:
        scene_token, rest = entry_name.split('__CAM_', 1)
        for n in range(args.multiplier):
            new_name = f'{scene_token}_x{n}__CAM_{rest}'
            dst_dir = os.path.join(dst_packets, new_name)
            if os.path.exists(dst_dir):
                skipped += 1
                continue
            tasks.append((src_dir, dst_dir))

    total = len(tasks)
    print(f'Copying {total} directories ({skipped} already exist)...')

    if total == 0:
        print('Nothing to do.')
        return

    # Execute copies in parallel
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(shutil.copytree, s, d): (s, d)
                   for s, d in tasks}
        for future in as_completed(futures):
            src, dst = futures[future]
            done += 1
            if done % 50 == 0 or done == total:
                print(f'  [{done}/{total}]')
            try:
                future.result()
            except Exception as e:
                failed.append((src, dst, str(e)))

    if failed:
        print(f'\nWARNING: {len(failed)} failures:')
        for s, d, err in failed[:10]:
            print(f'  {s} -> {d}: {err}')
    else:
        print(f'\nAll {total} directories copied successfully.')

    # Verify
    result_count = len([d for d in os.listdir(dst_packets)
                        if os.path.isdir(os.path.join(dst_packets, d))])
    expected = len(src_dirs) * args.multiplier
    status = 'OK' if result_count >= expected else 'MISMATCH'
    print(f'Verify: {result_count} directories in target ({expected} expected) [{status}]')


if __name__ == '__main__':
    main()
