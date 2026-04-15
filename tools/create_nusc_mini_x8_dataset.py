"""
create_nusc_mini_x8_dataset.py

Creates <out-dir>/nuscenes-mini-x8/ as an 8x physical copy of nuscenes-mini.

What gets multiplied:
  - nuScenes metadata tables: scene, sample, sample_data, ego_pose,
    sample_annotation, instance  (tokens → append _x{n})
  - Sensor files: samples/CAM_*/, samples/LIDAR_TOP/, sweeps/LIDAR_TOP/
    (filename: insert _x{n} before extension)
  - Video files: video_samples_mjpeg/ and video_samples_packets/
    (filename: {scene_token}_x{n}__CAM_*.mp4 / *.bin)

What is shared (not multiplied):
  - calibrated_sensor, sensor, category, attribute, visibility, log, map tables
  - maps/ directory (symlinked)
  - can_bus/ directory (not used by StreamPETR; symlinked)

Usage:
    python tools/create_nusc_mini_x8_dataset.py \\
        --src  /data/nuscenes-mini \\
        --out  /data/nuscenes-mini-x8 \\
        --multiplier 8 \\
        --workers 8

After this script, run:
    cd tools
    python create_data_nusc.py \\
        --root-path /data/nuscenes-mini-x8 \\
        --out-dir   /data/nuscenes-mini-x8/stream_petr \\
        --extra-tag nuscenes2d \\
        --version   v1.0-mini

    python create_video_data_nusc.py \\
        --root-path /data/nuscenes-mini-x8 \\
        --out-dir   /data/nuscenes-mini-x8/stream_petr \\
        --extra-tag nuscenes2d_video \\
        --version   v1.0-mini
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import path as osp

# Each copy gets a timestamp offset so that sorting by timestamp keeps
# all frames of copy-n before copy-(n+1).  Use ~115 days in microseconds.
TIMESTAMP_OFFSET_US = int(1e13)


# ── token suffix helpers ──────────────────────────────────────────────────────

def tok(token, n):
    """Return token with copy-index suffix, or '' if token is empty."""
    if not token:
        return ''
    return f'{token}_x{n}'


def tok_list(tokens, n):
    return [tok(t, n) for t in tokens]


def rename_filename(filename, n):
    """Insert _x{n} before the file extension.
    e.g. 'samples/CAM_FRONT/abc.jpg' → 'samples/CAM_FRONT/abc_x3.jpg'
    """
    root, ext = osp.splitext(filename)
    return f'{root}_x{n}{ext}'


# ── table multiplication ──────────────────────────────────────────────────────

def multiply_scene(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']              = tok(r['token'], n)
        c['name']               = r['name'] + f'_x{n}'
        c['first_sample_token'] = tok(r['first_sample_token'], n)
        c['last_sample_token']  = tok(r['last_sample_token'],  n)
        # log_token: shared → unchanged
        out.append(c)
    return out


def multiply_sample(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']       = tok(r['token'], n)
        c['scene_token'] = tok(r['scene_token'], n)
        c['prev']        = tok(r['prev'], n)
        c['next']        = tok(r['next'], n)
        c['timestamp']   = r['timestamp'] + n * TIMESTAMP_OFFSET_US
        # 'data' (sensor → sample_data token) and 'anns' are populated by the
        # SDK at runtime from sample_data.json / sample_annotation.json.
        # They may or may not be present in the raw JSON depending on SDK version.
        if 'data' in r:
            c['data'] = {k: tok(v, n) for k, v in r['data'].items()}
        if 'anns' in r:
            c['anns'] = tok_list(r['anns'], n)
        out.append(c)
    return out


def multiply_sample_data(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']              = tok(r['token'], n)
        c['sample_token']       = tok(r['sample_token'], n)
        c['ego_pose_token']     = tok(r['ego_pose_token'], n)
        c['prev']               = tok(r['prev'], n)
        c['next']               = tok(r['next'], n)
        c['timestamp']          = r['timestamp'] + n * TIMESTAMP_OFFSET_US
        # calibrated_sensor_token: shared → unchanged
        c['filename']           = rename_filename(r['filename'], n)
        out.append(c)
    return out


def multiply_ego_pose(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']     = tok(r['token'], n)
        c['timestamp'] = r['timestamp'] + n * TIMESTAMP_OFFSET_US
        out.append(c)
    return out


def multiply_sample_annotation(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']          = tok(r['token'], n)
        c['sample_token']   = tok(r['sample_token'], n)
        c['instance_token'] = tok(r['instance_token'], n)
        c['prev']           = tok(r['prev'], n)
        c['next']           = tok(r['next'], n)
        # attribute_tokens: shared → unchanged
        out.append(c)
    return out


def multiply_instance(records, n):
    out = []
    for r in records:
        c = dict(r)
        c['token']                  = tok(r['token'], n)
        c['first_annotation_token'] = tok(r['first_annotation_token'], n)
        c['last_annotation_token']  = tok(r['last_annotation_token'],  n)
        # category_token: shared → unchanged
        out.append(c)
    return out


MULTIPLY_FNS = {
    'scene':              multiply_scene,
    'sample':             multiply_sample,
    'sample_data':        multiply_sample_data,
    'ego_pose':           multiply_ego_pose,
    'sample_annotation':  multiply_sample_annotation,
    'instance':           multiply_instance,
}

SHARED_TABLES = [
    'calibrated_sensor', 'sensor', 'category', 'attribute',
    'visibility', 'log', 'map',
]


# ── file copy helpers ─────────────────────────────────────────────────────────

def copy_file(src, dst):
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def collect_sensor_copy_tasks(src_root, dst_root, multiplier):
    """
    For each file under samples/ and sweeps/ in src_root,
    produce (src_path, dst_path_for_copy_n) for n=0..multiplier-1.

    Only copies files that actually exist in src_root.
    """
    tasks = []
    for subdir in ['samples', 'sweeps']:
        src_subdir = osp.join(src_root, subdir)
        if not osp.isdir(src_subdir):
            continue
        for sensor_dir in os.listdir(src_subdir):
            sensor_path = osp.join(src_subdir, sensor_dir)
            if not osp.isdir(sensor_path):
                continue
            for fname in os.listdir(sensor_path):
                src_file = osp.join(sensor_path, fname)
                if not osp.isfile(src_file):
                    continue
                root, ext = osp.splitext(fname)
                for n in range(multiplier):
                    new_fname = f'{root}_x{n}{ext}'
                    dst_file  = osp.join(dst_root, subdir, sensor_dir, new_fname)
                    tasks.append((src_file, dst_file))
    return tasks


def collect_video_copy_tasks(src_root, dst_root, multiplier):
    """
    video_samples_mjpeg/ and video_samples_packets/ contain files named
    {scene_token}__CAM_*.mp4 / *.bin etc.

    For each source file, create copies with scene_token replaced by
    scene_token_x{n} for n=0..multiplier-1.

    The scene_token is everything before '__CAM_'.
    """
    tasks = []
    for video_dir in ['video_samples_mjpeg', 'video_samples', 'video_samples_packets']:
        src_video = osp.join(src_root, video_dir)
        if not osp.isdir(src_video):
            continue
        for fname in os.listdir(src_video):
            src_file = osp.join(src_video, fname)
            if not osp.isfile(src_file):
                continue
            # Split on '__CAM_' separator to extract scene_token
            if '__CAM_' not in fname:
                # Unknown naming convention — just copy as-is for n=0
                dst = osp.join(dst_root, video_dir, fname)
                tasks.append((src_file, dst))
                continue
            scene_token, rest = fname.split('__CAM_', 1)
            for n in range(multiplier):
                new_fname = f'{scene_token}_x{n}__CAM_{rest}'
                dst_file  = osp.join(dst_root, video_dir, new_fname)
                tasks.append((src_file, dst_file))
    return tasks


# ── main ──────────────────────────────────────────────────────────────────────

def run(src_root, dst_root, version, multiplier, workers, metadata_only=False):
    meta_src = osp.join(src_root, version)
    meta_dst = osp.join(dst_root, version)
    os.makedirs(meta_dst, exist_ok=True)

    # ── 1. Load all tables ────────────────────────────────────────────────────
    print(f'Loading metadata from {meta_src}')
    tables = {}
    for fname in os.listdir(meta_src):
        if not fname.endswith('.json'):
            continue
        table_name = fname[:-5]
        with open(osp.join(meta_src, fname)) as f:
            tables[table_name] = json.load(f)
        print(f'  {table_name}: {len(tables[table_name])} records')

    # ── 2. Build x8 metadata ─────────────────────────────────────────────────
    print(f'\nMultiplying metadata x{multiplier}...')
    new_tables = {}
    for table_name, records in tables.items():
        if table_name in MULTIPLY_FNS:
            multiplied = []
            for n in range(multiplier):
                multiplied.extend(MULTIPLY_FNS[table_name](records, n))
            new_tables[table_name] = multiplied
            print(f'  {table_name}: {len(records)} → {len(multiplied)}')
        else:
            # shared table: copy as-is
            new_tables[table_name] = records
            print(f'  {table_name}: {len(records)} (shared)')

    # ── 3. Write new metadata JSONs ───────────────────────────────────────────
    print(f'\nWriting metadata to {meta_dst}')
    for table_name, records in new_tables.items():
        out_path = osp.join(meta_dst, f'{table_name}.json')
        with open(out_path, 'w') as f:
            json.dump(records, f)
    print('  Done.')

    # ── 4. Collect file copy tasks ────────────────────────────────────────────
    if not metadata_only:
        # ── 4. Collect file copy tasks ─────────────────────────────────────────
        print(f'\nCollecting file copy tasks...')
        tasks = []
        tasks += collect_sensor_copy_tasks(src_root, dst_root, multiplier)
        tasks += collect_video_copy_tasks(src_root, dst_root, multiplier)

        # Symlink maps and can_bus (not per-scene data)
        for shared_dir in ['maps', 'can_bus']:
            src_d = osp.join(src_root, shared_dir)
            dst_d = osp.join(dst_root, shared_dir)
            if osp.isdir(src_d) and not osp.exists(dst_d):
                os.symlink(src_d, dst_d)
                print(f'  Symlinked {shared_dir}/')

        total = len(tasks)
        print(f'  {total} files to copy ({multiplier}x {total//multiplier} source files)')

        # ── 5. Execute copies in parallel ──────────────────────────────────────
        print(f'\nCopying files with {workers} workers...')
        done = 0
        failed = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(copy_file, src, dst): (src, dst)
                       for src, dst in tasks}
            for future in as_completed(futures):
                src, dst = futures[future]
                done += 1
                if done % 1000 == 0 or done == total:
                    print(f'  [{done}/{total}]')
                try:
                    future.result()
                except Exception as e:
                    failed.append((src, dst, str(e)))

        if failed:
            print(f'\nWARNING: {len(failed)} copy failures:')
            for src, dst, err in failed[:10]:
                print(f'  {src} → {dst}: {err}')
        else:
            print(f'  All {total} files copied successfully.')
    else:
        print('\n[--metadata-only] Skipping file copies.')

    # ── 6. Summary ────────────────────────────────────────────────────────────
    unique_scenes = len(new_tables.get('scene', []))
    seq_split_num = 2
    groups = unique_scenes * seq_split_num
    print(f'\n=== Dataset created at {dst_root} ===')
    print(f'  Scenes : {len(tables["scene"])} → {unique_scenes}')
    print(f'  Samples: {len(tables["sample"])} → {len(new_tables["sample"])}')
    print(f'  Groups (seq_split_num={seq_split_num}): {groups}')
    print()
    print('  Sampler check:')
    for ngpu, bs in [(1, 16), (8, 2), (8, 8), (8, 16), (8, 32)]:
        gbs = ngpu * bs
        ok = '✓' if groups >= gbs else '✗'
        print(f'    {ngpu} GPU × bs={bs:2d}  global_bs={gbs:3d}  {ok}')
    print()
    print('Next steps:')
    print(f'  cd tools')
    print(f'  # Image pkl:')
    print(f'  python create_data_nusc.py --root-path {dst_root} --out-dir {dst_root}/stream_petr --extra-tag nuscenes2d --version v1.0-mini')
    print(f'  # Video pkl:')
    print(f'  python create_video_data_nusc.py --root-path {dst_root} --out-dir {dst_root}/stream_petr --extra-tag nuscenes2d_video --version v1.0-mini')


parser = argparse.ArgumentParser()
parser.add_argument('--src',           type=str, default='/data/nuscenes-mini')
parser.add_argument('--out',           type=str, default='/data/nuscenes-mini-x8')
parser.add_argument('--version',       type=str, default='v1.0-mini')
parser.add_argument('--multiplier',    type=int, default=8)
parser.add_argument('--workers',       type=int, default=8,
                    help='parallel copy threads')
parser.add_argument('--metadata-only', action='store_true',
                    help='only regenerate metadata JSONs, skip file copying')
args = parser.parse_args()

if __name__ == '__main__':
    run(args.src, args.out, args.version, args.multiplier, args.workers,
        metadata_only=args.metadata_only)
