"""
Duplicate nuscenes-mini pkl x8 to simulate a larger dataset.

The actual sensor files are NOT copied — all 8 copies point to the same
real files on disk.  Only scene/sample tokens are made unique so that
InfiniteGroupEachSampleInBatchSampler sees enough groups for 8-GPU training.

Usage:
    python tools/create_data_nusc_mini_x8.py \
        --data-root /data/nuscenes-mini/stream_petr \
        --multiplier 8

Output (written next to the originals):
    nuscenes2d_temporal_infos_train_x8.pkl
    nuscenes2d_temporal_infos_val_x8.pkl   (val is also multiplied so
                                              val pipeline keeps working)
"""
import argparse
import copy
import os
import pickle


def duplicate_infos(original_infos, multiplier, suffix_fmt='_x{n}'):
    """Return a list that is multiplier× the original.

    Copy n=0 keeps original tokens unchanged.
    Copies n=1..multiplier-1 append suffix_fmt.format(n=n) to every token.
    File paths are preserved as-is.
    """
    new_infos = []
    for n in range(multiplier):
        suffix = '' if n == 0 else suffix_fmt.format(n=n)
        for info in original_infos:
            new_info = copy.deepcopy(info)
            new_info['scene_token'] = info['scene_token'] + suffix
            new_info['token']       = info['token']       + suffix
            new_info['prev']        = info['prev'] + suffix if info['prev'] else ''
            new_info['next']        = info['next'] + suffix if info['next'] else ''
            for cam in new_info['cams']:
                sd = new_info['cams'][cam]
                sd['sample_data_token'] = info['cams'][cam]['sample_data_token'] + suffix
                # data_path / cam_path unchanged → same real files on disk
            new_infos.append(new_info)
    return new_infos


def process_pkl(input_path, output_path, multiplier):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    original_infos = data['infos']
    new_infos = duplicate_infos(original_infos, multiplier)
    data['infos'] = new_infos

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    unique_scenes = len(set(info['scene_token'] for info in new_infos))
    print(f'  {os.path.basename(input_path)}: '
          f'{len(original_infos)} → {len(new_infos)} samples | '
          f'{unique_scenes} unique scenes | '
          f'groups(seq_split_num=2)={unique_scenes * 2}')
    print(f'  → {output_path}')
    return len(new_infos), unique_scenes


parser = argparse.ArgumentParser()
parser.add_argument('--data-root',  type=str, default='/data/nuscenes-mini/stream_petr')
parser.add_argument('--multiplier', type=int, default=8)
parser.add_argument('--tag',        type=str, default='nuscenes2d_temporal_infos',
                    help='pkl name prefix (without _train/_val.pkl)')
args = parser.parse_args()

if __name__ == '__main__':
    m = args.multiplier
    root = args.data_root

    for split in ('train', 'val'):
        src = os.path.join(root, f'{args.tag}_{split}.pkl')
        dst = os.path.join(root, f'{args.tag}_{split}_x{m}.pkl')
        if not os.path.exists(src):
            print(f'[skip] {src} not found')
            continue
        print(f'Processing {split}:')
        n_samples, n_scenes = process_pkl(src, dst, m)

    print()
    print('=== Sampler check (train) ===')
    print(f'  seq_split_num=2  → groups = {n_scenes} × 2 = {n_scenes * 2}')
    for ngpu, bs in [(1, 16), (8, 2), (8, 8), (8, 16)]:
        gbs = ngpu * bs
        ok = '✓' if n_scenes * 2 >= gbs else '✗'
        print(f'  {ngpu} GPU × bs={bs:2d}  global_bs={gbs:3d}  {ok}')
