# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from data_converter import video_nuscenes_converter as video_nuscenes_converter


def nuscenes_data_prep(root_path,
                       out_dir,
                       info_prefix,
                       version,
                       max_sweeps=10,
                       encode_mjpeg=True):
    """Prepare video data related to nuScenes dataset.

    Args:
        root_path (str): Path of dataset root.
        out_dir (str): Output directory for pkl files and video_samples_mjpeg/.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version (e.g. 'v1.0-trainval').
        max_sweeps (int, optional): Number of input consecutive frames. Default: 10.
        encode_mjpeg (bool): If True, encode jpg frames into video_samples_mjpeg/.
            If False, use pre-existing video_samples/ directly (skips encoding).
    """
    video_nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, info_prefix, version=version, max_sweeps=max_sweeps,
        encode_mjpeg=encode_mjpeg)


parser = argparse.ArgumentParser(description='Video data converter for nuScenes')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='/data/nuscenes',
    required=False,
    help='output directory for pkl files and video_samples_mjpeg/')
parser.add_argument('--extra-tag', type=str, default='nuscenes2d_video')
parser.add_argument('--encode-mjpeg', action='store_true', default=False,
                    help='encode jpg frames into video_samples_mjpeg/; '
                         'if omitted, use pre-existing video_samples/ directly')
args = parser.parse_args()

if __name__ == '__main__':
    if args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps,
            encode_mjpeg=args.encode_mjpeg)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps,
            encode_mjpeg=args.encode_mjpeg)
    elif args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps,
            encode_mjpeg=args.encode_mjpeg)

# Example usage:
# Mini:
#   python tools/create_video_data_nusc.py --root-path /data/nuscenes-mini --out-dir /data/nuscenes-mini --extra-tag nuscenes2d_video --version v1.0-mini
# Full:
#   python tools/create_video_data_nusc.py --root-path /data/nuscenes-full --out-dir /data/nuscenes-full --extra-tag nuscenes2d_video --version v1.0
