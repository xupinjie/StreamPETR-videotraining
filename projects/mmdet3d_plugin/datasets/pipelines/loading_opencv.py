# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.datasets.builder import PIPELINES
import cv2
import torch
import os
import random
import pdb
from collections import OrderedDict

import nvtx

@PIPELINES.register_module()
class LoadMultiViewImageFromVideo_opencv(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, batch_size, num_cameras, ida_aug_conf, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.batch_size = batch_size
        self.num_cameras = num_cameras
        self.ida_aug_conf = ida_aug_conf
        self.max_cache_size = batch_size * num_cameras
        self.video_cap_cache = None

    def __lazy_init__(self):
        if self.video_cap_cache is None:
            # Use OrderedDict as LRU cache for VideoCapture objects
            # Value: (cv2.VideoCapture, current_frame_position)
            self.video_cap_cache = OrderedDict()

    def _get_video_capture(self, video_path):
        """Get or create a VideoCapture object for the given video path.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tuple: (cv2.VideoCapture, current_frame_pos) for the video.
        """
        # Check if video_path is in cache
        if video_path in self.video_cap_cache:
            # Move to end (most recently used)
            self.video_cap_cache.move_to_end(video_path)
            return self.video_cap_cache[video_path]

        # Convert to absolute path
        abs_path = os.path.abspath(video_path)

        # Create new VideoCapture with absolute path
        cap = cv2.VideoCapture(abs_path)
        if not cap.isOpened():
            print(f"ERROR: Failed to open video file with OpenCV: {abs_path}")
            print(f"  OpenCV build info:")
            print(f"  Version: {cv2.__version__}")
            print(f"  Video backend: {cap.getBackendName()}")
            return None

        # Add to cache: (cap, current_position)
        # Position starts at 0 for a freshly opened capture
        entry = [cap, 0]
        self.video_cap_cache[video_path] = entry

        # Remove oldest if cache is full
        if len(self.video_cap_cache) > self.max_cache_size:
            _, (oldest_cap, _) = self.video_cap_cache.popitem(last=False)
            oldest_cap.release()

        return entry

    def _read_frame(self, video_path, frame_id):
        """Read a frame, using sequential reads to advance instead of seek.

        Args:
            video_path (str): Path to the video file.
            frame_id (int): Target frame index.

        Returns:
            np.ndarray: Decoded frame (H, W, 3) in BGR.
        """
        entry = self._get_video_capture(video_path)
        if entry is None:
            raise ValueError(f"Failed to get VideoCapture for {video_path}")

        cap, cur_pos = entry

        if cur_pos <= frame_id:
            # Forward: skip-read to target (avoids expensive keyframe seek)
            for _ in range(frame_id - cur_pos):
                cap.read()
        else:
            # Backward: must seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_id} from {video_path}")

        # Update tracked position (read advances by 1)
        entry[1] = frame_id + 1

        return frame

    @nvtx.annotate("LoadMultiViewImageFromVideo_opencv",color="green")
    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        if results["video_paths"] is None:
            raise ValueError("video_paths is not provided")

        self.__lazy_init__()

        H = self.ida_aug_conf["H"]
        W = self.ida_aug_conf["W"]
        C = 3

        filename = results['video_paths']
        video_frame_ids = results["video_frame_ids"]

        img = []

        # Decode each video frame using OpenCV
        for video_path, frame_id in zip(filename, video_frame_ids):
            try:
                frame = self._read_frame(video_path, frame_id)

                # Check frame shape
                if frame.shape[0] != H or frame.shape[1] != W or frame.shape[2] != C:
                    frame = cv2.resize(frame, (W, H))

            except Exception as e:
                print(f"ERROR: Failed to decode frame {frame_id} from {video_path}: {e}")
                frame = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)

            img.append(frame)

        # Stack along last axis: (H, W, C, N)
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __del__(self):
        """Release all cached VideoCapture objects."""
        if hasattr(self, 'video_cap_cache') and self.video_cap_cache is not None:
            for (cap, _) in self.video_cap_cache.values():
                if cap is not None:
                    cap.release()
            self.video_cap_cache.clear()

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'color_type={self.color_type}, '
        repr_str += f'batch_size={self.batch_size}, '
        repr_str += f'num_cameras={self.num_cameras}, '
        repr_str += f'ida_aug_conf={self.ida_aug_conf}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f'color_type={self.color_type})'
        return repr_str
