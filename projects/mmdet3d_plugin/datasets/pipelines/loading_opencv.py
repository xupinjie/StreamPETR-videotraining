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
            self.video_cap_cache = OrderedDict()
    
    def _get_video_capture(self, video_path):
        """Get or create a VideoCapture object for the given video path.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            cv2.VideoCapture: VideoCapture object for the video.
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
        
        # Print video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f"  Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Add to cache (use original path as key)
        self.video_cap_cache[video_path] = cap
        
        # Remove oldest if cache is full
        if len(self.video_cap_cache) > self.max_cache_size:
            oldest_path, oldest_cap = self.video_cap_cache.popitem(last=False)
            oldest_cap.release()
        
        return cap
    
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
        NUM_CAMERAS = self.num_cameras

        filename = results['video_paths']
        video_frame_ids = results["video_frame_ids"]

        img = []
        
        # Decode each video frame using OpenCV
        for video_path, frame_id in zip(filename, video_frame_ids):
            try:
                # Get VideoCapture from cache
                cap = self._get_video_capture(video_path)
                
                if cap is None:
                    raise ValueError(f"Failed to get VideoCapture for {video_path}")
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    raise ValueError(f"Failed to read frame {frame_id} from {video_path}")
                
                # Check frame shape
                if frame.shape[0] != H or frame.shape[1] != W or frame.shape[2] != C:
                    print(f"ERROR: Frame shape mismatch, resizing. Got {frame.shape}, expected ({H}, {W}, {C})")
                    frame = cv2.resize(frame, (W, H))
                
                # Convert to torch tensor (keep BGR format as requested)
                torch_frame = torch.from_numpy(frame).to(torch.cuda.current_device())
                
            except Exception as e:
                print(f"ERROR: Failed to decode frame {frame_id} from {video_path}: {e}")
                print(f"Generating random image as fallback")
                torch_frame = torch.randint(0, 256, (H, W, C), dtype=torch.uint8).to(torch.cuda.current_device())
            
            img.append(torch_frame)

        # print("img shape = ", torch_frame.shape)
        
        img = torch.stack(img, axis=-1)
        img = img.detach().cpu().numpy()
        if self.to_float32:
            img = img.astype(np.float32)
        # print(f"img shape = {img.shape}")
 
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
            for cap in self.video_cap_cache.values():
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
