# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet.datasets.builder import PIPELINES
import cv2
import torch
import os
import accvlab.on_demand_video_decoder as nvc
import random
import pdb

@PIPELINES.register_module()
class LoadMultiViewImageFromVideo_stream_access(object):
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
        self.nv_gop_dec = None
        self.batch_size = batch_size
        self.num_cameras = num_cameras
        self.ida_aug_conf = ida_aug_conf

    def __lazy_init__(self):
        if self.nv_gop_dec is None:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.nv_gop_dec = nvc.CreateSampleReader(
                num_of_set = self.batch_size,
                num_of_file = self.num_cameras,
                iGpu = local_rank,
            )
    
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
        try:
            decoded_frames = self.nv_gop_dec.DecodeN12ToRGB(filename, video_frame_ids, True)
        except Exception as e:
            print(f"ERROR: DecodeN12ToRGB failed, generated random rgb image: {e}")
            decoded_frames = [torch.randint(0, 256, (H, W, C),  dtype=torch.uint8).to(torch.cuda.current_device())]  * len(filename)
        
        for decoded_frame in decoded_frames:
            try:
                torch_bgr = torch.tensor(decoded_frame, dtype=torch.uint8).to(torch.cuda.current_device())
                if torch_bgr.shape[0] != H or torch_bgr.shape[1] != W or torch_bgr.shape[2] != C:
                    print("ERROR: img shape is not correct, generated random rgb image, get shape: ", torch_bgr.shape, ", target shape: ",H, W, C)
                    torch_bgr = torch.randint(0, 256, (H, W, 3),  dtype=torch.uint8).to(torch.cuda.current_device())
            except Exception as e:
                print(f"ERROR: convert to torch tensor failed, generated random rgb image: {e}")
                torch_bgr = torch.randint(0, 256, (H, W, 3),  dtype=torch.uint8).to(torch.cuda.current_device())
            img.append(torch_bgr)
        
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
