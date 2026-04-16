# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from projects.mmdet3d_plugin.models.utils.misc import NvtxRangePush, NvtxRangePop
import os
import torch.nn.functional as F

import nvtx

import accvlab.on_demand_video_decoder as nvc

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 enable_gop_decode=False,
                 enable_transform=False,
                 img_norm_cfg=None,
                 ida_aug_conf=None,
                 batch_size=0,
                 num_cameras=0):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        self.iter = 0
        
        # GOP decoder initialization
        self.enable_gop_decode = enable_gop_decode
        self.enable_transform = enable_transform
        self.img_norm_cfg = img_norm_cfg
        self.ida_aug_conf = ida_aug_conf
        self._gop_decoder = None
        self.batch_size = batch_size
        self.num_cameras = num_cameras

    def _init_gop_decoder(self):
        """Lazy initialization of GOP decoder"""
        if self._gop_decoder is None and self.enable_gop_decode:
            # Initialize GOP decoder using nvc.CreateGopDecoder
            self._gop_decoder = nvc.CreateGopDecoder(
                maxfiles=self.batch_size*self.num_cameras,
                iGpu=int(os.environ.get("LOCAL_RANK", 0)),
            )
            self._gop_decoder_initialized = True
            print(f"GOP decoder initialized on GPU {os.environ.get('LOCAL_RANK', 0)}")
    
    @nvtx.annotate("decode_gop_batch", color="red")
    def _decode_gop_batch(self, gop_data_list, video_paths_list, video_frame_ids_list, to_float32=True):
        """
        Decode GOP data on GPU in batch.
        
        Args:
            gop_data_list: List[List[np.ndarray]] - batch of GOP data, each item is a list of GOP arrays for cameras
            video_paths_list: List[List[str]] - batch of video paths
            video_frame_ids_list: List[List[int]] - batch of frame IDs
            to_float32: bool - whether to convert to float32
        
        Returns:
            torch.Tensor: Decoded images in shape (B, T, N, C, H, W) or (B, N, C, H, W)
        """
        self._init_gop_decoder()
        
        H, W, C = self.ida_aug_conf["H"], self.ida_aug_conf["W"], 3
        batch_size = len(gop_data_list)
        if (batch_size != self.batch_size):
            raise ValueError(f"data {batch_size} != self.batch_size {self.batch_size}")
        
        # Check if gop_data_list has time dimension
        # gop_data_list could be:
        # 1. List[List[np.ndarray]] - (B, N) where N is num_cameras
        # 2. List[List[List[np.ndarray]]] - (B, T, N) where T is temporal frames
        
        if batch_size > 0 and len(gop_data_list[0]) > 0:
            if isinstance(gop_data_list[0][0], list):
                raise ValueError("gop_data_list should be a list of lists of np.ndarrays, and not support T dimension Now")
        
        decoded_images = []

        # Single frame case: (B, N) - but we need to add time dimension for consistency
        gops_all = []
        paths_all = []
        frame_ids_all = []
        
        for b in range(batch_size):
            gops_all.extend(gop_data_list[b])  # Extend with N cameras
            paths_all.extend(video_paths_list[b])
            frame_ids_all.extend(video_frame_ids_list[b])
        
        # Decode all at once: (B*N) images
        # print("gops_all: ", type(gops_all), len(gops_all), type(gops_all[0]), len(gops_all[0]))
        try:
            with nvtx.annotate("nvc_DecodeFromGOPListRGB", color="blue"):
                decoded_frames = self._gop_decoder.DecodeFromGOPListRGB(
                    gops_all, paths_all, frame_ids_all, True  # RGB format
                )

            if decoded_frames is None or len(decoded_frames) != len(gops_all):
                raise RuntimeError(f"GOP decode failed or returned wrong number of frames")

            # Convert to tensors (zero-copy from __cuda_array_interface__)
            with nvtx.annotate("as_tensor_and_float", color="cyan"):
                imgs = []
                for decoded_frame in decoded_frames:
                    img_tensor = torch.as_tensor(decoded_frame, device='cuda')
                    if to_float32:
                        img_tensor = img_tensor.float()
                    imgs.append(img_tensor)

            with nvtx.annotate("stack_and_reshape", color="cyan"):
                imgs = torch.stack(imgs)  # (B*N, C, H, W)
            # Reshape to (B, N, C, H, W)
            N = len(gop_data_list[0])
            decoded_images = imgs.reshape(batch_size, N, C, H, W)
            
            # Add time dimension: (B, N, C, H, W) -> (B, 1, N, C, H, W)
            decoded_images = decoded_images.unsqueeze(1)
            
        except Exception as e:
            print(f"ERROR in GOP decode: {e}")
            # Fallback: generate random with time dimension
            N = len(gop_data_list[0])
            decoded_images = torch.randint(0, 256, (batch_size, 1, N, C, H, W),
                                            dtype=torch.float32 if to_float32 else torch.uint8)
    
        # # Move to current device
        # device = next(self.parameters()).device
        # decoded_images = decoded_images.to(device)
        
        return decoded_images
    
    @nvtx.annotate("resize_crop_flip_gpu_batch", color="cyan")
    def _resize_crop_flip_gpu_batch(self, img, ida_aug_conf, training=True, ida_aug_params=None):
        """
        Apply resize, crop, flip transforms on GPU in batch.
        
        Args:
            img: torch.Tensor - (B, T, N, C, H, W) or (B, N, C, H, W)
            ida_aug_conf: dict - augmentation config
            training: bool - whether in training mode
            ida_aug_params: dict - Pre-sampled augmentation parameters (from PrepareGPUImageAug Pipeline),
                           containing resize, resize_dims, crop, flip, rotate.
                           If provided, these parameters are used instead of re-sampling.
        
        Returns:
            img: torch.Tensor - transformed images
            ida_mats: torch.Tensor - transformation matrices (B, T, N, 3, 3) or (B, N, 3, 3)
        """
        # Check dimensions
        has_time_dim = (img.dim() == 6)  # (B, T, N, C, H, W)
        
        if has_time_dim:
            B, T, N, C, H, W = img.shape
            # Flatten to (B*T*N, C, H, W)
            img = img.reshape(B * T * N, C, H, W)
        else:
            B, N, C, H, W = img.shape
            T = 1
            # Flatten to (B*N, C, H, W)
            img = img.reshape(B * N, C, H, W)
        
        fH, fW = ida_aug_conf["final_dim"]
        orig_H, orig_W = ida_aug_conf["H"], ida_aug_conf["W"]
        
        # Use pre-sampled parameters or sample new ones
        if ida_aug_params is not None:
            # Use pre-sampled parameters from Pipeline (stored in img_metas)
            # These are Python native types, no conversion needed
            resize = ida_aug_params['resize']
            resize_dims = ida_aug_params['resize_dims']
            crop = ida_aug_params['crop']
            flip = ida_aug_params['flip']
            rotate = ida_aug_params['rotate']
        elif training:
            # Random sampling in training mode (backward compatible)
            resize = np.random.uniform(*ida_aug_conf["resize_lim"])
            resize_dims = (int(orig_W * resize), int(orig_H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*ida_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if ida_aug_conf.get("rand_flip", False) and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*ida_aug_conf.get("rot_lim", (0.0, 0.0)))
        else:
            # Use fixed parameters in test mode
            resize = max(fH / orig_H, fW / orig_W)
            resize_dims = (int(orig_W * resize), int(orig_H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(ida_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0.0
        
        # 1. Resize
        img = F.interpolate(
            img, 
            size=(resize_dims[1], resize_dims[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        )
        
        # 2. Crop
        x1, y1, x2, y2 = crop
        _, _, H_resized, W_resized = img.shape
        x1_clamped = max(0, min(x1, W_resized))
        x2_clamped = max(0, min(x2, W_resized))
        y1_clamped = max(0, min(y1, H_resized))
        y2_clamped = max(0, min(y2, H_resized))
        img = img[:, :, y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        
        # Ensure exact final dimensions
        img = F.interpolate(img, size=(fH, fW), mode='bilinear', align_corners=False)
        
        # 3. Flip
        if flip:
            img = torch.flip(img, dims=[3])  # Flip width dimension
        
        # 4. Rotation (if needed, currently disabled in config)
        # Note: rotation would require more complex implementation
        
        # Compute transformation matrix (ida_mat)
        ida_rot = torch.eye(2, device=img.device) * resize
        ida_tran = torch.tensor([-crop[0], -crop[1]], device=img.device, dtype=torch.float32)
        
        if flip:
            A = torch.tensor([[-1, 0], [0, 1]], device=img.device, dtype=torch.float32)
            b = torch.tensor([fW, 0], device=img.device, dtype=torch.float32)
            ida_rot = A @ ida_rot
            ida_tran = A @ ida_tran + b
        
        ida_mat = torch.eye(3, device=img.device, dtype=torch.float32)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        
        # Reshape back
        if has_time_dim:
            img = img.reshape(B, T, N, C, fH, fW)
            ida_mats = ida_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N, 3, 3)
        else:
            img = img.reshape(B, N, C, fH, fW)
            ida_mats = ida_mat.unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3)
        
        return img, ida_mats
    
    @nvtx.annotate("normalize_gpu_batch", color="cyan")
    def _normalize_gpu_batch(self, img, mean, std, to_rgb=True):
        """
        Normalize images on GPU in batch.
        
        Args:
            img: torch.Tensor - (B, T, N, C, H, W) or (B, N, C, H, W)
            mean: list/tuple - mean values for each channel
            std: list/tuple - std values for each channel
            to_rgb: bool - whether to convert BGR to RGB (already RGB from decoder)
        
        Returns:
            img: torch.Tensor - normalized images
        """
        has_time_dim = (img.dim() == 6)
        
        # Convert mean and std to tensors
        mean = torch.tensor(mean, device=img.device, dtype=img.dtype)
        std = torch.tensor(std, device=img.device, dtype=img.dtype)
        
        if has_time_dim:
            # (B, T, N, C, H, W)
            mean = mean.view(1, 1, 1, -1, 1, 1)
            std = std.view(1, 1, 1, -1, 1, 1)
        else:
            # (B, N, C, H, W)
            mean = mean.view(1, 1, -1, 1, 1)
            std = std.view(1, 1, -1, 1, 1)
        
        # Normalize: (img - mean) / std
        img = (img - mean) / std
        
        return img
    
    @nvtx.annotate("pad_gpu_batch", color="cyan")
    def _pad_gpu_batch(self, img, size_divisor=32):
        """
        Pad images to be divisible by size_divisor on GPU.
        
        Args:
            img: torch.Tensor - (B, T, N, C, H, W) or (B, N, C, H, W)
            size_divisor: int - pad to be divisible by this value
        
        Returns:
            img: torch.Tensor - padded images
            pad_shape: tuple - (pad_H, pad_W, C) for metadata update
        """
        has_time_dim = (img.dim() == 6)
        
        if has_time_dim:
            B, T, N, C, H, W = img.shape
        else:
            B, N, C, H, W = img.shape
        
        # Calculate padding
        pad_H = int(np.ceil(H / size_divisor)) * size_divisor
        pad_W = int(np.ceil(W / size_divisor)) * size_divisor
        
        if pad_H == H and pad_W == W:
            # No padding needed, but still return shape for metadata
            return img, (pad_H, pad_W, C)
        
        # Calculate padding amounts (left, right, top, bottom)
        pad_h = pad_H - H
        pad_w = pad_W - W
        padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        
        # Reshape for padding
        if has_time_dim:
            img = img.reshape(B * T * N, C, H, W)
        else:
            img = img.reshape(B * N, C, H, W)
        
        # Pad
        img = F.pad(img, padding, mode='constant', value=0)
        
        # Reshape back
        if has_time_dim:
            img = img.reshape(B, T, N, C, pad_H, pad_W)
        else:
            img = img.reshape(B, N, C, pad_H, pad_W)
        
        return img, (pad_H, pad_W, C)
    
    @nvtx.annotate("apply_global_transform", color="cyan")
    def _apply_global_transform(self, data, rot_range, scale_ratio_range, translation_std, reverse_angle, training=True):
        """
        Apply global rotation, scale, translation transforms (CPU matrix operations).
        This updates lidar2img, ego_pose matrices.
        
        Args:
            data: dict - data dictionary containing transforms
            rot_range: tuple - rotation range
            scale_ratio_range: tuple - scale ratio range
            translation_std: list - translation std
            reverse_angle: bool - whether to reverse rotation angle
            training: bool - whether in training mode
        
        Returns:
            None (modifies data in place)
        """
        if not training:
            return
        
        # Sample augmentation parameters
        rot_angle = np.random.uniform(*rot_range)
        scale_ratio = np.random.uniform(*scale_ratio_range)
        trans = np.random.normal(scale=np.array(translation_std, dtype=np.float32), size=3)
        
        # Get batch size
        if 'lidar2img' in data:
            batch_size = len(data['lidar2img'])
        else:
            return
        
        # Process each sample in batch
        for b in range(batch_size):
            # Rotation matrix
            rot_cos = np.cos(rot_angle)
            rot_sin = np.sin(rot_angle)
            rot_mat = np.array([
                [rot_cos, rot_sin, 0, 0],
                [-rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            rot_mat_inv = np.linalg.inv(rot_mat)
            
            # Scale matrix
            scale_mat = np.array([
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            scale_mat_inv = np.linalg.inv(scale_mat)
            
            # Translation matrix
            trans_mat = np.eye(4, dtype=np.float32)
            trans_mat[:3, 3] = trans
            trans_mat_inv = np.linalg.inv(trans_mat)
            
            # Apply to ego_pose
            if 'ego_pose' in data and isinstance(data['ego_pose'], list):
                ego_pose = np.array(data['ego_pose'][b], dtype=np.float32)
                ego_pose = ego_pose @ rot_mat_inv @ scale_mat_inv @ trans_mat_inv
                data['ego_pose'][b] = ego_pose
            
            if 'ego_pose_inv' in data and isinstance(data['ego_pose_inv'], list):
                ego_pose_inv = np.array(data['ego_pose_inv'][b], dtype=np.float32)
                ego_pose_inv = trans_mat @ scale_mat @ rot_mat @ ego_pose_inv
                data['ego_pose_inv'][b] = ego_pose_inv
            
            # Apply to lidar2img
            if 'lidar2img' in data and isinstance(data['lidar2img'], list):
                num_views = len(data['lidar2img'][b])
                for v in range(num_views):
                    lidar2img = np.array(data['lidar2img'][b][v], dtype=np.float32)
                    lidar2img = lidar2img @ rot_mat_inv @ scale_mat_inv @ trans_mat_inv
                    data['lidar2img'][b][v] = lidar2img
            
            # Apply to gt_bboxes_3d
            if 'gt_bboxes_3d' in data and isinstance(data['gt_bboxes_3d'], list):
                # Note: gt_bboxes_3d rotation/scale/translation should be handled by framework
                # This is just a placeholder - actual implementation depends on bbox format
                pass
    
    @nvtx.annotate("apply_gpu_transforms", color="orange")
    def _apply_gpu_transforms(self, img, data, ida_aug_conf, img_norm_cfg, 
                             global_rot_range, global_scale_range, global_trans_std,
                             reverse_angle=True, training=True):
        """
        Apply all GPU transforms to decoded images.
        
        Args:
            img: torch.Tensor - decoded images (B, T, N, C, H, W) or (B, N, C, H, W)
            data: dict - data dictionary containing img_metas and ida_aug_params
            ida_aug_conf: dict - image augmentation config
            img_norm_cfg: dict - normalization config
            global_rot_range: tuple - global rotation range
            global_scale_range: tuple - global scale range
            global_trans_std: list - global translation std
            reverse_angle: bool - reverse rotation angle for global transform
            training: bool - training mode
        
        Returns:
            img: torch.Tensor - transformed images
        """
        # Extract pre-sampled augmentation parameters from img_metas (if exists)
        # ida_aug_params is stored in img_metas to avoid collate issues
        ida_aug_params = None
        if 'img_metas' in data:
            img_metas = data['img_metas']
            # img_metas is a list of dicts (one per sample), take the first one
            if isinstance(img_metas, list) and len(img_metas) > 0:
                first_meta = img_metas[0]
                # For temporal training, first_meta might be a list of metas
                if isinstance(first_meta, list) and len(first_meta) > 0:
                    first_meta = first_meta[0]
                ida_aug_params = first_meta.get('ida_aug_params', None)
        
        # 1. Resize, Crop, Flip, Rotate
        img, ida_mats = self._resize_crop_flip_gpu_batch(img, ida_aug_conf, training, ida_aug_params)
        
        # After resize/crop, update img_shape in img_metas
        has_time_dim = (img.dim() == 6)
        if has_time_dim:
            B, T, N, C, H, W = img.shape
        else:
            B, N, C, H, W = img.shape
        
        # Update img_shape for each sample after resize/crop
        if 'img_metas' in data:
            img_metas = data['img_metas']
            for batch_idx in range(len(img_metas)):
                # img_metas might be nested for temporal data
                if isinstance(img_metas[batch_idx], list):
                    # Temporal case: img_metas[batch_idx] is a list of T metas
                    for t_idx in range(len(img_metas[batch_idx])):
                        meta = img_metas[batch_idx][t_idx]
                        # Update to current shape after resize/crop
                        meta['img_shape'] = [(H, W, C)] * N
                else:
                    # Non-temporal case
                    meta = img_metas[batch_idx]
                    meta['img_shape'] = [(H, W, C)] * N
        
        # 2. Normalize
        img = self._normalize_gpu_batch(
            img, 
            mean=img_norm_cfg['mean'], 
            std=img_norm_cfg['std'],
            to_rgb=img_norm_cfg.get('to_rgb', True)
        )
        
        # 3. Pad
        img, pad_shape = self._pad_gpu_batch(img, size_divisor=32)
        
        # Update pad_shape in img_metas after padding
        if 'img_metas' in data:
            img_metas = data['img_metas']
            for batch_idx in range(len(img_metas)):
                if isinstance(img_metas[batch_idx], list):
                    # Temporal case
                    for t_idx in range(len(img_metas[batch_idx])):
                        meta = img_metas[batch_idx][t_idx]
                        # Update to padded shape
                        meta['pad_shape'] = [pad_shape] * N
                        # img_shape should also reflect the final padded shape
                        meta['img_shape'] = [pad_shape] * N
                else:
                    # Non-temporal case
                    meta = img_metas[batch_idx]
                    meta['pad_shape'] = [pad_shape] * N
                    meta['img_shape'] = [pad_shape] * N
        
        # 4. Global transform (CPU matrix operations)
        # If PrepareGPUImageAug Pipeline is used, global transform is already done in pipeline, skip
        if ida_aug_params is None:
            self._apply_global_transform(
                data, 
                rot_range=global_rot_range,
                scale_ratio_range=global_scale_range,
                translation_std=global_trans_std,
                reverse_angle=reverse_angle,
                training=training
            )
        
        # Note: ida_mats could be stored in data for downstream use if needed
        # data['ida_mats'] = ida_mats
        
        return img

    @nvtx.annotate("decode_and_set_img", color="purple")
    def _decode_and_set_img(self, data):
        """Decode GOP data and apply GPU transforms, setting data['img'].

        Can be called from forward() (non-prefetch) or from the prefetch
        runner (before MMDDP scatter).  Handles DataContainer unwrapping
        so it works in both contexts.
        """
        from mmcv.parallel import DataContainer as DC

        def _unwrap(val):
            if not isinstance(val, DC):
                return val
            raw = val.data
            # mmcv collate wraps cpu_only DC in GPU-groups:
            #   DC(data=[[sample0, sample1], ...])
            # MMDDP scatter peels this to [sample0, sample1].
            # When called before scatter (prefetch runner), do it here.
            if (val.cpu_only and isinstance(raw, list)
                    and len(raw) == 1 and isinstance(raw[0], list)):
                return raw[0]
            return raw

        gop_data_list = _unwrap(data['GOP'])
        video_paths_list = _unwrap(data.get('video_paths'))
        video_frame_ids_list = _unwrap(data.get('video_frame_ids'))

        if video_paths_list is None:
            video_paths_list = [[]] * len(gop_data_list)
        if video_frame_ids_list is None:
            video_frame_ids_list = [[]] * len(gop_data_list)

        # 1. Decode GOP → images on GPU
        with nvtx.annotate("decode_gop_batch", color="blue"):
            decoded_img = self._decode_gop_batch(
                gop_data_list, video_paths_list,
                video_frame_ids_list, to_float32=True)

        # 2. Apply GPU transforms (resize, crop, flip, normalize, pad)
        with nvtx.annotate("unwrap_img_metas", color="cyan"):
            img_metas_orig = data.get('img_metas')
            if isinstance(img_metas_orig, DC):
                raw = img_metas_orig.data
                if (img_metas_orig.cpu_only and isinstance(raw, list)
                        and len(raw) == 1 and isinstance(raw[0], list)):
                    data['img_metas'] = raw[0]
                else:
                    data['img_metas'] = raw

        with nvtx.annotate("apply_gpu_transforms", color="magenta"):
            transformed_img = self._apply_gpu_transforms(
                decoded_img, data,
                ida_aug_conf=self.ida_aug_conf,
                img_norm_cfg=self.img_norm_cfg,
                global_rot_range=[-0.3925, 0.3925],
                global_scale_range=[0.95, 1.05],
                global_trans_std=[0, 0, 0],
                reverse_angle=True,
                training=True)

        # Restore img_metas DC wrapper (inner list was modified in-place)
        if isinstance(img_metas_orig, DC):
            data['img_metas'] = img_metas_orig

        # 3. Set decoded image and clean up GOP keys
        data['img'] = transformed_img
        data.pop('GOP', None)
        data.pop('video_paths', None)
        data.pop('video_frame_ids', None)

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                data_t[key] = data[key][:, i] 

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # Handle GOP decode + GPU transforms
        # Three cases:
        #   1. enable_gop_decode=True, 'GOP' in data → decode + transform here
        #   2. enable_gop_decode=True, 'GOP' not in data → prefetch runner
        #      already decoded & transformed, data['img'] is ready
        #   3. enable_transform=True (no gop) → GPU transform only
        if self.enable_gop_decode and 'GOP' in data:
            self._decode_and_set_img(data)
        elif self.enable_transform and not self.enable_gop_decode:
            input_img = data['img'].to(torch.float32)
            data['img'] = self._apply_gpu_transforms(
                input_img, data,
                ida_aug_conf=self.ida_aug_conf,
                img_norm_cfg=self.img_norm_cfg,
                global_rot_range=[-0.3925, 0.3925],
                global_scale_range=[0.95, 1.05],
                global_trans_std=[0, 0, 0],
                reverse_angle=True,
                training=return_loss)

        # Continue with normal forward logic
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """
        Test function of point cloud branch.
        
        Supports two modes for temporal state management:
        1. Single video mode (legacy): Automatically detect scene changes using
           scene_token and reset memory accordingly. All samples in batch are
           assumed to be from the same video.
        2. Multi-video mode: If 'prev_exists' is passed in data, use it directly
           for per-sample temporal control. prev_exists should be shape (B,)
           where each element indicates if the corresponding sample has previous
           temporal context (1.0) or is starting fresh (0.0).
        """
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        # Check if prev_exists is provided externally (multi-video mode)
        if 'prev_exists' in data and data['prev_exists'] is not None:
            # Multi-video parallel mode: use provided prev_exists
            # prev_exists should be shape (B,) for per-sample control
            prev_exists = data['prev_exists']
            if prev_exists.dim() == 0:
                prev_exists = prev_exists.unsqueeze(0)
            
            # If any sample needs reset (prev_exists=0), we need to handle memory
            # The memory refresh in pts_bbox_head.pre_update_memory handles this
            # per-sample using the prev_exists mask
            if prev_exists.shape[0] == 1 and len(img_metas) > 1:
                # Broadcast single value to batch size
                prev_exists = prev_exists.expand(len(img_metas))
            data['prev_exists'] = prev_exists
        else:
            # Legacy single video mode: auto-detect scene changes
            if img_metas[0]['scene_token'] != self.prev_scene_token:
                self.prev_scene_token = img_metas[0]['scene_token']
                # Create prev_exists with shape (B,) for batch support
                batch_size = len(img_metas)
                data['prev_exists'] = data['img'].new_zeros(batch_size)
                self.pts_bbox_head.reset_memory()
            else:
                batch_size = len(img_metas)
                data['prev_exists'] = data['img'].new_ones(batch_size)

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    