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

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
import torch.nn.functional as F
from PIL import Image

import nvtx

@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    @nvtx.annotate("PadMultiViewImage", color="green")
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    @nvtx.annotate("NormalizeMultiviewImage", color="green")
    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    @nvtx.annotate("ResizeCropFlipRotImage", color="green")
    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()


        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        return results

    @nvtx.annotate("bboxes_transform", color="red")
    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    @nvtx.annotate("img_transform", color="red")
    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class PrepareGPUImageAug():
    """
    Pipeline class to prepare for GPU-based image augmentation.
    
    This class executes the following operations in the dataloader worker:
    1. Sample image augmentation parameters (resize, crop, flip, rotate)
    2. Transform 2D annotations (gt_bboxes, centers2d, depths, gt_labels)
    3. Update camera intrinsics and projection matrices (lidar2img)
    4. Save augmentation parameters for GPU image transforms in main process
    
    Note: This class does NOT transform the images themselves. Image transforms
    are performed on GPU in model forward.
    
    Args:
        data_aug_conf (dict): Image augmentation configuration containing:
            - resize_lim: resize range
            - final_dim: final image dimensions (H, W)
            - bot_pct_lim: bottom crop percentage range
            - rot_lim: rotation angle range
            - rand_flip: whether to randomly flip
            - H, W: original image dimensions
        with_2d (bool): Whether to process 2D annotations
        filter_invisible (bool): Whether to filter invisible 2D boxes
        training (bool): Whether in training mode
    """
    
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    @nvtx.annotate("PrepareGPUImageAug", color="yellow")
    def __call__(self, results):
        """
        Execute augmentation parameter sampling, annotation transforms, and intrinsics update.
        
        Args:
            results (dict): Dictionary containing image and annotation data
            
        Returns:
            dict: Updated results dictionary with new 'ida_aug_params' key
        """
        N = len(results['intrinsics'])  # Number of cameras
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        # Sample augmentation parameters (same for all cameras)
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        
        # Compute ida_mat (for updating intrinsics)
        ida_mat = self._compute_ida_mat(resize, crop, flip, rotate)

        for i in range(N):
            # Transform 2D annotations
            if self.training and self.with_2d:
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
                        gt_bboxes, centers2d, gt_labels, depths
                    )

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            # Update camera intrinsics
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        
        # Update annotations
        if self.training and self.with_2d:
            results['gt_bboxes'] = new_gt_bboxes
            results['centers2d'] = new_centers2d
            results['gt_labels'] = new_gt_labels
            results['depths'] = new_depths
        
        # Update lidar2img
        results['lidar2img'] = [
            results['intrinsics'][i] @ results['extrinsics'][i] 
            for i in range(len(results['extrinsics']))
        ]
        
        # Save augmentation parameters for GPU image transforms
        # Will be collected into img_metas by Collect3D via meta_keys
        results['ida_aug_params'] = {
            'resize': resize,
            'resize_dims': resize_dims,
            'crop': crop,
            'flip': flip,
            'rotate': rotate,
        }

        return results

    def _compute_ida_mat(self, resize, crop, flip, rotate):
        """
        Compute image data augmentation transformation matrix (ida_mat).
        
        Args:
            resize: resize ratio
            crop: crop region (x1, y1, x2, y2)
            flip: whether to flip
            rotate: rotation angle
            
        Returns:
            numpy.ndarray: 3x3 transformation matrix
        """
        ida_rot = np.eye(2) * resize
        ida_tran = np.array([-crop[0], -crop[1]], dtype=np.float32)
        
        if flip:
            A = np.array([[-1, 0], [0, 1]], dtype=np.float32)
            b = np.array([crop[2] - crop[0], 0], dtype=np.float32)
            ida_rot = A @ ida_rot
            ida_tran = A @ ida_tran + b
        
        if rotate != 0:
            rot_rad = rotate / 180 * np.pi
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            A = np.array([[cos_r, sin_r], [-sin_r, cos_r]], dtype=np.float32)
            b = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float32) / 2
            b = A @ (-b) + b
            ida_rot = A @ ida_rot
            ida_tran = A @ ida_tran + b
        
        ida_mat = np.eye(3, dtype=np.float32)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        
        return ida_mat

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, resize, crop, flip):
        """
        Transform 2D bounding boxes and center points.
        """
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & \
               ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        """
        Filter invisible 2D bounding boxes.
        """
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _sample_augmentation(self):
        """
        Sample image augmentation parameters.
        """
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(data_aug_conf={self.data_aug_conf}, '
        repr_str += f'with_2d={self.with_2d}, '
        repr_str += f'filter_invisible={self.filter_invisible}, '
        repr_str += f'training={self.training})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImageGPU():
    """GPU-accelerated version of ResizeCropFlipRotImage.
    
    This is a copy of ResizeCropFlipRotImage that will be optimized for GPU processing.
    Currently identical to the CPU version - GPU optimization to be implemented.
    """
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    @nvtx.annotate("ResizeCropFlipRotImageGPU", color="blue")
    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()


        new_imgs = self._img_transform_batch(
            imgs,
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
        )

        # print(f"new_imgs: {len(new_imgs)}")
        # for i in range(len(new_imgs)):
        #     print(f"new_imgs[{i}]: {new_imgs[i].shape}")
        
        # Compute ida_mat separately (same for all N cameras)
        ida_mat = self._compute_ida_mat(resize, crop, flip, rotate)
        
        for i in range(N):
            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        return results

    @nvtx.annotate("bboxes_transform_gpu", color="cyan")
    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    @nvtx.annotate("filter_invisible_gpu", color="cyan")
    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    @nvtx.annotate("compute_ida_mat", color="cyan")
    def _compute_ida_mat(self, resize, crop, flip, rotate):
        """
        Compute intrinsic adjustment matrix based on augmentation parameters.
        
        This is independent of image content and only depends on geometric transforms.
        
        Args:
            resize: float, resize scale factor
            crop: tuple (x1, y1, x2, y2), crop coordinates
            flip: bool, whether to flip horizontally
            rotate: float, rotation angle in degrees
            
        Returns:
            ida_mat: numpy array (3, 3), intrinsic adjustment matrix
        """
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        
        # Apply resize
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        # Apply flip
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        # Apply rotation
        if abs(rotate) > 1e-6:
            A = self._get_rot(rotate / 180 * np.pi)
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        # Construct 3x3 intrinsic adjustment matrix
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        
        return ida_mat.numpy()

    @nvtx.annotate("img_transform_gpu", color="cyan")
    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    @nvtx.annotate("img_transform_batch_cpu", color="cyan")
    def _img_transform_batch(self, imgs, resize, resize_dims, crop, flip, rotate):
        """
        Batch process N images at once using torch operations.
        
        This processes all N camera images simultaneously instead of looping.
        Only handles image transformation - ida_mat is computed separately.
        
        OPTIMIZATION: If input images are already GPU tensors, process entirely on GPU
        to avoid GPU->CPU->GPU transfers.
        
        Args:
            imgs: list of N numpy arrays (H, W, C) or torch.Tensor(GPU) (H, W, C)
            resize: float, resize scale factor
            resize_dims: tuple (W, H), target size after resize
            crop: tuple (x1, y1, x2, y2), crop coordinates
            flip: bool, whether to flip horizontally
            rotate: float, rotation angle (must be 0.0, rotation not supported)
            
        Returns:
            imgs_out: list of N numpy arrays (H_out, W_out, C) in float32
                     where H_out, W_out = final_dim from data_aug_conf
        """
        N = len(imgs)
        
        # Get target output size from config
        fH, fW = self.data_aug_conf["final_dim"]  # Fixed output size (height, width)
        
        # Determine if inputs are GPU tensors
        input_is_gpu = isinstance(imgs[0], torch.Tensor) and imgs[0].is_cuda
        
        # Step 1: Convert to torch tensors and stack into batch
        imgs_tensors = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                # Convert numpy to torch, ensure float32
                if img.dtype == np.uint8:
                    img_t = torch.from_numpy(img.astype(np.float32))
                else:
                    img_t = torch.from_numpy(img).float()
                # Move to GPU if input was GPU
                if input_is_gpu:
                    img_t = img_t.cuda()
            else:
                # Already a torch tensor
                if input_is_gpu:
                    # Keep on GPU, ensure float32
                    img_t = img.float() if img.dtype != torch.float32 else img
                else:
                    # Move to CPU if needed
                    img_t = img.cpu().float() if img.dtype != torch.float32 else img.cpu()
            imgs_tensors.append(img_t)
        
        # Stack: list of (H, W, C) -> (N, H, W, C)
        imgs_batch = torch.stack(imgs_tensors, dim=0)
        
        # Permute to PyTorch format: (N, H, W, C) -> (N, C, H, W)
        imgs_batch = imgs_batch.permute(0, 3, 1, 2)
        
        # Step 2: Batch resize using torch.nn.functional.interpolate
        # All operations stay on same device (GPU or CPU)
        imgs_batch = F.interpolate(
            imgs_batch, 
            size=(resize_dims[1], resize_dims[0]),  # (height, width)
            mode='bilinear',
            align_corners=False
        )
        
        # Step 3: Batch crop using tensor slicing
        # IMPORTANT: Ensure output is EXACTLY final_dim size
        x1, y1, x2, y2 = crop
        _, _, H_resized, W_resized = imgs_batch.shape
        
        # Clamp crop coordinates to image bounds
        x1_clamped = max(0, min(x1, W_resized))
        x2_clamped = max(0, min(x2, W_resized))
        y1_clamped = max(0, min(y1, H_resized))
        y2_clamped = max(0, min(y2, H_resized))
        
        # Perform crop
        imgs_batch = imgs_batch[:, :, y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        
        # CRITICAL: Resize/pad to exact final_dim
        # The crop might not give us exact size due to boundary issues
        _, _, H_crop, W_crop = imgs_batch.shape
        
        if H_crop != fH or W_crop != fW:
            # Resize to exact final_dim to ensure all outputs have same size
            imgs_batch = F.interpolate(
                imgs_batch,
                size=(fH, fW),  # Force exact output size
                mode='bilinear',
                align_corners=False
            )
        
        # Step 4: Batch flip if needed
        if flip:
            imgs_batch = torch.flip(imgs_batch, dims=[3])  # flip along width dimension
        
        # Step 5: Convert back to list of numpy arrays
        # Permute back: (N, C, H, W) -> (N, H, W, C)
        imgs_batch = imgs_batch.permute(0, 2, 3, 1)
        
        # OPTIMIZATION: Move to CPU only once at the end
        if input_is_gpu:
            imgs_batch = imgs_batch.cpu()
        
        # Convert to numpy - mimic the original PIL conversion path to ensure compatibility
        imgs_out = []
        for i in range(N):
            # Extract single image
            img_tensor = imgs_batch[i]
            # Convert to numpy uint8 first (like PIL does)
            img_np = img_tensor.clamp(0, 255).numpy().astype(np.uint8)
            # Convert through PIL Image to match original behavior exactly
            img_pil = Image.fromarray(img_np)
            # Convert back to numpy array (this creates a clean, collate-friendly array)
            img_final = np.array(img_pil).astype(np.float32)
            imgs_out.append(img_final)
        
        return imgs_out

    @nvtx.annotate("sample_augmentation_gpu", color="cyan")
    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    @nvtx.annotate("GlobalRotScaleTransImage", color="green")
    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()


    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()
