# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# Image Transforms
# Reusable logic from ResizeCropFlipRotImage, NormalizeMultiviewImage, PadMultiViewImage
# ------------------------------------------------------------------------

import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any


class ImageTransforms:
    """
    Image transform class, replaces transform operations in mmdet3d pipeline
    
    Implements the following transforms:
    1. ResizeCropFlip - resize, crop (no flip during inference)
    2. Normalize - normalization
    3. Pad - pad to multiple of 32
    """
    
    def __init__(self, img_norm_cfg: Dict[str, Any], ida_aug_conf: Dict[str, Any]):
        """
        Args:
            img_norm_cfg: Normalization config
                - mean: [123.675, 116.28, 103.53]
                - std: [58.395, 57.12, 57.375]
                - to_rgb: True
            ida_aug_conf: Image augmentation config
                - resize_lim: (0.38, 0.55)
                - final_dim: (256, 704)
                - H: 900
                - W: 1600
                - rand_flip: True (not used during inference)
        """
        self.mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        self.std = np.array(img_norm_cfg['std'], dtype=np.float32)
        self.to_rgb = img_norm_cfg.get('to_rgb', True)
        
        self.final_dim = tuple(ida_aug_conf['final_dim'])  # (H, W) = (256, 704)
        self.original_H = ida_aug_conf['H']  # 900
        self.original_W = ida_aug_conf['W']  # 1600
        self.bot_pct_lim = ida_aug_conf.get('bot_pct_lim', (0.0, 0.0))
        
        # Use the same logic as ResizeCropFlipRotImage(training=False) during inference
        fH, fW = self.final_dim
        H, W = self.original_H, self.original_W
        self.resize = max(fH / H, fW / W)  # Ensure output size is at least final_dim
        
        # Pre-compute crop parameters (same as resize_crop method)
        newW, newH = int(W * self.resize), int(H * self.resize)
        bot_pct = np.mean(self.bot_pct_lim)
        self.crop_h = int((1 - bot_pct) * newH) - fH
        self.crop_w = int(max(0, newW - fW) / 2)
    
    def __call__(self, images: List[np.ndarray], metadata: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Apply all transforms
        
        Args:
            images: List[np.ndarray] - multi-view image list, each with shape (H, W, 3), BGR format
            metadata: dict - contains intrinsics, extrinsics and other info
        
        Returns:
            Tuple[List[np.ndarray], dict]: Transformed images and updated metadata
        """
        # 1. Resize + Crop
        images, metadata = self.resize_crop(images, metadata)
        
        # 2. Normalize
        images = self.normalize(images)
        
        # 3. Pad
        images = self.pad(images, divisor=32)
        
        return images, metadata
    
    def resize_crop(self, images: List[np.ndarray], metadata: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Resize + Crop transform
        Reuses logic from ResizeCropFlipRotImage (training=False branch)
        
        Args:
            images: Original image list
            metadata: Metadata containing intrinsics and extrinsics
        
        Returns:
            Transformed images and updated metadata
        """
        fH, fW = self.final_dim  # (256, 704)
        H, W = self.original_H, self.original_W  # (900, 1600)
        
        # Use the same logic as original code during inference
        resize = self.resize
        newW, newH = int(W * resize), int(H * resize)
        
        # Same crop calculation as ResizeCropFlipRotImage(training=False)
        bot_pct = np.mean(self.bot_pct_lim)
        crop_h = int((1 - bot_pct) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)  # Center crop
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (left, top, right, bottom)
        
        new_images = []
        intrinsics = metadata.get('intrinsics', [])
        new_intrinsics = []
        
        for i, img in enumerate(images):
            # Resize
            pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((newW, newH), Image.BILINEAR)
            
            # Crop
            pil_img = pil_img.crop(crop)
            new_images.append(np.array(pil_img).astype(np.float32))
            
            # Update intrinsics
            if len(intrinsics) > i:
                intr = intrinsics[i].copy()
                # Build IDA transform matrix
                ida_mat = np.eye(3, dtype=np.float32)
                ida_mat[0, 0] = resize  # scale x
                ida_mat[1, 1] = resize  # scale y
                ida_mat[0, 2] = -crop[0]  # translate x
                ida_mat[1, 2] = -crop[1]  # translate y
                intr[:3, :3] = ida_mat @ intr[:3, :3]
                new_intrinsics.append(intr)
        
        # Update metadata
        if new_intrinsics:
            metadata['intrinsics'] = new_intrinsics
            # Recalculate lidar2img
            extrinsics = metadata.get('extrinsics', [])
            if extrinsics:
                metadata['lidar2img'] = [
                    new_intrinsics[i] @ extrinsics[i] 
                    for i in range(len(extrinsics))
                ]
        
        # Update shape info
        metadata['img_shape'] = [(fH, fW, 3) for _ in range(len(new_images))]
        
        return new_images, metadata
    
    def normalize(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize images
        Reuses logic from NormalizeMultiviewImage
        
        Args:
            images: Image list, float32 format
        
        Returns:
            Normalized image list
        """
        normalized = []
        for img in images:
            img = img.copy()
            # BGR to RGB
            if self.to_rgb:
                img = img[..., ::-1].copy()
            # Normalize
            img = (img - self.mean) / self.std
            normalized.append(img.astype(np.float32))
        return normalized
    
    def pad(self, images: List[np.ndarray], divisor: int = 32) -> List[np.ndarray]:
        """
        Pad images to multiple of divisor
        Reuses logic from PadMultiViewImage
        
        Args:
            images: Image list
            divisor: Pad to multiple of this number
        
        Returns:
            Padded image list
        """
        padded = []
        for img in images:
            h, w = img.shape[:2]
            pad_h = int(np.ceil(h / divisor) * divisor) - h
            pad_w = int(np.ceil(w / divisor) * divisor) - w
            
            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img, 
                    ((0, pad_h), (0, pad_w), (0, 0)), 
                    mode='constant', 
                    constant_values=0
                )
            padded.append(img)
        return padded
    
    def get_pad_shape(self) -> Tuple[int, int]:
        """Get padded image dimensions"""
        fH, fW = self.final_dim
        pad_h = int(np.ceil(fH / 32) * 32)
        pad_w = int(np.ceil(fW / 32) * 32)
        return (pad_h, pad_w)

