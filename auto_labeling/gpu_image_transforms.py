# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# GPU Image Transforms
# GPU-accelerated image preprocessing using torchvision
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union


class GPUImageTransforms:
    """
    GPU-accelerated image transformation class using torchvision/torch.
    
    The following operations are executed on GPU:
    1. Resize - using F.interpolate
    2. Crop - using tensor slicing
    3. BGR→RGB - using tensor indexing
    4. Normalize - using tensor operations
    5. Pad - using F.pad
    
    Compatible with ImageTransforms interface for seamless replacement.
    """
    
    def __init__(
        self, 
        img_norm_cfg: Dict[str, Any], 
        ida_aug_conf: Dict[str, Any],
        device: str = 'cuda:0'
    ):
        """
        Initialize GPU image transformer.
        
        Args:
            img_norm_cfg: Image normalization configuration
                - mean: [123.675, 116.28, 103.53]
                - std: [58.395, 57.12, 57.375]
                - to_rgb: True
            ida_aug_conf: Image augmentation configuration
                - resize_lim: (0.38, 0.55)
                - final_dim: (256, 704)
                - H: 900
                - W: 1600
            device: GPU device
        """
        self.device = device
        
        # Pre-compute normalization parameters (on GPU)
        mean = img_norm_cfg['mean']
        std = img_norm_cfg['std']
        self.to_rgb = img_norm_cfg.get('to_rgb', True)
        
        # Create GPU tensor with shape (1, 3, 1, 1) for broadcasting
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        
        # Image dimension parameters
        self.final_dim = tuple(ida_aug_conf['final_dim'])  # (H, W) = (256, 704)
        self.original_H = ida_aug_conf['H']  # 900
        self.original_W = ida_aug_conf['W']  # 1600
        self.bot_pct_lim = ida_aug_conf.get('bot_pct_lim', (0.0, 0.0))
        
        # Use fixed resize ratio during inference
        fH, fW = self.final_dim
        H, W = self.original_H, self.original_W
        self.resize = max(fH / H, fW / W)
        
        # Pre-compute resized dimensions and crop parameters
        self.newW = int(W * self.resize)
        self.newH = int(H * self.resize)
        
        bot_pct = np.mean(self.bot_pct_lim)
        self.crop_h = int((1 - bot_pct) * self.newH) - fH
        self.crop_w = int(max(0, self.newW - fW) / 2)
        
        # Pre-compute pad parameters
        self.pad_h = int(np.ceil(fH / 32) * 32) - fH
        self.pad_w = int(np.ceil(fW / 32) * 32) - fW
    
    def __call__(
        self, 
        images: Union[List[np.ndarray], List[torch.Tensor]], 
        metadata: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute all image transforms on GPU.
        
        Args:
            images: Multi-view image list. Can be either:
                    - List[np.ndarray]: CPU numpy arrays, shape (H, W, 3), BGR format
                    - List[torch.Tensor]: GPU tensors, shape (H, W, 3), BGR format
                    When GPU tensors are provided, zero-copy is achieved.
            metadata: dict containing intrinsics, extrinsics, etc.
        
        Returns:
            Tuple[torch.Tensor, dict]: 
                - Transformed image tensor, shape (N, C, H, W), on GPU
                - Updated metadata
        """
        with torch.cuda.nvtx.range("GPU_Preprocess"):
            # 1. Convert to GPU tensor (handles both numpy and tensor inputs)
            with torch.cuda.nvtx.range("GPU_To_Tensor"):
                imgs = self._to_gpu_tensor(images)  # (N, C, H, W)
            
            # 2. Resize
            with torch.cuda.nvtx.range("GPU_Resize"):
                imgs = self._resize(imgs)
            
            # 3. Crop
            with torch.cuda.nvtx.range("GPU_Crop"):
                imgs = self._crop(imgs)
            
            # 4. BGR→RGB + Normalize
            with torch.cuda.nvtx.range("GPU_Normalize"):
                imgs = self._normalize(imgs)
            
            # 5. Pad
            with torch.cuda.nvtx.range("GPU_Pad"):
                imgs = self._pad(imgs)
            
            # 6. Update metadata
            with torch.cuda.nvtx.range("GPU_Update_Meta"):
                metadata = self._update_metadata(metadata, len(images))
        
        return imgs, metadata
    
    def _to_gpu_tensor(self, images: Union[List[np.ndarray], List[torch.Tensor]]) -> torch.Tensor:
        """
        Convert image list to GPU tensor (supports both numpy and tensor inputs).
        
        This method handles two input types:
        - List[np.ndarray]: Standard CPU numpy arrays (requires copy to GPU)
        - List[torch.Tensor]: GPU tensors (zero-copy if already on correct device)
        
        Args:
            images: Image list, each with shape (H, W, C), float32
        
        Returns:
            GPU tensor, shape (N, C, H, W)
        """
        # Check if inputs are already GPU tensors
        if isinstance(images[0], torch.Tensor):
            # Zero-copy path: stack GPU tensors directly
            with torch.cuda.nvtx.range("GPU_Stack_Tensors"):
                imgs = torch.stack(images, dim=0)  # (N, H, W, C)
                # Ensure on correct device (no-op if already there)
                if imgs.device != torch.device(self.device):
                    imgs = imgs.to(self.device, non_blocking=True)
                # NHWC -> NCHW
                imgs = imgs.permute(0, 3, 1, 2).contiguous()
                return imgs
        else:
            # Standard path: numpy arrays need to be copied to GPU
            with torch.cuda.nvtx.range("GPU_Copy_From_CPU"):
                imgs_np = np.stack(images, axis=0)  # (N, H, W, C)
                imgs = torch.from_numpy(imgs_np).to(self.device, non_blocking=True)
                # NHWC -> NCHW
                imgs = imgs.permute(0, 3, 1, 2).contiguous()
                return imgs
    
    def _resize(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Execute resize on GPU.
        
        Args:
            imgs: Input tensor with shape (N, C, H, W)
        
        Returns:
            Resized tensor
        """
        return F.interpolate(
            imgs, 
            size=(self.newH, self.newW), 
            mode='bilinear', 
            align_corners=False
        )
    
    def _crop(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Execute crop on GPU.
        
        Args:
            imgs: Input tensor with shape (N, C, H, W)
        
        Returns:
            Cropped tensor
        """
        fH, fW = self.final_dim
        top = self.crop_h
        left = self.crop_w
        return imgs[:, :, top:top+fH, left:left+fW].contiguous()
    
    def _normalize(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Execute BGR→RGB conversion and normalization on GPU.
        
        Args:
            imgs: Input tensor with shape (N, C, H, W), BGR format
        
        Returns:
            Normalized tensor in RGB format
        """
        # BGR -> RGB
        if self.to_rgb:
            imgs = imgs[:, [2, 1, 0], :, :]
        
        # Normalize: (img - mean) / std
        imgs = (imgs - self.mean) / self.std
        
        return imgs
    
    def _pad(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Execute padding on GPU to ensure dimensions are multiples of 32.
        
        Args:
            imgs: Input tensor with shape (N, C, H, W)
        
        Returns:
            Padded tensor
        """
        if self.pad_h > 0 or self.pad_w > 0:
            # F.pad padding format: (left, right, top, bottom)
            imgs = F.pad(imgs, (0, self.pad_w, 0, self.pad_h), mode='constant', value=0)
        return imgs
    
    def _update_metadata(self, metadata: Dict[str, Any], num_views: int) -> Dict[str, Any]:
        """
        Update intrinsics and lidar2img matrices.
        
        Args:
            metadata: Original metadata
            num_views: Number of views
        
        Returns:
            Updated metadata
        """
        fH, fW = self.final_dim
        
        intrinsics = metadata.get('intrinsics', [])
        new_intrinsics = []
        
        for i in range(len(intrinsics)):
            intr = intrinsics[i].copy()
            # Build IDA transformation matrix
            ida_mat = np.eye(3, dtype=np.float32)
            ida_mat[0, 0] = self.resize  # scale x
            ida_mat[1, 1] = self.resize  # scale y
            ida_mat[0, 2] = -self.crop_w  # translate x
            ida_mat[1, 2] = -self.crop_h  # translate y
            intr[:3, :3] = ida_mat @ intr[:3, :3]
            new_intrinsics.append(intr)
        
        if new_intrinsics:
            metadata['intrinsics'] = new_intrinsics
            # Recalculate lidar2img
            extrinsics = metadata.get('extrinsics', [])
            if extrinsics:
                metadata['lidar2img'] = [
                    new_intrinsics[i] @ extrinsics[i] 
                    for i in range(len(extrinsics))
                ]
        
        # Update shape information
        metadata['img_shape'] = [(fH, fW, 3) for _ in range(num_views)]
        
        return metadata
    
    def get_pad_shape(self) -> Tuple[int, int]:
        """
        Get image dimensions after padding.
        
        Returns:
            Tuple of (padded_height, padded_width)
        """
        fH, fW = self.final_dim
        pad_h = int(np.ceil(fH / 32) * 32)
        pad_w = int(np.ceil(fW / 32) * 32)
        return (pad_h, pad_w)

