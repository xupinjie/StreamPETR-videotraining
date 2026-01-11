# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# ImageDataProvider - Load images from image files using cv2.imread
# ------------------------------------------------------------------------

import cv2
import numpy as np
import torch
from typing import List, Dict, Any

from .data_provider_base import BaseDataProvider
from .batch_utils import (
    update_metadata_for_transform,
    build_img_metas,
    build_sample,
    collate_samples,
    images_to_tensor,
)


class ImageDataProvider(BaseDataProvider):
    """
    Data provider that loads images from image files using cv2.imread.
    
    Simple serial implementation - serves as reference implementation.
    For high-performance image loading, use TorchvisionDataProvider instead.
    
    Note: cv2.imread does NOT release GIL, so parallel loading has limited benefit.
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
        gpu_preprocess: bool = False,
        device: str = 'cuda:0',
    ):
        """
        Initialize the image data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device (only used when gpu_preprocess=True)
        """
        super().__init__(
            pkl_path=pkl_path,
            data_root=data_root,
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
            gpu_preprocess=gpu_preprocess,
            device=device,
        )
    
    def _load_images(self, metadata: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images from image files.
        
        Args:
            metadata: Sample metadata with 'img_filename' field
            
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        """
        images = []
        for filename in metadata['img_filename']:
            path = self.data_index.get_full_path(filename)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {path}")
            images.append(img.astype(np.float32))
        return images
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples with serial loading.
        
        Implements simple serial loading and per-sample processing.
        Each sample is loaded and transformed independently.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary
        """
        samples = []
        pad_shape = self.transforms.get_pad_shape()
        
        for idx in indices:
            with torch.cuda.nvtx.range("Process_Sample"):
                # 1. Get metadata
                with torch.cuda.nvtx.range("Get_Metadata"):
                    metadata = self.data_index.get_data_info(idx)
                
                # 2. Load images
                with torch.cuda.nvtx.range("Load_Images"):
                    images = self._load_images(metadata)
                
                # 3. Transform
                with torch.cuda.nvtx.range("Transform"):
                    if self.gpu_preprocess:
                        # Convert to GPU tensors
                        gpu_images = [
                            torch.from_numpy(img).to(self.device, non_blocking=True)
                            for img in images
                        ]
                        img_tensor, _ = self.transforms(gpu_images, metadata)
                    else:
                        images, _ = self.transforms(images, metadata)
                        img_tensor = images_to_tensor(images)
                
                # 4. Update metadata for transform
                with torch.cuda.nvtx.range("Update_Metadata"):
                    updated_meta = update_metadata_for_transform(
                        metadata,
                        resize=self.transforms.resize,
                        crop_h=self.transforms.crop_h,
                        crop_w=self.transforms.crop_w,
                        final_dim=self.transforms.final_dim,
                    )
                
                # 5. Build sample
                num_views = img_tensor.shape[0]
                img_metas = build_img_metas(
                    updated_meta, pad_shape, self.img_norm_cfg, 
                    self.ida_aug_conf, num_views
                )
                sample = build_sample(img_tensor, updated_meta, img_metas)
                samples.append(sample)
        
        # 6. Collate
        with torch.cuda.nvtx.range("Collate"):
            return collate_samples(samples)
