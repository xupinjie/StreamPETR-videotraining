# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# ImageDataProvider - Load images from image files using cv2.imread
# ------------------------------------------------------------------------

import cv2
import numpy as np
from typing import List, Dict, Any

from .data_provider_base import BaseDataProvider


class ImageDataProvider(BaseDataProvider):
    """
    Data provider that loads images from image files.
    
    Uses cv2.imread() to read images from disk.
    This is the traditional approach for loading data from image files.
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
            data_root: Root directory of the data (used as prefix for relative paths)
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
    
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images from image files using cv2.imread.
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'img_filename' field for image paths
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        
        Raises:
            FileNotFoundError: If an image file cannot be loaded
        """
        images = []
        filenames = data_info['img_filename']
        
        for filename in filenames:
            # Handle relative path by prepending data_root
            if not filename.startswith('/'):
                filename = f"{self.data_root}/{filename}"
            
            img = cv2.imread(filename)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {filename}")
            
            images.append(img.astype(np.float32))
        
        return images

