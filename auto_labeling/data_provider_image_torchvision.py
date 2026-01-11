# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# TorchvisionDataProvider - Load images using torchvision with parallel I/O
# ------------------------------------------------------------------------

import numpy as np
import torch
from typing import List, Dict, Any

from .data_provider_base import BaseDataProvider

# Import torchvision for image loading
try:
    from torchvision.io import read_image, ImageReadMode
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class TorchvisionDataProvider(BaseDataProvider):
    """
    Data provider that loads images using torchvision.io with parallel I/O.
    
    Uses torchvision.io.read_image() which internally uses libjpeg-turbo
    for JPEG decoding. The key advantage is that torchvision releases GIL
    during image reading, allowing true parallel I/O and decoding when
    combined with ThreadPoolExecutor.
    
    Benefits over cv2.imread:
        - Uses libjpeg-turbo (2-4x faster JPEG decoding)
        - Releases GIL during read, enabling true parallelism
        - Can load directly to GPU tensor (zero-copy path available)
    
    Requirements:
        - torchvision >= 0.8.0 (for torchvision.io.read_image)
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
        gpu_preprocess: bool = False,
        device: str = 'cuda:0',
        num_workers: int = 6,
        batch_size: int = 1,
    ):
        """
        Initialize the torchvision data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data (used as prefix for relative paths)
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device (only used when gpu_preprocess=True)
            num_workers: Number of parallel workers for image loading per sample.
                        Default is 6 (one per camera for nuScenes).
                        Total workers = num_workers * batch_size for batch loading.
            batch_size: Expected batch size. Used to scale thread pool for
                       optimal parallel loading. Total threads = num_workers * batch_size.
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision.io is not available. "
                "Please install torchvision >= 0.8.0 to use TorchvisionDataProvider."
            )
        
        super().__init__(
            pkl_path=pkl_path,
            data_root=data_root,
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
            gpu_preprocess=gpu_preprocess,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        
        print(f"TorchvisionDataProvider initialized with ThreadPool "
              f"({num_workers * batch_size} workers = "
              f"batch_size={batch_size} x num_workers={num_workers})")
    
    def load_single_image(self, filepath: str) -> np.ndarray:
        """
        Load a single image using torchvision and return as numpy array.
        
        This method is designed to be called in parallel via ThreadPoolExecutor.
        torchvision.io.read_image releases GIL during execution, enabling true
        parallel I/O and JPEG decoding across multiple threads.
        
        Args:
            filepath: Full path to the image file
            
        Returns:
            Image as numpy array, BGR format, dtype float32, shape (H, W, 3)
            
        Raises:
            FileNotFoundError: If the image file cannot be loaded
        """
        try:
            # read_image returns tensor in (C, H, W) format, RGB order
            img_tensor = read_image(filepath, mode=ImageReadMode.RGB)
            
            # Convert to numpy: (C, H, W) -> (H, W, C)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            # Convert RGB to BGR (to match cv2.imread behavior)
            img_bgr = img_np[:, :, ::-1].copy()
            
            return img_bgr.astype(np.float32)
            
        except Exception as e:
            raise FileNotFoundError(f"Cannot load image: {filepath}. Error: {e}")
    
    def load_single_image_to_gpu(self, filepath: str) -> torch.Tensor:
        """
        Load a single image and return as GPU tensor.
        
        This method overrides the base class implementation to provide
        optimized loading with direct GPU transfer using torchvision.
        
        Args:
            filepath: Full path to the image file
            
        Returns:
            Image as GPU tensor, BGR format, dtype float32, shape (H, W, 3)
        """
        try:
            # read_image returns tensor in (C, H, W) format, RGB order
            img_tensor = read_image(filepath, mode=ImageReadMode.RGB)
            
            # Convert to (H, W, C) and BGR order
            img_hwc = img_tensor.permute(1, 2, 0)
            img_bgr = img_hwc.flip(dims=[2])  # RGB -> BGR
            
            # Move to GPU and convert to float32
            img_gpu = img_bgr.to(device=self.device, dtype=torch.float32, non_blocking=True)
            
            return img_gpu
            
        except Exception as e:
            # Return a fallback tensor on error
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            print(f"ERROR: Cannot load image: {filepath}. Error: {e}")
            print("Generating random image as fallback")
            return torch.randint(
                0, 256, (H, W, 3),
                dtype=torch.float32,
                device=self.device
            )
    
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images using torchvision with parallel I/O.
        
        This method is kept for backward compatibility and direct usage.
        The optimized get_batch() in BaseDataProvider uses load_single_image()
        with ThreadPoolExecutor for parallel loading.
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'img_filename' field for image paths
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        """
        filenames = data_info['img_filename']
        full_paths = [self._get_full_path(f) for f in filenames]
        
        # Use thread pool for parallel loading
        images = list(self._thread_pool.map(self.load_single_image, full_paths))
        return images
    
    def load_images_to_gpu(self, data_info: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Load multi-view images directly to GPU tensors with parallel I/O.
        
        This method is kept for backward compatibility and direct usage.
        The optimized get_batch() in BaseDataProvider uses load_single_image_to_gpu()
        with ThreadPoolExecutor for parallel loading.
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'img_filename' field for image paths
        
        Returns:
            List of GPU tensors, BGR format, shape (H, W, 3), dtype float32
        """
        filenames = data_info['img_filename']
        full_paths = [self._get_full_path(f) for f in filenames]
        
        # Use thread pool for parallel loading to GPU
        gpu_images = list(self._thread_pool.map(self.load_single_image_to_gpu, full_paths))
        return gpu_images
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"data_root={self.data_root}, "
            f"num_workers={self.num_workers}, "
            f"batch_size={self.batch_size}, "
            f"gpu_preprocess={self.gpu_preprocess})"
        )
