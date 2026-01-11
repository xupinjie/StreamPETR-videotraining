# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# TorchvisionDataProvider - Load images using torchvision with parallel I/O
# ------------------------------------------------------------------------

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from .data_provider_base import BaseDataProvider
from .batch_utils import (
    update_metadata_for_transform,
    build_img_metas,
    build_sample,
    collate_samples,
    images_to_tensor,
)

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
    
    Key features:
        - Uses libjpeg-turbo (2-4x faster JPEG decoding)
        - Releases GIL during read, enabling true parallelism
        - Parallel loading across all samples in batch (batch_size * num_cameras)
        - Single GPU preprocessing call for all images in batch
    
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
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device (only used when gpu_preprocess=True)
            num_workers: Number of parallel workers for image loading per sample.
                        Default is 6 (one per camera for nuScenes).
            batch_size: Expected batch size. Used to scale thread pool.
                       Total threads = num_workers * batch_size.
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
        )
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Thread pool for parallel image loading
        total_workers = num_workers * batch_size
        self._thread_pool = ThreadPoolExecutor(max_workers=total_workers)
        
        print(f"TorchvisionDataProvider initialized: "
              f"ThreadPool({total_workers} workers = "
              f"batch_size={batch_size} x num_workers={num_workers})")
    
    def _load_image_cpu(self, filepath: str) -> np.ndarray:
        """
        Load a single image using torchvision and return as numpy array.
        
        Releases GIL during execution, enabling true parallel I/O and decoding.
        
        Args:
            filepath: Full path to the image file
            
        Returns:
            Image as numpy array, BGR format, dtype float32, shape (H, W, 3)
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
    
    def _load_image_gpu(self, filepath: str) -> torch.Tensor:
        """
        Load a single image and return as GPU tensor.
        
        Releases GIL during execution, enabling true parallel I/O and decoding.
        
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
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch with parallel loading and batch GPU preprocessing.
        
        Optimized data flow:
        1. Batch metadata collection
        2. Parallel image loading (batch_size * num_cameras images)
        3. Single GPU preprocessing call (one kernel launch for all images)
        4. Split and assemble results
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary
        """
        # Step 1: Collect all metadata
        with torch.cuda.nvtx.range("Get_All_Metadata"):
            all_metadata = self.data_index.get_batch_data_info(indices)
        
        # Step 2: Collect all image paths
        all_paths = []
        path_counts = []
        for metadata in all_metadata:
            filenames = metadata['img_filename']
            paths = [self.data_index.get_full_path(f) for f in filenames]
            all_paths.extend(paths)
            path_counts.append(len(paths))
        
        # Step 3: Parallel load ALL images (batch_size * num_cameras)
        with torch.cuda.nvtx.range("Parallel_Load_All_Images"):
            if self.gpu_preprocess:
                all_images = list(self._thread_pool.map(self._load_image_gpu, all_paths))
            else:
                all_images = list(self._thread_pool.map(self._load_image_cpu, all_paths))
        
        # Step 4: Batch transform
        with torch.cuda.nvtx.range("Batch_Transform"):
            if self.gpu_preprocess:
                # GPU mode: Transform ALL images at once (single kernel launch)
                with torch.cuda.nvtx.range("GPU_Transform_All"):
                    all_transformed, _ = self.transforms(all_images, all_metadata[0])
                    # all_transformed shape: (total_images, C, H, W)
                
                # Split and build samples
                all_img_tensors = []
                all_updated_metadata = []
                
                img_idx = 0
                for i, metadata in enumerate(all_metadata):
                    num_imgs = path_counts[i]
                    sample_tensor = all_transformed[img_idx:img_idx + num_imgs]
                    img_idx += num_imgs
                    
                    updated_meta = update_metadata_for_transform(
                        metadata,
                        resize=self.transforms.resize,
                        crop_h=self.transforms.crop_h,
                        crop_w=self.transforms.crop_w,
                        final_dim=self.transforms.final_dim,
                    )
                    all_img_tensors.append(sample_tensor)
                    all_updated_metadata.append(updated_meta)
            else:
                # CPU mode: Transform per sample
                all_img_tensors = []
                all_updated_metadata = []
                
                img_idx = 0
                for i, metadata in enumerate(all_metadata):
                    num_imgs = path_counts[i]
                    sample_images = all_images[img_idx:img_idx + num_imgs]
                    img_idx += num_imgs
                    
                    images, _ = self.transforms(sample_images, metadata)
                    img_tensor = images_to_tensor(images)
                    
                    updated_meta = update_metadata_for_transform(
                        metadata,
                        resize=self.transforms.resize,
                        crop_h=self.transforms.crop_h,
                        crop_w=self.transforms.crop_w,
                        final_dim=self.transforms.final_dim,
                    )
                    all_img_tensors.append(img_tensor)
                    all_updated_metadata.append(updated_meta)
        
        # Step 5: Build samples
        with torch.cuda.nvtx.range("Build_Samples"):
            pad_shape = self.transforms.get_pad_shape()
            samples = []
            
            for img_tensor, metadata in zip(all_img_tensors, all_updated_metadata):
                num_views = img_tensor.shape[0]
                img_metas = build_img_metas(
                    metadata, pad_shape, self.img_norm_cfg,
                    self.ida_aug_conf, num_views
                )
                sample = build_sample(img_tensor, metadata, img_metas)
                samples.append(sample)
        
        # Step 6: Collate
        with torch.cuda.nvtx.range("Collate"):
            return collate_samples(samples)
    
    def __del__(self):
        """Clean up thread pool on deletion."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"num_workers={self.num_workers}, "
            f"batch_size={self.batch_size}, "
            f"gpu_preprocess={self.gpu_preprocess})"
        )
