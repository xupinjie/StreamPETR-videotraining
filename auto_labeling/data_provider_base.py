# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# BaseDataProvider - Lightweight abstract base class for data providers
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

from .data_index import DataIndex
from .multi_video_sampler import MultiVideoSampler
from .image_transforms import ImageTransforms
from .gpu_image_transforms import GPUImageTransforms


class BaseDataProvider(ABC):
    """
    Lightweight abstract base class for data providers.
    
    Provides:
    - Shared initialization (data index, transforms)
    - Abstract get_batch() interface that subclasses must implement
    - Multi-video sampling support
    
    Each subclass implements its own get_batch() method with the optimal
    loading strategy for its data source (image files, video files, etc.).
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
        Initialize the data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device (only used when gpu_preprocess=True)
        """
        # Data index
        self.data_index = DataIndex(pkl_path, data_root)
        
        # Store config
        self.img_norm_cfg = img_norm_cfg
        self.ida_aug_conf = ida_aug_conf
        self.gpu_preprocess = gpu_preprocess
        self.device = device
        
        # Initialize transforms
        if gpu_preprocess:
            self.transforms = GPUImageTransforms(img_norm_cfg, ida_aug_conf, device=device)
        else:
            self.transforms = ImageTransforms(img_norm_cfg, ida_aug_conf)
        
        # Multi-video sampler (initialized when needed)
        self._sampler: Optional[MultiVideoSampler] = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_index)
    
    @abstractmethod
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples.
        
        Each subclass must implement this method with its own optimal
        loading and preprocessing strategy.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary with keys:
                - img: Tensor (B, N, C, H, W)
                - img_metas: List of metadata dicts
                - lidar2img, intrinsics, extrinsics: Tensors (B, N, 4, 4)
                - timestamp: Tensor (B,)
                - img_timestamp: Tensor (B, N)
                - ego_pose, ego_pose_inv: Tensors (B, 4, 4)
        """
        pass
    
    # =========================================================================
    # Multi-video sampling support
    # =========================================================================
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        Get information about videos in the dataset.
        
        Returns:
            Dict containing num_videos, total_frames, video_lengths
        """
        return self.data_index.get_video_info()
    
    def create_multi_video_sampler(self, batch_size: int) -> MultiVideoSampler:
        """
        Create a multi-video sampler for parallel video processing.
        
        Args:
            batch_size: Number of parallel video streams
            
        Returns:
            MultiVideoSampler instance
        """
        self._sampler = MultiVideoSampler(self.data_index.video_indices, batch_size)
        return self._sampler
    
    def get_multi_video_batch(self) -> Tuple[Dict[str, Any], List[bool]]:
        """
        Get a batch of samples for multi-video parallel inference.
        
        Returns:
            Tuple of (batch_data, is_first_frame_flags)
        """
        if self._sampler is None:
            raise RuntimeError("Call create_multi_video_sampler() first.")
        
        indices, is_first_frame = self._sampler.get_batch_indices()
        batch_data = self.get_batch(indices)
        
        return batch_data, is_first_frame
    
    def advance_multi_video_frame(self) -> bool:
        """
        Advance to next frame for all video streams.
        
        Returns:
            True if there are more frames, False if exhausted
        """
        if self._sampler is None:
            raise RuntimeError("Call create_multi_video_sampler() first.")
        
        return self._sampler.advance_frame()
    
    # =========================================================================
    # Convenience methods
    # =========================================================================
    
    def iterate_batches(self, batch_size: int = 1):
        """
        Iterate and generate batches.
        
        Args:
            batch_size: Batch size
        
        Yields:
            Batch data dictionary
        """
        num_samples = len(self)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            indices = list(range(start_idx, end_idx))
            yield self.get_batch(indices)
