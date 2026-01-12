# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# VideoDataProviderACCVStreamAsync - Async video decoder with batch prefetching
# ------------------------------------------------------------------------

import os
import torch
from typing import List, Dict, Any, Optional

from .data_provider_base import BaseDataProvider
from .batch_utils import (
    update_metadata_for_transform,
    build_img_metas,
    build_sample,
    collate_samples,
)

# Import NVC video decoder library
try:
    import accvlab.on_demand_video_decoder as nvc
    NVC_AVAILABLE = True
except ImportError:
    NVC_AVAILABLE = False


class VideoDataProviderACCVStreamAsync(BaseDataProvider):
    """
    Async video data provider with batch prefetching capability.
    
    Uses accvlab.on_demand_video_decoder library for GPU-accelerated async
    video decoding. While processing batch N, asynchronously prefetches
    batch N+1 to overlap decode and compute operations for higher throughput.
    
    Key Features:
        - Batch decoding: All frames (bs * num_cameras) decoded in one call
        - Async prefetch: Next batch is decoded while current batch is processed
        - Zero-copy: Decoded frames stay on GPU
    
    Usage Pattern:
        for i, batch_indices in enumerate(all_batches):
            # Prefetch next batch (non-blocking)
            if i + 1 < len(all_batches):
                provider.prefetch_batch(all_batches[i + 1])
            
            # Get current batch (uses prefetched data if available)
            batch = provider.get_batch(batch_indices)
            
            # Process batch...
    
    Requirements:
        - accvlab.on_demand_video_decoder library must be installed
        - CUDA-capable GPU
        - gpu_preprocess=True recommended for full zero-copy benefit
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
        video_base_path: Optional[str] = None,
        num_cameras: int = 6,
        batch_size: int = 1,
        gpu_preprocess: bool = False,
        device: str = 'cuda:0',
    ):
        """
        Initialize the async video data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            video_base_path: Base path for video files. If None, uses data_root.
            num_cameras: Number of cameras per sample. Default is 6 for nuScenes.
            batch_size: Batch size for inference. Used for decoder initialization.
            gpu_preprocess: Whether to use GPU for image preprocessing.
            device: GPU device for decoding and preprocessing
        """
        if not NVC_AVAILABLE:
            raise ImportError(
                "accvlab.on_demand_video_decoder is not available. "
                "Please install the NVC library to use VideoDataProviderACCVStreamAsync."
            )
        
        super().__init__(
            pkl_path=pkl_path,
            data_root=data_root,
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
            gpu_preprocess=gpu_preprocess,
            device=device,
        )
        
        self.video_base_path = video_base_path if video_base_path else data_root
        self.num_cameras = num_cameras
        self.batch_size = batch_size
        
        # Parse GPU ID from device string
        if device.startswith('cuda:'):
            self._gpu_id = int(device.split(':')[1])
        elif device == 'cuda':
            self._gpu_id = 0
        else:
            self._gpu_id = 0
        
        # NVC async decoder (lazy initialization)
        self._nvc_decoder = None
        
        # Prefetch state for batch-level async decoding
        self._prefetch_state = {
            'pending': False,
            'indices': None,           # Tuple of indices for comparison
            'all_paths': None,         # Flattened absolute paths
            'all_frame_ids': None,     # Flattened frame IDs
            'image_counts': None,      # Images per sample
            'all_metadata': None,      # Metadata list
        }
    
    def _lazy_init_decoder(self):
        """
        Lazy initialization of the NVC async decoder.
        
        Initializes decoder with capacity for batch_size * num_cameras files.
        """
        if self._nvc_decoder is None:
            with torch.cuda.nvtx.range("NVC_Async_Decoder_Init"):
                gpu_id = int(os.environ.get("LOCAL_RANK", self._gpu_id))
                
                self._nvc_decoder = nvc.CreateSampleReader(
                    num_of_set=1,
                    num_of_file=self.batch_size * self.num_cameras,
                    iGpu=gpu_id,
                )
    
    def _build_absolute_paths(self, video_paths: List[str]) -> List[str]:
        """
        Build absolute paths for video files.
        
        Args:
            video_paths: List of video paths (may be relative or absolute)
            
        Returns:
            List of absolute video paths
        """
        abs_paths = []
        for video_path in video_paths:
            if not video_path.startswith('/'):
                abs_path = os.path.join(self.video_base_path, video_path)
            else:
                abs_path = video_path
            abs_paths.append(os.path.abspath(abs_path))
        return abs_paths
    
    def _collect_batch_info(self, indices: List[int]):
        """
        Collect all video paths and frame IDs for a batch.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Tuple of (all_metadata, all_paths, all_frame_ids, image_counts)
        """
        all_metadata = []
        all_paths = []
        all_frame_ids = []
        image_counts = []
        
        for idx in indices:
            metadata = self.data_index.get_data_info(idx)
            all_metadata.append(metadata)
            
            video_paths = metadata.get('video_paths')
            video_frame_ids = metadata.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(f"video_paths and video_frame_ids required for sample {idx}")
            
            all_paths.extend(video_paths)
            all_frame_ids.extend(video_frame_ids)
            image_counts.append(len(video_paths))
        
        abs_paths = self._build_absolute_paths(all_paths)
        return all_metadata, abs_paths, all_frame_ids, image_counts
    
    def prefetch_batch(self, indices: List[int]):
        """
        Start async prefetch for the given batch indices.
        
        This method initiates asynchronous decoding for the next batch
        without blocking. Call this before get_batch() of current batch
        to overlap decode with compute.
        
        Args:
            indices: List of sample indices to prefetch
        """
        self._lazy_init_decoder()
        
        with torch.cuda.nvtx.range("Async_Prefetch_Start"):
            try:
                all_metadata, abs_paths, all_frame_ids, image_counts = \
                    self._collect_batch_info(indices)
                
                # Start async decode (non-blocking)
                self._nvc_decoder.DecodeN12ToRGBAsync(
                    abs_paths,
                    all_frame_ids,
                    True  # RGB format
                )
                
                # Update prefetch state
                self._prefetch_state = {
                    'pending': True,
                    'indices': tuple(indices),
                    'all_paths': abs_paths,
                    'all_frame_ids': all_frame_ids,
                    'image_counts': image_counts,
                    'all_metadata': all_metadata,
                }
                
            except Exception as e:
                print(f"WARNING: Failed to start prefetch: {e}")
                self._prefetch_state['pending'] = False
    
    def reset_prefetch_state(self):
        """
        Reset prefetch state.
        
        Call this when:
        - Starting a new iteration from the beginning
        - Switching to random access pattern
        - After scene changes
        """
        self._prefetch_state = {
            'pending': False,
            'indices': None,
            'all_paths': None,
            'all_frame_ids': None,
            'image_counts': None,
            'all_metadata': None,
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples, using prefetched data if available.
        
        If prefetch_batch() was called with matching indices, retrieves
        the already-decoded frames. Otherwise, falls back to sync decode.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary with collated samples
        """
        self._lazy_init_decoder()
        
        H = self.ida_aug_conf.get('H', 900)
        W = self.ida_aug_conf.get('W', 1600)
        
        # Check if prefetch matches current request
        prefetch_hit = (
            self._prefetch_state['pending'] and
            self._prefetch_state['indices'] == tuple(indices)
        )
        
        if prefetch_hit:
            # Use prefetched data
            with torch.cuda.nvtx.range("Async_Get_Prefetched"):
                try:
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGBAsyncGetBuffer(
                        self._prefetch_state['all_paths'],
                        self._prefetch_state['all_frame_ids'],
                        True
                    )
                    all_metadata = self._prefetch_state['all_metadata']
                    image_counts = self._prefetch_state['image_counts']
                except Exception as e:
                    print(f"ERROR: Failed to get prefetched buffer: {e}")
                    decoded_frames = None
                finally:
                    self._prefetch_state['pending'] = False
            
            # Fallback if get buffer failed
            if decoded_frames is None:
                prefetch_hit = False
        
        if not prefetch_hit:
            # Sync decode (first batch or prefetch miss)
            with torch.cuda.nvtx.range("Async_Sync_Decode"):
                all_metadata, abs_paths, all_frame_ids, image_counts = \
                    self._collect_batch_info(indices)
                
                try:
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGB(
                        abs_paths,
                        all_frame_ids,
                        True
                    )
                except Exception as e:
                    print(f"ERROR: NVC DecodeN12ToRGB failed: {e}")
                    decoded_frames = [
                        torch.randint(0, 256, (H, W, 3), dtype=torch.uint8,
                                      device=f'cuda:{self._gpu_id}')
                        for _ in abs_paths
                    ]
        
        # Convert RGBFrame to torch tensors with deep copy
        # Deep copy is required because decode buffer will be reused
        with torch.cuda.nvtx.range("Convert_And_Clone"):
            gpu_images = []
            for frame in decoded_frames:
                try:
                    tensor = torch.tensor(frame, dtype=torch.uint8).to(
                        device=f'cuda:{self._gpu_id}', dtype=torch.float32, non_blocking=True
                    ).clone()  # Deep copy for async safety
                    gpu_images.append(tensor)
                except Exception as e:
                    print(f"ERROR: Failed to convert frame: {e}")
                    fallback = torch.randint(0, 256, (H, W, 3), dtype=torch.float32,
                                             device=f'cuda:{self._gpu_id}')
                    gpu_images.append(fallback)
        
        # Batch transform
        with torch.cuda.nvtx.range("Batch_Transform"):
            if self.gpu_preprocess:
                # GPU mode: Transform ALL images at once
                with torch.cuda.nvtx.range("GPU_Transform_All"):
                    all_transformed, _ = self.transforms(gpu_images, all_metadata[0])
                
                all_img_tensors = []
                all_updated_metadata = []
                
                img_idx = 0
                for i, metadata in enumerate(all_metadata):
                    num_imgs = image_counts[i]
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
                # CPU mode fallback
                import numpy as np
                from .batch_utils import images_to_tensor
                
                all_img_tensors = []
                all_updated_metadata = []
                
                all_images_np = [img.cpu().numpy() for img in gpu_images]
                
                img_idx = 0
                for i, metadata in enumerate(all_metadata):
                    num_imgs = image_counts[i]
                    sample_images = all_images_np[img_idx:img_idx + num_imgs]
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
        
        # Build samples
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
        
        # Collate
        with torch.cuda.nvtx.range("Collate"):
            return collate_samples(samples)
    
    def supports_prefetch(self) -> bool:
        """Check if this provider supports async prefetching."""
        return True
    
    def prefetch_next_batch(self) -> bool:
        """
        Prefetch the next batch based on sampler state.
        
        Uses the sampler to peek at next frame's indices and starts
        async decode. Call this after get_multi_video_batch() to overlap
        decode with inference.
        
        Returns:
            True if prefetch was initiated, False otherwise
        """
        if self._sampler is None:
            return False
        
        # Peek at next batch indices without advancing
        next_indices = self._sampler.peek_next_indices()
        if next_indices is None:
            return False
        
        self.prefetch_batch(next_indices)
        return self._prefetch_state['pending']
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"video_base_path={self.video_base_path}, "
            f"num_cameras={self.num_cameras}, "
            f"batch_size={self.batch_size}, "
            f"gpu_id={self._gpu_id}, "
            f"gpu_preprocess={self.gpu_preprocess}, "
            f"prefetch_pending={self._prefetch_state['pending']})"
        )
