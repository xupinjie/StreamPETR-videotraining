# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# OpenCVVideoDataProvider - Load images by decoding video frames using OpenCV
# ------------------------------------------------------------------------

import os
import cv2
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Dict, Any, Optional

from .data_provider_base import BaseDataProvider
from .batch_utils import (
    update_metadata_for_transform,
    build_img_metas,
    build_sample,
    collate_samples,
    images_to_tensor,
)


class OpenCVVideoDataProvider(BaseDataProvider):
    """
    Data provider that loads images by decoding video frames using OpenCV.
    
    Uses OpenCV VideoCapture to seek and decode frames from video files.
    Implements LRU cache for VideoCapture objects to avoid frequently
    opening and closing video files.
    
    This approach is useful when:
        - Data is stored as video files instead of individual images
        - Video paths and frame IDs are available in the pkl metadata
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
        video_base_path: Optional[str] = None,
        max_cache_size: int = 6,
        gpu_preprocess: bool = False,
        device: str = 'cuda:0',
    ):
        """
        Initialize the OpenCV video data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            video_base_path: Base path for video files. If None, uses data_root.
            max_cache_size: Maximum number of VideoCapture objects to cache.
                           Default is 6 (one per camera for nuScenes).
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
        
        self.video_base_path = video_base_path if video_base_path else data_root
        self.max_cache_size = max_cache_size
        
        # LRU cache for VideoCapture objects (lazy initialization)
        self._video_cap_cache: Optional[OrderedDict] = None
    
    def _lazy_init_cache(self):
        """Lazy initialization of the VideoCapture cache."""
        if self._video_cap_cache is None:
            self._video_cap_cache = OrderedDict()
    
    def _get_video_path(self, video_path: str) -> str:
        """
        Get absolute path for a video file.
        
        Args:
            video_path: Video path (may be relative or absolute)
            
        Returns:
            Absolute path to the video file
        """
        if not video_path.startswith('/'):
            return os.path.abspath(os.path.join(self.video_base_path, video_path))
        return video_path
    
    def _get_video_capture(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Get or create a VideoCapture object for the given video path.
        
        Uses LRU cache to manage VideoCapture objects efficiently.
        
        Args:
            video_path: Path to the video file (may be relative or absolute)
            
        Returns:
            cv2.VideoCapture object for the video, or None if failed to open
        """
        self._lazy_init_cache()
        
        # Check if video_path is in cache
        if video_path in self._video_cap_cache:
            with torch.cuda.nvtx.range("Video_Cache_Hit"):
                self._video_cap_cache.move_to_end(video_path)
                return self._video_cap_cache[video_path]
        
        # Cache miss - create new VideoCapture
        with torch.cuda.nvtx.range("Video_Cache_Miss"):
            abs_path = self._get_video_path(video_path)
            
            with torch.cuda.nvtx.range("Video_Open"):
                cap = cv2.VideoCapture(abs_path)
                if not cap.isOpened():
                    print(f"ERROR: Failed to open video: {abs_path}")
                    return None
            
            self._video_cap_cache[video_path] = cap
            
            # Evict oldest if cache is full
            if len(self._video_cap_cache) > self.max_cache_size:
                with torch.cuda.nvtx.range("Video_Cache_Evict"):
                    _, oldest_cap = self._video_cap_cache.popitem(last=False)
                    oldest_cap.release()
            
            return cap
    
    def _decode_frame(self, video_path: str, frame_id: int) -> np.ndarray:
        """
        Decode a single frame from a video file.
        
        Args:
            video_path: Path to the video file
            frame_id: Frame index to decode
            
        Returns:
            Decoded frame as numpy array, BGR format, dtype float32
        """
        H = self.ida_aug_conf.get('H', 900)
        W = self.ida_aug_conf.get('W', 1600)
        
        try:
            cap = self._get_video_capture(video_path)
            if cap is None:
                raise ValueError(f"Failed to get VideoCapture for {video_path}")
            
            # Seek to frame
            with torch.cuda.nvtx.range("Video_Seek"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            
            # Decode frame
            with torch.cuda.nvtx.range("Video_Decode"):
                ret, frame = cap.read()
            
            if not ret or frame is None:
                raise ValueError(f"Failed to read frame {frame_id} from {video_path}")
            
            # Resize if needed
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            
            return frame.astype(np.float32)
            
        except Exception as e:
            print(f"ERROR: Failed to decode frame {frame_id} from {video_path}: {e}")
            print("Generating random image as fallback")
            return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8).astype(np.float32)
    
    def _load_sample_images(self, metadata: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images for a single sample by decoding video frames.
        
        Args:
            metadata: Sample metadata with 'video_paths' and 'video_frame_ids'
            
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        """
        video_paths = metadata.get('video_paths')
        video_frame_ids = metadata.get('video_frame_ids')
        
        if video_paths is None or video_frame_ids is None:
            raise ValueError(
                "video_paths and video_frame_ids are required for OpenCVVideoDataProvider. "
                "Make sure the pkl file contains video path information."
            )
        
        images = []
        for video_path, frame_id in zip(video_paths, video_frame_ids):
            frame = self._decode_frame(video_path, frame_id)
            images.append(frame)
        
        return images
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples by decoding video frames.
        
        For video-based providers, images are loaded per-sample since
        VideoCapture requires sequential access within each video file.
        GPU preprocessing is still batched across all images for efficiency.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary
        """
        # Step 1: Collect all metadata and load images
        all_metadata = []
        all_images = []
        image_counts = []
        
        with torch.cuda.nvtx.range("Load_All_Frames"):
            for idx in indices:
                with torch.cuda.nvtx.range("Get_Metadata"):
                    metadata = self.data_index.get_data_info(idx)
                    all_metadata.append(metadata)
                
                with torch.cuda.nvtx.range("Decode_Sample_Frames"):
                    images = self._load_sample_images(metadata)
                    all_images.extend(images)
                    image_counts.append(len(images))
        
        # Step 2: Batch transform
        with torch.cuda.nvtx.range("Batch_Transform"):
            if self.gpu_preprocess:
                # Convert to GPU tensors
                gpu_images = [
                    torch.from_numpy(img).to(self.device, non_blocking=True)
                    for img in all_images
                ]
                
                # GPU mode: Transform ALL images at once
                with torch.cuda.nvtx.range("GPU_Transform_All"):
                    all_transformed, _ = self.transforms(gpu_images, all_metadata[0])
                
                # Split and build samples
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
                # CPU mode: Transform per sample
                all_img_tensors = []
                all_updated_metadata = []
                
                img_idx = 0
                for i, metadata in enumerate(all_metadata):
                    num_imgs = image_counts[i]
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
        
        # Step 3: Build samples
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
        
        # Step 4: Collate
        with torch.cuda.nvtx.range("Collate"):
            return collate_samples(samples)
    
    def __del__(self):
        """Release all cached VideoCapture objects."""
        if hasattr(self, '_video_cap_cache') and self._video_cap_cache is not None:
            for cap in self._video_cap_cache.values():
                if cap is not None:
                    cap.release()
            self._video_cap_cache.clear()
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"video_base_path={self.video_base_path}, "
            f"max_cache_size={self.max_cache_size}, "
            f"gpu_preprocess={self.gpu_preprocess})"
        )
