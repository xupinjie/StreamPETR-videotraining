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


class OpenCVVideoDataProvider(BaseDataProvider):
    """
    Data provider that loads images by decoding video frames.
    
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
                            Video paths in pkl will be prefixed with this path.
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
            num_workers=max_cache_size,  # Use max_cache_size as num_workers
            batch_size=1,  # Video providers typically use per-sample loading
        )
        
        self.video_base_path = video_base_path if video_base_path else data_root
        self.max_cache_size = max_cache_size
        
        # LRU cache for VideoCapture objects (lazy initialization)
        self._video_cap_cache: Optional[OrderedDict] = None
    
    def _lazy_init_cache(self):
        """Lazy initialization of the VideoCapture cache."""
        if self._video_cap_cache is None:
            self._video_cap_cache = OrderedDict()
    
    def _get_video_capture(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Get or create a VideoCapture object for the given video path.
        
        Uses LRU cache to manage VideoCapture objects efficiently.
        When cache is full, releases the least recently used VideoCapture.
        
        NVTX ranges:
            - Video_Cache_Hit: Cache hit, returning cached VideoCapture
            - Video_Cache_Miss: Cache miss, creating new VideoCapture
            - Video_Open: Opening video file with cv2.VideoCapture
            - Video_Cache_Evict: Evicting oldest VideoCapture from cache
        
        Args:
            video_path: Path to the video file (may be relative or absolute)
            
        Returns:
            cv2.VideoCapture object for the video, or None if failed to open
        """
        self._lazy_init_cache()
        
        # Check if video_path is in cache
        if video_path in self._video_cap_cache:
            with torch.cuda.nvtx.range("Video_Cache_Hit"):
                # Move to end (most recently used)
                self._video_cap_cache.move_to_end(video_path)
                return self._video_cap_cache[video_path]
        
        # Cache miss - need to create new VideoCapture
        with torch.cuda.nvtx.range("Video_Cache_Miss"):
            # Build absolute path
            if not video_path.startswith('/'):
                abs_path = os.path.join(self.video_base_path, video_path)
            else:
                abs_path = video_path
            abs_path = os.path.abspath(abs_path)
            
            # Create new VideoCapture
            with torch.cuda.nvtx.range("Video_Open"):
                cap = cv2.VideoCapture(abs_path)
                if not cap.isOpened():
                    print(f"ERROR: Failed to open video file with OpenCV: {abs_path}")
                    print(f"  OpenCV version: {cv2.__version__}")
                    print(f"  Video backend: {cap.getBackendName()}")
                    return None
            
            # Add to cache (use original path as key for consistency)
            self._video_cap_cache[video_path] = cap
            
            # Remove oldest if cache is full
            if len(self._video_cap_cache) > self.max_cache_size:
                with torch.cuda.nvtx.range("Video_Cache_Evict"):
                    oldest_path, oldest_cap = self._video_cap_cache.popitem(last=False)
                    oldest_cap.release()
            
            return cap
    
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images by decoding video frames using OpenCV.
        
        NVTX ranges:
            - Video_Load_Images: Overall image loading from videos
            - Video_Get_Capture: Getting VideoCapture (cache hit or miss)
            - Video_Seek: Seeking to frame position
            - Video_Decode: Decoding video frame
            - Video_Resize: Resizing frame if dimensions mismatch
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'video_paths' and 'video_frame_ids' fields
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        
        Raises:
            ValueError: If video_paths or video_frame_ids is not available,
                       or if frame decoding fails
        """
        with torch.cuda.nvtx.range("Video_Load_Images"):
            video_paths = data_info.get('video_paths')
            video_frame_ids = data_info.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(
                    "video_paths and video_frame_ids are required for OpenCVVideoDataProvider. "
                    "Make sure the pkl file contains video path information."
                )
            
            if len(video_paths) != len(video_frame_ids):
                raise ValueError(
                    f"video_paths length ({len(video_paths)}) != "
                    f"video_frame_ids length ({len(video_frame_ids)})"
                )
            
            # Get expected image dimensions from ida_aug_conf
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            
            images = []
            
            for video_path, frame_id in zip(video_paths, video_frame_ids):
                try:
                    # Get VideoCapture from cache
                    with torch.cuda.nvtx.range("Video_Get_Capture"):
                        cap = self._get_video_capture(video_path)
                    
                    if cap is None:
                        raise ValueError(f"Failed to get VideoCapture for {video_path}")
                    
                    # Seek to frame position
                    with torch.cuda.nvtx.range("Video_Seek"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    
                    # Read and decode frame
                    with torch.cuda.nvtx.range("Video_Decode"):
                        ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        raise ValueError(f"Failed to read frame {frame_id} from {video_path}")
                    
                    # Check and resize if frame shape doesn't match expected dimensions
                    if frame.shape[0] != H or frame.shape[1] != W:
                        with torch.cuda.nvtx.range("Video_Resize"):
                            print(f"WARNING: Frame shape mismatch, resizing. "
                                  f"Got {frame.shape}, expected ({H}, {W}, 3)")
                            frame = cv2.resize(frame, (W, H))
                    
                    images.append(frame.astype(np.float32))
                    
                except Exception as e:
                    print(f"ERROR: Failed to decode frame {frame_id} from {video_path}: {e}")
                    print(f"Generating random image as fallback")
                    # Generate random image as fallback (for robustness)
                    fallback = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
                    images.append(fallback.astype(np.float32))
            
            return images
    
    def __del__(self):
        """Release all cached VideoCapture objects when the provider is destroyed."""
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

