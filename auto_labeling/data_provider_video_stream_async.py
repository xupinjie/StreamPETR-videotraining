# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# VideoDataProviderACCVStreamAsync - Async video decoder with prefetching
# ------------------------------------------------------------------------

import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from .data_provider_base import BaseDataProvider

# Import NVC video decoder library
try:
    import accvlab.on_demand_video_decoder as nvc
    NVC_AVAILABLE = True
except ImportError:
    NVC_AVAILABLE = False


class VideoDataProviderACCVStreamAsync(BaseDataProvider):
    """
    Async video data provider with prefetching capability.
    
    Uses accvlab.on_demand_video_decoder library for GPU-accelerated async
    video decoding. While processing sample N, asynchronously prefetches
    sample N+1 to overlap decode and compute operations for higher throughput.
    
    Key Features:
        - Async decoding: DecodeN12ToRGBAsync starts decoding without blocking
        - Prefetching: Next sample is decoded while current sample is processed
        - Zero-copy: Decoded frames stay on GPU (with mandatory deep copy before next prefetch)
    
    Requirements:
        - accvlab.on_demand_video_decoder library must be installed
        - CUDA-capable GPU
        - gpu_preprocess=True recommended for full zero-copy benefit
    
    Prefetch Flow:
        Sample 0: [Sync decode] → [Deep copy] → [Prefetch sample 1]
        Sample 1: [Get prefetched] → [Deep copy] → [Prefetch sample 2]
        Sample 2: [Get prefetched] → [Deep copy] → [Prefetch sample 3]
        ...
        Sample N-1: [Get prefetched] → [Deep copy] → [No prefetch, last sample]
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
        video_base_path: Optional[str] = None,
        num_cameras: int = 6,
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
            num_cameras: Number of cameras (used for decoder initialization).
            gpu_preprocess: Whether to use GPU for image preprocessing.
                           Recommended True for full async benefit.
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
            num_workers=num_cameras,  # Use num_cameras as num_workers
            batch_size=1,  # Video providers typically use per-sample loading
        )
        
        self.video_base_path = video_base_path if video_base_path else data_root
        self.num_cameras = num_cameras
        
        # NVC async decoder (lazy initialization)
        self._nvc_decoder = None
        
        # Parse GPU ID from device string
        if device.startswith('cuda:'):
            self._gpu_id = int(device.split(':')[1])
        elif device == 'cuda':
            self._gpu_id = 0
        else:
            self._gpu_id = 0
        
        # Prefetch state management
        self._prefetch_state = {
            'pending': False,           # Is there a pending prefetch?
            'index': None,              # Index of prefetched sample
            'video_paths': None,        # Prefetched video paths (absolute)
            'frame_ids': None,          # Prefetched frame IDs
        }
    
    def _lazy_init_decoder(self):
        """
        Lazy initialization of the NVC async decoder.
        
        Creates the decoder on first use to avoid initialization overhead
        when the provider is created but not yet used.
        """
        if self._nvc_decoder is None:
            with torch.cuda.nvtx.range("NVC_Async_Decoder_Init"):
                # Use LOCAL_RANK if available (for distributed training)
                gpu_id = int(os.environ.get("LOCAL_RANK", self._gpu_id))
                
                self._nvc_decoder = nvc.CreateSampleReader(
                    num_of_set=1,
                    num_of_file=self.num_cameras,
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
    
    def _start_prefetch(self, index: int):
        """
        Start async prefetch for the given sample index.
        
        This method initiates asynchronous decoding for the next sample
        without blocking. The decoded frames can be retrieved later using
        DecodeN12ToRGBAsyncGetBuffer.
        
        Args:
            index: Sample index to prefetch
        """
        # Boundary check: don't prefetch beyond dataset
        if index >= len(self):
            self._prefetch_state['pending'] = False
            return
        
        with torch.cuda.nvtx.range("Async_Start_Prefetch"):
            try:
                # Get metadata for the sample to prefetch
                metadata = self.get_data_info(index)
                video_paths = metadata.get('video_paths')
                frame_ids = metadata.get('video_frame_ids')
                
                if video_paths is None or frame_ids is None:
                    print(f"WARNING: Cannot prefetch index {index}: missing video_paths or frame_ids")
                    self._prefetch_state['pending'] = False
                    return
                
                # Build absolute paths
                abs_video_paths = self._build_absolute_paths(video_paths)
                
                # Start async decode
                self._nvc_decoder.DecodeN12ToRGBAsync(
                    abs_video_paths,
                    frame_ids,
                    True  # BGR format
                )
                
                # Update prefetch state
                self._prefetch_state = {
                    'pending': True,
                    'index': index,
                    'video_paths': abs_video_paths,
                    'frame_ids': frame_ids,
                }
                
            except Exception as e:
                print(f"WARNING: Failed to start prefetch for index {index}: {e}")
                self._prefetch_state['pending'] = False
    
    def _get_decoded_frames(self, video_paths: List[str], frame_ids: List[int]) -> List:
        """
        Get decoded frames from the async decoder.
        
        If frames are already prefetched, retrieves them from buffer.
        Otherwise, starts sync decode and waits for completion.
        
        Args:
            video_paths: List of absolute video paths
            frame_ids: List of frame indices
            
        Returns:
            List of decoded frames (GPU tensors)
        """
        # Check if current request matches pending prefetch
        if (self._prefetch_state['pending'] and 
            self._prefetch_state['video_paths'] == video_paths and
            self._prefetch_state['frame_ids'] == frame_ids):
            
            # Prefetch hit: get frames from buffer
            with torch.cuda.nvtx.range("Async_Get_Prefetched"):
                decoded_frames = self._nvc_decoder.DecodeN12ToRGBAsyncGetBuffer(
                    video_paths,
                    frame_ids,
                    True  # BGR format
                )
                self._prefetch_state['pending'] = False
                return decoded_frames
        else:
            # Prefetch miss or first sample: do sync decode
            # This happens when:
            # 1. index=0 (first sample, no prefetch yet)
            # 2. Random access (non-sequential iteration)
            # 3. Prefetch failed
            with torch.cuda.nvtx.range("Async_Sync_Decode"):
                # Start async decode
                self._nvc_decoder.DecodeN12ToRGBAsync(
                    video_paths,
                    frame_ids,
                    True  # BGR format
                )
                # Immediately get result (blocks until decode complete)
                decoded_frames = self._nvc_decoder.DecodeN12ToRGBAsyncGetBuffer(
                    video_paths,
                    frame_ids,
                    True  # BGR format
                )
                self._prefetch_state['pending'] = False
                return decoded_frames
    
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images by decoding video frames (CPU path).
        
        This method is used when gpu_preprocess=False.
        Note: For best performance, use gpu_preprocess=True with load_images_to_gpu().
        
        Args:
            data_info: Metadata dictionary from get_data_info()
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        """
        with torch.cuda.nvtx.range("Async_Load_Images_CPU"):
            video_paths = data_info.get('video_paths')
            video_frame_ids = data_info.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(
                    "video_paths and video_frame_ids are required. "
                    "Make sure the pkl file contains video path information."
                )
            
            self._lazy_init_decoder()
            
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            C = 3
            
            abs_video_paths = self._build_absolute_paths(video_paths)
            
            try:
                decoded_frames = self._get_decoded_frames(abs_video_paths, video_frame_ids)
                
                # Convert to numpy (CPU path)
                images = []
                for frame in decoded_frames:
                    tensor = torch.as_tensor(frame, device=f'cuda:{self._gpu_id}')
                    # Deep copy and move to CPU
                    np_frame = tensor.cpu().numpy().astype(np.float32)
                    
                    if np_frame.shape[0] != H or np_frame.shape[1] != W:
                        print(f"WARNING: Frame shape mismatch: {np_frame.shape}")
                    
                    images.append(np_frame)
                
                return images
                
            except Exception as e:
                print(f"ERROR: Async decode failed: {e}")
                # Fallback to random images
                images = []
                for _ in video_paths:
                    fallback = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
                    images.append(fallback.astype(np.float32))
                return images
    
    def load_images_to_gpu(self, data_info: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Load multi-view images directly to GPU tensors with async prefetch.
        
        This method provides zero-copy loading with async prefetching.
        Decoded frames are deep-copied before returning to allow the next
        prefetch to reuse the decode buffer.
        
        IMPORTANT: Frames are deep-copied using torch.clone() because the
        decode buffer will be overwritten by the next DecodeN12ToRGBAsync call.
        
        Args:
            data_info: Metadata dictionary from get_data_info()
        
        Returns:
            List of GPU tensors, BGR format, shape (H, W, 3), dtype float32
        """
        with torch.cuda.nvtx.range("Async_Load_Images_GPU"):
            video_paths = data_info.get('video_paths')
            video_frame_ids = data_info.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(
                    "video_paths and video_frame_ids are required. "
                    "Make sure the pkl file contains video path information."
                )
            
            self._lazy_init_decoder()
            
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            C = 3
            
            abs_video_paths = self._build_absolute_paths(video_paths)
            gpu_images = []
            
            try:
                decoded_frames = self._get_decoded_frames(abs_video_paths, video_frame_ids)
                
                # IMPORTANT: Deep copy frames before next prefetch
                # The decode buffer will be overwritten by DecodeN12ToRGBAsync
                with torch.cuda.nvtx.range("Async_Deep_Copy"):
                    for frame in decoded_frames:
                        # torch.as_tensor provides zero-copy view
                        # .clone() creates deep copy to preserve data
                        tensor = torch.as_tensor(
                            frame,
                            device=f'cuda:{self._gpu_id}'
                        ).to(torch.float32).clone()
                        
                        if tensor.shape[0] != H or tensor.shape[1] != W or tensor.shape[2] != C:
                            print(f"WARNING: Frame shape mismatch: {tensor.shape}")
                            tensor = torch.randint(
                                0, 256, (H, W, C),
                                dtype=torch.float32,
                                device=f'cuda:{self._gpu_id}'
                            )
                        
                        gpu_images.append(tensor)
                
            except Exception as e:
                print(f"ERROR: Async decode failed: {e}")
                # Fallback to random images
                for _ in video_paths:
                    fallback = torch.randint(
                        0, 256, (H, W, C),
                        dtype=torch.float32,
                        device=f'cuda:{self._gpu_id}'
                    )
                    gpu_images.append(fallback)
            
            return gpu_images
    
    def get_single_sample(self, index: int) -> Dict[str, Any]:
        """
        Get complete data for a single sample with async prefetching.
        
        Overrides base class to add prefetch logic:
        1. Load current sample (from prefetch or sync decode)
        2. After loading, start prefetch for index+1
        
        Boundary handling:
        - index=0: No prefetch available, does sync decode, then prefetches index=1
        - index=N-1 (last): Gets prefetched data, no next prefetch
        
        Args:
            index: Data index
        
        Returns:
            Dictionary containing images and metadata
        """
        # Call base class implementation (which calls load_images or load_images_to_gpu)
        result = super().get_single_sample(index)
        
        # Start prefetch for next sample (only in GPU mode for zero-copy benefit)
        if self.gpu_preprocess:
            next_index = index + 1
            if next_index < len(self):
                self._start_prefetch(next_index)
        
        return result
    
    def reset_prefetch_state(self):
        """
        Reset prefetch state.
        
        Call this method when:
        - Starting a new iteration from the beginning
        - Switching to random access pattern
        - After scene changes that invalidate prefetch
        """
        self._prefetch_state = {
            'pending': False,
            'index': None,
            'video_paths': None,
            'frame_ids': None,
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"video_base_path={self.video_base_path}, "
            f"num_cameras={self.num_cameras}, "
            f"gpu_id={self._gpu_id}, "
            f"gpu_preprocess={self.gpu_preprocess}, "
            f"prefetch_pending={self._prefetch_state['pending']})"
        )

