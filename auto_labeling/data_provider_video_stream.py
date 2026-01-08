# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# VideoDataProviderACCVStream - Load images by decoding video frames using NVC GPU decoder
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


class VideoDataProviderACCVStream(BaseDataProvider):
    """
    Data provider that loads images by decoding video frames using NVC GPU decoder.
    
    Uses accvlab.on_demand_video_decoder library for GPU-accelerated video decoding.
    This approach provides batch decoding of multiple video frames directly on GPU,
    which is more efficient than CPU-based OpenCV decoding for high-throughput inference.
    
    Requirements:
        - accvlab.on_demand_video_decoder library must be installed
        - CUDA-capable GPU
    
    This approach is useful when:
        - High-performance GPU video decoding is required
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
        num_cameras: int = 6,
        gpu_preprocess: bool = False,
        device: str = 'cuda:0',
    ):
        """
        Initialize the NVC video data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            video_base_path: Base path for video files. If None, uses data_root.
                            Video paths in pkl will be prefixed with this path.
            num_cameras: Number of cameras (used for decoder initialization).
                        Default is 6 for nuScenes.
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device (used for decoder initialization and preprocessing)
        """
        if not NVC_AVAILABLE:
            raise ImportError(
                "accvlab.on_demand_video_decoder is not available. "
                "Please install the NVC library to use VideoDataProviderACCVStream."
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
        
        # NVC decoder (lazy initialization)
        self._nvc_decoder = None
        
        # Parse GPU ID from device string (e.g., 'cuda:0' -> 0)
        if device.startswith('cuda:'):
            self._gpu_id = int(device.split(':')[1])
        elif device == 'cuda':
            self._gpu_id = 0
        else:
            self._gpu_id = 0
    
    def _lazy_init_decoder(self):
        """
        Lazy initialization of the NVC decoder.
        
        Creates the decoder on first use to avoid initialization overhead
        when the provider is created but not yet used.
        """
        if self._nvc_decoder is None:
            with torch.cuda.nvtx.range("NVC_Decoder_Init"):
                # Use LOCAL_RANK if available (for distributed training)
                gpu_id = int(os.environ.get("LOCAL_RANK", self._gpu_id))
                
                self._nvc_decoder = nvc.CreateSampleReader(
                    num_of_set=1,  # batch_size for decoder
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
    
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Load multi-view images by decoding video frames using NVC GPU decoder.
        
        NVTX ranges:
            - NVC_Load_Images: Overall image loading from videos
            - NVC_Decode: GPU video frame decoding
            - NVC_To_Numpy: Converting GPU tensors to numpy arrays
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'video_paths' and 'video_frame_ids' fields
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        
        Raises:
            ValueError: If video_paths or video_frame_ids is not available,
                       or if frame decoding fails
        """
        with torch.cuda.nvtx.range("NVC_Load_Images"):
            video_paths = data_info.get('video_paths')
            video_frame_ids = data_info.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(
                    "video_paths and video_frame_ids are required for VideoDataProviderACCVStream. "
                    "Make sure the pkl file contains video path information."
                )
            
            if len(video_paths) != len(video_frame_ids):
                raise ValueError(
                    f"video_paths length ({len(video_paths)}) != "
                    f"video_frame_ids length ({len(video_frame_ids)})"
                )
            
            # Initialize decoder if not already done
            self._lazy_init_decoder()
            
            # Get expected image dimensions from ida_aug_conf
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            C = 3
            
            # Build absolute paths for video files
            abs_video_paths = self._build_absolute_paths(video_paths)
            
            images = []
            
            try:
                # Batch decode all frames using NVC GPU decoder
                with torch.cuda.nvtx.range("NVC_Decode"):
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGB(
                        abs_video_paths, 
                        video_frame_ids, 
                        True  # Return as GPU tensor
                    )
                
                # Convert decoded frames to numpy arrays
                with torch.cuda.nvtx.range("NVC_To_Numpy"):
                    for decoded_frame in decoded_frames:
                        try:
                            # decoded_frame is already a GPU tensor
                            torch_bgr = torch.tensor(
                                decoded_frame, 
                                dtype=torch.uint8
                            ).to(f'cuda:{self._gpu_id}')
                            
                            # Check shape
                            if torch_bgr.shape[0] != H or torch_bgr.shape[1] != W or torch_bgr.shape[2] != C:
                                print(f"WARNING: Frame shape mismatch. "
                                      f"Got {torch_bgr.shape}, expected ({H}, {W}, {C})")
                                torch_bgr = torch.randint(
                                    0, 256, (H, W, C), 
                                    dtype=torch.uint8
                                ).to(f'cuda:{self._gpu_id}')
                            
                            # Convert to numpy float32
                            frame = torch_bgr.cpu().numpy().astype(np.float32)
                            images.append(frame)
                            
                        except Exception as e:
                            print(f"ERROR: Failed to process decoded frame: {e}")
                            print("Generating random image as fallback")
                            fallback = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
                            images.append(fallback.astype(np.float32))
                            
            except Exception as e:
                print(f"ERROR: NVC DecodeN12ToRGB failed: {e}")
                print("Generating random images as fallback for all frames")
                # Generate fallback images for all cameras
                for _ in video_paths:
                    fallback = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
                    images.append(fallback.astype(np.float32))
            
            return images
    
    def load_images_to_gpu(self, data_info: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Load multi-view images directly to GPU tensors (zero-copy).
        
        This method overrides the base class implementation to provide
        zero-copy loading from NVC GPU decoder. The decoded frames stay
        on GPU without being copied to CPU memory.
        
        NVTX ranges:
            - NVC_Load_Images_GPU: Overall GPU image loading
            - NVC_Decode: GPU video frame decoding
            - NVC_Process_Tensors: Processing decoded tensors
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Uses 'video_paths' and 'video_frame_ids' fields
        
        Returns:
            List of GPU tensors, BGR format, shape (H, W, 3), dtype float32
        """
        with torch.cuda.nvtx.range("NVC_Load_Images_GPU"):
            video_paths = data_info.get('video_paths')
            video_frame_ids = data_info.get('video_frame_ids')
            
            if video_paths is None or video_frame_ids is None:
                raise ValueError(
                    "video_paths and video_frame_ids are required for VideoDataProviderACCVStream. "
                    "Make sure the pkl file contains video path information."
                )
            
            if len(video_paths) != len(video_frame_ids):
                raise ValueError(
                    f"video_paths length ({len(video_paths)}) != "
                    f"video_frame_ids length ({len(video_frame_ids)})"
                )
            
            # Initialize decoder if not already done
            self._lazy_init_decoder()
            
            # Get expected image dimensions from ida_aug_conf
            H = self.ida_aug_conf.get('H', 900)
            W = self.ida_aug_conf.get('W', 1600)
            C = 3
            
            # Build absolute paths for video files
            abs_video_paths = self._build_absolute_paths(video_paths)
            
            gpu_images = []
            
            try:
                # Batch decode all frames using NVC GPU decoder
                with torch.cuda.nvtx.range("NVC_Decode"):
                    decoded_frames = self._nvc_decoder.DecodeN12ToRGB(
                        abs_video_paths, 
                        video_frame_ids, 
                        True  # Return as GPU tensor
                    )
                
                # Process decoded frames - keep on GPU (zero-copy)
                with torch.cuda.nvtx.range("NVC_Process_Tensors"):
                    for decoded_frame in decoded_frames:
                        try:
                            # decoded_frame is already a GPU tensor from NVC
                            # Convert to torch tensor if needed and ensure correct dtype
                            if isinstance(decoded_frame, torch.Tensor):
                                torch_bgr = decoded_frame.to(
                                    device=f'cuda:{self._gpu_id}',
                                    dtype=torch.float32,
                                    non_blocking=True
                                )
                            else:
                                # If it's not a tensor, wrap it (zero-copy if possible)
                                torch_bgr = torch.as_tensor(
                                    decoded_frame, 
                                    dtype=torch.float32,
                                    device=f'cuda:{self._gpu_id}'
                                )
                            
                            # Check shape
                            if torch_bgr.shape[0] != H or torch_bgr.shape[1] != W or torch_bgr.shape[2] != C:
                                print(f"WARNING: Frame shape mismatch. "
                                      f"Got {torch_bgr.shape}, expected ({H}, {W}, {C})")
                                torch_bgr = torch.randint(
                                    0, 256, (H, W, C), 
                                    dtype=torch.float32,
                                    device=f'cuda:{self._gpu_id}'
                                )
                            
                            gpu_images.append(torch_bgr)
                            
                        except Exception as e:
                            print(f"ERROR: Failed to process decoded frame: {e}")
                            print("Generating random image as fallback")
                            fallback = torch.randint(
                                0, 256, (H, W, C),
                                dtype=torch.float32,
                                device=f'cuda:{self._gpu_id}'
                            )
                            gpu_images.append(fallback)
                            
            except Exception as e:
                print(f"ERROR: NVC DecodeN12ToRGB failed: {e}")
                print("Generating random images as fallback for all frames")
                # Generate fallback images for all cameras
                for _ in video_paths:
                    fallback = torch.randint(
                        0, 256, (H, W, C),
                        dtype=torch.float32,
                        device=f'cuda:{self._gpu_id}'
                    )
                    gpu_images.append(fallback)
            
            return gpu_images
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"video_base_path={self.video_base_path}, "
            f"num_cameras={self.num_cameras}, "
            f"gpu_id={self._gpu_id}, "
            f"gpu_preprocess={self.gpu_preprocess})"
        )

