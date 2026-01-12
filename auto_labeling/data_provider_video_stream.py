# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# VideoDataProviderACCVStream - Load images by decoding video frames using NVC GPU decoder
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


class VideoDataProviderACCVStream(BaseDataProvider):
    """
    Data provider that loads images by decoding video frames using NVC GPU decoder.
    
    Uses accvlab.on_demand_video_decoder library for GPU-accelerated video decoding.
    This approach provides batch decoding of multiple video frames directly on GPU,
    which is more efficient than CPU-based OpenCV decoding for high-throughput inference.
    
    Key features:
        - Batch decoding: All frames (bs * num_cameras) are decoded in one call
        - Zero-copy: Decoded frames stay on GPU without CPU round-trip
        - C++ multi-threading: accvlab uses C++ threads for parallel GPU decoding
    
    Requirements:
        - accvlab.on_demand_video_decoder library must be installed
        - CUDA-capable GPU
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
        Initialize the NVC video data provider.
        
        Args:
            pkl_path: Path to the data index pkl file
            data_root: Root directory of the data
            img_norm_cfg: Image normalization configuration
            ida_aug_conf: Image augmentation configuration
            video_base_path: Base path for video files. If None, uses data_root.
            num_cameras: Number of cameras per sample. Default is 6 for nuScenes.
            batch_size: Batch size for inference. Used for decoder initialization.
            gpu_preprocess: Whether to use GPU for image preprocessing
            device: GPU device for decoding and preprocessing
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
        self.batch_size = batch_size
        
        # Parse GPU ID from device string
        if device.startswith('cuda:'):
            self._gpu_id = int(device.split(':')[1])
        elif device == 'cuda':
            self._gpu_id = 0
        else:
            self._gpu_id = 0
        
        # NVC decoder (lazy initialization)
        self._nvc_decoder = None
    
    def _lazy_init_decoder(self):
        """
        Lazy initialization of the NVC decoder.
        
        Initializes decoder with capacity for batch_size * num_cameras files
        to enable efficient batch decoding.
        """
        if self._nvc_decoder is None:
            with torch.cuda.nvtx.range("NVC_Decoder_Init"):
                gpu_id = int(os.environ.get("LOCAL_RANK", self._gpu_id))
                
                # Initialize for max batch capacity
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
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples by batch decoding video frames on GPU.
        
        This method leverages accvlab's C++ multi-threaded GPU decoding by:
        1. Collecting all video paths and frame IDs from all samples
        2. Calling DecodeN12ToRGB once with all paths (bs * num_cameras)
        3. Stacking decoded GPU tensors
        4. Applying GPU transforms in batch
        5. Splitting and building sample dictionaries
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary with collated samples
        """
        self._lazy_init_decoder()
        
        # Step 1: Collect all paths and frame IDs
        all_metadata = []
        all_paths = []
        all_frame_ids = []
        image_counts = []
        
        with torch.cuda.nvtx.range("Collect_Paths"):
            for idx in indices:
                metadata = self.data_index.get_data_info(idx)
                all_metadata.append(metadata)
                
                video_paths = metadata.get('video_paths')
                video_frame_ids = metadata.get('video_frame_ids')
                
                if video_paths is None or video_frame_ids is None:
                    raise ValueError(
                        f"video_paths and video_frame_ids required for sample {idx}"
                    )
                
                all_paths.extend(video_paths)
                all_frame_ids.extend(video_frame_ids)
                image_counts.append(len(video_paths))
        
        # Step 2: Batch decode all frames (C++ multi-threaded GPU decoding)
        H = self.ida_aug_conf.get('H', 900)
        W = self.ida_aug_conf.get('W', 1600)
        
        with torch.cuda.nvtx.range("NVC_Batch_Decode"):
            abs_paths = self._build_absolute_paths(all_paths)
            
            try:
                decoded_frames = self._nvc_decoder.DecodeN12ToRGB(
                    abs_paths, all_frame_ids, True
                )
            except Exception as e:
                print(f"ERROR: NVC DecodeN12ToRGB failed: {e}")
                # Generate fallback frames
                decoded_frames = [
                    torch.randint(0, 256, (H, W, 3), dtype=torch.uint8,
                                  device=f'cuda:{self._gpu_id}')
                    for _ in all_paths
                ]
        
        # Step 3: Convert RGBFrame to torch tensors (keep on GPU)
        with torch.cuda.nvtx.range("Convert_To_Tensors"):
            gpu_images = []
            for frame in decoded_frames:
                try:
                    # RGBFrame needs torch.tensor() to convert, then move to GPU
                    tensor = torch.tensor(frame, dtype=torch.uint8).to(
                        device=f'cuda:{self._gpu_id}', dtype=torch.float32, non_blocking=True
                    )
                    gpu_images.append(tensor)
                except Exception as e:
                    print(f"ERROR: Failed to convert frame: {e}")
                    fallback = torch.randint(0, 256, (H, W, 3), dtype=torch.float32,
                                             device=f'cuda:{self._gpu_id}')
                    gpu_images.append(fallback)
        
        # Step 4: Batch transform
        with torch.cuda.nvtx.range("Batch_Transform"):
            if self.gpu_preprocess:
                # GPU mode: Transform ALL images at once
                with torch.cuda.nvtx.range("GPU_Transform_All"):
                    all_transformed, _ = self.transforms(gpu_images, all_metadata[0])
                
                # Split and collect
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
                # CPU mode (fallback): Convert to numpy and process
                import numpy as np
                from .batch_utils import images_to_tensor
                
                all_img_tensors = []
                all_updated_metadata = []
                
                # Convert GPU tensors to numpy
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
    
    def __repr__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"video_base_path={self.video_base_path}, "
            f"num_cameras={self.num_cameras}, "
            f"batch_size={self.batch_size}, "
            f"gpu_id={self._gpu_id}, "
            f"gpu_preprocess={self.gpu_preprocess})"
        )
