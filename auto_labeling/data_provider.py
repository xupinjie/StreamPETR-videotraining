# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# DataProvider - Factory function and backward compatibility
# ------------------------------------------------------------------------

from typing import Dict, Any, Union

from .data_provider_base import BaseDataProvider
from .data_provider_image import ImageDataProvider
from .data_provider_image_torchvision import TorchvisionDataProvider
from .data_provider_video_cv2 import OpenCVVideoDataProvider
from .data_provider_video_stream import VideoDataProviderACCVStream
from .data_provider_video_stream_async import VideoDataProviderACCVStreamAsync


def create_data_provider(
    mode: str = 'image',
    pkl_path: str = None,
    data_root: str = None,
    img_norm_cfg: Dict[str, Any] = None,
    ida_aug_conf: Dict[str, Any] = None,
    gpu_preprocess: bool = False,
    device: str = 'cuda:0',
    **kwargs
) -> BaseDataProvider:
    """
    Factory function to create a data provider instance.
    
    This function provides a unified interface to create different types
    of data providers based on the specified mode.
    
    Supported modes:
        - 'image': Load images from image files using cv2.imread
        - 'torchvision_image': Load images using torchvision.io with parallel I/O
        - 'opencv_video': Decode video frames using OpenCV VideoCapture (CPU)
        - 'nvc_video': Decode video frames using NVC GPU decoder (sync)
        - 'nvc_video_async': Decode video frames using NVC GPU decoder with async prefetch
    
    Args:
        mode: Data provider mode. One of 'image', 'opencv_video', 'nvc_video', 'nvc_video_async'.
        pkl_path: Path to the data index pkl file
        data_root: Root directory of the data
        img_norm_cfg: Image normalization configuration
        ida_aug_conf: Image augmentation configuration
        gpu_preprocess: Whether to use GPU for image preprocessing
        device: GPU device (only used when gpu_preprocess=True)
        **kwargs: Additional arguments passed to specific provider class.
                  For 'torchvision_image' mode:
                      - num_workers: Number of parallel workers (default 6)
                  For 'opencv_video' mode:
                      - video_base_path: Base path for video files
                      - max_cache_size: Max VideoCapture cache size
                  For 'nvc_video' and 'nvc_video_async' mode:
                      - video_base_path: Base path for video files
                      - num_cameras: Number of cameras (default 6)
    
    Returns:
        A data provider instance (subclass of BaseDataProvider)
    
    Raises:
        ValueError: If an unknown mode is specified
    
    Examples:
        # Create image data provider
        provider = create_data_provider(
            mode='image',
            pkl_path='data.pkl',
            data_root='/data/nuscenes',
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
        )
        
        # Create OpenCV video data provider
        provider = create_data_provider(
            mode='opencv_video',
            pkl_path='data.pkl',
            data_root='/data/nuscenes',
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
            video_base_path='/data/nuscenes/videos',
            max_cache_size=6,
        )
        
        # Create NVC GPU video data provider
        provider = create_data_provider(
            mode='nvc_video',
            pkl_path='data.pkl',
            data_root='/data/nuscenes',
            img_norm_cfg=img_norm_cfg,
            ida_aug_conf=ida_aug_conf,
            video_base_path='/data/nuscenes/videos',
            num_cameras=6,
        )
    """
    common_args = {
        'pkl_path': pkl_path,
        'data_root': data_root,
        'img_norm_cfg': img_norm_cfg,
        'ida_aug_conf': ida_aug_conf,
        'gpu_preprocess': gpu_preprocess,
        'device': device,
    }
    
    if mode == 'image':
        # ImageDataProvider uses simple serial loading, no extra args needed
        return ImageDataProvider(**common_args)
    
    elif mode == 'torchvision_image':
        # Extract torchvision-specific arguments
        tv_args = {
            'num_workers': kwargs.get('num_workers', 6),
            'batch_size': kwargs.get('batch_size', 1),
        }
        return TorchvisionDataProvider(**common_args, **tv_args)
    
    elif mode == 'opencv_video':
        # Extract video-specific arguments
        video_args = {
            'video_base_path': kwargs.get('video_base_path'),
            'max_cache_size': kwargs.get('max_cache_size', 6),
        }
        return OpenCVVideoDataProvider(**common_args, **video_args)
    
    elif mode == 'nvc_video':
        # Extract NVC-specific arguments
        nvc_args = {
            'video_base_path': kwargs.get('video_base_path'),
            'num_cameras': kwargs.get('num_cameras', 6),
            'batch_size': kwargs.get('batch_size', 1),
        }
        return VideoDataProviderACCVStream(**common_args, **nvc_args)
    
    elif mode == 'nvc_video_async':
        # Extract NVC async-specific arguments
        nvc_args = {
            'video_base_path': kwargs.get('video_base_path'),
            'num_cameras': kwargs.get('num_cameras', 6),
            'batch_size': kwargs.get('batch_size', 1),
        }
        return VideoDataProviderACCVStreamAsync(**common_args, **nvc_args)
    
    else:
        available_modes = ['image', 'torchvision_image', 'opencv_video', 'nvc_video', 'nvc_video_async']
        raise ValueError(
            f"Unknown data provider mode: '{mode}'. "
            f"Available modes: {available_modes}"
        )


# Backward compatibility: DataProvider as alias for ImageDataProvider
# This ensures existing code that uses DataProvider directly continues to work
DataProvider = ImageDataProvider


# Re-export classes for direct import
__all__ = [
    'BaseDataProvider',
    'ImageDataProvider',
    'TorchvisionDataProvider',
    'OpenCVVideoDataProvider',
    'VideoDataProviderACCVStream',
    'VideoDataProviderACCVStreamAsync',
    'DataProvider',
    'create_data_provider',
]
