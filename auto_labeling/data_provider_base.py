# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# BaseDataProvider - Abstract base class for data providers
# ------------------------------------------------------------------------

import pickle
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from mmdet3d.core.bbox import get_box_type

from .image_transforms import ImageTransforms
from .gpu_image_transforms import GPUImageTransforms


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 egopose matrix.
    
    Args:
        egopose: 4x4 transformation matrix
        
    Returns:
        Inverse of the transformation matrix
    """
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Convert rotation and translation to a 4x4 transformation matrix.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3-element translation vector
        
    Returns:
        4x4 transformation matrix
    """
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: Quaternion in [w, x, y, z] format
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion
    
    # Normalize
    n = w*w + x*x + y*y + z*z
    if n == 0:
        return np.eye(3, dtype=np.float32)
    
    s = 2.0 / n
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    
    rotation_matrix = np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)]
    ], dtype=np.float32)
    
    return rotation_matrix


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.
    
    Replaces Dataset + DataLoader, directly provides inference data.
    Subclasses must implement the load_images() method to define
    how images are loaded (from files, videos, etc.).
    
    Functions:
        1. Read metadata from pkl file
        2. Load multi-view images (abstract, implemented by subclasses)
        3. Apply image transforms
        4. Assemble batch
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
        # Load data index
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.data_infos = data['infos']
        self.data_root = data_root
        self.img_norm_cfg = img_norm_cfg
        self.ida_aug_conf = ida_aug_conf
        self.gpu_preprocess = gpu_preprocess
        self.device = device
        
        # Initialize image transforms (CPU or GPU)
        if gpu_preprocess:
            self.transforms = GPUImageTransforms(img_norm_cfg, ida_aug_conf, device=device)
        else:
            self.transforms = ImageTransforms(img_norm_cfg, ida_aug_conf)
        
        # Cache scene token for temporal inference
        self.prev_scene_token = None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_infos)
    
    def get_data_info(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for a single sample.
        
        Parses the pkl info and extracts:
            - Image paths (img_filename)
            - Video paths and frame IDs (if available)
            - Camera intrinsics and extrinsics
            - Ego pose information
        
        Args:
            index: Data index
        
        Returns:
            Dictionary containing metadata
        """
        index = index % len(self.data_infos)
        info = self.data_infos[index]
        
        # Parse ego pose
        e2g_rotation = quaternion_to_rotation_matrix(info['ego2global_rotation'])
        e2g_translation = np.array(info['ego2global_translation'], dtype=np.float32)
        l2e_rotation = quaternion_to_rotation_matrix(info['lidar2ego_rotation'])
        l2e_translation = np.array(info['lidar2ego_translation'], dtype=np.float32)
        
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose = e2g_matrix @ l2e_matrix  # lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        input_dict = {
            'sample_idx': info['token'],
            'pts_filename': info.get('lidar_path', ''),
            'sweeps': info.get('sweeps', []),
            'ego_pose': ego_pose,
            'ego_pose_inv': ego_pose_inv,
            'prev_idx': info.get('prev', ''),
            'next_idx': info.get('next', ''),
            'scene_token': info['scene_token'],
            'frame_idx': info.get('frame_idx', 0),
            'timestamp': info['timestamp'] / 1e6,
        }
        
        # Parse camera info
        image_paths = []
        video_paths = []
        video_frame_ids = []
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        
        for cam_type, cam_info in info['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])
            
            # Extract video path and frame ID if available
            if 'video_path' in cam_info:
                video_paths.append(cam_info['video_path'])
                video_frame_ids.append(cam_info['video_frame_id'])
            
            # Compute lidar to image transformation matrix
            cam2lidar_r = np.array(cam_info['sensor2lidar_rotation'], dtype=np.float32)
            cam2lidar_t = np.array(cam_info['sensor2lidar_translation'], dtype=np.float32)
            cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
            lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)
            
            intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
            viewpad = np.eye(4, dtype=np.float32)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt
            
            intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt)
            lidar2img_rts.append(lidar2img_rt)
        
        input_dict.update({
            'img_timestamp': img_timestamp,
            'img_filename': image_paths,
            'lidar2img': lidar2img_rts,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            # Video-related fields (may be empty lists)
            'video_paths': video_paths if len(video_paths) > 0 else None,
            'video_frame_ids': video_frame_ids if len(video_frame_ids) > 0 else None,
        })
        
        return input_dict
    
    @abstractmethod
    def load_images(self, data_info: Dict[str, Any]) -> List[np.ndarray]:
        """
        Abstract method: Load multi-view images.
        
        Subclasses must implement this method to define how images are loaded.
        For example:
            - ImageDataProvider: uses cv2.imread() to load from image files
            - OpenCVVideoDataProvider: uses cv2.VideoCapture to decode from videos
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
                       Contains 'img_filename', 'video_paths', 'video_frame_ids', etc.
        
        Returns:
            List of images, BGR format, np.ndarray with dtype float32
        """
        pass
    
    def load_images_to_gpu(self, data_info: Dict[str, Any]) -> List[torch.Tensor]:
        """
        Load multi-view images directly to GPU tensors.
        
        This method is used when gpu_preprocess=True to enable zero-copy
        data loading for providers that can decode directly to GPU memory.
        
        Default implementation calls load_images() and converts to GPU tensors.
        Subclasses (e.g., VideoDataProviderACCVStream) can override this for zero-copy.
        
        Args:
            data_info: Metadata dictionary returned by get_data_info()
        
        Returns:
            List of GPU tensors, BGR format, shape (H, W, 3), dtype float32
        """
        # Default implementation: load to CPU numpy then transfer to GPU
        images = self.load_images(data_info)
        gpu_images = []
        for img in images:
            # img is np.ndarray float32 (H, W, C)
            tensor = torch.from_numpy(img).to(self.device, non_blocking=True)
            gpu_images.append(tensor)
        return gpu_images
    
    def get_single_sample(self, index: int) -> Dict[str, Any]:
        """
        Get complete data for a single sample (replaces __getitem__).
        
        NVTX ranges for performance profiling:
            - Get_Data_Info: Metadata parsing
            - Load_Images: Image loading (implementation depends on subclass)
            - Apply_Transforms: Image preprocessing transforms (CPU mode)
            - GPU_Preprocess: GPU image preprocessing (GPU mode)
            - To_Tensor: NumPy to Tensor conversion (CPU mode only)
            - Build_Img_Metas: Metadata dictionary construction
        
        Args:
            index: Data index
        
        Returns:
            Dictionary containing images and metadata
        """
        # 1. Get metadata
        with torch.cuda.nvtx.range("Get_Data_Info"):
            metadata = self.get_data_info(index)
        
        # 2. Load images and 3. Apply transforms (CPU or GPU)
        if self.gpu_preprocess:
            # GPU mode: use load_images_to_gpu for potential zero-copy loading
            with torch.cuda.nvtx.range("Load_Images_GPU"):
                gpu_images = self.load_images_to_gpu(metadata)
            
            # GPU transforms directly processes GPU tensors
            img_tensor, metadata = self.transforms(gpu_images, metadata)
            num_views = img_tensor.shape[0]
        else:
            # CPU mode: load images to numpy
            with torch.cuda.nvtx.range("Load_Images"):
                images = self.load_images(metadata)
            
            # CPU transforms returns numpy list, need manual conversion
            with torch.cuda.nvtx.range("Apply_Transforms"):
                images, metadata = self.transforms(images, metadata)
            
            # Convert to tensor (HWC -> CHW)
            with torch.cuda.nvtx.range("To_Tensor"):
                img_tensor = self._to_tensor(images)  # (N, C, H, W)
            num_views = len(images)
        
        # 5. Build img_metas
        with torch.cuda.nvtx.range("Build_Img_Metas"):
            pad_shape = self.transforms.get_pad_shape()
            box_type_3d, box_mode_3d = get_box_type('LiDAR')
            img_metas = {
                'filename': metadata['img_filename'],
                'ori_shape': (self.ida_aug_conf['H'], self.ida_aug_conf['W'], 3),
                'img_shape': metadata.get('img_shape', [(pad_shape[0], pad_shape[1], 3)] * num_views),
                'pad_shape': [(pad_shape[0], pad_shape[1], 3)] * num_views,
                'scale_factor': 1.0,
                'flip': False,
                'box_mode_3d': box_mode_3d,
                'box_type_3d': box_type_3d,
                'img_norm_cfg': self.img_norm_cfg,
                'scene_token': metadata['scene_token'],
                'sample_idx': metadata['sample_idx'],
            }
        
        return {
            'img': img_tensor,
            'img_metas': img_metas,
            'lidar2img': np.stack(metadata['lidar2img'], axis=0),  # (N, 4, 4)
            'intrinsics': np.stack(metadata['intrinsics'], axis=0),  # (N, 4, 4)
            'extrinsics': np.stack(metadata['extrinsics'], axis=0),  # (N, 4, 4)
            'timestamp': np.array(metadata['timestamp'], dtype=np.float64),
            'img_timestamp': np.array(metadata['img_timestamp'], dtype=np.float64),
            'ego_pose': metadata['ego_pose'],  # (4, 4)
            'ego_pose_inv': metadata['ego_pose_inv'],  # (4, 4)
        }
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Assemble batch (replaces collate_fn).
        
        NVTX ranges for performance profiling:
            - Get_Single_Samples: Per-sample data loading and processing
            - Collate_Batch: Batch assembly and tensor stacking
        
        Args:
            indices: List of sample indices
        
        Returns:
            Batch data dictionary
        """
        with torch.cuda.nvtx.range("Get_Single_Samples"):
            samples = [self.get_single_sample(i) for i in indices]
        with torch.cuda.nvtx.range("Collate_Batch"):
            return self._collate(samples)
    
    def _to_tensor(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert image list to tensor.
        
        Args:
            images: List of images, each with shape (H, W, C)
        
        Returns:
            Tensor with shape (N, C, H, W)
        """
        # HWC -> CHW
        imgs = [img.transpose(2, 0, 1) for img in images]
        return torch.from_numpy(np.stack(imgs, axis=0)).float()
    
    def _collate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate multiple samples into a batch.
        
        Args:
            samples: List of samples
        
        Returns:
            Batch dictionary
        """
        batch_size = len(samples)
        
        # Stack tensors
        img = torch.stack([s['img'] for s in samples], dim=0)  # (B, N, C, H, W)
        
        # Stack numpy arrays and convert to tensor
        lidar2img = torch.from_numpy(
            np.stack([s['lidar2img'] for s in samples], axis=0)
        ).float()  # (B, N, 4, 4)
        
        intrinsics = torch.from_numpy(
            np.stack([s['intrinsics'] for s in samples], axis=0)
        ).float()  # (B, N, 4, 4)
        
        extrinsics = torch.from_numpy(
            np.stack([s['extrinsics'] for s in samples], axis=0)
        ).float()  # (B, N, 4, 4)
        
        timestamp = torch.from_numpy(
            np.stack([s['timestamp'] for s in samples], axis=0)
        ).double()  # (B,)
        
        img_timestamp = torch.from_numpy(
            np.stack([s['img_timestamp'] for s in samples], axis=0)
        ).double()  # (B, N)
        
        ego_pose = torch.from_numpy(
            np.stack([s['ego_pose'] for s in samples], axis=0)
        ).float()  # (B, 4, 4)
        
        ego_pose_inv = torch.from_numpy(
            np.stack([s['ego_pose_inv'] for s in samples], axis=0)
        ).float()  # (B, 4, 4)
        
        # Collect img_metas as list
        img_metas = [s['img_metas'] for s in samples]
        
        return {
            'img': img,
            'img_metas': img_metas,
            'lidar2img': lidar2img,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'timestamp': timestamp,
            'img_timestamp': img_timestamp,
            'ego_pose': ego_pose,
            'ego_pose_inv': ego_pose_inv,
        }
    
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

