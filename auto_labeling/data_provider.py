# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# DataProvider - Data Provider
# Replaces Dataset + DataLoader, directly provides inference data
# ------------------------------------------------------------------------

import pickle
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional, Tuple

from mmdet3d.core.bbox import get_box_type

from .image_transforms import ImageTransforms


def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 4x4 egopose matrix"""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def convert_egopose_to_matrix_numpy(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Convert rotation and translation to a 4x4 transformation matrix"""
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    Quaternion format: [w, x, y, z]
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


class DataProvider:
    """
    Data provider, replaces Dataset + DataLoader
    
    Functions:
    1. Read metadata from pkl file
    2. Load multi-view images
    3. Apply image transforms
    4. Assemble batch
    """
    
    def __init__(
        self,
        pkl_path: str,
        data_root: str,
        img_norm_cfg: Dict[str, Any],
        ida_aug_conf: Dict[str, Any],
    ):
        """
        Args:
            pkl_path: Path to data index pkl file
            data_root: Data root directory
            img_norm_cfg: Image normalization config
            ida_aug_conf: Image augmentation config
        """
        # Load data index
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.data_infos = data['infos']
        self.data_root = data_root
        self.img_norm_cfg = img_norm_cfg
        self.ida_aug_conf = ida_aug_conf
        
        # Initialize image transforms
        self.transforms = ImageTransforms(img_norm_cfg, ida_aug_conf)
        
        # Cache scene token for temporal inference
        self.prev_scene_token = None
    
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def get_data_info(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for a single sample
        Reuses logic from CustomNuScenesDataset.get_data_info
        
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
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        
        for cam_type, cam_info in info['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            image_paths.append(cam_info['data_path'])
            
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
        })
        
        return input_dict
    
    def load_images(self, filenames: List[str]) -> List[np.ndarray]:
        """
        Load multi-view images
        
        Args:
            filenames: List of image file paths
        
        Returns:
            List of images, BGR format, np.ndarray
        """
        images = []
        for filename in filenames:
            # Handle relative path
            if not filename.startswith('/'):
                filename = f"{self.data_root}/{filename}"
            
            img = cv2.imread(filename)
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {filename}")
            images.append(img.astype(np.float32))
        
        return images
    
    def get_single_sample(self, index: int) -> Dict[str, Any]:
        """
        Get complete data for a single sample (replaces __getitem__)
        
        Args:
            index: Data index
        
        Returns:
            Dictionary containing images and metadata
        """
        # 1. Get metadata
        metadata = self.get_data_info(index)
        
        # 2. Load images
        images = self.load_images(metadata['img_filename'])
        
        # 3. Apply transforms
        images, metadata = self.transforms(images, metadata)
        
        # 4. Convert to tensor (HWC -> CHW)
        img_tensor = self._to_tensor(images)  # (N, C, H, W)
        
        # 5. Build img_metas
        pad_shape = self.transforms.get_pad_shape()
        box_type_3d, box_mode_3d = get_box_type('LiDAR')
        img_metas = {
            'filename': metadata['img_filename'],
            'ori_shape': (self.ida_aug_conf['H'], self.ida_aug_conf['W'], 3),
            'img_shape': metadata.get('img_shape', [(pad_shape[0], pad_shape[1], 3)] * len(images)),
            'pad_shape': [(pad_shape[0], pad_shape[1], 3)] * len(images),  # Pad shape for each camera view
            'scale_factor': 1.0,
            'flip': False,
            'box_mode_3d': box_mode_3d,
            'box_type_3d': box_type_3d,  # LiDARInstance3DBoxes class, used to create 3D bbox
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
        Assemble batch (replaces collate_fn)
        
        Args:
            indices: List of sample indices
        
        Returns:
            Batch data dictionary
        """
        samples = [self.get_single_sample(i) for i in indices]
        return self._collate(samples)
    
    def _to_tensor(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Convert image list to tensor
        
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
        Collate multiple samples into a batch
        
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
        Iterate and generate batches
        
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

