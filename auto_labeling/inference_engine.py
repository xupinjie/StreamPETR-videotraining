# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# InferenceEngine - Inference Engine
# Encapsulates model loading and inference logic
# ------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

# Add project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class InferenceEngine:
    """
    Inference engine for multi-video parallel inference.
    
    Functions:
    1. Load model config and weights
    2. Execute inference with per-sample temporal state control
    3. Return detection results
    
    Multi-video mode: Each batch position processes a different video
    independently with its own temporal state.
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
        batch_size: int = 1,
    ):
        """
        Args:
            config_path: Config file path
            checkpoint_path: Checkpoint file path
            device: Inference device
            batch_size: Batch size (number of parallel video streams)
        """
        self.device = torch.device(device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        
        # Load model
        self.model = self._build_model()
        self.model.eval()
        
        # State for multi-video parallel inference
        # Each batch position has its own scene token tracker
        self.prev_scene_tokens: List[Optional[str]] = [None] * batch_size
    
    def _build_model(self) -> nn.Module:
        """
        Build and load model
        
        Returns:
            Model with loaded weights
        """
        # Load config using mmcv
        from mmcv import Config
        cfg = Config.fromfile(self.config_path)
        
        # Import plugin module (register custom modules)
        if hasattr(cfg, 'plugin') and cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                importlib.import_module(_module_path)
        
        # Build model
        from mmdet3d.models import build_model
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        
        # Load weights
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Process state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove possible 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        
        print(f"Model loaded from {self.checkpoint_path}")
        
        return model
    
    def infer(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute inference
        
        Args:
            batch_data: Batch data provided by DataProvider
        
        Returns:
            Detection results list
        """
        # Move data to device
        batch_data = self._to_device(batch_data)
        
        with torch.no_grad():
            # Call model's simple_test
            results = self.model.simple_test(
                img_metas=batch_data['img_metas'],
                img=batch_data['img'],
                lidar2img=batch_data['lidar2img'],
                intrinsics=batch_data['intrinsics'],
                extrinsics=batch_data['extrinsics'],
                timestamp=batch_data['timestamp'],
                img_timestamp=batch_data['img_timestamp'],
                ego_pose=batch_data['ego_pose'],
                ego_pose_inv=batch_data['ego_pose_inv'],
            )
        
        return results
    
    def infer_multi_video(
        self, 
        batch_data: Dict[str, Any], 
        is_first_frame: List[bool]
    ) -> List[Dict[str, Any]]:
        """
        Inference for multi-video parallel mode.
        
        Each batch position is an independent video stream. Scene reset is
        controlled per-position using the is_first_frame flags.
        
        Args:
            batch_data: Batch data provided by DataProvider
            is_first_frame: List of flags indicating if each batch position
                           is at the first frame of its video (needs reset)
        
        Returns:
            Detection results list
        """
        batch_size = len(batch_data['img_metas'])
        
        # Build prev_exists tensor based on is_first_frame flags
        # prev_exists[i] = 0 if first frame (need reset), 1 otherwise
        prev_exists = torch.ones(batch_size, device=self.device)
        for i, is_first in enumerate(is_first_frame):
            if is_first:
                prev_exists[i] = 0.0
        
        # Move data to device
        batch_data = self._to_device(batch_data)
        
        # Add prev_exists to batch_data for model to use
        batch_data['prev_exists'] = prev_exists
        
        with torch.no_grad():
            # Call model's simple_test with prev_exists
            results = self.model.simple_test(
                img_metas=batch_data['img_metas'],
                img=batch_data['img'],
                lidar2img=batch_data['lidar2img'],
                intrinsics=batch_data['intrinsics'],
                extrinsics=batch_data['extrinsics'],
                timestamp=batch_data['timestamp'],
                img_timestamp=batch_data['img_timestamp'],
                ego_pose=batch_data['ego_pose'],
                ego_pose_inv=batch_data['ego_pose_inv'],
                prev_exists=batch_data['prev_exists'],
            )
        
        # Update scene tokens for each position
        for i, meta in enumerate(batch_data['img_metas']):
            self.prev_scene_tokens[i] = meta.get('scene_token')
        
        return results
    
    def reset_multi_video_state(self):
        """
        Reset temporal state for multi-video mode.
        
        Call this when starting a new round of videos.
        """
        self.prev_scene_tokens = [None] * self.batch_size
        self._reset_temporal_state()
    
    def _reset_temporal_state(self):
        """Reset model's temporal state (memory buffers)."""
        if hasattr(self.model, 'pts_bbox_head'):
            head = self.model.pts_bbox_head
            if hasattr(head, 'reset_memory'):
                head.reset_memory()
            if hasattr(head, 'memory_embedding'):
                head.memory_embedding = None
            if hasattr(head, 'memory_reference_point'):
                head.memory_reference_point = None
            if hasattr(head, 'memory_timestamp'):
                head.memory_timestamp = None
            if hasattr(head, 'memory_egopose'):
                head.memory_egopose = None
            if hasattr(head, 'memory_velo'):
                head.memory_velo = None
    
    def _to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move data to specified device
        
        Args:
            data: Data dictionary
        
        Returns:
            Data dictionary after moving to device
        """
        result = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, list):
                # img_metas is list of dict, keep unchanged
                result[k] = v
            else:
                result[k] = v
        return result
    
    @staticmethod
    def parse_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse inference results to a more friendly format
        
        Args:
            results: Raw results from model output
        
        Returns:
            Parsed results list
        """
        parsed_results = []
        
        for result in results:
            if 'pts_bbox' in result:
                pts_bbox = result['pts_bbox']
                
                # Extract detection boxes
                boxes_3d = pts_bbox['boxes_3d']  # LiDARInstance3DBoxes
                scores_3d = pts_bbox['scores_3d']  # tensor
                labels_3d = pts_bbox['labels_3d']  # tensor
                
                # Convert to numpy
                if hasattr(boxes_3d, 'tensor'):
                    boxes_np = boxes_3d.tensor.cpu().numpy()
                else:
                    boxes_np = boxes_3d.cpu().numpy()
                
                scores_np = scores_3d.cpu().numpy()
                labels_np = labels_3d.cpu().numpy()
                
                parsed_results.append({
                    'boxes_3d': boxes_np,  # (N, 9) - x, y, z, w, l, h, yaw, vx, vy
                    'scores_3d': scores_np,  # (N,)
                    'labels_3d': labels_np,  # (N,)
                })
            else:
                parsed_results.append(result)
        
        return parsed_results

