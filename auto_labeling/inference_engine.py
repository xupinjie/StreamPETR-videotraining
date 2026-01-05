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
    Inference engine, encapsulates model loading and inference
    
    Functions:
    1. Load model config and weights
    2. Execute inference
    3. Return detection results
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = 'cuda:0',
    ):
        """
        Args:
            config_path: Config file path
            checkpoint_path: Checkpoint file path
            device: Inference device
        """
        self.device = torch.device(device)
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = self._build_model()
        self.model.eval()
        
        # State for temporal inference
        self.prev_scene_token = None
    
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
    
    def infer_with_scene_reset(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Inference with scene reset (for temporal models)
        
        Reset model's temporal state when a new scene is detected
        
        Args:
            batch_data: Batch data provided by DataProvider
        
        Returns:
            Detection results list
        """
        # Check if scene reset is needed
        current_scene_token = batch_data['img_metas'][0].get('scene_token')
        if current_scene_token != self.prev_scene_token:
            self._reset_temporal_state()
            self.prev_scene_token = current_scene_token
        
        return self.infer(batch_data)
    
    def _reset_temporal_state(self):
        """Reset model's temporal state"""
        if hasattr(self.model, 'prev_scene_token'):
            self.model.prev_scene_token = None
        if hasattr(self.model, 'pts_bbox_head'):
            head = self.model.pts_bbox_head
            if hasattr(head, 'reset_memory'):
                head.reset_memory()
            if hasattr(head, 'memory'):
                head.memory = None
    
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

