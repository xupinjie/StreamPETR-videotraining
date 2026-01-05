# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# Visualizer - 3D Detection Result Visualizer
# Project 3D bbox onto multi-view images and draw
# ------------------------------------------------------------------------

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# nuScenes class names
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Color for each class (BGR format)
CLASS_COLORS = {
    'car': (0, 255, 0),           # Green
    'truck': (0, 200, 255),       # Orange
    'construction_vehicle': (0, 165, 255),  # Orange-yellow
    'bus': (255, 255, 0),         # Cyan
    'trailer': (255, 0, 255),     # Pink
    'barrier': (128, 128, 128),   # Gray
    'motorcycle': (255, 0, 0),    # Blue
    'bicycle': (255, 128, 0),     # Light blue
    'pedestrian': (0, 0, 255),    # Red
    'traffic_cone': (0, 128, 255),  # Orange-red
}

# Camera names (in nuScenes standard order)
CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def get_3d_box_corners(box: np.ndarray) -> np.ndarray:
    """
    Get 8 corner coordinates of 3D bbox (LiDAR coordinate system)
    
    Note: The z coordinate from model output is the bottom center (not geometric center)
    
    Args:
        box: 3D bbox parameters [x, y, z, w, l, h, yaw, ...]
             where (x, y, z) is bottom center, (w, l, h) is size, yaw is yaw angle
    
    Returns:
        corners: shape (3, 8) - 3D coordinates of 8 corners
    """
    x, y, z = box[0], box[1], box[2]
    w, l, h = box[3], box[4], box[5]
    yaw = box[6]
    
    # Half size (xy direction)
    dx = l / 2
    dy = w / 2
    
    # 8 corners (unrotated)
    # z is bottom center, so bottom z=0, top z=h
    # Order: front-right-bottom, front-left-bottom, back-left-bottom, back-right-bottom,
    #        front-right-top, front-left-top, back-left-top, back-right-top
    corners = np.array([
        [ dx,  dy, 0],  # 0: front-right-bottom
        [ dx, -dy, 0],  # 1: front-left-bottom
        [-dx, -dy, 0],  # 2: back-left-bottom
        [-dx,  dy, 0],  # 3: back-right-bottom
        [ dx,  dy, h],  # 4: front-right-top
        [ dx, -dy, h],  # 5: front-left-top
        [-dx, -dy, h],  # 6: back-left-top
        [-dx,  dy, h],  # 7: back-right-top
    ]).T  # shape: (3, 8)
    
    # Rotation matrix (around Z axis)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Rotate and translate (z is bottom center)
    corners = rot_matrix @ corners
    corners[0, :] += x
    corners[1, :] += y
    corners[2, :] += z
    
    return corners


def project_to_image(points_3d: np.ndarray, lidar2img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D image
    
    Args:
        points_3d: shape (3, N) - N 3D points
        lidar2img: shape (4, 4) - LiDAR to image transformation matrix
    
    Returns:
        points_2d: shape (2, N) - Projected 2D coordinates
        valid_mask: shape (N,) - Valid point mask (in front of camera)
    """
    num_points = points_3d.shape[1]
    
    # Convert to homogeneous coordinates
    points_3d_homo = np.vstack([points_3d, np.ones((1, num_points))])  # (4, N)
    
    # Project
    points_2d_homo = lidar2img @ points_3d_homo  # (4, N)
    
    # Depth (Z coordinate)
    depths = points_2d_homo[2, :]
    
    # Valid points: positive depth (in front of camera)
    valid_mask = depths > 0.1
    
    # Normalize
    points_2d = np.zeros((2, num_points))
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        points_2d[0, valid_indices] = points_2d_homo[0, valid_indices] / depths[valid_indices]
        points_2d[1, valid_indices] = points_2d_homo[1, valid_indices] / depths[valid_indices]
    
    return points_2d, valid_mask


def draw_3d_box(
    image: np.ndarray,
    corners_2d: np.ndarray,
    valid_mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw 3D bbox projection on image
    
    Args:
        image: Input image (H, W, 3)
        corners_2d: shape (2, 8) - 2D coordinates of 8 corners
        valid_mask: shape (8,) - Valid corner mask
        color: Color (B, G, R)
        thickness: Line width
    
    Returns:
        Image with drawing
    """
    image = image.copy()
    
    # 12 edges of 3D bbox connection relationship
    # Bottom face: 0-1, 1-2, 2-3, 3-0
    # Top face: 4-5, 5-6, 6-7, 7-4
    # Vertical edges: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    # Highlight front face with thicker lines
    front_edges = [(0, 1), (4, 5), (0, 4), (1, 5)]
    
    h, w = image.shape[:2]
    
    for i, j in edges:
        if valid_mask[i] and valid_mask[j]:
            pt1 = (int(corners_2d[0, i]), int(corners_2d[1, i]))
            pt2 = (int(corners_2d[0, j]), int(corners_2d[1, j]))
            
            # Check if points are within image bounds
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                line_thickness = thickness + 1 if (i, j) in front_edges else thickness
                cv2.line(image, pt1, pt2, color, line_thickness)
    
    return image


def draw_label(
    image: np.ndarray,
    corners_2d: np.ndarray,
    valid_mask: np.ndarray,
    label: str,
    score: float,
    color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw label next to 3D bbox
    
    Args:
        image: Input image
        corners_2d: shape (2, 8) - 2D coordinates of 8 corners
        valid_mask: shape (8,) - Valid corner mask
        label: Class name
        score: Confidence score
        color: Color
    
    Returns:
        Image with drawing
    """
    image = image.copy()
    h, w = image.shape[:2]
    
    # Find minimum y position of valid corners (top)
    valid_corners = corners_2d[:, valid_mask]
    if valid_corners.shape[1] == 0:
        return image
    
    # Use the topmost point among valid corners as label position
    min_y_idx = np.argmin(valid_corners[1, :])
    label_x = int(valid_corners[0, min_y_idx])
    label_y = int(valid_corners[1, min_y_idx]) - 10
    
    # Ensure label is within image bounds
    label_x = max(0, min(label_x, w - 50))
    label_y = max(15, min(label_y, h - 5))
    
    # Draw text background
    text = f"{label}: {score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.rectangle(image, (label_x, label_y - text_h - 2), 
                  (label_x + text_w, label_y + 2), color, -1)
    
    # Draw text
    cv2.putText(image, text, (label_x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return image


class DetectionVisualizer:
    """
    3D detection result visualizer
    
    Project 3D bbox onto multi-view images and draw
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        class_colors: Dict[str, Tuple[int, int, int]] = None,
        score_threshold: float = 0.3,
        line_thickness: int = 2,
    ):
        """
        Args:
            class_names: Class name list
            class_colors: Class color dictionary
            score_threshold: Confidence threshold
            line_thickness: Line width
        """
        self.class_names = class_names or CLASS_NAMES
        self.class_colors = class_colors or CLASS_COLORS
        self.score_threshold = score_threshold
        self.line_thickness = line_thickness
    
    def get_color(self, label_idx: int) -> Tuple[int, int, int]:
        """
        Get color for the corresponding class
        
        Args:
            label_idx: Class index
        
        Returns:
            Color (B, G, R)
        """
        if label_idx < len(self.class_names):
            class_name = self.class_names[label_idx]
            return self.class_colors.get(class_name, (0, 255, 0))
        return (0, 255, 0)
    
    def visualize_camera(
        self,
        image: np.ndarray,
        boxes_3d: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        lidar2img: np.ndarray,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Visualize 3D detection results on a single camera image
        
        Args:
            image: Input image (H, W, 3), BGR format
            boxes_3d: 3D bbox array, shape (N, 9) - [x, y, z, w, l, h, yaw, vx, vy]
            labels: Class labels, shape (N,)
            scores: Confidence scores, shape (N,)
            lidar2img: LiDAR to image transformation matrix, shape (4, 4)
            draw_labels: Whether to draw labels
        
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        for i in range(len(boxes_3d)):
            if scores[i] < self.score_threshold:
                continue
            
            box = boxes_3d[i]
            label_idx = int(labels[i])
            score = scores[i]
            
            # Get 3D corners
            corners_3d = get_3d_box_corners(box)
            
            # Project to 2D
            corners_2d, valid_mask = project_to_image(corners_3d, lidar2img)
            
            # Skip if no valid corners
            if not valid_mask.any():
                continue
            
            # Get color
            color = self.get_color(label_idx)
            
            # Draw 3D bbox
            vis_image = draw_3d_box(vis_image, corners_2d, valid_mask, 
                                    color, self.line_thickness)
            
            # Draw label
            if draw_labels:
                label_name = self.class_names[label_idx] if label_idx < len(self.class_names) else 'unknown'
                vis_image = draw_label(vis_image, corners_2d, valid_mask, 
                                       label_name, score, color)
        
        return vis_image
    
    def visualize_sample(
        self,
        images: np.ndarray,
        boxes_3d: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        lidar2img: np.ndarray,
        output_path: Optional[str] = None,
        camera_names: List[str] = None,
    ) -> np.ndarray:
        """
        Visualize all camera views for a single sample
        
        Args:
            images: Multi-view images, shape (N_cams, H, W, 3) or (N_cams, C, H, W)
            boxes_3d: 3D bbox array, shape (N_boxes, 9)
            labels: Class labels, shape (N_boxes,)
            scores: Confidence scores, shape (N_boxes,)
            lidar2img: LiDAR to image transformation matrix, shape (N_cams, 4, 4)
            output_path: Save path (optional)
            camera_names: Camera name list
        
        Returns:
            Concatenated visualization image
        """
        camera_names = camera_names or CAMERA_NAMES
        num_cameras = len(images)
        
        vis_images = []
        for cam_idx in range(num_cameras):
            # Get current camera image
            img = images[cam_idx]
            
            # If CHW format, convert to HWC
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            
            # Ensure uint8
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Get current camera's lidar2img
            l2i = lidar2img[cam_idx]
            
            # Visualize
            vis_img = self.visualize_camera(img, boxes_3d, labels, scores, l2i)
            
            # Add camera name
            if cam_idx < len(camera_names):
                cv2.putText(vis_img, camera_names[cam_idx], (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            vis_images.append(vis_img)
        
        # Concatenate images (2 rows, 3 columns)
        # Order: FRONT_LEFT, FRONT, FRONT_RIGHT
        #        BACK_LEFT,  BACK,  BACK_RIGHT
        if num_cameras == 6:
            # Reorder
            # Original order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
            # Target order: FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_LEFT, BACK, BACK_RIGHT
            reorder = [2, 0, 1, 4, 3, 5]
            vis_images = [vis_images[i] for i in reorder]
            
            row1 = np.hstack(vis_images[:3])
            row2 = np.hstack(vis_images[3:])
            combined = np.vstack([row1, row2])
        else:
            # Single row layout
            combined = np.hstack(vis_images)
        
        # Save
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, combined)
            print(f"Visualization saved to {output_path}")
        
        return combined
    
    def visualize_batch(
        self,
        batch_images: np.ndarray,
        batch_results: List[Dict[str, np.ndarray]],
        batch_lidar2img: np.ndarray,
        output_dir: str,
        batch_idx_offset: int = 0,
    ) -> List[str]:
        """
        Visualize results for a batch
        
        Args:
            batch_images: shape (B, N_cams, C, H, W) or (B, N_cams, H, W, C)
            batch_results: Result list containing 'boxes_3d', 'labels_3d', 'scores_3d'
            batch_lidar2img: shape (B, N_cams, 4, 4)
            output_dir: Output directory
            batch_idx_offset: Batch index offset
        
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        batch_size = len(batch_results)
        for i in range(batch_size):
            sample_idx = batch_idx_offset + i
            
            images = batch_images[i]  # (N_cams, C, H, W) or (N_cams, H, W, C)
            result = batch_results[i]
            lidar2img = batch_lidar2img[i]  # (N_cams, 4, 4)
            
            output_path = os.path.join(output_dir, f"sample_{sample_idx:04d}.jpg")
            
            self.visualize_sample(
                images=images,
                boxes_3d=result['boxes_3d'],
                labels=result['labels_3d'],
                scores=result['scores_3d'],
                lidar2img=lidar2img,
                output_path=output_path,
            )
            
            saved_paths.append(output_path)
        
        return saved_paths

