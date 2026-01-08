#!/usr/bin/env python
# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# Main entry point - Execute inference and save results
# ------------------------------------------------------------------------

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, List

import numpy as np
import torch

# Add project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from auto_labeling.data_provider import create_data_provider
from auto_labeling.inference_engine import InferenceEngine


# nuScenes class names
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Default config
DEFAULT_CONFIG = {
    # Data paths
    'data_root': '/data/nuscenes-mini/',
    # 'pkl_path': '/data/nuscenes-mini/stream_petr/nuscenes2d_temporal_infos_val.pkl', #iamge
    'pkl_path': '/data/nuscenes-mini/stream_petr/nuscenes2d_video_temporal_infos_val.pkl', #video
    
    # Model config
    'config_path': 'projects/configs/StreamPETR_video/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py',
    'checkpoint_path': 'work_dirs/StreamPETR_R50-428q_NuImg_60ep/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth',
    
    # Image normalization config
    'img_norm_cfg': {
        'mean': [123.675, 116.28, 103.53],
        'std': [58.395, 57.12, 57.375],
        'to_rgb': True,
    },
    
    # Image augmentation config
    'ida_aug_conf': {
        'resize_lim': (0.38, 0.55),
        'final_dim': (256, 704),
        'bot_pct_lim': (0.0, 0.0),
        'rot_lim': (0.0, 0.0),
        'H': 900,
        'W': 1600,
        'rand_flip': False,  # No flip during inference
    },
    
    # Inference config
    'batch_size': 1,
    'device': 'cuda:0',
    'output_dir': 'auto_labeling_results',
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Auto Labeling Inference')
    
    parser.add_argument('--config', type=str, 
                        default=DEFAULT_CONFIG['config_path'],
                        help='Model config file path')
    parser.add_argument('--checkpoint', type=str,
                        default=DEFAULT_CONFIG['checkpoint_path'],
                        help='Checkpoint file path')
    parser.add_argument('--pkl-path', type=str,
                        default=DEFAULT_CONFIG['pkl_path'],
                        help='Data pkl file path')
    parser.add_argument('--data-root', type=str,
                        default=DEFAULT_CONFIG['data_root'],
                        help='Data root directory')
    parser.add_argument('--output-dir', type=str,
                        default=DEFAULT_CONFIG['output_dir'],
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int,
                        default=DEFAULT_CONFIG['batch_size'],
                        help='Batch size for inference')
    parser.add_argument('--device', type=str,
                        default=DEFAULT_CONFIG['device'],
                        help='Device for inference')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None for all)')
    parser.add_argument('--score-threshold', type=float, default=0.3,
                        help='Score threshold for filtering detections')
    parser.add_argument('--gpu-preprocess', action='store_true',
                        default=True,
                        help='Use GPU for image preprocessing (resize, normalize, pad)')
    
    # Data loading mode arguments
    parser.add_argument('--data-mode', type=str, default='nvc_video',
                        choices=['image', 'opencv_video', 'nvc_video', 'nvc_video_async'],
                        help='Data loading mode: "image" for loading from image files, '
                             '"opencv_video" for decoding from video files using OpenCV (CPU), '
                             '"nvc_video" for decoding from video files using NVC (GPU sync), '
                             '"nvc_video_async" for NVC with async prefetching')
    parser.add_argument('--video-base-path', type=str, default='/data/nuscenes-mini/video_samples',
                        help='Base path for video files (only used when data-mode uses video). '
                             'If not specified, uses data-root.')
    parser.add_argument('--max-cache-size', type=int, default=6,
                        help='Maximum number of VideoCapture objects to cache '
                             '(only used when data-mode=opencv_video)')
    parser.add_argument('--num-cameras', type=int, default=6,
                        help='Number of cameras for NVC decoder initialization '
                             '(only used when data-mode=nvc_video or nvc_video_async)')
    
    return parser.parse_args()


def save_results_json(results: List[Dict[str, Any]], 
                      img_metas: List[Dict[str, Any]],
                      output_path: str,
                      score_threshold: float = 0.3):
    """
    Save results in JSON format
    
    Args:
        results: Parsed detection results
        img_metas: Image metadata
        output_path: Output file path
        score_threshold: Score threshold
    """
    output_data = []
    
    for i, (result, meta) in enumerate(zip(results, img_metas)):
        boxes_3d = result['boxes_3d']
        scores_3d = result['scores_3d']
        labels_3d = result['labels_3d']
        
        # Filter low score detections
        mask = scores_3d >= score_threshold
        boxes_3d = boxes_3d[mask]
        scores_3d = scores_3d[mask]
        labels_3d = labels_3d[mask]
        
        # Build detection list
        detections = []
        for j in range(len(boxes_3d)):
            box = boxes_3d[j]
            det = {
                'translation': box[:3].tolist(),  # x, y, z
                'size': box[3:6].tolist(),  # w, l, h
                'rotation_yaw': float(box[6]),
                'velocity': box[7:9].tolist() if len(box) > 7 else [0.0, 0.0],
                'score': float(scores_3d[j]),
                'label': int(labels_3d[j]),
                'label_name': CLASS_NAMES[int(labels_3d[j])] if int(labels_3d[j]) < len(CLASS_NAMES) else 'unknown',
            }
            detections.append(det)
        
        sample_data = {
            'sample_idx': meta.get('sample_idx', str(i)),
            'scene_token': meta.get('scene_token', ''),
            'filename': meta.get('filename', []),
            'num_detections': len(detections),
            'detections': detections,
        }
        output_data.append(sample_data)
    
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    """
    Main function
    
    NVTX ranges are added for performance profiling with Nsight Systems/Compute:
    - DataProvider_Init: Data provider initialization
    - InferenceEngine_Init: Inference engine initialization
    - Inference_Loop: Overall inference loop
    - Get_Batch: Batch data loading
    - Model_Inference: Model forward pass
    - Parse_Results: Result parsing
    - Save_Results: Result saving to JSON
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Auto Labeling Inference")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"PKL path: {args.pkl_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU preprocess: {args.gpu_preprocess}")
    print(f"Data mode: {args.data_mode}")
    if args.data_mode == 'opencv_video':
        print(f"Video base path: {args.video_base_path or args.data_root}")
        print(f"Max cache size: {args.max_cache_size}")
    elif args.data_mode in ['nvc_video', 'nvc_video_async']:
        print(f"Video base path: {args.video_base_path or args.data_root}")
        print(f"Num cameras: {args.num_cameras}")
        if args.data_mode == 'nvc_video_async':
            print(f"Async prefetch: enabled")
    print("=" * 60)
    
    # Initialize data provider
    print("\nInitializing DataProvider...")
    with torch.cuda.nvtx.range("DataProvider_Init"):
        # Build provider kwargs based on mode
        provider_kwargs = {
            'mode': args.data_mode,
            'pkl_path': args.pkl_path,
            'data_root': args.data_root,
            'img_norm_cfg': DEFAULT_CONFIG['img_norm_cfg'],
            'ida_aug_conf': DEFAULT_CONFIG['ida_aug_conf'],
            'gpu_preprocess': args.gpu_preprocess,
            'device': args.device,
        }
        
        # Add video-specific arguments based on mode
        if args.data_mode == 'opencv_video':
            provider_kwargs['video_base_path'] = args.video_base_path
            provider_kwargs['max_cache_size'] = args.max_cache_size
        elif args.data_mode in ['nvc_video', 'nvc_video_async']:
            provider_kwargs['video_base_path'] = args.video_base_path
            provider_kwargs['num_cameras'] = args.num_cameras
        
        data_provider = create_data_provider(**provider_kwargs)
    
    total_samples = len(data_provider)
    num_samples = args.num_samples if args.num_samples else total_samples
    num_samples = min(num_samples, total_samples)
    print(f"Total samples: {total_samples}, processing: {num_samples}")
    
    # Initialize inference engine
    print("\nInitializing InferenceEngine...")
    with torch.cuda.nvtx.range("InferenceEngine_Init"):
        engine = InferenceEngine(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    
    # Inference loop
    print("\nStarting inference...")
    all_results = []
    all_img_metas = []
    
    # Warmup tracking: skip first batch for throughput calculation
    warmup_done = False
    warmup_samples = 0
    start_time = time.time()
    
    # Timing statistics (excluding warmup)
    data_times = []    # Time for data loading
    infer_times = []   # Time for model inference
    parse_times = []   # Time for result parsing
    
    with torch.cuda.nvtx.range("Inference_Loop"):
        for i in range(0, num_samples, args.batch_size):
            batch_end = min(i + args.batch_size, num_samples)
            indices = list(range(i, batch_end))
            
            # Get batch data
            with torch.cuda.nvtx.range("Get_Batch"):
                t_data_start = time.time()
                batch_data = data_provider.get_batch(indices)
                t_data_end = time.time()
            
            # Execute inference
            with torch.cuda.nvtx.range("Model_Inference"):
                t_infer_start = time.time()
                results = engine.infer_with_scene_reset(batch_data)
                t_infer_end = time.time()
            
            # Parse results
            with torch.cuda.nvtx.range("Parse_Results"):
                t_parse_start = time.time()
                parsed_results = InferenceEngine.parse_results(results)
                t_parse_end = time.time()
            
            all_results.extend(parsed_results)
            all_img_metas.extend(batch_data['img_metas'])
            
            # Reset timer after first batch (warmup)
            if not warmup_done:
                warmup_done = True
                warmup_samples = len(indices)
                start_time = time.time()
                print(f"\rWarmup completed ({warmup_samples} samples), starting timing...")
                continue
            
            # Collect timing statistics (excluding warmup)
            data_times.append(t_data_end - t_data_start)
            infer_times.append(t_infer_end - t_infer_start)
            parse_times.append(t_parse_end - t_parse_start)
            
            # Print progress (excluding warmup samples)
            processed_samples = i + len(indices) - warmup_samples
            elapsed = time.time() - start_time
            fps = processed_samples / elapsed if elapsed > 0 else 0
            print(f"\rProcessed {i + len(indices)}/{num_samples} samples "
                  f"({100 * (i + len(indices)) / num_samples:.1f}%) "
                  f"- {fps:.2f} samples/sec (excl. warmup)", end='')
    
    print()  # Newline
    
    total_time = time.time() - start_time
    timed_samples = num_samples - warmup_samples
    
    # Calculate timing statistics
    total_data_time = sum(data_times)
    total_infer_time = sum(infer_times)
    total_parse_time = sum(parse_times)
    total_measured_time = total_data_time + total_infer_time + total_parse_time
    
    avg_data_time = total_data_time / len(data_times) if data_times else 0
    avg_infer_time = total_infer_time / len(infer_times) if infer_times else 0
    avg_parse_time = total_parse_time / len(parse_times) if parse_times else 0
    
    # Print performance report
    print("\n" + "=" * 60)
    print("Performance Report (excluding warmup)")
    print("=" * 60)
    print(f"Warmup samples: {warmup_samples}")
    print(f"Timed samples:  {timed_samples}")
    print(f"Total time:     {total_time:.3f} sec")
    print("-" * 60)
    print(f"{'Stage':<20} {'Total (sec)':<15} {'Avg (ms)':<15} {'Ratio (%)':<10}")
    print("-" * 60)
    print(f"{'Data Loading':<20} {total_data_time:<15.3f} {avg_data_time*1000:<15.3f} {100*total_data_time/total_measured_time if total_measured_time > 0 else 0:<10.1f}")
    print(f"{'Model Inference':<20} {total_infer_time:<15.3f} {avg_infer_time*1000:<15.3f} {100*total_infer_time/total_measured_time if total_measured_time > 0 else 0:<10.1f}")
    print(f"{'Result Parsing':<20} {total_parse_time:<15.3f} {avg_parse_time*1000:<15.3f} {100*total_parse_time/total_measured_time if total_measured_time > 0 else 0:<10.1f}")
    print("-" * 60)
    print(f"{'Total Measured':<20} {total_measured_time:<15.3f}")
    print("=" * 60)
    print(f"Throughput: {timed_samples / total_time:.2f} samples/sec")
    print("=" * 60)
    
    # Save results
    output_path = os.path.join(args.output_dir, 'detection_results.json')
    with torch.cuda.nvtx.range("Save_Results"):
        save_results_json(
            all_results, 
            all_img_metas, 
            output_path,
            score_threshold=args.score_threshold
        )
    
    # Count total detections
    total_detections = sum(len(r['boxes_3d']) for r in all_results)
    print(f"\nTotal detections: {total_detections}")
    print(f"Average detections per sample: {total_detections / num_samples:.2f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

