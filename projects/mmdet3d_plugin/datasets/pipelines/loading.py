# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Custom video loading pipeline for StreamPETR with GOP-based video decoding
# ------------------------------------------------------------------------

import mmcv
import numpy as np
import torch
import os
import time
import nvtx
from dataclasses import dataclass
from typing import List, Optional

from mmdet.datasets.builder import PIPELINES
import accvlab.on_demand_video_decoder as nvc


@dataclass
class GOPInfo:
    """
    Information about a single GOP file.

    Attributes:
        file_path (str): Full path to the GOP binary file
        first_frame_id (int): Index of the first frame in this GOP
        gop_len (int): Number of frames in this GOP
    """

    file_path: str
    first_frame_id: int
    gop_len: int


@dataclass
class GOPCacheLine:
    """
    Single GOP cache line.

    Attributes:
        video_path (str): Video file path
        first_frame_id (int): First frame ID of this GOP
        gop_len (int): Number of frames in this GOP
        gop_data (np.ndarray): GOP packet data
        last_access_time (float): Last access timestamp for LRU
        valid (bool): Whether this cache line is valid
    """

    video_path: str
    first_frame_id: int
    gop_len: int
    gop_data: np.ndarray
    last_access_time: float
    valid: bool = True


class GOPCache:
    """
    LRU cache for GOP data.

    Caches GOP packet data to avoid repeated disk reads.
    Uses simple linear scan for lookup (suitable for capacity < 200).
    """

    def __init__(self, capacity: int):
        """
        Initialize GOP cache.

        Args:
            capacity (int): Maximum number of GOP cache lines
        """
        self.capacity = capacity
        self.cache_lines: List[GOPCacheLine] = []
        self.hits = 0
        self.misses = 0
        
        # Get worker ID for debug printing
        import multiprocessing
        self.worker_id = multiprocessing.current_process().name

    def lookup(self, video_path: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Look up GOP data containing the specified frame.

        Args:
            video_path (str): Video file path
            frame_id (int): Target frame ID

        Returns:
            Optional[np.ndarray]: GOP data if found (cache hit), None otherwise (cache miss)
        """
        for cache_line in self.cache_lines:
            if (cache_line.valid and
                cache_line.video_path == video_path and
                cache_line.first_frame_id <= frame_id < cache_line.first_frame_id + cache_line.gop_len):
                # Cache hit
                cache_line.last_access_time = time.time()
                self.hits += 1
                return cache_line.gop_data

        # Cache miss
        self.misses += 1
        return None

    def insert(self, video_path: str, first_frame_id: int, gop_len: int, gop_data: np.ndarray):
        """
        Insert GOP data into cache.

        If cache is full, evicts the least recently used (LRU) cache line.

        Args:
            video_path (str): Video file path
            first_frame_id (int): First frame ID of the GOP
            gop_len (int): Number of frames in the GOP
            gop_data (np.ndarray): GOP packet data
        """
        current_time = time.time()

        # Check if this GOP already exists in cache
        for cache_line in self.cache_lines:
            if (cache_line.valid and
                cache_line.video_path == video_path and
                cache_line.first_frame_id == first_frame_id):
                # Update existing cache line
                cache_line.gop_data = gop_data
                cache_line.gop_len = gop_len
                cache_line.last_access_time = current_time
                return

        # Cache is not full, add new cache line
        if len(self.cache_lines) < self.capacity:
            new_cache_line = GOPCacheLine(
                video_path=video_path,
                first_frame_id=first_frame_id,
                gop_len=gop_len,
                gop_data=gop_data,
                last_access_time=current_time,
                valid=True
            )
            self.cache_lines.append(new_cache_line)
            return

        # Cache is full, evict LRU cache line
        lru_idx = 0
        lru_time = self.cache_lines[0].last_access_time
        for i in range(1, len(self.cache_lines)):
            if self.cache_lines[i].last_access_time < lru_time:
                lru_time = self.cache_lines[i].last_access_time
                lru_idx = i

        # Replace LRU cache line with new data
        self.cache_lines[lru_idx] = GOPCacheLine(
            video_path=video_path,
            first_frame_id=first_frame_id,
            gop_len=gop_len,
            gop_data=gop_data,
            last_access_time=current_time,
            valid=True
        )

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Dictionary containing cache statistics
        """
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            'capacity': self.capacity,
            'size': len(self.cache_lines),
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': total_accesses,
            'hit_rate': hit_rate,
        }

    def clear(self):
        """Clear all cache entries and reset statistics."""
        self.cache_lines.clear()
        self.hits = 0
        self.misses = 0


class GOPStorageManager:
    """
    Manager for storing and loading GOP (Group of Pictures) data using binary format.

    This class provides a hierarchical storage system for video GOP data.
    """

    def __init__(
        self, nv_gop_demuxer, video_base_path: str, gop_base_path: str, clip_size: int, 
        cache_capacity: int = 0
    ):
        """
        Initialize the GOP Storage Manager.

        Args:
            nv_gop_demuxer: GOP demuxer instance
            video_base_path (str): Base directory containing video files
            gop_base_path (str): Base directory for storing GOP data
            clip_size (int): Maximum number of files to use
            cache_capacity (int): Number of GOP cache lines. 0 disables caching. Default: 0
        """
        self.video_base_path = video_base_path
        self.gop_base_path = gop_base_path
        self.nv_gop_demuxer = nv_gop_demuxer
        
        # Initialize GOP cache if capacity > 0
        self.gop_cache = GOPCache(cache_capacity) if cache_capacity > 0 else None
        if self.gop_cache:
            import multiprocessing
            worker_id = multiprocessing.current_process().name
            print(f"[Worker: {worker_id}] GOP cache initialized with capacity {cache_capacity}")

    def _load_persistent_index(self, video_path: str) -> List[GOPInfo]:
        """
        Load GOP index from persistent index file or scan directory for GOP files.

        Args:
            video_path (str): Path to the video file

        Returns:
            List[GOPInfo]: Sorted list of GOP information
        """
        # Convert to real path (resolves symlinks) to handle both relative and absolute paths correctly
        video_realpath = os.path.realpath(video_path)
        video_base_realpath = os.path.realpath(self.video_base_path)
        
        # Calculate relative path from base
        video_relpath = os.path.relpath(video_realpath, video_base_realpath)
        gop_dir = os.path.join(self.gop_base_path, video_relpath)

        if not os.path.exists(gop_dir):
            print(f"WARNING: GOP directory not found:")
            print(f"         Video path (input): {video_path}")
            print(f"         Video path (real): {video_realpath}")
            print(f"         Video base path (real): {video_base_realpath}")
            print(f"         Video relative path: {video_relpath}")
            print(f"         Expected GOP directory: {gop_dir}")
            print(f"         Please run GOP generation first or check path configuration.")
            return []

        gop_infos = []

        # Scan directory and parse filenames directly
        torch.cuda.nvtx.range_push("scan_gop_dir")
        try:
            files = os.listdir(gop_dir)
            for file in files:
                if file.endswith(".bin") and file.startswith("gop."):
                    file_path = os.path.join(gop_dir, file)
                    # Parse filename: gop.{first_frame_id}.{gop_len}.bin
                    file_name_parts = file.split(".")
                    if len(file_name_parts) >= 4:  # gop, first_frame_id, gop_len, bin
                        try:
                            first_frame_id = int(file_name_parts[-3])
                            gop_len = int(file_name_parts[-2])
                            gop_infos.append(
                                GOPInfo(
                                    file_path=file_path, first_frame_id=first_frame_id, gop_len=gop_len
                                )
                            )
                        except ValueError:
                            continue  # Skip invalid filenames
        except OSError:
            pass  # Directory might not be readable
        torch.cuda.nvtx.range_pop()  # scan_gop_dir

        # Sort by first_frame_id for efficient searching
        gop_infos.sort(key=lambda x: x.first_frame_id)
        return gop_infos

    def _find_gop_for_frame(self, gop_infos: List[GOPInfo], frame_idx: int) -> Optional[GOPInfo]:
        """
        Find the GOP that contains the specified frame using binary search.

        Args:
            gop_infos (List[GOPInfo]): Sorted list of GOP information
            frame_idx (int): Target frame index

        Returns:
            Optional[GOPInfo]: GOP info if found, None otherwise
        """
        # Use binary search since gop_infos is sorted by first_frame_id
        left, right = 0, len(gop_infos) - 1

        while left <= right:
            mid = (left + right) // 2
            gop_info = gop_infos[mid]

            if gop_info.first_frame_id <= frame_idx < gop_info.first_frame_id + gop_info.gop_len:
                return gop_info
            elif frame_idx < gop_info.first_frame_id:
                right = mid - 1
            else:
                left = mid + 1

        return None

    def load_gops(self, frame_ids: List[int], video_paths: List[str]) -> Optional[List[np.ndarray]]:
        """
        Load multiple GOPs with hybrid cache + batch loading strategy.

        This method implements:
        1. Batch cache lookup for all requests
        2. Batch load cache misses using LoadGopsToList
        3. Insert newly loaded GOPs into cache
        4. Reorder results to match input order

        Args:
            frame_ids (List[int]): List of target frame indices
            video_paths (List[str]): List of video file paths

        Returns:
            Optional[List[np.ndarray]]: List of numpy arrays, one per video path,
                                       or None if any GOP is not found
        """
        torch.cuda.nvtx.range_push("load_gops_hybrid")

        if len(frame_ids) != len(video_paths):
            print(f"Error: frame_ids length ({len(frame_ids)}) != video_paths length ({len(video_paths)})")
            torch.cuda.nvtx.range_pop()
            return None

        # ========== Phase 1: Batch cache lookup ==========
        torch.cuda.nvtx.range_push("batch_cache_lookup")
        cache_hits = []  # [(index, gop_data), ...]
        cache_misses = []  # [(index, video_path, frame_id, gop_info), ...]

        for i, (video_path, frame_id) in enumerate(zip(video_paths, frame_ids)):
            # Try cache if enabled
            if self.gop_cache is not None:
                cached_data = self.gop_cache.lookup(video_path, frame_id)
                if cached_data is not None:
                    # Cache hit
                    cache_hits.append((i, cached_data))
                    continue

            # Cache miss - need to find GOP info from index
            gop_infos = self._load_persistent_index(video_path)
            if not gop_infos:
                print(f"Error: No GOP index found for video {video_path}")
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
                return None

            # Find the GOP containing the target frame
            gop_info = self._find_gop_for_frame(gop_infos, frame_id)
            if gop_info is None:
                print(f"Error: Could not find GOP for video {video_path}, frame {frame_id}")
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
                return None

            cache_misses.append((i, video_path, frame_id, gop_info))

        torch.cuda.nvtx.range_pop()  # batch_cache_lookup

        # ========== Phase 2: Batch load cache misses ==========
        miss_results = []
        if cache_misses:
            torch.cuda.nvtx.range_push("batch_load_misses")
            
            miss_gop_files = [gop_info.file_path for _, _, _, gop_info in cache_misses]
            
            try:
                # Batch load using LoadGopsToList - KEY OPTIMIZATION!
                loaded_gop_list = self.nv_gop_demuxer.LoadGopsToList(miss_gop_files)
                
                # Store results and insert into cache
                torch.cuda.nvtx.range_push("cache_insert_batch")
                for (idx, video_path, frame_id, gop_info), gop_data in \
                    zip(cache_misses, loaded_gop_list):
                    
                    miss_results.append((idx, gop_data))
                    
                    # Insert into cache if enabled
                    if self.gop_cache is not None:
                        self.gop_cache.insert(
                            video_path=video_path,
                            first_frame_id=gop_info.first_frame_id,
                            gop_len=gop_info.gop_len,
                            gop_data=gop_data
                        )
                torch.cuda.nvtx.range_pop()  # cache_insert_batch
                
            except Exception as e:
                print(f"Error batch loading GOPs: {e}")
                torch.cuda.nvtx.range_pop()  # batch_load_misses
                torch.cuda.nvtx.range_pop()  # load_gops_hybrid
                return None
            
            torch.cuda.nvtx.range_pop()  # batch_load_misses

        # ========== Phase 3: Reorder to match input order ==========
        torch.cuda.nvtx.range_push("reorder_results")
        gop_data_list = [None] * len(video_paths)

        # Fill cache hits
        for idx, gop_data in cache_hits:
            gop_data_list[idx] = gop_data

        # Fill newly loaded
        for idx, gop_data in miss_results:
            gop_data_list[idx] = gop_data

        # Verify completeness
        if any(gop is None for gop in gop_data_list):
            print(f"Error: Some GOP data is missing after reordering")
            torch.cuda.nvtx.range_pop()  # reorder_results
            torch.cuda.nvtx.range_pop()  # load_gops_hybrid
            return None

        torch.cuda.nvtx.range_pop()  # reorder_results
        torch.cuda.nvtx.range_pop()  # load_gops_hybrid
        return gop_data_list

    def get_cache_stats(self) -> Optional[dict]:
        """
        Get GOP cache statistics.

        Returns:
            Optional[dict]: Cache statistics if cache is enabled, None otherwise
        """
        if self.gop_cache is None:
            return None
        return self.gop_cache.get_stats()

    def clear_cache(self):
        """Clear GOP cache if enabled."""
        if self.gop_cache is not None:
            self.gop_cache.clear()


@PIPELINES.register_module()
class LoadMultiViewImageFromGOP_demuxer_free(object):
    """
    Load multi-view images from video files using GOP (Group of Pictures) decoding.
    
    This is a custom loader that supports both:
    - Traditional image file loading (when video_paths is None)
    - Video GOP-based loading with GPU decoding (when video_paths is provided)

    Expects results['img_filename'] or results['video_paths'] to be a list of filenames/paths.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
        cache_capacity (int, optional): Number of GOP cache lines for caching.
            Defaults to 192 (16 * 6 * 2).
    """

    def __init__(self, batch_size, num_cameras, ida_aug_conf, video_base_path, gop_base_path, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.cache_capacity = batch_size * num_cameras * 2
        self.batch_size = batch_size
        self.num_cameras = num_cameras
        self.ida_aug_conf = ida_aug_conf
        self._storage: GOPStorageManager = None
        self._nv_gop_dec = None
        
        # For cache statistics
        self._call_count = 0
        self._stats_print_interval = 100  # Print stats every 100 calls

        self.video_base_path = video_base_path
        self.gop_base_path = gop_base_path

    def __lazy_init__(self):
        """Lazy initialization of GOP decoder and storage manager."""
        if self._nv_gop_dec is not None:
            return

        self._nv_gop_dec = nvc.CreateGopDecoder(
            maxfiles=self.num_cameras,  # Maximum number of files to use
            iGpu=int(os.environ["LOCAL_RANK"]),  # GPU ID
        )

        self._storage = GOPStorageManager(
            nv_gop_demuxer=self._nv_gop_dec,
            video_base_path=self.video_base_path,
            gop_base_path=self.gop_base_path,
            clip_size=self.num_cameras,
            cache_capacity=self.cache_capacity,  # Enable GOP cache
        )
        self._is_initialized = True

    @nvtx.annotate("LoadMultiViewImageWithGOP", color="green")
    def __call__(self, results):
        """
        Call function to load multi-view image from files or videos.

        Args:
            results (dict): Result dict containing multi-view image filenames or video paths.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        H = self.ida_aug_conf["H"]
        W = self.ida_aug_conf["W"]
        C = 3
        NUM_CAMERAS = self.num_cameras
        
        if results["video_paths"] is None:
            raise ValueError("video_paths is not provided")
            
        # Video GOP-based loading path
        filename = results['video_paths']
        video_frame_ids = results["video_frame_ids"]

        self.__lazy_init__()
        
        # Load GOPs (returns List[np.ndarray] with cache support)
        gop_data_list = self._storage.load_gops(video_frame_ids, filename)
        if gop_data_list is None:
            print(f"ERROR: Failed to load GOPs for videos {filename}, frames {video_frame_ids}")
            print(f"       Generating random images as fallback")
            # Generate random fallback images
            img = np.random.randint(0, 256, (H, W, C, NUM_CAMERAS), dtype=np.uint8)
            if self.to_float32:
                img = img.astype(np.float32)
        else:
            # Decode frames with error handling using DecodeFromGOPListRGB
            try:
                decoded_frames = self._nv_gop_dec.DecodeFromGOPListRGB(
                    gop_data_list, filename, video_frame_ids, True  # BGR
                )
                
                # Validate decoded frames
                if decoded_frames is None or len(decoded_frames) == 0:
                    raise RuntimeError(f"DecodeFromGOPListRGB returned empty result for {len(filename)} videos")
                
                if len(decoded_frames) != len(filename):
                    raise RuntimeError(
                        f"DecodeFromGOPListRGB returned {len(decoded_frames)} frames "
                        f"but expected {len(filename)} frames"
                    )
                
            except Exception as e:
                print(f"ERROR: DecodeFromGOPListRGB failed with exception: {e}")
                print(f"       Videos: {filename}")
                print(f"       Frame IDs: {video_frame_ids}")
                print(f"       Generating random images as fallback")
                # Generate random fallback images
                img = np.random.randint(0, 256, (H, W, C, NUM_CAMERAS), dtype=np.uint8)
                if self.to_float32:
                    img = img.astype(np.float32)
            else:
                # Successfully decoded, now convert to tensors
                img = []
                for i, decoded_frame in enumerate(decoded_frames):
                    try:
                        torch_bgr = torch.tensor(decoded_frame, dtype=torch.uint8).to(torch.cuda.current_device())
                        
                        # Validate frame shape
                        if torch_bgr.shape[0] != H or torch_bgr.shape[1] != W or torch_bgr.shape[2] != C:
                            print(f"WARNING: Frame {i} shape mismatch: got {torch_bgr.shape}, expected ({H}, {W}, {C})")
                            print(f"         Generating random image for this frame")
                            torch_bgr = torch.randint(0, 256, (H, W, C), dtype=torch.uint8).to(torch.cuda.current_device())
                    
                    except Exception as e:
                        print(f"ERROR: Failed to convert frame {i} to tensor: {e}")
                        print(f"       Generating random image for this frame")
                        torch_bgr = torch.randint(0, 256, (H, W, C), dtype=torch.uint8).to(torch.cuda.current_device())
                    img.append(torch_bgr)
        
        img = torch.stack(img, axis=-1)
        img = img.detach().cpu().numpy()
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results
        
        # Print cache statistics periodically
        if self._storage is not None:
            self._call_count += 1
            if self._call_count % self._stats_print_interval == 0:
                cache_stats = self._storage.get_cache_stats()
                if cache_stats:
                    import multiprocessing
                    worker_id = multiprocessing.current_process().name
                    print(f"[{worker_id}] GOP Cache Stats after {self._call_count} calls: "
                          f"hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                          f"hit_rate={cache_stats['hit_rate']:.2%}, "
                          f"size={cache_stats['size']}/{cache_stats['capacity']}")
        
        return results

    def __repr__(self):
        """Return a string representation of the object."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'batch_size={self.batch_size}, '
        repr_str += f'num_cameras={self.num_cameras}, '
        repr_str += f'ida_aug_conf={self.ida_aug_conf}, '
        repr_str += f'to_float32={self.to_float32}, '
        repr_str += f'color_type={self.color_type})'
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromGOP_demuxer_free_gop_output(object):
    """
    Load multi-view images from video files using GOP (Group of Pictures) decoding.
    
    This is a custom loader that supports both:
    - Traditional image file loading (when video_paths is None)
    - Video GOP-based loading with GPU decoding (when video_paths is provided)

    Expects results['img_filename'] or results['video_paths'] to be a list of filenames/paths.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
        final_dim (tuple, optional): Final image dimensions after GPU transforms (H, W).
            Used to set correct metadata in worker. Defaults to (256, 704).
        cache_capacity (int, optional): Number of GOP cache lines for caching.
            Defaults to 192 (16 * 6 * 2).
    """

    def __init__(self, batch_size, num_cameras, ida_aug_conf, video_base_path, gop_base_path, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.cache_capacity = batch_size * num_cameras * 2
        self.batch_size = batch_size
        self.num_cameras = num_cameras
        self.ida_aug_conf = ida_aug_conf
        self.final_dim = ida_aug_conf["final_dim"]
        self._storage: GOPStorageManager = None
        self._nv_gop_dec = None
        
        # For cache statistics
        self._call_count = 0
        self._stats_print_interval = 100  # Print stats every 100 calls

        self.video_base_path = video_base_path
        self.gop_base_path = gop_base_path
    def __lazy_init__(self):
        """Lazy initialization of GOP decoder and storage manager."""
        if self._nv_gop_dec is not None:
            return

        self._nv_gop_dec = nvc.CreateGopDecoder(
            maxfiles=self.num_cameras,  # Maximum number of files to use
            iGpu=int(os.environ["LOCAL_RANK"]),  # GPU ID
        )
        self._storage = GOPStorageManager(
            nv_gop_demuxer=self._nv_gop_dec,
            video_base_path=self.video_base_path,
            gop_base_path=self.gop_base_path,
            clip_size=self.num_cameras,
            cache_capacity=self.cache_capacity,  # Enable GOP cache
        )

    @nvtx.annotate("LoadMultiViewImageWithGOP", color="green")
    def __call__(self, results):
        """
        Call function to load multi-view GOP data from videos (no decoding in worker).

        Args:
            results (dict): Result dict containing multi-view image filenames or video paths.

        Returns:
            dict: The result dict containing GOP data (for GPU decoding) or images (for traditional path).
                Added keys and values are described below.

                - filename (str): Multi-view image filenames or video paths.
                - GOP (list): GOP byte data (only for video path, to be decoded on GPU).
                - video_frame_ids (list): Frame IDs for each GOP (only for video path).
                - img_shape (tuple[int]): Expected shape after decoding.
                - ori_shape (tuple[int]): Original shape.
        """
        if results["video_paths"] is None:
            raise ValueError("video_paths is not provided")
        
        # Video GOP-based loading path - ONLY LOAD GOP DATA, NO DECODING
        filename = results['video_paths']
        video_frame_ids = results["video_frame_ids"]

        self.__lazy_init__()

        H = self.ida_aug_conf["H"]
        W = self.ida_aug_conf["W"]
        C = 3
        NUM_CAMERAS = self.num_cameras
        
        # Load GOPs (returns List[np.ndarray] - raw GOP bytes)
        gop_data_list = self._storage.load_gops(video_frame_ids, filename)
        
        # Store GOP data for GPU decoding in main process
        results['GOP'] = gop_data_list  # List of GOP byte arrays
        results['filename'] = filename
        results['video_frame_ids'] = video_frame_ids
        
        # Set expected shapes based on final_dim (after GPU transforms)
        # IMPORTANT: Worker sets the FINAL shape (after transforms), not original
        # This way Model forward doesn't need to update metadata
        final_H, final_W = self.final_dim
        
        # Calculate padded dimensions (divisible by 32)
        pad_H = int(np.ceil(final_H / 32)) * 32
        pad_W = int(np.ceil(final_W / 32)) * 32
        
        # Set final shape for each camera
        final_cam_shape = (pad_H, pad_W, C)
        original_cam_shape = (H, W, C)
        
        results['img_shape'] = [final_cam_shape] * NUM_CAMERAS  # Final shape after transforms
        results['ori_shape'] = [original_cam_shape] * NUM_CAMERAS  # Original video frame shape
        results['pad_shape'] = [final_cam_shape] * NUM_CAMERAS  # Same as img_shape
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.zeros(C, dtype=np.float32),
            std=np.ones(C, dtype=np.float32),
            to_rgb=False)
        
        # NOTE: results['img'] is NOT set here - will be decoded on GPU in main process
        
        # Print cache statistics periodically
        self._call_count += 1
        if self._call_count % self._stats_print_interval == 0:
            cache_stats = self._storage.get_cache_stats()
            if cache_stats:
                import multiprocessing
                worker_id = multiprocessing.current_process().name
                print(f"[{worker_id}] GOP Cache Stats after {self._call_count} calls: "
                        f"hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                        f"hit_rate={cache_stats['hit_rate']:.2%}, "
                        f"size={cache_stats['size']}/{cache_stats['capacity']}")
    
        return results

    def __repr__(self):
        """Return a string representation of the object."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}', "
        repr_str += f'cache_capacity={self.cache_capacity})'
        return repr_str

