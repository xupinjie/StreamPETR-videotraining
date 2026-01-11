# ------------------------------------------------------------------------
# Auto Labeling Inference Framework
# MultiVideoSampler - Sampler for parallel multi-video inference
# ------------------------------------------------------------------------

from typing import List, Dict, Tuple


class MultiVideoSampler:
    """
    Sampler for parallel multi-video inference.
    
    Manages multiple video streams simultaneously, where each batch position
    corresponds to an independent video. Supports:
    - Automatic video repetition when batch_size > num_videos
    - Automatic padding when remaining videos < batch_size at the end
    - Independent frame pointer for each batch position
    
    Example:
        - 10 videos, batch_size=4: Creates 4 parallel streams, 
          processes ceil(10/4)=3 rounds, last round pads 2 empty slots
        - 10 videos, batch_size=16: Repeats videos to fill 16 slots
    """
    
    def __init__(self, video_indices: Dict[str, List[int]], batch_size: int):
        """
        Initialize the multi-video sampler.
        
        Args:
            video_indices: Dict mapping scene_token to list of sample indices
                          (sorted by frame order within each video)
            batch_size: Number of parallel video streams
        """
        self.batch_size = batch_size
        
        # Convert to list of (scene_token, frame_indices) sorted by scene_token
        self.videos = [(scene_token, indices) 
                       for scene_token, indices in video_indices.items()]
        self.num_videos = len(self.videos)
        
        if self.num_videos == 0:
            raise ValueError("No videos found in dataset")
        
        # Assign videos to batch positions with repetition if needed
        self.batch_assignments = self._assign_videos_to_batch()
        
        # Frame pointer for each batch position
        self.frame_pointers = [0] * batch_size
        
        # Track which batch positions are active (not padding)
        self.active_mask = [True] * batch_size
        
        # Current round (for cycling through videos when num_videos > batch_size)
        self.current_round = 0
        self.total_rounds = (self.num_videos + batch_size - 1) // batch_size
        
        print(f"MultiVideoSampler initialized:")
        print(f"  - Total videos: {self.num_videos}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Total rounds: {self.total_rounds}")
    
    def _assign_videos_to_batch(self) -> List[int]:
        """
        Assign video indices to batch positions.
        
        If batch_size > num_videos, repeat videos to fill all positions.
        
        Returns:
            List of video indices for each batch position
        """
        assignments = []
        for i in range(self.batch_size):
            video_idx = i % self.num_videos
            assignments.append(video_idx)
        return assignments
    
    def set_round(self, round_idx: int):
        """
        Set the current round and update batch assignments.
        
        Args:
            round_idx: Round index (0 to total_rounds-1)
        """
        self.current_round = round_idx
        self.frame_pointers = [0] * self.batch_size
        
        start_video_idx = round_idx * self.batch_size
        
        for i in range(self.batch_size):
            global_video_idx = start_video_idx + i
            if global_video_idx < self.num_videos:
                self.batch_assignments[i] = global_video_idx
                self.active_mask[i] = True
            else:
                pad_source = i % min(self.batch_size, self.num_videos - start_video_idx)
                if self.num_videos - start_video_idx > 0:
                    self.batch_assignments[i] = start_video_idx + pad_source
                else:
                    self.batch_assignments[i] = i % self.num_videos
                self.active_mask[i] = False
    
    def get_video_length(self, batch_pos: int) -> int:
        """
        Get the total number of frames for the video at given batch position.
        
        Args:
            batch_pos: Batch position index
            
        Returns:
            Number of frames in the video
        """
        video_idx = self.batch_assignments[batch_pos]
        return len(self.videos[video_idx][1])
    
    def get_max_frames_in_round(self) -> int:
        """
        Get the maximum number of frames among all videos in current round.
        
        Returns:
            Maximum frame count
        """
        max_frames = 0
        for i in range(self.batch_size):
            if self.active_mask[i]:
                max_frames = max(max_frames, self.get_video_length(i))
        return max_frames
    
    def get_batch_indices(self) -> Tuple[List[int], List[bool]]:
        """
        Get sample indices for current batch (one frame per video stream).
        
        Returns:
            Tuple of:
                - List of sample indices (length = batch_size)
                - List of is_first_frame flags (True if this is first frame of video)
        """
        indices = []
        is_first_frame = []
        
        for i in range(self.batch_size):
            video_idx = self.batch_assignments[i]
            frame_ptr = self.frame_pointers[i]
            scene_token, frame_indices = self.videos[video_idx]
            
            actual_frame_ptr = frame_ptr % len(frame_indices)
            sample_idx = frame_indices[actual_frame_ptr]
            
            indices.append(sample_idx)
            is_first_frame.append(frame_ptr == 0)
        
        return indices, is_first_frame
    
    def advance_frame(self) -> bool:
        """
        Advance frame pointer for all batch positions.
        
        Returns:
            True if there are more frames to process in this round,
            False if all videos in this round are exhausted
        """
        all_exhausted = True
        
        for i in range(self.batch_size):
            self.frame_pointers[i] += 1
            
            if self.active_mask[i]:
                video_length = self.get_video_length(i)
                if self.frame_pointers[i] < video_length:
                    all_exhausted = False
        
        return not all_exhausted
    
    def get_scene_tokens(self) -> List[str]:
        """
        Get scene tokens for all batch positions.
        
        Returns:
            List of scene tokens
        """
        tokens = []
        for i in range(self.batch_size):
            video_idx = self.batch_assignments[i]
            scene_token, _ = self.videos[video_idx]
            tokens.append(scene_token)
        return tokens
    
    def get_total_frames(self) -> int:
        """
        Get total number of frames across all videos.
        
        Returns:
            Total frame count
        """
        return sum(len(indices) for _, indices in self.videos)
