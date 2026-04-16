"""
SharedGOPPool: Cross-worker shared GOP cache backed by SharedMemory.

Each GOP is stored as an independent SharedMemory block.
A small SharedMemory block holds the metadata table (index).
File-based locking (flock) for cross-process synchronization.

Usage:
    # Main process: create pool before DataLoader workers start
    pool = SharedGOPPool(capacity=36, pool_id=0, create=True)

    # Worker: attach to existing pool
    pool = SharedGOPPool(capacity=36, pool_id=0, create=False)
    hit = pool.lookup(video_path, frame_id)
    if hit is None:
        data = load_from_disk(...)
        shm_name, data_size = pool.put(video_path, first_frame_id, gop_len, data)
    else:
        shm_name, data_size = hit[0], hit[1]
    gop_np = pool.read_view(shm_name, data_size)  # zero-copy

    # Main process: read views, no need to release
    gop_np = pool.read_view(shm_name, data_size)

    # Main process: cleanup on shutdown
    pool.cleanup()
"""

import fcntl
import hashlib
import multiprocessing
import os
import time
from multiprocessing import shared_memory

import numpy as np


# Metadata entry layout (96 bytes per entry, fits in L1 cache for capacity < 500)
ENTRY_DTYPE = np.dtype([
    ('video_path_hash', np.uint64),  # hash of video_path string
    ('first_frame_id', np.int32),    # first frame ID of this GOP
    ('gop_len', np.int32),           # number of frames in this GOP
    ('data_size', np.int32),         # actual GOP data size in bytes
    ('state', np.uint8),             # 0=FREE, 1=USED
    ('_pad', np.uint8, (3,)),        # alignment padding
    ('access_tick', np.int64),       # monotonic counter for LRU ordering
    ('shm_name', 'S64'),            # SharedMemory name for the data block
])

_STATE_FREE = np.uint8(0)
_STATE_USED = np.uint8(1)


def _hash_video_path(video_path):
    """Deterministic uint64 hash for video path string.

    Uses MD5 (truncated to 8 bytes) instead of Python's built-in hash(),
    because hash() uses a per-process random seed under PYTHONHASHSEED != 0,
    which breaks cross-process consistency with spawn'd DataLoader workers.
    """
    digest = hashlib.md5(video_path.encode()).digest()[:8]
    return np.uint64(int.from_bytes(digest, 'little'))


class SharedGOPPool:
    """
    Cross-worker shared GOP cache.

    Stores GOP packet data in per-GOP SharedMemory blocks.
    Metadata table lives in a single small SharedMemory block.
    Uses file-based locking (flock) for cross-process safety under spawn mode.
    """

    def __init__(self, capacity, pool_id, create=True):
        """
        Args:
            capacity: Maximum number of GOPs to cache (bs * num_cameras * scale).
            pool_id: Unique pool identifier, typically LOCAL_RANK.
            create: True in main process (allocates), False in workers (attaches).
        """
        self.capacity = capacity
        self.pool_id = pool_id

        # --- Metadata SharedMemory ---
        meta_name = f"gop_pool_meta_{pool_id}"
        meta_size = capacity * ENTRY_DTYPE.itemsize
        if create:
            # Clean up stale SharedMemory from previous runs
            try:
                stale = shared_memory.SharedMemory(name=meta_name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._meta_shm = shared_memory.SharedMemory(
                name=meta_name, create=True, size=meta_size)
            # Zero-initialize all entries (state=FREE)
            np.frombuffer(self._meta_shm.buf, dtype=np.uint8)[:] = 0
        else:
            self._meta_shm = shared_memory.SharedMemory(
                name=meta_name, create=False)

        self._entries = np.ndarray(
            capacity, dtype=ENTRY_DTYPE, buffer=self._meta_shm.buf)

        # --- File-based lock (works across spawn'd processes) ---
        self._lock_path = f"/dev/shm/gop_pool_lock_{pool_id}"
        if create:
            # Create lock file
            with open(self._lock_path, 'w'):
                pass
        self._lock_fd = os.open(self._lock_path, os.O_RDWR)

        # --- Monotonic tick counter in a tiny SharedMemory ---
        tick_name = f"gop_pool_tick_{pool_id}"
        if create:
            try:
                stale = shared_memory.SharedMemory(name=tick_name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._tick_shm = shared_memory.SharedMemory(
                name=tick_name, create=True, size=8)
            np.frombuffer(self._tick_shm.buf, dtype=np.int64)[:] = 0
        else:
            self._tick_shm = shared_memory.SharedMemory(
                name=tick_name, create=False)
        self._tick_arr = np.ndarray(1, dtype=np.int64, buffer=self._tick_shm.buf)

        # --- Per-process handle cache (not shared) ---
        self._local_shm_handles = {}

        # --- Stats (per-process, not shared) ---
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

        self._is_creator = create

    # ------------------------------------------------------------------ #
    #  Locking helpers (flock)
    # ------------------------------------------------------------------ #

    def _lock(self):
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _unlock(self):
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _next_tick(self):
        """Atomically increment and return the monotonic tick counter."""
        # Not truly atomic across processes without lock, but tick is only
        # used for LRU ordering — a rare duplicate is harmless.
        self._tick_arr[0] += 1
        return int(self._tick_arr[0])

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def lookup(self, video_path, frame_id):
        """
        Look up a cached GOP containing frame_id.

        Lock-free read: worst case is a stale miss (causes an extra disk read,
        no correctness issue).

        Args:
            video_path: Video file path.
            frame_id: Target frame index.

        Returns:
            (shm_name, data_size, first_frame_id, gop_len) on hit, None on miss.
        """
        vp_hash = _hash_video_path(video_path)
        for i in range(self.capacity):
            e = self._entries[i]
            if (e['state'] == _STATE_USED
                    and e['video_path_hash'] == vp_hash
                    and e['first_frame_id'] <= frame_id
                    < e['first_frame_id'] + e['gop_len']):
                # Update access tick (benign race — LRU ordering only)
                e['access_tick'] = self._next_tick()
                self._hits += 1
                return (e['shm_name'].decode(), int(e['data_size']),
                        int(e['first_frame_id']), int(e['gop_len']))
        self._misses += 1
        return None

    def put(self, video_path, first_frame_id, gop_len, data):
        """
        Store GOP data into the pool.

        Holds flock during eviction + insertion to guarantee atomicity.

        Args:
            video_path: Video file path.
            first_frame_id: First frame ID of this GOP.
            gop_len: Number of frames in this GOP.
            data: np.ndarray of raw GOP packet bytes.

        Returns:
            (shm_name, data_size) reference to the stored data.
        """
        self._lock()
        try:
            # Double-check: another worker may have inserted while we waited
            existing = self.lookup(video_path, first_frame_id)
            if existing is not None:
                # Undo the miss count from the outer lookup that triggered put()
                return existing[0], existing[1]

            # Find free slot or evict LRU
            slot_idx = self._find_free_or_evict()

            # Name by content identity (video_path + first_frame_id), not slot index.
            # This way eviction + reuse never reuses a name for different data.
            vp_hash = _hash_video_path(video_path)
            shm_name = f"gop_{self.pool_id}_{vp_hash}_{first_frame_id}"
            data_size = int(data.nbytes)

            # Clean up if same GOP was previously cached then evicted
            try:
                stale = shared_memory.SharedMemory(name=shm_name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass

            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=data_size)
            shm.buf[:data_size] = data.tobytes()
            shm.close()  # close handle; shm persists until unlink

            # Write metadata entry
            self._entries[slot_idx]['video_path_hash'] = vp_hash
            self._entries[slot_idx]['first_frame_id'] = first_frame_id
            self._entries[slot_idx]['gop_len'] = gop_len
            self._entries[slot_idx]['data_size'] = data_size
            self._entries[slot_idx]['access_tick'] = self._next_tick()
            self._entries[slot_idx]['shm_name'] = shm_name.encode()
            # Set state last (acts as a publish barrier for lock-free readers)
            self._entries[slot_idx]['state'] = _STATE_USED

            self._puts += 1
            return shm_name, data_size
        finally:
            self._unlock()

    def read_view(self, shm_name, data_size):
        """
        Zero-copy numpy view of GOP data in SharedMemory.

        Caches SharedMemory handles per-process to avoid repeated shm_open.
        Periodically cleans up stale handles from evicted slots.

        Args:
            shm_name: SharedMemory name.
            data_size: Number of bytes to read.

        Returns:
            np.ndarray: uint8 view into shared memory (zero-copy).
        """
        if shm_name not in self._local_shm_handles:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            self._local_shm_handles[shm_name] = shm
        else:
            shm = self._local_shm_handles[shm_name]

        return np.frombuffer(shm.buf[:data_size], dtype=np.uint8)

    def resolve_batch(self, gop_refs):
        """
        Atomically open SharedMemory handles for an entire batch of GOP refs,
        then immediately unlink any orphaned shm blocks.

        Holds flock during the entire operation so that no worker can evict
        a block while we are opening handles.  After opening, orphaned blocks
        (evicted but not yet unlinked) are safe to remove because:
        1. Pool capacity > max in-flight GOPs, so LRU eviction only targets
           entries that no queued batch references.
        2. Any handle already opened by this or prior resolve_batch calls
           keeps its backing memory alive via POSIX fd semantics.

        Args:
            gop_refs: list[list[(shm_name, data_size)]]
                Outer list = batch samples, inner list = cameras per sample.

        Returns:
            list[list[np.ndarray]]: zero-copy uint8 views, same shape as input.
        """
        self._lock()
        try:
            result = [
                [self.read_view(shm_name, data_size)
                 for shm_name, data_size in sample_refs]
                for sample_refs in gop_refs
            ]
            self._unlink_orphans()
            return result
        finally:
            self._unlock()

    def _unlink_orphans(self):
        """Unlink shm blocks that were evicted but not yet cleaned up.

        Called under flock by resolve_batch().  Scans /dev/shm for blocks
        belonging to this pool that are no longer in the metadata table
        (i.e., their slot was evicted).  Safe to unlink immediately because
        pool capacity is sized to exceed the DataLoader queue depth, so
        evicted entries are never referenced by any in-flight batch.
        """
        import glob as glob_mod

        # Active shm names from metadata table
        active_names = set()
        for i in range(self.capacity):
            if self._entries[i]['state'] == _STATE_USED:
                name = self._entries[i]['shm_name'].decode()
                if name:
                    active_names.add(name)

        # Scan /dev/shm and unlink orphans
        prefix = f"gop_{self.pool_id}_"
        for path in glob_mod.glob(f"/dev/shm/{prefix}*"):
            name = os.path.basename(path)
            if name not in active_names:
                # Close our cached handle if any
                if name in self._local_shm_handles:
                    try:
                        self._local_shm_handles[name].close()
                    except Exception:
                        pass
                    del self._local_shm_handles[name]
                # Unlink
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    def get_stats(self):
        """
        Get per-process cache statistics.

        Returns:
            dict with hits, misses, hit_rate, puts, evictions, pool_usage.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        used_slots = int(np.count_nonzero(self._entries['state'] == _STATE_USED))
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'puts': self._puts,
            'evictions': self._evictions,
            'pool_usage': f"{used_slots}/{self.capacity}",
        }

    def reset_stats(self):
        """Reset per-process statistics."""
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _find_free_or_evict(self):
        """
        Find a FREE slot, or evict the LRU (lowest access_tick) entry.

        Must be called with flock held.

        Returns:
            int: slot index to use.
        """
        min_tick = np.iinfo(np.int64).max
        min_idx = 0
        for i in range(self.capacity):
            if self._entries[i]['state'] == _STATE_FREE:
                return i
            if self._entries[i]['access_tick'] < min_tick:
                min_tick = int(self._entries[i]['access_tick'])
                min_idx = i

        # Evict the LRU slot.  Do NOT unlink the shm block here — workers
        # don't know which blocks the main process still needs.  The main
        # process will clean up orphans in resolve_batch() → _unlink_orphans().
        self._entries[min_idx]['state'] = _STATE_FREE

        self._evictions += 1
        return min_idx

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """
        Unlink all SharedMemory blocks (active + orphaned) and lock file.

        Call from main process on shutdown.
        """
        # Close local handles first
        for shm in self._local_shm_handles.values():
            try:
                shm.close()
            except Exception:
                pass
        self._local_shm_handles.clear()

        # Glob-clean ALL gop shm blocks for this pool (catches orphans from eviction)
        import glob
        prefix = f"gop_{self.pool_id}_"
        for path in glob.glob(f"/dev/shm/{prefix}*"):
            name = os.path.basename(path)
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        # Unlink meta and tick SharedMemory
        self._meta_shm.close()
        self._tick_shm.close()
        if self._is_creator:
            try:
                self._meta_shm.unlink()
            except FileNotFoundError:
                pass
            try:
                self._tick_shm.unlink()
            except FileNotFoundError:
                pass
            # Remove lock file
            try:
                os.close(self._lock_fd)
                os.unlink(self._lock_path)
            except OSError:
                pass

    def close(self):
        """
        Close SharedMemory handles without unlinking.

        Call from worker processes before exit.
        """
        for shm in self._local_shm_handles.values():
            try:
                shm.close()
            except Exception:
                pass
        self._local_shm_handles.clear()

        try:
            self._meta_shm.close()
        except Exception:
            pass
        try:
            self._tick_shm.close()
        except Exception:
            pass
        try:
            os.close(self._lock_fd)
        except OSError:
            pass
