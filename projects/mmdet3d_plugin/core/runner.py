"""
Custom IterBasedRunner with thread-based GOP decode prefetch.

Overlaps GOP decode of batch N+1 with training of batch N using a
background thread + dedicated CUDA stream.  Both decode and training
happen in the main process (single CUDA context), avoiding the
time-slicing overhead of multi-process GPU decode (C-3 finding).

Timeline (steady state):
    Main thread:   [wait N] [train_step N] [backward N ===============]  [wait N+1] ...
    BG thread:                [next(dl) N+1] [NVC decode N+1] [xform N+1]
    GPU SM:                   [forward N===] [backward N ================]
    NVDEC HW:                                [==== decode N+1 ====]

The background thread runs both next(data_loader) and GOP decode,
so the main thread goes straight from train_step to backward without
any CPU-side blocking for data fetch or decode.

Requires the model to have:
  - enable_gop_decode = True
  - _decode_and_set_img(data) method

Falls back to standard IterBasedRunner when enable_gop_decode is False.
"""
import threading

import torch
from mmcv.runner import RUNNERS, IterBasedRunner

import nvtx


class _PrefetchWorker:
    """Background thread: fetch next batch from DataLoader + GOP decode.

    Usage:
        worker.submit(data_loader)    # non-blocking: spawns thread
        ...                           # main thread does backward etc.
        decoded = worker.wait()       # blocks until thread + GPU done
    """

    def __init__(self, model, decode_stream):
        self._model = model
        self._decode_stream = decode_stream
        self._thread = None
        self._batch = None

    def submit(self, data_loader):
        """Spawn background thread to fetch + decode next batch."""
        assert self._thread is None or not self._thread.is_alive(), \
            'previous prefetch still running — call wait() first'
        self._thread = threading.Thread(
            target=self._run, args=(data_loader,), daemon=True)
        self._thread.start()

    def _run(self, data_loader):
        """Background thread: fetch from DataLoader then decode on GPU.

        - next(data_loader):  mostly waits on IPC from worker processes,
          which releases the GIL (I/O wait).
        - NVC decode:  blocks this thread but releases GIL via
          py::gil_scoped_release, so main thread can run backward.
        - GPU transforms:  launched on *decode_stream* to avoid
          interfering with training on the default stream.
        """
        with nvtx.annotate("bg_fetch_data", color="orange"):
            data_batch = next(data_loader)
        self._batch = data_batch
        with torch.cuda.stream(self._decode_stream):
            with nvtx.annotate("bg_decode_xform", color="purple"):
                self._model._decode_and_set_img(data_batch)

    def wait(self):
        """Block until background fetch + decode is done.

        Returns the decoded batch with data['img'] set as a GPU tensor.
        """
        if self._thread is None:
            return None
        self._thread.join()
        torch.cuda.current_stream().wait_stream(self._decode_stream)
        batch = self._batch
        self._thread = None
        self._batch = None
        return batch


@RUNNERS.register_module()
class IterBasedRunnerWithPrefetch(IterBasedRunner):
    """IterBasedRunner with thread-based double-buffer GOP decode prefetch."""

    def _get_model(self):
        """Unwrap DDP/DP to get the underlying model."""
        model = self.model
        if hasattr(model, 'module'):
            model = model.module
        return model

    def _should_prefetch(self):
        """Check if the model supports GOP prefetch decode."""
        model = self._get_model()
        return getattr(model, 'enable_gop_decode', False)

    def train(self, data_loader, **kwargs):
        if not self._should_prefetch():
            return super().train(data_loader, **kwargs)

        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch

        # Lazy init
        if not hasattr(self, '_prefetch_worker'):
            self._decode_stream = torch.cuda.Stream()
            self._prefetch_worker = _PrefetchWorker(
                self._get_model(), self._decode_stream)
            self._prefetch_started = False
            self.logger.info(
                'GOP thread-based prefetch enabled: decode_stream=%s',
                self._decode_stream)

        model_unwrapped = self._get_model()

        # ── 1. Get current batch ──────────────────────────────────
        if not self._prefetch_started:
            # First iteration: synchronous fetch + decode (no overlap)
            with nvtx.annotate("prefetch_warmup", color="yellow"):
                data_batch = next(data_loader)
                model_unwrapped._decode_and_set_img(data_batch)
            self._prefetch_started = True
        else:
            with nvtx.annotate("prefetch_wait", color="gray"):
                data_batch = self._prefetch_worker.wait()

        # ── 2. Forward pass ───────────────────────────────────────
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        with nvtx.annotate("train_step", color="green"):
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

        # ── 3. Submit next fetch+decode (non-blocking, just spawns thread)
        with nvtx.annotate("prefetch_submit", color="yellow"):
            self._prefetch_worker.submit(data_loader)

        # ── 4. Backward + optimizer (overlaps with BG fetch+decode) ──
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'],
                                   outputs['num_samples'])
        self.outputs = outputs
        with nvtx.annotate("backward_and_optim", color="red"):
            self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1
