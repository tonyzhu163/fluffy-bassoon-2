"""Bounded background writes of owned CPU snapshots, including checkpoints."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import sys
import time

import numpy as np
import torch

from .rlinf_native_ppo import cpu_tree


def _bytes(value):
    # Count aliases repeatedly: conservative credit for every serialized view.
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Background snapshots require numeric NumPy arrays')
        return value.nbytes
    if isinstance(value, dict):
        return sys.getsizeof(value)+sum(_bytes(k)+_bytes(v) for k,v in value.items())
    if isinstance(value, (tuple,list)):
        return sys.getsizeof(value)+sum(map(_bytes,value))
    return sys.getsizeof(value)


class BoundedWrites:
    """Single producer, bounded running/queued payloads, no background GPU work.

    Admission waits before cloning the next payload. The caller may immediately
    reuse its tensors after submit returns. Flush propagates write errors before
    a recovery boundary can be published. Limits cover snapshot payload bytes;
    compression/serialization workspace and the caller's live tree are separate.
    """
    def __init__(self, *, maximum_jobs=2, maximum_bytes=512*2**20):
        if any(type(v) is not int or v < 1 for v in (maximum_jobs,maximum_bytes)):
            raise ValueError('Require positive write-job and payload-byte limits')
        self.maximum_jobs,self.maximum_bytes = maximum_jobs,maximum_bytes
        self.pool = ThreadPoolExecutor(max_workers=min(2,maximum_jobs))
        self.pending = deque()
        self.pending_bytes = self.peak_bytes = self.peak_jobs = self.submitted = 0
        self.blocked_seconds = 0.
        self.failure,self.closed = None,False

    def _finish(self):
        future,size = self.pending.popleft()
        try:
            return future.result()
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self.pending_bytes -= size

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError('Background writer is closed')
        if self.failure is not None:
            raise RuntimeError('A previous background write failed') from self.failure
        size = _bytes(args)
        if size > self.maximum_bytes:
            raise ValueError('One snapshot exceeds the background payload-byte limit')
        while self.pending and self.pending[0][0].done():
            self._finish()
        started = time.monotonic()
        while len(self.pending) >= self.maximum_jobs or self.pending_bytes+size > self.maximum_bytes:
            self._finish()
        self.blocked_seconds += time.monotonic()-started
        snapshot = cpu_tree(args)  # GPU→CPU, detach and clone happen on the caller.
        future = self.pool.submit(function,*snapshot)
        self.pending.append((future,size));self.pending_bytes += size;self.submitted += 1
        self.peak_bytes = max(self.peak_bytes,self.pending_bytes)
        self.peak_jobs = max(self.peak_jobs,len(self.pending))
        return future

    def flush(self):
        if self.failure is not None:
            raise RuntimeError('A previous background write failed') from self.failure
        while self.pending:
            self._finish()

    def stats(self):
        return dict(submitted=self.submitted,pending_jobs=len(self.pending),pending_bytes=self.pending_bytes,
            peak_jobs=self.peak_jobs,peak_bytes=self.peak_bytes,blocked_seconds=self.blocked_seconds,
            maximum_jobs=self.maximum_jobs,maximum_bytes=self.maximum_bytes)

    def close(self):
        try:
            self.flush()
        finally:
            self.pool.shutdown(wait=True)
            self.pending.clear();self.pending_bytes = 0;self.closed = True
