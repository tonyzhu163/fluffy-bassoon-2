"""Hash-verified random access to the observations behind physical source queries."""
from collections import OrderedDict
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import warnings

import numpy as np
import torch

from .robocasa_residual import combine_observations, public_observation
from .vla_history import file_sha256


class FlowSourceQueries:
    def __init__(self, sources, codec, *, cached_members=16, read_attempts=5, retry_delay=.25):
        self.codec, self.sources, self.members, self.cache = codec, {}, {}, OrderedDict()
        self.archive_paths = {}
        if type(read_attempts) is not int or read_attempts < 1 or not 0 <= retry_delay <= 10:
            raise ValueError("Require bounded source-read attempts and retry delay")
        self.read_attempts, self.retry_delay = read_attempts, retry_delay
        self.cached_members = cached_members
        if type(cached_members) is not int or cached_members < 1:
            raise ValueError("Require positive bounded observation cache capacity")
        for task, source in sources.items():
            index = Path(source["index_path"])
            if file_sha256(index) != source["index_sha256"]:
                raise ValueError("Source tar query index changed")
            rows = json.loads(index.read_text())
            lookup = {(r["rank"], r["version"], r["chunk"]): r for r in rows}
            if len(lookup) != len(rows):
                raise ValueError("Ambiguous chronological source member identity")
            self.archive_paths[task] = source["archive"]
            self.sources[task] = None
            self.members[task] = lookup

    def _close_source(self, task):
        fd = self.sources.pop(task, None)
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError as error:
            # These descriptors are read-only: there are no writes to flush.
            # On Linux a failed close may already have released the descriptor;
            # retrying that integer could close an unrelated, reused descriptor.
            warnings.warn(f"Read-only source close failed for {task}: {error}; "
                          "descriptor retired without retry", RuntimeWarning)

    def close(self):
        for task in list(self.sources):
            self._close_source(task)
        self.cache.clear()

    def _verified_bytes(self, task, row):
        transient = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN, errno.EBUSY, errno.EINTR}
        for attempt in range(self.read_attempts):
            try:
                if self.sources.get(task) is None:
                    self.sources[task] = os.open(self.archive_paths[task], os.O_RDONLY)
                parts, received = [], 0
                while received < row["bytes"]:
                    part = os.pread(self.sources[task], row["bytes"] - received,
                                    row["offset_data"] + received)
                    if not part:
                        break
                    parts.append(part)
                    received += len(part)
                raw = b"".join(parts)
                if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError(f"Raw source query member failed byte/hash verification: "
                                     f"{task} offset={row['offset_data']} "
                                     f"expected_bytes={row['bytes']} received_bytes={len(raw)}")
                return raw
            except (OSError, ValueError) as error:
                if isinstance(error, OSError) and error.errno not in transient:
                    raise
                self._close_source(task)
                if attempt + 1 == self.read_attempts:
                    raise
                warnings.warn(f"Source read {attempt + 1}/{self.read_attempts} failed for "
                              f"{task} offset={row['offset_data']}: {error}; reopening and "
                              "retrying the same member with byte/hash verification", RuntimeWarning)
                time.sleep(min(self.retry_delay * 2**attempt, 2.))

    def read_member(self, task, rank, version, chunk):
        key = (task, rank, version, chunk)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        row = self.members[task][rank, version, chunk]
        raw = self._verified_bytes(task, row)
        value = torch.load(io.BytesIO(gzip.decompress(raw)), map_location="cpu", weights_only=False)
        self.cache[key] = value
        while len(self.cache) > self.cached_members:
            self.cache.popitem(last=False)
        return value

    def observations(self, queries, trials, trial_ids):
        result = []
        for trial_idx, query_idx in queries:
            trial = trials[trial_idx]
            task, identity = trial_ids[trial_idx].split("/", 1)
            match = re.fullmatch(r"flow-source-slot-(\d{3})", identity)
            if match is None or task not in self.members:
                raise ValueError("Require a task-namespaced chronological collection slot")
            slot = int(match[1])
            if not 0 <= slot < 128 or not 0 <= query_idx < len(trial["rewards"]):
                raise ValueError("Invalid source slot/query position")
            version, chunk = divmod(query_idx, 12)
            rank, lane = divmod(slot, 16)
            payload = self.read_member(task, rank, version, chunk + 1)
            observation = public_observation(payload["current_observation"])
            one = {k: (v[lane:lane + 1] if isinstance(v, (torch.Tensor, np.ndarray, list, tuple))
                       else v) for k, v in observation.items()}
            state = np.asarray(one["states"], np.float32)[0]
            commands = np.asarray(payload["decoded_actions"], np.float32)[lane]
            if (not np.array_equal(state, trial["observations"][query_idx, :25])
                    or not np.array_equal(self.codec.encode(commands), trial["executed_noise"][query_idx])
                    or int(trial["teacher_steps"][query_idx]) != version):
                raise ValueError("Saved RGB query does not match the source state, behavior action and version")
            result.append(one)
        return combine_observations(result)
