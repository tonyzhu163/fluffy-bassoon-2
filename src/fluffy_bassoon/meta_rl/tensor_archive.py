"""Lossless, optionally compressed Torch artifacts with atomic publication."""
import gzip
import io
from pathlib import Path

import torch


def write_tensor(path, value):
    """Stream directly to the requested .pt or .pt.gz; no raw disk staging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        if path.suffix == '.gz':
            with gzip.open(temporary, 'wb', compresslevel=1) as handle:
                torch.save(value, handle)
        else:
            torch.save(value, temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_tensor(path, *, map_location='cpu'):
    """Read gzip in one pass; hold serialized bytes in RAM, never on disk."""
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as handle:
            payload = io.BytesIO(handle.read())
        return torch.load(payload, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location, weights_only=False)
