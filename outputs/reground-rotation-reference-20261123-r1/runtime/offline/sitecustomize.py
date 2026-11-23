"""Serve the one unversioned metadata request made by upstream ReGround while offline."""
import os
from pathlib import Path
from types import SimpleNamespace

if os.environ.get('REGROUND_PINNED_OFFLINE') == '1':
    from huggingface_hub import HfApi

    _original_model_info = HfApi.model_info
    _repo = 'lerobot/pi05_libero_finetuned'
    _revision = '7152f13e34c6c115d808aea0e5d08aca74eac14b'

    def _model_info(self, repo_id, *args, **kwargs):
        if repo_id == _repo and not args and not kwargs:
            ref = Path(os.environ['HF_HUB_CACHE']) / 'models--lerobot--pi05_libero_finetuned' / 'refs/main'
            if ref.read_text() != _revision:
                raise RuntimeError(f'Offline policy ref disagrees with {_revision}')
            return SimpleNamespace(sha=_revision)
        return _original_model_info(self, repo_id, *args, **kwargs)

    HfApi.model_info = _model_info
