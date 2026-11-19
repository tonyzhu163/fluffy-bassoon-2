"""Shared LIBERO environment, timing, and latent-steering utilities."""

from .history_encoder import HistoryEncoderConfig, pack_control_tokens
from .latent_adapter import AdapterConfig

__all__ = [
    "AdapterConfig",
    "HistoryEncoderConfig",
    "estimate_command_lag",
    "pack_control_tokens",
]


def __getattr__(name):
    if name == "estimate_command_lag":
        from .timing import estimate_command_lag

        return estimate_command_lag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
