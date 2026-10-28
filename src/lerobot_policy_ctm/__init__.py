"""Register LeRobot policies when its supported registry API is available.

Standalone recurrent cores also run in native motor environments with no
LeRobot registry (or an older LeRobot API). Importing a core must not import
unrelated policy configs, processors, or visual-language dependencies there.
"""
from importlib import import_module as _import_module

try:
    _configs = _import_module("lerobot.configs")
except ModuleNotFoundError as _error:
    if _error.name not in {"lerobot", "lerobot.configs"}:
        raise
    _configs = None

if hasattr(_configs, "PreTrainedConfig"):
    from . import _registration

    # Preserve all existing public package attributes and registration side
    # effects in the supported LeRobot environment, including processor steps.
    globals().update({key: value for key, value in vars(_registration).items()
                      if not key.startswith("__")})
    __all__ = _registration.__all__
else:
    __all__ = []
