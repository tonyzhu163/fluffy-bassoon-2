"""Core imports must work without the optional LeRobot policy registry."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("legacy_api", [False, True])
def test_cmm_import_does_not_require_lerobot_policy_registration(legacy_api):
    source = r'''
import importlib.abc
import sys
import types

if LEGACY_API:
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = []
    configs = types.ModuleType("lerobot.configs")
    lerobot.configs = configs
    sys.modules.update({"lerobot": lerobot, "lerobot.configs": configs})
else:
    class NoLeRobot(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "lerobot" or fullname.startswith("lerobot."):
                raise ModuleNotFoundError("No optional LeRobot", name=fullname)
    sys.meta_path.insert(0, NoLeRobot())

from fluffy_bassoon.meta_rl.cmm import CMMCore
from lerobot_policy_ctm.architectures.continuous_memory_machine.core import ContinuousMemoryMachine
import lerobot_policy_ctm
assert issubclass(CMMCore, ContinuousMemoryMachine)
assert lerobot_policy_ctm.__all__ == []
assert "lerobot_policy_ctm._registration" not in sys.modules
assert "lerobot_policy_ctm.shared.processor" not in sys.modules
assert not any(name.startswith("lerobot_policy_ctm.architectures.lyco_") for name in sys.modules)
'''.replace("LEGACY_API", repr(legacy_api))
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, "-c", source], cwd=root, env=environment,
                   check=True, capture_output=True, text=True, timeout=60)
