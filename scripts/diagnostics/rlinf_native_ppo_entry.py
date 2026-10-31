#!/usr/bin/env python
"""Run the pinned official entry with narrowly extended workers and runner."""
import importlib.util
import os
from pathlib import Path

from fluffy_bassoon.meta_rl.rlinf_native_ppo_runtime import (
    PilotActor, PilotEnvironment, PilotRollout, PilotRunner,
)

# The entry imports its actor inside main; patch that module as well as the
# entry's already imported classes. No source checkout is edited.
import rlinf.workers.actor.embodied_fsdp_actor_worker as actor_module
actor_module.EmbodiedFSDPActor = PilotActor
path = Path(os.environ['EMBODIED_PATH']) / 'train_embodied_agent.py'
spec = importlib.util.spec_from_file_location('native_official_embodied_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.EmbodiedRunner = PilotRunner
entry.EnvWorker = PilotEnvironment
entry.MultiStepRolloutWorker = PilotRollout

if __name__ == '__main__':
    entry.main()
