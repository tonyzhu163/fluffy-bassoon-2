"""Wire the one-chunk prompt to native IPC. Actions stay fixed for this interface check."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from ur5e_motor import public_digest
from ur5e_sim_worker import UR5eSimulator

from common import save, sha
from phase_prompt import PhasePrompt


def run(model, output):
    data = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-native-transfer-20261123-r1/gpu01/libero_goal-3-15-Kinova3.json")
    commands = json.loads(data.read_bytes())["chunks"][:18]
    config = dict(arm="Kinova3",suite="libero_goal",task_id=3,variant="BASE",embodiment="nom",
                  init_state_id=30,seed=42710330)
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    result = dict(model_sha256=sha(model),commands_sha256=sha(data),case=config,calls={})
    with UR5eSimulator(allow_mesa=True) as sim:
        for mode in ("raw","visual","clock"):
            gate = PhasePrompt(sim,json.loads(model.read_bytes()),mode)
            _,first = gate.call("create",config)
            assert "llvmpipe" in str(first["renderer"]).lower()
            trace = [public_digest(first["observation"])]
            rewards = []
            prompts = [first["language_instruction"]]
            for chunk in commands:
                step = gate.call("chunk",np.asarray(chunk["cmd"],np.float32))
                assert not (step["terminated"] or step["truncated"])
                trace.append(public_digest(step["observation"]))
                rewards.extend(step["rewards"])
                prompts.append(step["language_instruction"])
            if mode == "raw":
                reference = trace,rewards,first["environment_reproducibility"]
                result["renderer"] = first["renderer"]
            assert (trace,rewards,first["environment_reproducibility"]) == reference
            fires = [v["t"] for v in gate.events if v["fire"]]
            assert fires == [5*i for i,p in enumerate(prompts) if p != first["language_instruction"]]
            result["calls"][mode] = dict(fires=fires,prompts=prompts,events=gate.events)
    result.update(matching_public_observations=len(trace),matching_native_rewards=len(rewards),
                  released_scene=reference[2]["scene_fingerprint"])
    save(output,result)


if __name__ == "__main__":
    run(*map(Path,sys.argv[1:]))
