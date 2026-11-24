"""Run the preregistered fixed tapes on independently reset, locally matched Kinova3 starts."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

from material import DEV, HERE, digest, read, sha
sys.path.insert(0, str(DEV))
import replay as prior


def play(env, commands):
    """Audit the seven float32 channels at the underlying OSC simulator call, as in donor replay."""
    native = env._env
    base_step = native.step
    issued = []
    flags = []
    def audited_step(value):
        assert value.dtype == np.float32 and value.shape == (7,)
        assert value.tobytes() == commands[len(issued)].tobytes()
        issued.append(value.copy())
        return base_step(value)

    with patch.object(native.sim, "render", return_value=np.zeros((256,256,3), np.uint8)), \
         patch.object(native, "step", side_effect=audited_step):
        for n, a in enumerate(commands, 1):
            step = env.step(a)
            assert step["issued_action"].tobytes() == a.tobytes() and step["worker_step_index"] == n == len(issued)
            reward, success, term, trunc = (step[k] for k in ("reward","success","terminated","truncated"))
            assert reward == float(success) and success == term and trunc == (n == 300 and not success)
            flags.append([float(reward), bool(success), bool(term), bool(trunc)])
            if term or trunc:
                break
    assert len(flags) == len(commands) or term or trunc
    return dict(controls=n, tape_length=len(commands), reward_sum=sum(f[0] for f in flags),
                first_success=n if success else None, stop="success" if success else "horizon" if trunc else "tape_end",
                # Every earlier vector was asserted to be [0,false,false,false].
                final_reward_success_terminated_truncated=flags[-1], preceding_zero_nonterminal_controls=n-1,
                action_sha256=prior.fingerprint(commands), submitted_sha256=prior.fingerprint(np.array(issued)))


def main(out, pin, cpu, rows):
    # Reuse the original engine/runtime, release bank, hardware and recorded-source validation.
    assert sha((DEV/"replay.py").read_bytes()) == "3f04e516e0d575099a39a6f18867dbaef985dbd2de2bcf3fe514fd33eed5238d"
    prior.main(out, pin, cpu, [])
    runtime_path = out/f"runtime-{cpu}.json"
    runtime = json.loads(runtime_path.read_bytes())
    prior_runtime = json.loads((DEV/"evidence/cases/runtime-30.json").read_bytes())
    assert runtime["module_sha256"] == prior_runtime["module_sha256"]
    threads = {k:os.environ[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LP_NUM_THREADS","NUMEXPR_NUM_THREADS")}
    assert set(threads.values()) == {"1"}
    prior.save(runtime_path, dict(runtime, rows=rows, threads=threads))
    bank = Path(runtime["release_source"])/"simulator-source/libero/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init"
    states = torch.load(bank, weights_only=False, map_location="cpu")
    source = prior.inputs()
    target, manifest = read()
    assert manifest == json.loads((HERE/"inputs/manifest.json").read_bytes())
    for (row,mode),(x,rec,_,_) in target.items():
        noise = np.random.default_rng(rec["reset"]["noise_seed"]).standard_normal((60,10,32)).astype(np.float32)
        assert prior.fingerprint(noise) == rec["reset"]["noise_sha256"]
        if row == 31:
            assert x["noise_shape"] == [10,32] and all(c["noise_sha256"] == prior.fingerprint(noise[i]) for i,c in enumerate(x["chunks"]))
    body = target[14,"raw"][0]["model"]
    for row in rows:
        historical = (row,"raw") in target
        seed = (41810300 if historical else 42710300) + row
        expected_start = None
        branches = (["raw"] if historical else []) + (["tuned"] if row == 31 else []) + ["IIWA35","IIWA34"]
        for branch in branches:
            if branch.startswith("IIWA"):
                d = int(branch[4:])
                tape, commands = source["IIWA",d]
                assert tape["native"]["success"] and len(commands) == {35:184,34:171}[d]
                dev_case = json.loads((DEV/f"evidence/cases/Kinova3-34-from-IIWA-{d}.json").read_bytes())
                assert prior.fingerprint(commands) == dev_case["donor_action_sha256"]
                conditioned = tape["native"]["reset"]["noise_sha256"]
            else:
                _, rec, _, values = target[row,branch]
                commands = np.asarray(values, dtype=np.float32)
                assert np.array_equal(commands, values) and prior.fingerprint(commands) == digest(values)
                conditioned = rec["reset"]["noise_sha256"]
            ref = target.get((row,branch), target.get((row,"raw")))
            with prior.arm_env.ArmEnv("Kinova3","libero_goal",3,case_id="BASE") as env:
                for attempt in range(1,4):
                    first = env.reset(row, seed)
                    repro = first["environment_reproducibility"]
                    assert repro["released_row_fingerprint"] == prior.numerical_state_fingerprint(states[row])
                    assert repro["source_pin"] == runtime["release_pin"] and first["model_identity"] == body
                    assert repro["init_state_id"] == row and repro["seed"] == seed and repro["perturbation_plan"]["case_id"] == "BASE"
                    assert first["language_instruction"] == target[14,"raw"][1]["released_prompt"]
                    assert first["worker_step_index"] == 0 and first["max_steps"] == 300
                    if ref:
                        assert ref[0]["model"] == body and all(repro[k] == ref[2][k] for k in prior.SCENE)
                    start = dict(environment=repro, public={k:prior.fingerprint(v) for k,v in first["observation"].items()})
                    if expected_start is None or expected_start == start:
                        break
                else:
                    raise ValueError(f"Unmatched local start {row} {branch}")
                if expected_start is None:
                    prior.save(out/f"start-{row}.json", dict(start, body=body))
                    expected_start = start
                result = play(env, commands)
            original = None
            if ref:
                original = dict(physics_equal=repro["initial_state_fingerprint"]==ref[2]["initial_state_fingerprint"],
                                post_settle_equal=repro["post_settle_scene_fingerprint"]==ref[2]["post_settle_scene_fingerprint"],
                                public_equal={k:v==ref[1]["reset"]["public"][k] for k,v in start["public"].items()})
            result.update(row=row, branch=branch, seed=seed, reset_attempt=attempt, start_sha256=sha((out/f"start-{row}.json").read_bytes()),
                          tape_noise_sha256=conditioned, historical_reset_comparison=original)
            prior.save(out/f"case-{row}-{branch}.json", result)
            print(row, branch, result["stop"], result["controls"], flush=True)


if __name__ == "__main__":
    out, pin, core, *rows = sys.argv[1:]
    main(Path(out), pin, int(core), list(map(int, rows)))
