"""Replay eight original, already-issued tapes at the native OSC step boundary."""
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tarfile
from unittest.mock import patch

import numpy as np
import arm_env
import sim_reset
import ur5e_sim_worker
from libero_remote.reproducibility import numerical_state_fingerprint

HERE = Path(__file__).resolve().parent
ROWS = (34, 35, 36, 37)
ARMS = ("IIWA", "Kinova3")
CUE = Path("/mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-cue-20261124-r1")
RECORD_PIN = "e15eefb3f76e475acdf0795e2e17ba0e1ee46fae"
NATIVE_PIN = "a8a997e7d06bb131d3f9710204c58e106f0f0904"
BANK_SHA = "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"
SCENE = ("source_pin", "manifest_sha256", "bddl_sha256", "suite", "task_id", "init_state_id",
         "seed", "released_row_fingerprint", "scene_fingerprint", "perturbation_plan",
         "perturbation_plan_fingerprint")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(x):
    a = np.ascontiguousarray(x)
    return sha(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def head(tree):
    return subprocess.check_output(["git", "-C", tree, "rev-parse", "HEAD"], text=True).strip()


def inputs():
    manifest = json.loads((HERE/"inputs/manifest.json").read_bytes())
    path = HERE/"inputs/raw-tapes.tar.gz"
    assert sha(path.read_bytes()) == manifest["raw_archive_sha256"]
    tapes = {}
    with tarfile.open(path) as archive:
        for arm in ARMS:
            for row in ROWS:
                name = f"gpu01/episodes/{arm}-{row}-raw.json"
                blob = archive.extractfile(name).read()
                assert sha(blob) == manifest["members"][name] == sha((CUE/name).read_bytes())
                tape = json.loads(blob)
                q, rec = tape["query"], tape["native"]
                assert q == dict(arm=arm, suite="libero_goal", task_id=3,
                                 init_state_id=row, seed=42710300+row, variant="BASE")
                assert tape["pin"] == RECORD_PIN and tape["mode"] == "raw" and not tape["fixture"]
                assert rec["inferred"] == dict(yaw_deg=0., xy_scale=1., gripper_sign=1)
                reset = rec["reset"]
                bank = np.random.default_rng(q["seed"]+817117).standard_normal((60, *tape["shape"])).astype(np.float32)
                assert reset["horizon"] == 300 and reset["noise_sha256"] == fingerprint(bank)
                assert reset["noise_seed"] == q["seed"]+817117
                commands, rewards = [], []
                for i, call in enumerate(tape["calls"]):
                    assert call["t"] == len(commands) == i*5 and call["noise"] == fingerprint(bank[i])
                    assert len(call["sent"]) == len(call["rewards"]) and 1 <= len(call["sent"]) <= 5
                    commands.extend(call["sent"])
                    rewards.extend(call["rewards"])
                a = np.array(commands, dtype=np.float32)
                assert a.shape == (rec["controls"], 7) and np.array_equal(a, commands)
                assert np.isfinite(a).all() and np.max(np.abs(a)) <= 1
                assert rewards == rec["native_rewards"] and sum(rewards) == rec["native_return"] == int(rec["success"])
                assert rec["terminated"] == rec["success"] and rec["truncated"] != rec["terminated"]
                assert rec["success"] or rec["controls"] == 300
                tapes[arm,row] = (tape, a)
    for row in ROWS:
        left, right = [tapes[arm,row][0] for arm in ARMS]
        assert all(left["env"][k] == right["env"][k] for k in SCENE)
        assert left["native"]["reset"]["noise_sha256"] == right["native"]["reset"]["noise_sha256"]
    return tapes


def replay(out, own, donor, expected_start, release_row):
    q, rec = own["query"], own["native"]
    arm, row = q["arm"], q["init_state_id"]
    source, commands = donor
    donor_arm, donor_row = source["query"]["arm"], source["query"]["init_state_id"]
    own_tape = donor_arm == arm and donor_row == row
    group = ("own" if own_tape else "correct" if donor_row == row else
             "sham" if donor_row == ROWS[(ROWS.index(row)+1) % 4] else "other")
    case = f"{arm}-{row}-from-{donor_arm}-{donor_row}"
    with arm_env.ArmEnv(arm, q["suite"], q["task_id"], case_id=q["variant"]) as env:
        for attempt in range(1, 4):
            first = env.reset(row, q["seed"])
            reset = first["environment_reproducibility"]
            assert reset["released_row_fingerprint"] == release_row
            assert first["worker_step_index"] == 0 and first["max_steps"] == 300
            assert first["language_instruction"] == rec["released_prompt"]
            assert first["model_identity"] == own["body"] and all(reset[k] == own["env"][k] for k in SCENE)
            start = dict(environment=reset, public={k:fingerprint(v) for k,v in first["observation"].items()})
            if expected_start is None or start == expected_start:
                break
        else:
            raise ValueError(f"Unmatched local target start {case}")
        native = env._env
        base_step = native.step
        issued = []
        rewards, terminated, truncated, successes = [], [], [], []
        proprio_at_query = []
        def audited_step(value):
            assert value.dtype == np.float32 and value.shape == (7,)
            assert value.tobytes() == commands[len(issued)].tobytes()
            issued.append(value.copy())
            return base_step(value)
        obs = first["observation"]
        with patch.object(native.sim, "render", return_value=np.zeros((256,256,3), np.uint8)), \
             patch.object(native, "step", side_effect=audited_step):
            for t, action in enumerate(commands, 1):
                if own_tape and (t-1) % 5 == 0:
                    gpu_public = own["calls"][(t-1)//5]["public"]
                    proprio_at_query.append(dict(t=t-1, equal={k:fingerprint(v)==gpu_public[k]
                                              for k,v in obs.items() if "image" not in k}))
                step = env.step(action)
                assert step["issued_action"].tobytes() == action.tobytes()
                assert step["worker_step_index"] == t == len(issued)
                rewards.append(step["reward"])
                terminated.append(bool(step["terminated"]))
                truncated.append(bool(step["truncated"]))
                successes.append(bool(step["success"]))
                obs = step["observation"]
                if terminated[-1] or truncated[-1]:
                    break
    n = len(rewards)
    terminal = terminated[-1] or truncated[-1]
    first_success = next((i for i,s in enumerate(successes, 1) if s), None)
    assert rewards == list(map(float, successes)) and not any(successes[:-1])
    assert successes[-1] == terminated[-1] and truncated[-1] == (n == 300 and not successes[-1])
    assert terminal or n == len(commands)
    submitted = fingerprint(np.array(issued))
    assert submitted == fingerprint(commands[:n])
    fixture_path = CUE/f"cpu03/episodes/{arm}-{row}-raw.json"
    fixture = json.loads(fixture_path.read_bytes())
    save(out/f"{case}.json", dict(case=case, target=arm, layout=row, donor=donor_arm, donor_layout=donor_row,
         group=group, donor_gpu_success=source["native"]["success"], own_gpu_success=rec["success"],
         own_gpu_controls=rec["controls"], donor_controls=len(commands), controls=n,
         reward_sum=sum(rewards), success=successes[-1], terminal_control=n if terminal else None,
         first_success_control=first_success, stop=("success" if successes[-1] else "horizon" if terminal else "tape_end"),
         native_rewards=rewards, native_success=successes, native_terminated=terminated, native_truncated=truncated,
         donor_action_sha256=fingerprint(commands), submitted_action_sha256=submitted,
         reset=start, reset_attempt=attempt, body=first["model_identity"], cpu03_reference_sha256=sha(fixture_path.read_bytes()),
         cpu03_physics_equal=reset["initial_state_fingerprint"]==fixture["env"]["initial_state_fingerprint"],
         gpu_physics_equal=reset["initial_state_fingerprint"]==own["env"]["initial_state_fingerprint"],
         gpu_post_settle_scene_equal=reset["post_settle_scene_fingerprint"]==own["env"]["post_settle_scene_fingerprint"],
         noise_sha256=rec["reset"]["noise_sha256"], donor_noise_sha256=source["native"]["reset"]["noise_sha256"],
         own_proprio_at_original_queries=proprio_at_query))
    print(case, group, n, terminal, successes[-1], flush=True)
    return start


def main(out, pin, cpu, rows):
    renderer = ur5e_sim_worker._probe_egl(True)
    assert head(HERE) == pin and os.sched_getaffinity(0) == {cpu}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and "llvmpipe" in renderer[1].lower()
    module = Path(arm_env.__file__).resolve()
    native = module.parents[2]
    assert head(native) == NATIVE_PIN and Path(ur5e_sim_worker.__file__).resolve().is_relative_to(native)
    release = Path(os.environ["CROSSARM_RELEASE"])
    assert head(release) == arm_env.SOURCE_PIN
    bank = release/"simulator-source/libero/libero/init_files/libero_goal/open_the_top_drawer_and_put_the_bowl_inside.pruned_init"
    assert sha(bank.read_bytes()) == BANK_SHA
    import torch
    states = torch.load(bank, map_location="cpu", weights_only=False)
    assert states.shape == (50,79)
    modules = [module, Path(sim_reset.__file__), Path(ur5e_sim_worker.__file__), native/"outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py"]
    save(out/f"runtime-{cpu}.json", dict(host=socket.gethostname(), platform=platform.platform(),
         cpu=cpu, rows=rows, cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"], renderer=renderer, gpu_hours=0,
         interpreter=sys.executable, numpy=np.__version__, physics_source=str(native), physics_pin=head(native),
         release_source=str(release), release_pin=head(release), study_pin=pin, release_bank_sha256=BANK_SHA,
         module_sha256={str(p):sha(p.read_bytes()) for p in modules}))
    tapes = inputs()
    for row in rows:
        for target in ARMS:
            opposite = next(a for a in ARMS if a != target)
            next_row = ROWS[(ROWS.index(row)+1) % 4]
            donors = [(target,row), (opposite,row), (opposite,next_row)]
            if target == "Kinova3":
                donors.extend((opposite,i) for i in ROWS if i not in (row,next_row))
            start = None
            for key in donors:
                start = replay(out, tapes[target,row][0], tapes[key], start, numerical_state_fingerprint(states[row]))


if __name__ == "__main__":
    out, pin, cpu, *rows = sys.argv[1:]
    main(Path(out), pin, int(cpu), list(map(int, rows)))
