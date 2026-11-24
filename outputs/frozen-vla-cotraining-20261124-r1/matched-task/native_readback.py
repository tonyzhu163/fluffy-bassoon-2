"""Read the sealed matched-task job using its own pinned episode validator and the registered CPU starts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

import matched
from panel import key, require
from ur5e_motor import array_digest
import verify


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def divergence(joint, other):
    a, b = joint["calls"], other["calls"]
    d = verify.first_change(a, b)
    action_at = d["first_different_issued_control"]
    obs_at = next((x["t"] for x, y in zip(a, b) if x["obs"] != y["obs"]), None)
    magnitude = None
    if action_at is not None:
        i, j = divmod(action_at, 5)
        magnitude = float(np.abs(verify.f32(a[i]["issued"][j])-verify.f32(b[i]["issued"][j])).max())
    return {**d, "first_different_observation_replan": obs_at,
            "same_observations_until_first_action_difference": obs_at is None or
                (action_at is not None and obs_at > action_at),
            "max_abs_at_first_issued_difference": magnitude}


def run(run_dir, job, pin, fit_sha, output):
    out = run_dir / "matched-task/gpu01"
    fit = run_dir / "model/gpu02"
    rc, start, end = map(int, (out/"exit.txt").read_text().split())
    require(rc == 0 and end > start, "native job exit")
    code = matched.HERE
    probe, metadata = read(out/"gpu.json"), read(out/"run.json")
    dev, = probe["devices"]
    require(probe["status"] == "passed" and probe["purpose"] == "evaluation" and
            probe["hostname"] == metadata["host"] == job+"-master-0" and not metadata["fixture"]
            and dev["cuda_name"] == "NVIDIA GeForce RTX 4090" and
            dev["allocation_check"]["own_pid_gpu_uuids"] == [dev["uuid"]], "Baige allocation")
    require(metadata["pin"] == pin and metadata["plan_sha256"] == sha(code/"PLAN.md")
            and metadata["contract_sha256"] == sha(code/"contract.json"), "native panel pin")
    contract, cases, tasks = matched.contract()
    source_manifest = run_dir/"data/dataset-v1/manifest.json"
    training = read(fit/"fit.json")
    admission = metadata["offline_fit"]
    require((fit/"exit.txt").read_text().split()[0] == "0" and
            sha(fit/"fit.json") == admission["fit_sha256"] == fit_sha and
            training["code_revision"] == admission["training_revision"] and
            training["data_manifest"] == metadata["source_manifest"] == sha(source_manifest) == contract["data_manifest"]
            and metadata["train_inputs_sha256"] == sha(fit/"train-inputs.json"), "fit/source lock")
    before, after = read(out/"frozen-before.json"), read(out/"frozen-after.json")
    require(before == after and sha(out/"frozen-before.json") == sha(out/"frozen-after.json"),
            "resident weights or files changed")
    require(before["fit_file"] == fit_sha and before["heads_file"] == training["heads_sha256"] == sha(fit/"heads.npz")
            and all(before[k] == v for k, v in training["base"].items()) and bool(before["resident_stock"]),
            "checkpoint or normalization identity")
    for mode, group in training["parameters"].items():
        require(before["resident_effective_kernels"][mode] == group["action_out_proj/kernel"] and
                all(before["resident_heads"][mode][k] == v for k, v in group["fitted"].items()), "installed head")
    expected = {f"{key(c)}-{mode}.json" for c in cases for mode in matched.MODES}
    require(expected == {p.name for p in (out/"episodes").glob("*.json")} and len(expected) == 40, "episode coverage")
    with patch.object(verify, "TASKS", tasks):
        summary = matched.summarize(out, cases, False)
    require(read(out/"native-summary.json") == summary, "saved summary does not replay")
    episode_rows = []
    for c in cases:
        t = contract["tasks"][c["suite"]]
        label = key(c)
        files = {m:read(out/"episodes"/f"{label}-{m}.json") for m in matched.MODES}
        cpu = read(run_dir/"matched-task/cpu02/episodes"/f"{label}-original.json")
        compact = {}
        for mode, x in files.items():
            env, body, native = x["environment"], x["body"], x["native"]
            reset = native["reset"]
            require(x["case"] == c and x["mode"] == mode and not x["fixture"] and
                    env["released_row_fingerprint"] == t["layouts"][str(c["init_state_id"])] and
                    env["bddl_sha256"] == t["bddl"]["bddl_sha256"] and reset["renderer_vendor"] == "NVIDIA Corporation"
                    and reset["worker_step_index"] == 0 and body["arm"] == c["arm"] and
                    body["model_class"] == cpu["body"]["model_class"], "registered simulator identity")
            require(all(env[k] == cpu["environment"][k] for k in
                        ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256", "perturbation_plan_fingerprint"))
                    and all(reset[k] == cpu["native"]["reset"][k] for k in ("prompt","horizon","noise_sha256","action_schema")),
                    "registered CPU/GPU start mismatch")
            rewards = native["native_rewards"]
            positive = [i for i, v in enumerate(rewards) if v]
            require(rewards == [0.0]*(len(rewards)-1)+[float(native["success"])] and
                    native["native_return"] == float(native["success"]) and native["controls"] == len(rewards) and
                    (native["terminated"] and not native["truncated"] if native["success"] else
                     native["truncated"] and not native["terminated"] and len(rewards) == reset["horizon"]),
                    "terminal reward, success, truncation or control-count mismatch")
            compact[mode] = dict(success=native["success"], native_return=native["native_return"],
                                 controls=len(rewards), terminated=native["terminated"], truncated=native["truncated"],
                                 nonzero_reward_indices=positive, accepted_reset_attempt=x["accepted_reset_attempt"])
        joint = files["joint"]
        m = joint["mechanism"]
        if m:
            calls = joint["calls"]
            require(m["at_control"] == 10 and np.any(calls[1]["history"]) and
                    np.any(calls[2]["history"]) and m["current"] == array_digest(verify.f32(calls[2]["history"])) and
                    m["alternatives"]["lagged"]["history"] == array_digest(verify.f32(calls[1]["history"])),
                    "same-graph nonempty history counters")
        episode_rows.append(dict(case=c, modes=compact, same_input_control_10=m,
                                 stock_parity=files["original"]["parity"],
                                 differences={mode:divergence(joint, files[mode]) for mode in matched.MODES if mode!="joint"},
                                 scene=joint["environment"]["scene_fingerprint"],
                                 reset_physics=joint["native"]["reset"]["physics"],
                                 noise=joint["native"]["reset"]["noise_sha256"],
                                 renderer=joint["native"]["reset"]["renderer"]))
    files = [p for p in out.iterdir() if p.is_file()] + list((out/"episodes").glob("*.json"))
    files += [fit/k for k in ("fit.json","heads.npz","train-inputs.json","config.json","exit.txt")] + [source_manifest]
    data = dict(job=job, pin=pin, endpoint_directory=str(out), training_revision=training["code_revision"],
                start_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
                exit_utc=datetime.fromtimestamp(end,timezone.utc).isoformat(), exit_code=rc, job_step_seconds=end-start,
                source_windows=metadata["all_source_examples_fitted"], fixed_digests=before, probe_host=probe["hostname"],
                cuda_device=dev["cuda_name"], gpu_uuid=dev["uuid"], probe_utc=probe["created_at_utc"],
                rows=episode_rows, joint_minus_comparator=summary["joint_minus_comparator"],
                source_artifacts_sha256={str(p.relative_to(run_dir)):sha(p) for p in sorted(files)},
                readback_affinity=sorted(os.sched_getaffinity(0)))
    with output.open("x") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("job")
    p.add_argument("pin")
    p.add_argument("fit_sha")
    p.add_argument("output",type=Path)
    run(**vars(p.parse_args()))
