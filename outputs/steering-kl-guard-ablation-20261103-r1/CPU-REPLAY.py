#!/usr/bin/env python3
"""Counterfactual one-epoch KL ablation on real source banks; allocated Linux only."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    p = Path(path)
    temporary = p.with_name(p.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--expected-source", required=True)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-hostname", required=True)
    args = ap.parse_args()
    if (platform.system() != "Linux" or socket.gethostname() != args.expected_hostname
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("Require the bound allocated Linux CPU container and hidden CUDA")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.code, text=True).strip() != args.expected_source:
        raise RuntimeError("Code commit differs from the reviewed source")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=args.code, text=True).strip():
        raise RuntimeError("Diagnostic checkout is dirty")
    args.output.mkdir(parents=True, exist_ok=False)
    probe = args.output / ".write-probe"
    probe.write_bytes(b"kl-ablation")
    probe.unlink()
    fs = os.statvfs(args.output)
    if fs.f_bavail * fs.f_frsize < 2 * 2**30:
        raise RuntimeError("Less than2GiB free for independent optimizer artifacts")
    sys.path.insert(0, str(args.code / "src"))
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, update
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, tensor_hash
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU replay must not allocate an accelerator")
    started = time.time()
    result = dict(format="real_first_bank_kl_ablation_v1", completed=False,
        hostname=socket.gethostname(), pid=os.getpid(), source_commit=args.expected_source,
        script_sha256=sha(__file__), started_unix=started, torch_version=torch.__version__,
        CUDA_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device="cpu",
        cpu_threads=torch.get_num_threads(), new_simulator_controls=0, scientific_source_updates=0,
        resource=dict(cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            storage_free_GiB=fs.f_bavail * fs.f_frsize / 2**30), arms={})
    write(args.output / "RESULT.json", result)
    source_hashes = {}
    for arm, source_name in (("velocity", "velocity"), ("kv_all_layers", "kv")):
        source = args.sources / f"robocasa-joint-steering-{source_name}-scale50-20261103-r1"
        bank_path = source / "banks/round-001-before-sgd.pt"
        boundary_path = source / "checkpoints/event-003-dev-000.pt"
        final_path = source / "checkpoints/event-005-train-002.pt"
        source_hashes.update({str(p): sha(p) for p in (bank_path, boundary_path, final_path)})
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        boundary = torch.load(boundary_path, map_location="cpu", weights_only=False)
        if (bank["policy_version"] != 0 or boundary["iteration"] != 0 or boundary["sequence"] != 3
                or boundary["arm"] != arm or boundary["optimizer"]["state"]):
            raise RuntimeError("Require the real initial batch and pre-first-update empty Adam state")
        config = ResidualPPOConfig(**boundary["policy_config"])
        if config.target_kl != .02 or config.epochs != 10 or config.minibatch_size != 32:
            raise RuntimeError("Historical production optimizer contract differs")
        rollout = {k: cpu_tree(v) for k, v in bank["rollout"].items()}
        bank_policy_hash = tensor_hash(bank["policy"])
        row = dict(source=str(source), source_hashes={str(p): source_hashes[str(p)]
            for p in (bank_path, boundary_path, final_path)}, policy_version=bank["policy_version"],
            original_config=asdict(config), valid_chunk_samples=int(rollout["rewards"].numel()),
            episodes=len(bank["episodes"]), positive_rewards=int((rollout["rewards"] > 0).sum()),
            normalized_initial_policy_sha256=bank_policy_hash, variants={})
        directory = args.output / arm
        directory.mkdir()
        for name, epochs, target in (("guarded1", 1, .02), ("unguarded1", 1, None),
                                     ("guarded10_reference", 10, .02)):
            candidate_config = replace(config, epochs=epochs, target_kl=target)
            random.setstate(boundary["global_rng"]["python"])
            np.random.set_state(boundary["global_rng"]["numpy"])
            torch.set_rng_state(boundary["global_rng"]["torch"])
            policy = ResidualActorCritic(candidate_config)
            policy.load_state_dict(deepcopy(bank["policy"]), strict=True)
            optimizer = torch.optim.Adam(policy.parameters(), lr=candidate_config.learning_rate)
            optimizer.load_state_dict(deepcopy(boundary["optimizer"]))
            generator = torch.Generator()
            generator.set_state(boundary["generator"])
            initial_model_hash = tensor_hash(policy.state_dict())
            initial_adam_hash = tensor_hash(optimizer.state_dict())
            initial_sampler_hash = tensor_hash(generator.get_state())
            if initial_model_hash != bank_policy_hash:
                raise RuntimeError("Candidate does not start from the same actual pre-SGD model")
            with torch.no_grad():
                initial_dist, _ = policy._distribution(rollout["features"].reshape(-1, config.feature_dim))
                initial_mean = initial_dist.mean.clone()
                initial_std = initial_dist.stddev.clone()
            metrics = update(policy, optimizer, deepcopy(rollout), generator=generator)
            if metrics["initial_metrics"]["joint_gaussian_kl"] > 1e-5:
                raise RuntimeError("Saved behavior distribution does not match the pre-update policy")
            with torch.no_grad():
                final_dist, _ = policy._distribution(rollout["features"].reshape(-1, config.feature_dim))
                shift = dict(mean_delta_RMS=float((final_dist.mean - initial_mean).square().mean().sqrt()),
                    mean_delta_RMS_in_behavior_std=float(((final_dist.mean-initial_mean)/initial_std).square().mean().sqrt()),
                    final_std_min=float(final_dist.stddev.min()), final_std_mean=float(final_dist.stddev.mean()),
                    final_std_max=float(final_dist.stddev.max()),
                    deterministic_tanh_mean_delta_RMS=float((final_dist.mean.tanh()-initial_mean.tanh()).square().mean().sqrt()))
            candidate = dict(format="steering_kl_counterfactual_v1", arm=arm, candidate=name,
                source_policy_version=0, diagnostic_update_count=1, source_commit=args.expected_source,
                source_bank=dict(path=str(bank_path), sha256=source_hashes[str(bank_path)]),
                policy_config=asdict(candidate_config), interface_spec=bank["interface_spec"],
                policy=cpu_tree(policy.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                generator=generator.get_state(), global_rng=dict(python=random.getstate(),
                    numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=[]), metrics=metrics)
            candidate_path = directory / (name + ".pt")
            torch.save(candidate, candidate_path)
            row["variants"][name] = dict(config=asdict(candidate_config), metrics=metrics, policy_shift=shift,
                initial_model_sha256=initial_model_hash, initial_adam_sha256=initial_adam_hash,
                initial_sampler_sha256=initial_sampler_hash, final_model_sha256=tensor_hash(policy.state_dict()),
                final_adam_sha256=tensor_hash(optimizer.state_dict()), final_sampler_sha256=tensor_hash(generator.get_state()),
                artifact=dict(path=str(candidate_path), sha256=sha(candidate_path)),
                gradient_sample_fraction=metrics["samples_processed"] / metrics["samples"])
            write(directory / (name + ".json"), row["variants"][name])
            result["arms"][arm] = row
            write(args.output / "RESULT.json", result)
            print(json.dumps(dict(arm=arm, variant=name, steps=metrics["optimizer_steps"],
                samples=metrics["samples_processed"], total=metrics["samples"],
                KL=metrics["final_metrics"]["joint_gaussian_kl"],
                stop=metrics["stop_reason"], elapsed=metrics["elapsed_seconds"])), flush=True)
        variants = row["variants"]
        if len({v["initial_adam_sha256"] for v in variants.values()}) != 1 or len({
                v["initial_sampler_sha256"] for v in variants.values()}) != 1:
            raise RuntimeError("Counterfactual candidates have different initial Adam or sampler state")
        first, reference = variants["guarded1"], variants["guarded10_reference"]
        row["guarded_epoch_limit_equivalence"] = dict(
            stopped_during_first_epoch=first["metrics"]["early_stopped"] and reference["metrics"]["epochs_started"] == 1,
            model_exact=first["final_model_sha256"] == reference["final_model_sha256"],
            adam_exact=first["final_adam_sha256"] == reference["final_adam_sha256"],
            sampler_exact=first["final_sampler_sha256"] == reference["final_sampler_sha256"])
        if row["guarded_epoch_limit_equivalence"]["stopped_during_first_epoch"] and not all(
                row["guarded_epoch_limit_equivalence"][k] for k in ("model_exact", "adam_exact", "sampler_exact")):
            raise RuntimeError("Epoch1 stop should reproduce the original guarded epoch limit exactly")
        unguarded = variants["unguarded1"]["metrics"]
        if (unguarded["early_stopped"] or unguarded["epochs_completed"] != 1
                or unguarded["samples_processed"] != unguarded["samples"]):
            raise RuntimeError("Unguarded candidate failed to process exactly one complete epoch")
        # Keep the exact common pre-SGD initial policy for same-batch native baseline evaluation.
        initial_path = directory / "initial.pt"
        torch.save(dict(format="steering_kl_counterfactual_v1", arm=arm, candidate="initial",
            source_policy_version=0, diagnostic_update_count=0, source_commit=args.expected_source,
            policy_config=asdict(config), interface_spec=bank["interface_spec"], policy=cpu_tree(bank["policy"])), initial_path)
        row["initial_artifact"] = dict(path=str(initial_path), sha256=sha(initial_path))
        write(directory / "RESULT.json", row)
    result.update(completed=True, elapsed_seconds=time.time()-started,
        source_files_unchanged=all(sha(p) == h for p, h in source_hashes.items()),
        source_hashes_before_and_after=source_hashes, finished_unix=time.time(),
        interpretation="Numerical optimizer diagnostic only; native success evaluation is separate")
    if not result["source_files_unchanged"]:
        raise RuntimeError("A source input changed during the independent replay")
    write(args.output / "RESULT.json", result)
    print(json.dumps(dict(completed=True, elapsed_seconds=result["elapsed_seconds"],
        source_files_unchanged=True)), flush=True)


if __name__ == "__main__":
    main()
