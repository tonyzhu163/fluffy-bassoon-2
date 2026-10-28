"""Frozen collector-by-reader diagnostics on identical saved public histories."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.rl import maniskill_pi05 as ms
from .checkpoint import rng_state
from .pi05_bridge_train import admit_gpu
from .pi05_fresh_probe import (ProbeBoundary, _zero_bridge_policy, clone_boundary,
                               run_live_attempt, validate_probe_plans)
from .pi05_support_probe_run import bridge_policy, native_contract, load_direct_candidate
from .warmstart import NATIVE, atomic_save, checked_load, initialize_from_warmstart


def public_interface_sha256(contract):
    """Architecture-independent identity for exactly the same frozen public stream."""
    keys = ("task", "motor_sha256", "norm_stats_sha256", "rlinf_sources", "feedback", "reward",
            "attempts", "episode_steps", "action_chunk", "denoise_steps")
    data = {key: contract[key] for key in keys}
    data.update(kv_layer=-1, kv_dim=512, feedback_dim=11, active_dims=7, residual_scale=1.)
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_reader(path, digest, policy, contract):
    saved = checked_load(path, digest, weights_only=False)
    if saved.get("format") == "ctm_benchmark_warmstart_v1":
        return initialize_from_warmstart(path, digest, family=NATIVE, policy=policy, contract=contract)
    return load_direct_candidate(path, digest, policy, contract)


def paired_probe_validity(branches):
    """Every branch must start from the same observation and native noise stream."""
    prefixes = [row.get("first_prefix_sha256") for row in branches]
    noises = [row.get("final_noise_state_sha256") for row in branches]
    return dict(branches=len(branches), first_prefix_identical=bool(prefixes) and None not in prefixes
                and len(set(prefixes)) == 1,
                native_noise_streams_identical=bool(noises) and None not in noises and len(set(noises)) == 1)


def own_reader_parity(reconstructed, original):
    if reconstructed.memory.keys() != original.memory.keys():
        raise RuntimeError("Replayed own-reader memory schema differs from collection")
    errors = {}
    for key, actual in reconstructed.memory.items():
        expected = original.memory[key].to(actual)
        if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Replayed own-reader memory differs from collection")
        errors[key] = float((actual-expected).abs().max())
    return dict(max_abs_error=max(errors.values()), per_field=errors, atol=1e-5, rtol=1e-5)


def parser():
    from .pi05_bridge_train import parser as live_parser
    p = live_parser()
    p.description = __doc__
    p.set_defaults(num_envs=4, attempts=4, max_elapsed_seconds=2400.)
    for short in ("ctm", "xl"):
        p.add_argument(f"--reader-{short}", type=Path, required=True)
        p.add_argument(f"--reader-{short}-sha256", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (sys.platform != "linux" or args.num_envs != 4 or args.attempts != 4 or args.width != 64
            or args.objects != [0, 1] or args.noise_scale != .05 or args.residual_scale != 1.
            or args.resume or args.warmstart or args.collect_dpt or args.threads < 1
            or not math.isfinite(args.max_elapsed_seconds) or not 0 < args.max_elapsed_seconds <= 2400):
        raise ValueError("Require the bounded allocated-Linux four-lane K4 crossed-probe contract")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (source != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require an exact clean reviewed source commit")
    if sha256(args.model_path / "model.safetensors") != args.expected_model_sha256:
        raise ValueError("Frozen motor input checksum differs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, stop, env = time.monotonic(), False, None
    costs = dict(simulator_steps=0, action_chunks=0, native_prefix_queries=0,
                 native_suffix_queries=0, bridge_context_updates=0, cached_reader_context_updates=0)
    histories, rows, support_rows, pairing, parents, replay_checks = {}, [], [], {}, {}, {}
    def should_stop():
        return stop or time.monotonic()-started >= args.max_elapsed_seconds
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), request_stop)
    def add_costs(counts):
        for key in counts:
            costs[key] += counts[key]
    def progress(phase):
        result = dict(phase=phase, source_commit=source, histories=histories, rows=rows,
            support_rows=support_rows, pairing=pairing, parents=parents, costs=costs,
            replay_checks=replay_checks,
            elapsed_seconds=time.monotonic()-started,
            interpretation="crossed diagnostic; reader familiarity with another collector's histories is not established; no active-experimentation claim")
        write_json(args.output_dir / "progress.json", result)
        print(json.dumps(dict(phase=phase, histories=list(histories), probe_rows=len(rows), costs=costs,
                              elapsed_seconds=result["elapsed_seconds"]), allow_nan=False), flush=True)
        return result
    try:
        admit_gpu(args, root, args.output_dir)
        sys.path.insert(0, str(args.rlinf_root))
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
        import rlinf.envs.sim.maniskill.tasks  # noqa: F401
        from rlinf.models import get_model
        from .pi05_balanced_plan import make_heldout_probe_plan
        from .pi05_public_history import (PublicHistoryRecorder, replay_public_history,
                                          save_public_history)
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env = ManiskillEnv(ms.env_config(args.rlinf_root, 4, args.seed), 4, seed_offset=0,
                          total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        motor = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval().requires_grad_(False)
        if (motor.num_steps, motor.action_horizon, motor.action_dim) != (4, 8, 32):
            raise ValueError("Native motor differs from qualified four-Euler model")
        motor_before = ms.model_fingerprint(motor)
        readers, contracts = {}, {}
        for name, core in (("ctm", "persistent_ctm"), ("xl", "transformer_xl")):
            policy, config = bridge_policy(core)
            contract = native_contract(args, source, motor_before, policy, config)
            parents[name] = load_reader(getattr(args, f"reader_{name}"),
                getattr(args, f"reader_{name}_sha256"), policy, contract)
            readers[name], contracts[name] = policy, contract
        interface = public_interface_sha256(contracts["ctm"])
        if public_interface_sha256(contracts["xl"]) != interface:
            raise ValueError("Readers must consume identical frozen public stream interfaces")
        plan = make_heldout_probe_plan(env.env.unwrapped.basic_obj_infos, seed=510000 + args.seed)
        support, probes = plan["support_plan"], plan["probe_plans"]
        lanes = validate_probe_plans(support, probes, plan["probe_seeds"], plan["opposite_indices"])
        write_json(args.output_dir / "plan.json", plan)
        write_json(args.output_dir / "config.json", dict(source_commit=source, protocol="pi05_crossed_public_history_v1",
            motor_sha256=args.expected_model_sha256, public_interface_sha256=interface, parents=parents,
            collectors=["frozen_motor", "ctm"], readers=list(readers), conditions=["normal", "reset_memory"],
            collector_sampling="deterministic residual mean; paired native noise generator",
            reader_training_coverage={name: dict(parent_kind=parent.get("kind", "direct_ppo"),
                familiarity="cross-collector coverage not established; retain parent training provenance") for name, parent in parents.items()},
            independent_history_batches=1, max_elapsed_seconds=args.max_elapsed_seconds,
            seed=args.seed, hostname=os.uname().nodename, job_id=ms.job_id()))
        zero = _zero_bridge_policy(readers["ctm"])
        fingerprints = {name: ms.model_fingerprint(policy) for name, policy in readers.items()}
        for collector_name, collector in (("frozen_motor", zero), ("ctm", readers["ctm"])):
            if should_stop():
                break
            history_ids = tuple(f"support-{collector_name}/lane-{lane}" for lane in range(lanes))
            reference = next(collector.parameters())
            state = ProbeBoundary(None, reference.new_zeros(lanes, 7), torch.zeros(lanes), 0, history_ids)
            recorder = PublicHistoryRecorder(motor_sha256=args.expected_model_sha256,
                interface_sha256=interface, history_ids=history_ids)
            generator = torch.Generator(device="cuda").manual_seed(520000 + args.seed)
            complete = True
            for attempt in range(3):
                if should_stop():
                    complete = False
                    break
                result = run_live_attempt(collector, motor, env, support["scenes"][attempt],
                    support["theta_degrees"], support["reset_seeds"][attempt], state,
                    generator=generator, on_public_chunk=recorder.on_chunk)
                state = result["state"]
                recorder.on_boundary(state)
                add_costs(result["counts"])
                support_rows.append(dict(collector=collector_name, attempt_index=attempt,
                                         success_once=result["success_once"], counts=result["counts"]))
                atomic_save(args.output_dir / f"{collector_name}-boundary-{attempt+1}.pt",
                    dict(state=state, generator_state=generator.get_state(), rng=rng_state(),
                         public_history_partial=recorder.partial_snapshot(),
                         collector_fingerprint=ms.model_fingerprint(collector), source_commit=source))
                progress(f"collect-{collector_name}-{attempt+1}")
            if not complete:
                break
            history = recorder.finish()
            saved = save_public_history(args.output_dir / f"history-{collector_name}", history,
                source_commit=source, collector_id=collector_name,
                collector_checkpoint_sha256=args.reader_ctm_sha256)
            histories[collector_name] = saved
            histories[collector_name].update(collector_fingerprint=ms.model_fingerprint(collector),
                collector_derivation="zeroed residual output of CTM parent" if collector_name == "frozen_motor" else "unchanged CTM parent")
            progress(f"saved-{collector_name}")
            for reader_name, reader in readers.items():
                if should_stop():
                    break
                boundary = replay_public_history(reader, history,
                    expected_motor_sha256=args.expected_model_sha256, expected_interface_sha256=interface)
                costs["cached_reader_context_updates"] += 3 * 16 * lanes
                if reader_name == "ctm":
                    replay_checks[collector_name] = own_reader_parity(boundary, state)
                # Replay adds reader compute, not simulator or frozen motor queries.
                for probe_index, probe in enumerate(probes):
                    for condition in ("normal", "reset_memory"):
                        if should_stop():
                            break
                        copied = clone_boundary(boundary, reset_memory=condition == "reset_memory")
                        result = run_live_attempt(reader, motor, env, probe["scenes"], support["theta_degrees"],
                            probe["reset_seed"], copied,
                            generator=torch.Generator(device="cuda").manual_seed(plan["probe_seeds"][probe_index]))
                        add_costs(result["counts"])
                        branch = dict(collector=collector_name, reader=reader_name, condition=condition,
                            probe_index=probe_index, first_prefix_sha256=result["first_prefix_sha256"],
                            final_noise_state_sha256=result["final_noise_state_sha256"])
                        pairing.setdefault(str(probe_index), []).append(branch)
                        for lane in range(lanes):
                            rows.append(dict(collector=collector_name, reader=reader_name, condition=condition,
                                probe_index=probe_index, lane=lane, history_id=history_ids[lane],
                                cluster_id="support-batch-0", theta_degrees=support["theta_degrees"][lane],
                                object=probe["scenes"][lane]["object"], success_once=result["success_once"][lane],
                                success_at_end=result["success_at_end"][lane], first_feedback=result["first_feedback"][lane]))
                        progress(f"probe-{collector_name}-{reader_name}-{condition}")
            del history
        # A single frozen-motor fresh probe is a common reference across readers.
        for probe_index, probe in enumerate(probes):
            if should_stop():
                break
            reference = next(zero.parameters())
            boundary = ProbeBoundary(None, reference.new_zeros(lanes, 7), torch.zeros(lanes), 3,
                tuple(f"reference/lane-{lane}" for lane in range(lanes)), attempt_end=True)
            result = run_live_attempt(zero, motor, env, probe["scenes"], support["theta_degrees"],
                probe["reset_seed"], boundary,
                generator=torch.Generator(device="cuda").manual_seed(plan["probe_seeds"][probe_index]))
            add_costs(result["counts"])
            pairing.setdefault(str(probe_index), []).append(dict(collector="none", reader="frozen_motor",
                condition="reference", probe_index=probe_index, first_prefix_sha256=result["first_prefix_sha256"],
                final_noise_state_sha256=result["final_noise_state_sha256"]))
            for lane in range(lanes):
                rows.append(dict(collector="none", reader="frozen_motor", condition="reference", probe_index=probe_index,
                    lane=lane, cluster_id="support-batch-0", history_id=f"reference/lane-{lane}",
                    success_once=result["success_once"][lane], success_at_end=result["success_at_end"][lane]))
        if any(ms.model_fingerprint(readers[name]) != fingerprint for name, fingerprint in fingerprints.items()):
            raise RuntimeError("Frozen crossed evaluation changed reader weights")
        if ms.model_fingerprint(motor) != motor_before:
            raise RuntimeError("Frozen crossed evaluation changed motor weights")
        result = progress("finished")
        result.update(completed=len(rows) == lanes*len(probes)*9,
            pairing_checks={key: paired_probe_validity(value) for key, value in pairing.items()},
            frozen_motor_fingerprint=motor_before, reader_fingerprints=fingerprints,
            bounded_stop=should_stop(), independent_clusters=1)
        write_json(args.output_dir / "summary.json", result)
    except BaseException as error:
        write_json(args.output_dir / "failure.json", dict(error_type=type(error).__name__, error=str(error),
            costs=costs, source_commit=source, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
