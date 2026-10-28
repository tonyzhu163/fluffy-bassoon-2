"""Fit a privileged bounded target tracker, then test independent live scenes.

Targets are private teacher information. Executed commands always come from
the actual four-Euler residual sampler, never directly from inverse labels.
"""

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
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge
from .checkpoint import rng_state
from .pi05_bridge_train import admit_gpu, collect_live_trials, parser as live_parser
from .pi05_privileged_tracker import PrivilegedTargetTracker
from .pi05_support_probe_run import bridge_policy, native_contract
from .warmstart import atomic_save


GAINS = (.25, .5, 1.)
THETAS = (-45., 0., 45.)


def review_tracker_gate(raw):
    """Exploratory small-screen rule declared in the launch plan before execution."""
    required = {str(theta) for theta in THETAS}
    if set(raw) != required or any(set(value) != {"bridge", "zero_bridge", "external_oracle"} for value in raw.values()):
        return dict(passed=False, status="incomplete", reason="all paired live conditions are required")
    totals, per_theta = {key: 0 for key in ("bridge", "zero_bridge", "external_oracle")}, {}
    for theta, comparisons in raw.items():
        counts = {}
        for condition, values in comparisons.items():
            if (not isinstance(values, list) or len(values) != 1 or len(values[0]) != 2
                    or any(value not in (0, 1) for value in values[0])):
                raise ValueError("Tracker gate requires one complete paired two-object attempt per condition")
            counts[condition] = sum(values[0])
            totals[condition] += counts[condition]
        per_theta[theta] = counts
    no_degradation = all(row["bridge"] >= row["zero_bridge"] for row in per_theta.values())
    perturbed_success = all(per_theta[str(theta)]["bridge"] >= 1 for theta in (-45., 45.))
    improvement = totals["bridge"] > totals["zero_bridge"]
    passed = bool(no_degradation and perturbed_success and improvement)
    return dict(passed=passed, status="provisional_capacity_screen_passed" if passed else "unresolved",
        per_theta=per_theta, totals=totals, no_per_theta_degradation=no_degradation,
        perturbed_success_each=perturbed_success, aggregate_improvement=improvement,
        interpretation="six paired episodes per condition; privileged constrained controller only; no learned-CTM or ICRL claim")


def main(argv=None):
    p = live_parser()
    p.description = __doc__
    p.set_defaults(num_envs=2, max_elapsed_seconds=2700.)
    p.add_argument("--reference-plan-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if (sys.platform != "linux" or args.num_envs != 2 or args.width != 64 or args.objects != [0, 1]
            or args.resume or args.warmstart or args.collect_dpt or args.balanced_paired_plan
            or not math.isfinite(args.max_elapsed_seconds) or not 0 < args.max_elapsed_seconds <= 2700):
        raise ValueError("Require allocated Linux and the bounded two-object tracker gate")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed source")
    if sha256(args.model_path / "model.safetensors") != args.expected_model_sha256:
        raise ValueError("Frozen motor input hash differs")
    reference_plans = {}
    for theta in THETAS:
        path = args.reference_plan_dir / f"plan-{int(theta):+d}.json"
        plan = json.loads(path.read_text())
        if (plan.get("theta_degrees") != [theta, theta] or len(plan.get("scenes", [])) != 1
                or [row["object"] for row in plan["scenes"][0]] != [0, 1]
                or any(row["overlay"] not in range(12, 16) for row in plan["scenes"][0])):
            raise ValueError("Held-out reference plans must retain paired theta/object and held-out overlays")
        reference_plans[str(theta)] = dict(plan=plan, sha256=sha256(path))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env, stop, started = None, False, time.monotonic()
    raw, calibration, selected, diagnostics = {}, {}, {"0.0": 1.}, {}
    counters = dict(simulator_steps=0, policy_chunks=0, native_prefix_queries=0,
                    native_suffix_queries=0, extra_label_motor_calls=0)
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), request_stop)
    def should_stop():
        return stop or time.monotonic()-started >= args.max_elapsed_seconds
    def save_progress(phase):
        receipt = dict(source_commit=source, phase=phase, calibration=calibration, selected_gains=selected,
            comparisons=raw, diagnostics=diagnostics, counters=counters, elapsed_seconds=time.monotonic()-started)
        write_json(args.output_dir / "progress.json", receipt)
        print(json.dumps(dict(phase=phase, selected_gains=selected, comparisons=raw, counters=counters), allow_nan=False), flush=True)
        return receipt
    try:
        admit_gpu(args, root, args.output_dir)
        sys.path.insert(0, str(args.rlinf_root))
        from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
        import rlinf.envs.sim.maniskill.tasks  # noqa: F401
        from rlinf.models import get_model
        from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
        from .pi05_balanced_plan import make_balanced_history_manifest
        def preprocess(observation):
            prepared = preprocess_observation(observation, train=False)
            if not torch.equal(prepared.state, observation.state):
                raise ValueError("Tracker decoder requires the preprocessor to preserve public state")
            return prepared
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env = ManiskillEnv(ms.env_config(args.rlinf_root, 2, args.seed), 2, seed_offset=0,
                          total_num_processes=1, worker_info=None, record_metrics=True)
        ms.record_renderer_binding(env, args.output_dir)
        motor = get_model(ms.model_config(args.model_path, args.norm_stats)).cuda().eval().requires_grad_(False)
        if (motor.num_steps, motor.action_horizon, motor.action_dim) != (4, 8, 32):
            raise ValueError("Motor differs from the qualified four-Euler contract")
        frozen_before = ms.model_fingerprint(motor)
        tracker = PrivilegedTargetTracker(Pi05FeatureBridge(core="persistent_ctm", width=64,
            ticks=4, residual_scale=1.)).cuda().eval()
        zero, bridge_config = bridge_policy("persistent_ctm")
        contract = native_contract(args, source, frozen_before, zero, bridge_config)
        write_json(args.output_dir / "config.json", dict(contract=contract, tracker=tracker.config(),
            gains=list(GAINS), heldout_reference_plans=reference_plans, hostname=os.uname().nodename,
            job_id=ms.job_id(), max_elapsed_seconds=args.max_elapsed_seconds,
            gain_selection="maximize training success, then prefer higher gain; no held-out tuning",
            pass_rule="all per-theta bridge >= zero; >=1 perturbed success per angle; strict aggregate gain",
            qualification="privileged controller capacity screen; no automatic PPO admission"))
        generator = torch.Generator(device="cuda")
        def run(policy, plan, name, *, execute_labels=False):
            before = ms.model_fingerprint(policy)
            generator.manual_seed(810000)
            rollout = collect_live_trials(policy, motor, env, plan, generator=generator,
                deterministic=True, execute_labels=execute_labels, preprocess=preprocess)
            if ms.model_fingerprint(policy) != before:
                raise RuntimeError("Frozen tracker evaluation changed policy tensors")
            chunks = rollout["policy_chunks"]
            counters["simulator_steps"] += rollout["simulator_steps"]
            counters["policy_chunks"] += chunks
            counters["native_prefix_queries"] += chunks
            counters["native_suffix_queries"] += 4*chunks
            if execute_labels:
                counters["extra_label_motor_calls"] += chunks
                counters["native_prefix_queries"] += chunks
                counters["native_suffix_queries"] += 4*chunks
            rows = [record["privileged_tracker"] for record in rollout["records"] if "privileged_tracker" in record]
            for row in rows:
                if row["residual_mean_abs_max"] > 1. or row["actual_executed_action_steps"] != 5:
                    raise RuntimeError("Tracker escaped the actual bounded intervention path")
                counters["native_prefix_queries"] += row["extra_native_prefix_queries"]
                counters["native_suffix_queries"] += row["extra_native_suffix_queries"]
            if rows:
                diagnostics[name] = rows
            write_json(args.output_dir / f"audit-{name}.json", rollout["audit"])
            atomic_save(args.output_dir / "latest.pt", dict(source_commit=source, phase=name,
                selected_gains=selected, calibration=calibration, comparisons=raw, counters=counters,
                policy=tracker.state_dict(), motor_rng=generator.get_state(), rng=rng_state(),
                last_outcomes=rollout["outcomes"].tolist(),
                optimization="finite scalar gain selection; no gradient optimizer", reference_plans=reference_plans))
            return rollout["outcomes"].tolist()
        calibration_source = make_balanced_history_manifest(env.env.unwrapped.basic_obj_infos,
                                                           seed=610000 + args.seed)["plans"][0]
        for theta in (-45., 45.):
            if should_stop():
                break
            indices = [i for i, value in enumerate(calibration_source["theta_degrees"]) if value == theta]
            plan = dict(theta_degrees=[theta, theta],
                scenes=[[calibration_source["scenes"][0][i] for i in indices]],
                reset_seeds=[calibration_source["reset_seeds"][0]],
                split="tracker_gain_calibration_train_overlays",
                balanced_source_plan_sha256=calibration_source["plan_sha256"])
            write_json(args.output_dir / f"calibration-plan-{int(theta):+d}.json", plan)
            calibration[str(theta)] = {}
            for gain in GAINS:
                if should_stop():
                    break
                tracker.set_gain(gain).select_theta([theta, theta])
                calibration[str(theta)][str(gain)] = run(tracker, plan, f"calibrate-{int(theta):+d}-{gain}")
                save_progress("gain_calibration")
            if len(calibration[str(theta)]) == len(GAINS):
                selected[str(theta)] = max(GAINS, key=lambda gain: (
                    sum(sum(row) for row in calibration[str(theta)][str(gain)]), gain))
        for theta in THETAS:
            if should_stop() or str(theta) not in selected:
                break
            plan = reference_plans[str(theta)]["plan"]
            raw[str(theta)] = {}
            for condition in ("bridge", "zero_bridge", "external_oracle"):
                if should_stop():
                    break
                tracker.set_gain(selected[str(theta)]).select_theta([theta, theta])
                raw[str(theta)][condition] = run(tracker if condition == "bridge" else zero, plan,
                    f"heldout-{int(theta):+d}-{condition}", execute_labels=condition == "external_oracle")
                save_progress("heldout_live_comparison")
        if ms.model_fingerprint(motor) != frozen_before:
            raise RuntimeError("Tracker gate changed frozen motor")
        result = save_progress("finished")
        result.update(review=review_tracker_gate(raw), bounded_stop=should_stop(),
                      frozen_motor_before=frozen_before, frozen_motor_after=ms.model_fingerprint(motor))
        write_json(args.output_dir / "summary.json", result)
    except BaseException as error:
        write_json(args.output_dir / "failure.json", dict(error_type=type(error).__name__, error=str(error),
            counters=counters, source_commit=source, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if env is not None:
            env.env.close()


if __name__ == "__main__":
    main()
