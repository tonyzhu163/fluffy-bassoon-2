"""Shared collection order and complete-boundary state for frozen steering PPO.

The learned action is the steering coefficient vector. The separately recorded
native Gaussian is independent, has a fixed law, and cancels from PPO ratios.
Physical post-success suffixes remain in the archive and action ledger, while
GAE sees complete logical episodes ending at the first-success chunk.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .residual_ppo import ResidualActorCritic, ResidualPPOConfig
from .rlinf_native_ppo import cpu_tree, snapshot_rng

TASKS = ("OpenDrawer", "PickPlaceCounterToCabinet", "PickPlaceDrawerToCounter",
         "TurnOffStove", "CoffeeSetupMug")
OOD_TASKS = ("CloseFridge", "OpenCabinet", "TurnOnMicrowave", "TurnOnSinkFaucet",
             "OpenStandMixerHead")
TRAIN_LIMIT = 11_650_000
EVAL_LIMIT = 1_200_000
UPDATES = 50
TRAIN_EPISODES = 384
ROLLOUT_LANES = 48
WORKER_COUNT = 8
DEV_ROUNDS = tuple(range(0, UPDATES + 1, 5))

# Frozen public-observation action sensitivity grid; rewards never select scale.
SCALE_FRACTIONS = {
    "velocity": (.025, .05, .1, .125, .15, .2, .3),
    "kv_all_layers": (.01, .02, .25 / 6, .08, .16, .24, .32, .48, .64, .96, 1.28),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class JointActionJournal:
    """Constant-size locked escrow; individual physical calls stay durable.

    Reserve before dispatch. Unresolved calls remain fully charged. A finish
    writes its detailed call first, so an interrupted finish can overcharge
    but cannot permit excess work. Never silently release an old reservation.
    """
    def __init__(self, root, *, rank, category, scope):
        if category not in ("train", "eval"):
            raise ValueError("Require a bound train/eval action category")
        self.root, self.rank, self.category, self.scope = Path(root), int(rank), category, str(scope)
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = self.root / f"rank-{self.rank:02d}"
        self.calls.mkdir(exist_ok=True)
        self.state_path = self.root / "escrow.json"
        with self._locked():
            if not self.state_path.exists():
                write_json(self.state_path, dict(format="joint_steering_action_escrow_v1",
                    limits=dict(train=TRAIN_LIMIT, eval=EVAL_LIMIT),
                    charged=dict(train=0, eval=0), pending={}))
            state = self._read()
            if state["limits"] != dict(train=TRAIN_LIMIT, eval=EVAL_LIMIT):
                raise ValueError("Preserve the approved action ceilings")

    @contextmanager
    def _locked(self):
        with (self.root / ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _read(self):
        return json.loads(self.state_path.read_text())

    def begin(self, requested, *, phase):
        if type(requested) is not int or requested <= 0:
            raise ValueError("A physical call needs a positive finite reservation")
        with self._locked():
            state = self._read()
            if state["charged"][self.category] + requested > state["limits"][self.category]:
                raise RuntimeError(f"Approved {self.category} action ceiling reached")
            token = f"r{self.rank:02d}-{time.time_ns()}"
            row = dict(token=token, rank=self.rank, category=self.category, scope=self.scope,
                       phase=phase, requested=requested, resolved=False, started_unix=time.time())
            state["charged"][self.category] += requested
            state["pending"][token] = row
            write_json(self.state_path, state)
            write_json(self.calls / (token + ".json"), row)
            return token

    def finish(self, token, executed):
        with self._locked():
            state = self._read()
            row = state["pending"].get(token)
            if (row is None or row["rank"] != self.rank or type(executed) is not int
                    or not 0 <= executed <= row["requested"]):
                raise ValueError("Invalid or duplicate physical-call resolution")
            row = dict(row, resolved=True, executed_or_conservative=executed, ended_unix=time.time())
            write_json(self.calls / (token + ".json"), row)
            state["charged"][row["category"]] -= row["requested"] - executed
            del state["pending"][token]
            write_json(self.state_path, state)

    def charged(self):
        with self._locked():
            return sum(self._read()["charged"].values())

    def snapshot(self):
        with self._locked():
            return deepcopy(self._read())


def spec_arguments(arm):
    if arm == "velocity":
        return dict(kind="velocity", steering_dim=32, fraction=.05)
    if arm == "kv_all_layers":
        return dict(kind="kv_all_layers", steering_dim=1152, kv_layers=18, fraction=.25 / 6)
    raise ValueError("Only the two human-approved frozen steering arms")


def cases_for(phase, iteration=0):
    """All tasks in each bank; task identity never acts as a curriculum clock."""
    from .rlinf_robocasa_ppo import case_bank
    if phase == "train":
        if type(iteration) is not int or not 1 <= iteration <= UPDATES:
            raise ValueError("Require an approved prospective training round")
        # One task has76 cases, the other four77. Rotate the smaller task;
        # consecutive per-task slots preserve every episode without planned holes.
        prior = iteration - 1
        banks = []
        for task_index, task in enumerate(TASKS):
            count = 76 if task_index == prior % len(TASKS) else 77
            offset = prior * 77 - (prior + 4 - task_index) // len(TASKS)
            banks.append([dict(bank="joint_steering_train", task=task, split="target",
                native_seed=1_700_000 + task_index * 100_000 + offset + slot,
                reset_ordinal=0, slot=offset + slot, counted=True,
                layout_source="native_target_constructor_seed_first_reset") for slot in range(count)])
    elif phase == "dev":
        banks = [case_bank(task, "DEV", count=8) for task in TASKS]
    elif phase in ("test_before", "test_after"):
        banks = [case_bank(task, "ID", count=50) for task in TASKS]
        banks += [case_bank(task, "OOD", count=50) for task in OOD_TASKS]
    else:
        raise ValueError("Unknown scientific collection phase")
    return [bank[slot] for slot in range(max(map(len, banks))) for bank in banks if slot < len(bank)]


def case_key(case):
    return case["task"], int(case["native_seed"])


def chronological_records(results, cases):
    """Regroup interleaved lanes before GAE, without mixing task episodes."""
    by_case = {case_key(case): [] for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("A bank may not repeat a native case")
    for result in results:
        for row in result["records"]:
            key = case_key(row["case"])
            if key not in by_case:
                raise ValueError("A worker returned an unrequested case")
            by_case[key].append(row)
    ordered = []
    for case in cases:
        rows = sorted(by_case[case_key(case)], key=lambda row: row["chunk_index"])
        if (not rows or not rows[0]["episode_start"] or not rows[-1]["done"]
                or any(row["done"] for row in rows[:-1])
                or [row["chunk_index"] for row in rows] != list(range(len(rows)))):
            raise ValueError("GAE requires a complete contiguous first-success episode")
        ordered.extend(rows)
    return ordered


def summarize_episodes(episodes):
    grouped = {}
    identities = set()
    for row in episodes:
        key = case_key(row["case"])
        if key in identities:
            raise ValueError("Never count an episode twice")
        identities.add(key)
        task = row["case"]["task"]
        counts = grouped.setdefault(task, dict(episodes=0, success_once=0, success_at_end=0))
        counts["episodes"] += 1
        for metric in ("success_once", "success_at_end"):
            counts[metric] += bool(row[metric])
    def macro(tasks, metric):
        present = [task for task in tasks if task in grouped]
        return float(np.mean([grouped[t][metric] / grouped[t]["episodes"] for t in present])) if present else None
    return dict(episodes=len(episodes), per_task=grouped,
        macro_ID_success_once=macro(TASKS, "success_once"),
        macro_ID_success_at_end=macro(TASKS, "success_at_end"),
        macro_OOD_success_once=macro(OOD_TASKS, "success_once"),
        macro_OOD_success_at_end=macro(OOD_TASKS, "success_at_end"))


def make_policy(arm, *, device="cpu", seed=1234, optimizer_config=None):
    settings = dict(learning_rate=3e-4 if arm == "velocity" else 3e-5,
                    epochs=10, minibatch_size=32, target_kl=.02,
                    kl_check_cadence="minibatch")
    if optimizer_config is not None:
        if set(optimizer_config) != set(settings):
            raise ValueError("Bind the complete qualified steering optimizer configuration")
        settings.update(optimizer_config)
    config = ResidualPPOConfig(feature_dim=2064, residual_dim=spec_arguments(arm)["steering_dim"],
        gamma=.99, gae_lambda=.95, **settings)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        policy = ResidualActorCritic(config).to(device)
    return policy


@torch.no_grad()
def normalize_first_bank(policy, records):
    """Fit public-feature statistics before update 1 without changing behavior.

    Zero actor output makes the steering distribution independent of these
    statistics. Recompute the value baseline with the fitted statistics, and
    preserve original rollout values separately; no success labels fit them.
    """
    weight = next(policy.parameters())
    features = torch.as_tensor(np.asarray([row["features"] for row in records]),
                               device=weight.device, dtype=weight.dtype)
    controls = torch.as_tensor(np.asarray([row["control"] for row in records]),
                               device=weight.device, dtype=weight.dtype)
    if any(torch.count_nonzero(value) for value in policy.actor[-1].parameters()):
        raise ValueError("Feature normalization must be fitted before actor learning")
    before, _ = policy._distribution(features)
    original_mean, original_std = before.mean.clone(), before.stddev.clone()
    policy.fit_feature_normalization(features)
    after, _ = policy._distribution(features)
    if not torch.equal(after.mean, original_mean) or not torch.equal(after.stddev, original_std):
        raise ValueError("Fitting feature statistics changed the behavior distribution")
    log_probs, _, values = policy.evaluate(features, controls)
    old_log_probs = torch.tensor([row["log_prob"] for row in records], device=weight.device)
    if not torch.isfinite(log_probs).all() or not torch.isfinite(old_log_probs).all():
        raise ValueError("Nonfinite initial Gaussian likelihood")
    # Preserve the actual behavior log-probabilities. GPU row reductions can
    # differ slightly between worker batches1..4 and the master bank batch.
    return [dict(row, pre_normalization_value=row["value"], value=float(value),
                 normalization_log_prob_absolute_delta=float(delta))
            for row, value, delta in zip(records, values, (log_probs - old_log_probs).abs())]


@torch.no_grad()
def native_scale_calibration(motor, observations, spec):
    """Choose scale from public TRAIN observations and action sensitivity only.

    No reward, success label, simulator step or optimizer update is involved.
    Candidate fractions and reference action-residual RMS are declared before
    the first policy update. The original native inference is also compared
    against zero steering on the same actual GPU and explicit Gaussian.
    """
    from dataclasses import replace
    from .robocasa_residual import public_observation
    if np.shape(observations["states"]) != (1, 25):
        raise ValueError("Calibrate one actual public TRAIN reset at a time")
    fractions = SCALE_FRACTIONS[spec.kind]
    generator = torch.Generator(device=motor.model.device).manual_seed(921002)
    noise = torch.randn((1, 50, 32), generator=generator, device=motor.model.device)
    original, _ = motor.model.predict_action_batch(public_observation(observations), mode="eval", noise=noise)
    original = original.detach().cpu().numpy() if torch.is_tensor(original) else np.asarray(original)
    _, prepared = motor.prepare_batch(observations)
    zero = motor.execute_batch(prepared, noise=noise,
        steering=np.zeros((1, spec.steering_dim), np.float32), spec=spec)
    if not np.array_equal(original, zero):
        raise ValueError("Zero steering differs from original native inference")
    code = np.random.default_rng(921001).normal(0, .5, (1, spec.steering_dim)).astype(np.float32)
    target = float(np.sqrt(np.mean((.25 * np.tanh(code[0, :6])) ** 2) * .5))
    action_std = np.asarray(motor.native.stats["actions"]["std"], np.float32)[:12]
    rows = []
    for fraction in fractions:
        _, prepared = motor.prepare_batch(observations)
        altered = motor.execute_batch(prepared, noise=noise, steering=code,
                                      spec=replace(spec, fraction=fraction))
        effect = float(np.sqrt(np.mean(((altered - zero) / (action_std + 1e-6)) ** 2)))
        if not np.isfinite(effect):
            raise FloatingPointError("Nonfinite native steering action sensitivity")
        rows.append(dict(fraction=fraction, normalized_action_rms=effect))
    if not any(row["normalized_action_rms"] > 0 for row in rows):
        raise ValueError("Steering has no measured effect on native actions")
    return dict(completed=True, zero_native_bitwise_equal=True, target_normalized_action_rms=target,
        candidates=rows, selection="Task-balanced public TRAIN action-RMS aggregation; bracket reference before selection",
        extra_policy_controls=0, reward_or_success_used=False)


def qualify_learning_rates(arm, features, interface_spec, *, device):
    """Bounded head-only numerical qualification, with no scientific labels.

    Reuse public reset features; synthetic ends/rewards define a numerical
    stress batch only. Compare correctly summed joint KL and next-Adam recovery
    for every preregistered candidate. No live production learner is mutated.
    """
    from .residual_ppo import update
    rates = (3e-4, 1e-4, 3e-5) if arm == "velocity" else (3e-5, 1e-5, 3e-6)
    public = torch.as_tensor(features, dtype=torch.float32, device=device).reshape(-1, 2064)
    if len(public) < 1 or not torch.isfinite(public).all():
        raise ValueError("Require finite public TRAIN features for numerical qualification")
    public = public.repeat(((24 + len(public) - 1) // len(public), 1))[:24]
    trials = []
    for rate in rates:
        settings = dict(learning_rate=rate, epochs=10, minibatch_size=32,
                        target_kl=.02, kl_check_cadence="minibatch")
        policy = make_policy(arm, device=device, seed=921071, optimizer_config=settings)
        policy.fit_feature_normalization(public)
        adam = torch.optim.Adam(policy.parameters(), lr=rate)
        generator = torch.Generator().manual_seed(921072)
        noise = torch.randn((24, policy.config.residual_dim),
                            generator=torch.Generator().manual_seed(921073)).to(device)
        controls, logs, values = policy.act(public, action_noise=noise)
        with torch.no_grad():
            behavior, _ = policy._distribution(public)
            means, stds = behavior.mean.clone(), behavior.stddev.clone()
        ends = torch.tensor([False] * 5 + [True], device=device)[:, None].repeat(4, 1)
        rollout = dict(features=public[:, None], controls=controls[:, None], log_probs=logs[:, None],
            values=values[:, None], ends=ends, rewards=ends.float(),
            behavior_mean=means[:, None], behavior_std=stds[:, None])
        metrics = update(policy, adam, rollout, generator=generator)
        recovery = qualify_optimizer_restore(policy, adam, generator, public,
            arm=arm, interface_spec=interface_spec)
        peak = max([metrics["final_metrics"]["joint_gaussian_kl"],
                    recovery["final_metrics"]["joint_gaussian_kl"],
                    *[row["kl"] for row in metrics["minibatch_kl_checks"]]])
        passed = metrics["optimizer_steps"] >= 1 and math.isfinite(peak) and peak <= .03
        trials.append(dict(optimizer_config=settings, passed=passed,
            maximum_observed_joint_kl=peak, metrics=metrics, recovery=recovery))
    passing = [row for row in trials if row["passed"]]
    if not passing:
        return dict(completed=True, passed=False, trials=trials,
                    physical_controls=0, scientific_updates=0, reward_or_success_used=False)
    return dict(completed=True, passed=True, selected_optimizer_config=passing[0]["optimizer_config"],
        trials=trials, physical_controls=0, scientific_updates=0, reward_or_success_used=False,
        selection="Largest preregistered LR with peak jointKL<=.03 and exact next-Adam restore; target stoppingKL=.02")


def select_scale(receipts, *, selected_fraction=None):
    """Select from a bracketing public-observation grid, without rewards.

    Qualification supplies one receipt per TRAIN task, with equal task weight.
    On production admission, a supplied fraction is the already qualified
    binding; device-local fresh reset diagnostics do not tune it again.
    """
    if not receipts:
        raise ValueError("Require public TRAIN calibration receipts")
    receipts = [row.get("calibration", row) for row in receipts]
    fractions = [row["fraction"] for row in receipts[0]["candidates"]]
    if not fractions or any([row["fraction"] for row in receipt["candidates"]] != fractions
                            for receipt in receipts):
        raise ValueError("Every public reset must qualify the same declared fractions")
    if not all(receipt["completed"] and receipt["zero_native_bitwise_equal"]
               and not receipt["reward_or_success_used"] for receipt in receipts):
        raise ValueError("Require original-native parity and reward-free calibration")
    target = float(np.mean([receipt["target_normalized_action_rms"] for receipt in receipts]))
    candidates = [dict(fraction=fraction, normalized_action_rms=float(np.mean([
        receipt["candidates"][index]["normalized_action_rms"] for receipt in receipts])))
        for index, fraction in enumerate(fractions)]
    effects = [row["normalized_action_rms"] for row in candidates]
    if not math.isfinite(target) or target <= 0 or not all(math.isfinite(x) and x >= 0 for x in effects):
        raise ValueError("Require finite positive action-RMS reference and finite effects")
    bracketed = min(effects) <= target <= max(effects)
    if selected_fraction is None:
        if not bracketed:
            raise ValueError("Public TRAIN scale grid does not bracket the action-RMS reference")
        chosen = min(candidates, key=lambda row: abs(row["normalized_action_rms"] - target))
    else:
        matches = [row for row in candidates if row["fraction"] == selected_fraction]
        if len(matches) != 1:
            raise ValueError("Qualified fraction is absent from the declared admission grid")
        chosen = matches[0]
    if chosen["normalized_action_rms"] <= 0:
        raise ValueError("Qualified map has no finite measured action influence")
    return dict(completed=True, selected_fraction=chosen["fraction"], candidates=candidates,
        target_normalized_action_rms=target, selected_normalized_action_rms=chosen["normalized_action_rms"],
        relative_RMS_error=abs(chosen["normalized_action_rms"] / target - 1),
        reference_bracketed=bracketed, receipts=receipts, reward_or_success_used=False,
        selection="Nearest equal-task mean public action RMS" if selected_fraction is None else
                  "Previously qualified fraction retained; fresh device diagnostics only")


def boundary_state(policy, optimizer, generator, *, arm, iteration, sequence, worker_states, escrow,
                   interface_spec=None):
    from .interface_ppo import InterfaceSpec
    specification = interface_spec or spec_arguments(arm)
    return cpu_tree(dict(format="joint_frozen_steering_boundary_v1", arm=arm,
        spec=specification, interface=InterfaceSpec(**specification).record(),
        policy_config=asdict(policy.config), policy=policy.state_dict(),
        optimizer=optimizer.state_dict(), generator=generator.get_state(), global_rng=snapshot_rng(),
        iteration=iteration, sequence=sequence, worker_states=worker_states, escrow=escrow))


def restore_boundary(state, policy, optimizer, generator, *, arm, interface_spec=None):
    if (state["format"] != "joint_frozen_steering_boundary_v1" or state["arm"] != arm
            or state["spec"] != (interface_spec or spec_arguments(arm))
            or state["policy_config"] != asdict(policy.config)):
        raise ValueError("Cannot change the steering/optimizer contract on recovery")
    policy.load_state_dict(state["policy"], strict=True)
    # Optimizer.load_state_dict may retain CPU moment storage by reference.
    # Loading one boundary twice must not mutate the reusable checkpoint.
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    generator.set_state(state["generator"])
    from .rlinf_native_ppo import restore_rng
    restore_rng(state["global_rng"])
    return state["iteration"], state["sequence"], state["worker_states"]


def qualify_optimizer_restore(policy, optimizer, generator, features, *, arm, interface_spec):
    """Compare two next Adam updates on the actual device without advancing PPO.

    The shadow batch uses retained public TRAIN features with synthetic actor
    draws/rewards. It never enters the scientific history or controls an env.
    The live actor, optimizer, sampler and global RNG remain unchanged.
    """
    from .rlinf_native_ppo import restore_rng, tensor_hash
    from .residual_ppo import update
    saved = boundary_state(policy, optimizer, generator, arm=arm, iteration=1, sequence=0,
        worker_states={}, escrow={}, interface_spec=interface_spec)
    weight = next(policy.parameters())
    observations = torch.as_tensor(features, device=weight.device, dtype=weight.dtype)[:24]
    if len(observations) != 24:
        raise ValueError("Recovery qualification requires24 retained public TRAIN feature rows")
    try:
        states, diagnostics, rollout = [], [], None
        for _ in range(2):
            shadow = ResidualActorCritic(policy.config).to(weight.device)
            adam = torch.optim.Adam(shadow.parameters(), lr=policy.config.learning_rate)
            sampler = torch.Generator()
            restore_boundary(saved, shadow, adam, sampler, arm=arm, interface_spec=interface_spec)
            if rollout is None:
                noise = torch.randn((24, policy.config.residual_dim),
                    generator=torch.Generator().manual_seed(921019)).to(weight.device)
                controls, logs, values = shadow.act(observations, action_noise=noise)
                behavior, _ = shadow._distribution(observations)
                ends = torch.tensor([False] * 5 + [True], device=weight.device)[:, None].repeat(4, 1)
                rollout = dict(features=observations[:, None], controls=controls[:, None],
                    log_probs=logs[:, None], values=values[:, None], ends=ends,
                    rewards=ends.to(weight.dtype), behavior_mean=behavior.mean.detach()[:, None],
                    behavior_std=behavior.stddev.detach()[:, None])
            diagnostic = update(shadow, adam, rollout, generator=sampler)
            if diagnostic["optimizer_steps"] < 1:
                raise ValueError("Shadow recovery check did not exercise an Adam step")
            diagnostics.append(diagnostic["final_metrics"])
            states.append(tensor_hash(dict(policy=shadow.state_dict(), optimizer=adam.state_dict(),
                                           generator=sampler.get_state())))
        if states[0] != states[1] or diagnostics[0] != diagnostics[1]:
            raise ValueError("Restored next Adam update differs from its same-device reference")
        live = tensor_hash(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                                generator=generator.get_state()))
        before = tensor_hash({key: saved[key] for key in ("policy", "optimizer", "generator")})
        if live != before:
            raise ValueError("Shadow qualification modified the scientific learner")
        return dict(completed=True, device=str(weight.device), feature_dim=policy.config.feature_dim,
            steering_dim=policy.config.residual_dim, next_update_exact=True,
            live_learner_unchanged=True, optimizer_steps_each=diagnostic["optimizer_steps"],
            final_metrics=diagnostics[0], state_sha256=states[0], physical_controls=0,
            scope="Two discarded numerical shadow updates on synthetic rewards, outside scientific PPO")
    finally:
        restore_rng(saved["global_rng"])
