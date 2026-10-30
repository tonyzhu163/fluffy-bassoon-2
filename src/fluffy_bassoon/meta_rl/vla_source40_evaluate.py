"""Frozen source40 controllers: actual support episodes, then a separate query.

One invocation evaluates one explicit target case over independent shot budgets.
Simulator RPC is shared with standard LIBERO90 and the Plus/Pro workers. This
module never constructs a learner or adds target experience to source training.
"""
import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_history import FrozenHistoryController, file_sha256, load_policy
from .vla_libero import FrozenOpenPI, SimulatorClient, expand_noise, write_json
from .vla_source40_corpus import PI05_RECEIPT, save_motor_rng
from .vla_source40_schedule import content_hash
from .vla_source40_training import NOISE_CONTRACT, load_assembly


SUPPORT_BUDGETS = (0, 1, 2, 4, 8)
METHODS = ("native", "tied_gaussian", "ad_full", "ad_episode_reset", "ad_static",
           "dpt_full", "dpt_episode_reset", "dpt_static")
SUPPORTED_METHODS = METHODS + ('frozen_source_sac', 'frozen_target_sac', 'zero', 'tied_gaussian_clipped',
    'ppo_full', 'ppo_episode_reset', 'ppo_trained_episode',
    'ad_zero_mean_same_std', 'dpt_zero_mean_same_std', 'ad_original_mean_std1', 'dpt_original_mean_std1')
FORMAT = "openpi_pi05_source40_support_query_v1"


@dataclass(frozen=True)
class TargetCase:
    benchmark: str
    case_id: str
    suite: str
    task_id: int
    initial_state: int
    scene_seed: int

    def validate(self, metadata):
        if (self.benchmark not in ("libero90", "plus", "pro") or not self.case_id
                or type(self.initial_state) is not int or not 0 <= self.initial_state < metadata["initial_states"]
                or type(self.scene_seed) is not int or not 0 <= self.scene_seed < 2**31
                or self.suite != metadata["suite"] or self.task_id != metadata["task_id"]):
            raise ValueError("Explicit target case differs from simulator metadata")
        if self.benchmark == "libero90":
            if self.suite != "libero_90" or not 0 <= self.task_id < 90 or metadata.get("benchmark_kind") in ("plus", "pro"):
                raise ValueError("LIBERO90 evaluation accepts every task0..89 without a semantic exclusion gate")
        elif metadata.get("benchmark_kind") != self.benchmark:
            raise ValueError("Target benchmark and simulator implementation differ")
        if not (metadata.get("fresh_environment_on_reset") is True
                or (self.benchmark == "pro" and metadata.get("reset", "").startswith("Fresh environment per episode;"))):
            raise ValueError("Target evaluation requires a fresh physical environment at every episode reset")
        if (type(metadata.get("horizon")) is not int or metadata["horizon"] < 1
                or type(metadata.get("settle_steps")) is not int or metadata["settle_steps"] < 0):
            raise ValueError("Simulator must declare its episode horizon and settling cost")


def _seed(case, role, ordinal):
    # Query randomness is paired across support budgets and methods. It does
    # not depend on how many stochastic support actions were consumed.
    return int(content_hash(dict(case=asdict(case), role=role, ordinal=ordinal))[:8], 16)


def _policy_signature(policy):
    if hasattr(policy, 'frozen_signature'):
        return policy.frozen_signature()
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _motor_signature(motor):
    if hasattr(motor, "frozen_signature"):
        return motor.frozen_signature()
    # The pinned OpenPI module_jit captures frozen immutable model state.
    # Track those callables; there is no optimizer or mutable training path.
    return (id(motor.policy._sample_actions), id(motor.feature_encoder.hidden_apply),
            motor.metadata["receipt_sha256"])


def load_candidates(index_path, expected_sha256, assembly_path, motor, *, seed, device, allow_small_pilot=False):
    if file_sha256(index_path) != expected_sha256:
        raise ValueError("Student matrix index checksum differs")
    index = json.loads(Path(index_path).read_text())
    assembly = load_assembly(assembly_path, allow_small_pilot=allow_small_pilot)
    if (index.get("completed") is not True
            or index.get("contract", {}).get("assembly_sha256") != assembly["assembly_sha256"]
            or motor.metadata.get("receipt_sha256") != PI05_RECEIPT
            or motor.feature_encoder.contract != assembly["feature_contract"]):
        raise ValueError("Require a completed matched matrix and the same frozen OpenPI40 motor/features")
    selected = [row for row in index["controllers"] if row["seed"] == seed]
    if {(row["objective"], row["mode"]) for row in selected} != {(o, m) for o in ("ad", "dpt") for m in ("full", "static")} or len(selected) != 4:
        raise ValueError("Require exactly four matched AD/DPT full/static final controllers for this seed")
    candidates, signatures = {}, set()
    for row in selected:
        policy, manifest = load_policy(row["final_path"], expected_sha256=row["final_sha256"], device=device)
        contract = manifest["contract"]
        metadata = contract["train_metadata"]
        if (manifest.get("selection") != "fixed requested final update" or manifest.get("update") != row["updates"]
                or manifest.get("assembly_sha256") != assembly["assembly_sha256"]
                or manifest.get("full_recovery_sha256") != row["full_recovery_sha256"]
                or contract.get("objective") != row["objective"] or contract.get("mode") != row["mode"]
                or contract.get("seed") != seed or asdict(policy.config) != contract["config"]
                or (policy.config.observation_dim, policy.config.noise_dim) != (4104, 32)
                or metadata.get("motor_sha256") != PI05_RECEIPT or metadata.get("noise_contract") != NOISE_CONTRACT
                or metadata.get("feature_contract") != motor.feature_encoder.contract
                or metadata.get("global_task_split") != "source40"
                or metadata.get("source_task_keys") != assembly["source_task_keys"]
                or bool(metadata.get("small_pilot")) != assembly["small_pilot"]):
            raise ValueError("Final controller contract differs from source40 assembly/motor/memory mode")
        for partition in ("train", "validation"):
            if contract.get(partition + "_sha256") != assembly["datasets"][partition][row["objective"]]["sha256"]:
                raise ValueError("Controller train/validation data differ from the declared assembly")
        signatures.add(content_hash(dict(config=contract["config"], seed=seed, updates=row["updates"],
            batch_size=contract["batch_size"], learning_rate=contract["learning_rate"], max_context=contract["max_context"])))
        spec = dict(policy=policy, mode=row["mode"], path=row["final_path"], sha256=row["final_sha256"])
        candidates[f"{row['objective']}_{row['mode']}"] = spec
        if row["mode"] == "full":
            candidates[f"{row['objective']}_episode_reset"] = dict(spec, mode="episode")
    if len(signatures) != 1:
        raise ValueError("Controller architectures or optimization budgets are not matched")
    costs = assembly["collection_costs_count_once"]
    if any(type(costs[partition].get(key)) is not int or costs[partition][key] < 1
           for partition in ("train", "validation") for key in ("actual_primitive_steps", "source_updates")):
        raise ValueError("Require explicit nonzero source collection/update costs")
    provenance = dict(assembly_sha256=assembly["assembly_sha256"], small_pilot=assembly["small_pilot"],
        source_collection_costs_count_once=costs, source_primitive_steps_total=sum(row["actual_primitive_steps"] for row in costs.values()),
        source_settling_steps_total=sum(row["actual_settling_steps"] for row in costs.values()),
        student_index_sha256=expected_sha256, student_seed=seed,
        all_matrix_optimizer_updates=sum(row["updates"] for row in index["controllers"]),
        all_matrix_training_seconds=sum(row["elapsed_seconds"] for row in index["controllers"]),
        ordinary_dsrl_baseline="pending separate source40 baseline wiring; native/tied Gaussian are not DSRL")
    return candidates, provenance


class CostLedger:
    """Journal RPC intents before execution; unresolved intents count at ceiling."""
    def __init__(self, output, *, max_primitive_steps, max_settling_steps, max_total_actions=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_primitive_steps, self.max_settling_steps = max_primitive_steps, max_settling_steps
        self.max_total_actions = max_total_actions
        self.rows = {path.name: json.loads(path.read_text()) for path in self.output.glob("*.json")}

    def costs(self):
        totals = {role: dict(primitive_steps=0, uncertain_primitive_steps_ceiling=0,
            settling_steps=0, uncertain_settling_steps_ceiling=0, reset_attempts=0) for role in ("support", "query")}
        for row in self.rows.values():
            value = totals[row["role"]]
            if row["kind"] == "reset":
                value["reset_attempts"] += 1
                key = "settling_steps" if row["returned"] else "uncertain_settling_steps_ceiling"
            else:
                key = "primitive_steps" if row["returned"] else "uncertain_primitive_steps_ceiling"
            value[key] += row["actual_steps"] if row["returned"] else row["maximum_steps"]
        return totals

    def intent(self, *, kind, role, episode_id, maximum_steps):
        costs = self.costs()
        key = "primitive" if kind == "step" else "settling"
        charged = sum(value[key + "_steps"] + value["uncertain_" + key + "_steps_ceiling"] for value in costs.values())
        ceiling = self.max_primitive_steps if kind == "step" else self.max_settling_steps
        if charged + maximum_steps > ceiling:
            raise RuntimeError(f"Target {key} interaction ceiling would be exceeded")
        joint = sum(value[name] for value in costs.values() for name in
                    ('primitive_steps', 'uncertain_primitive_steps_ceiling',
                     'settling_steps', 'uncertain_settling_steps_ceiling'))
        if self.max_total_actions is not None and joint + maximum_steps > self.max_total_actions:
            raise RuntimeError('Target joint physical-action ceiling would be exceeded')
        name = f"{time.time_ns()}-{kind}.json"
        row = dict(kind=kind, role=role, episode_id=episode_id, returned=False,
                   maximum_steps=maximum_steps, actual_steps=None)
        write_json(self.output / name, row)
        self.rows[name] = row
        return name

    def returned(self, event, actual_steps):
        row = self.rows[event]
        if type(actual_steps) is not int or not 0 <= actual_steps <= row["maximum_steps"]:
            raise ValueError("RPC actual cost exceeds its declared execution intent")
        row = dict(row, returned=True, actual_steps=actual_steps)
        write_json(self.output / event, row)
        self.rows[event] = row


def _controller(spec):
    return FrozenHistoryController(spec["policy"], mode=spec["mode"])


def _episode_rng(controller, seed):
    if hasattr(controller, "set_episode_seed"):
        controller.set_episode_seed(seed)
    else:
        import torch
        device = next(controller.policy.parameters()).device
        controller.generator = torch.Generator(device=device).manual_seed(seed)


def _snapshot(controller):
    if controller is None:
        return None
    if hasattr(controller, "snapshot"):
        return controller.snapshot()
    import torch
    def cpu(value):
        return {key: cpu(item) for key, item in value.items()} if isinstance(value, dict) else value.detach().cpu().clone() if torch.is_tensor(value) else value
    return dict(state=cpu(controller.state), previous_noise=controller.previous_noise.copy(),
        previous_reward=controller.previous_reward, previous_done=controller.previous_done,
        episode_start=controller.episode_start, pending=controller.pending,
        generator=controller.generator.get_state() if controller.generator is not None else None)


def _restore(controller, saved):
    if controller is None or saved is None:
        return
    if hasattr(controller, "restore"):
        controller.restore(saved)
        return
    import torch
    device = next(controller.policy.parameters()).device
    def target(value):
        return {key: target(item) for key, item in value.items()} if isinstance(value, dict) else value.to(device) if torch.is_tensor(value) else value
    controller.state = target(saved["state"])
    for key in ("previous_noise", "previous_reward", "previous_done", "episode_start", "pending"):
        setattr(controller, key, saved[key])
    if controller.pending:
        raise ValueError("Evaluation recovery must be at a completed episode boundary")
    if saved["generator"] is not None:
        controller.generator = torch.Generator(device=device)
        controller.generator.set_state(saved["generator"])


def _save_recovery(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=5)
    temporary.replace(path)


def rollout_episode(sim, motor, controller, *, case, method, role, ordinal, episode_id,
                    ledger, reset_motor_rng, qualify=None, deadline=float("inf")):
    started = time.monotonic()
    if started >= deadline:
        raise RuntimeError("Target evaluation elapsed-time ceiling reached")
    reset_event = ledger.intent(kind="reset", role=role, episode_id=episode_id, maximum_steps=sim.metadata["settle_steps"])
    initial = sim.reset(case.initial_state, case.scene_seed)
    ledger.returned(reset_event, sim.metadata["settle_steps"])
    if not initial.get("initial_observation_sha256"):
        raise ValueError("Real reset must expose the actual initial-observation identity")
    if not isinstance(initial.get('initial_state_vector_sha256'), str) or not initial['initial_state_vector_sha256']:
        raise ValueError('Real reset must expose the actual nominal physical-state identity')
    seed = _seed(case, role, ordinal)
    reset_motor_rng(seed)
    if controller is not None:
        _episode_rng(controller, seed)
    if qualify is not None:
        qualify(initial["observation"])
        reset_motor_rng(seed)
    rng = np.random.default_rng(seed)
    observation, rows, primitives = initial["observation"], [], 0
    while True:
        if time.monotonic() >= deadline:
            raise RuntimeError("Target evaluation elapsed-time ceiling reached")
        feature = None
        if controller is not None:
            if getattr(controller, 'public_observation_input', False):
                proposed = np.asarray(controller.act(observation), np.float32)
            else:
                feature = motor.feature_encoder(observation, sim.metadata["instruction"])
                proposed = np.asarray(controller.act(feature), np.float32)
            if proposed.shape != (32,) or not np.isfinite(proposed).all():
                raise ValueError("Controller must return a finite32D steering coordinate")
            executed = np.clip(proposed, -1., 1.).astype(np.float32)
            noise = expand_noise(executed, motor.horizon)
        elif method == 'zero':
            proposed = executed = np.zeros(32, np.float32)
            noise = expand_noise(executed, motor.horizon)
        elif method == "native":
            proposed = executed = None
            noise = rng.standard_normal((motor.horizon, 32)).astype(np.float32)
        else:
            proposed = rng.standard_normal(32).astype(np.float32)
            executed = np.clip(proposed, -1., 1.).astype(np.float32) if method == 'tied_gaussian_clipped' else proposed
            noise = expand_noise(executed, motor.horizon)
        actions = np.asarray(motor.infer(observation, sim.metadata["instruction"], noise), np.float32)
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError("Require native H10 finite decoded motor actions")
        actions = actions[:min(10, sim.metadata["horizon"] - primitives)]
        if not len(actions):
            raise RuntimeError("Simulator did not finish at its declared horizon")
        event = ledger.intent(kind="step", role=role, episode_id=episode_id, maximum_steps=len(actions))
        outcome = sim.step(actions)
        steps = int(outcome["primitive_steps"])
        ledger.returned(event, steps)
        if (steps < 1 or np.asarray(outcome["executed_actions"]).shape != (steps, 7)
                or not np.isfinite(np.asarray(outcome["executed_actions"])).all()
                or len(outcome["rewards"]) != steps or not np.isfinite(outcome["rewards"]).all()
                or type(outcome["done"]) not in (bool, np.bool_) or type(outcome["success"]) not in (bool, np.bool_)):
            raise ValueError("Simulator omitted actual decoded-action/reward/boundary receipts")
        reward = 0. if outcome["success"] else -1.
        # The current transition's reward is visible only after actual execution.
        if controller is not None:
            controller.observe(executed_noise=executed, reward=reward, done=bool(outcome["done"]))
        rows.append(dict(motor_observation=observation, motor_next_observation=outcome['observation'],
            public_features=feature, proposed_noise=proposed, executed_noise=executed,
            native_initial_noise=noise, executed_actions=np.asarray(outcome["executed_actions"], np.float32),
            reward=reward, raw_rewards=list(outcome["rewards"]), done=bool(outcome["done"]),
            success=bool(outcome["success"]), primitive_steps=steps))
        distribution = getattr(controller, 'last_distribution', None) if controller is not None else None
        if distribution is not None:
            rows[-1]['steering_distribution'] = distribution
        primitives += steps
        observation = outcome["observation"]
        if outcome["done"]:
            break
    receipt = dict(episode_id=episode_id, role=role, role_episode_ordinal=ordinal,
        initial_state=case.initial_state, scene_seed=case.scene_seed, episode_random_seed=seed,
        initial_observation_sha256=initial["initial_observation_sha256"],
        initial_state_vector_sha256=initial.get("initial_state_vector_sha256"),
        success=bool(outcome["success"]), primitive_steps=primitives, settling_steps=sim.metadata["settle_steps"],
        chunks=len(rows), raw_return=sum(sum(row["raw_rewards"]) for row in rows),
        elapsed_seconds=time.monotonic() - started, completed=True)
    return rows, receipt, rng.bit_generator.state


def evaluate_case(sim, motor, candidates, *, case, output, provenance, max_primitive_steps,
                  max_settling_steps, support_budgets=SUPPORT_BUDGETS, methods=METHODS,
                  controller_factory=_controller, reset_motor_rng=lambda seed: None,
                  motor_rng_state=lambda: None, qualify=None, resume=False, max_elapsed_seconds=3600.,
                  external_deadline=float('inf'), method_support_budgets=None):
    case.validate(sim.metadata)
    if (motor.horizon != 10 or len(set(methods)) != len(methods) or not methods or not set(methods).issubset(SUPPORTED_METHODS)
            or not support_budgets or len(set(support_budgets)) != len(support_budgets)
            or any(type(value) is not int or value not in SUPPORT_BUDGETS for value in support_budgets)
            or max_primitive_steps < 1 or max_settling_steps < 0
            or not np.isfinite(max_elapsed_seconds) or max_elapsed_seconds <= 0):
        raise ValueError("Invalid native horizon, method, support grid or physical budget")
    if any(method not in candidates for method in methods if method not in ("native", "tied_gaussian", 'zero', 'tied_gaussian_clipped')):
        raise ValueError("Every learned method needs a declared fixed controller")
    if method_support_budgets is not None and (set(method_support_budgets) != set(methods)
            or any(not values or len(set(values)) != len(values)
                or any(type(value) is not int or value not in SUPPORT_BUDGETS for value in values)
                for values in method_support_budgets.values())):
        raise ValueError('Explicit per-method support budgets must cover exactly the declared methods')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    traces = output / "traces"
    traces.mkdir(exist_ok=True)
    ledger = CostLedger(output / "physical-journal", max_primitive_steps=max_primitive_steps,
        max_settling_steps=max_settling_steps, max_total_actions=provenance.get('group_maximum_total_actions'))
    simulator_contract = {key: value for key, value in sim.metadata.items() if key not in ("pid", "hostname")}
    contract = dict(format=FORMAT, case=asdict(case), simulator=simulator_contract, support_budgets=list(support_budgets), methods=list(methods),
        max_primitive_steps=max_primitive_steps, max_settling_steps=max_settling_steps,
        max_elapsed_seconds_per_invocation=max_elapsed_seconds,
        motor_receipt_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT, provenance=provenance,
        candidates={key: {name: value for name, value in spec.items() if name != "policy"} for key, spec in candidates.items()},
        one_shot="One actual completed support episode followed by a separate reset/query episode; no gradient updates",
        query_randomness="Paired by case and role; independent of support draw count",
        motor_parameters="Pinned OpenPI module_jit captures immutable parameters; no optimizer updates during deployment",
        suite_scope="One declared case; no semantic filtering. Prior development declarations remain development results.")
    if method_support_budgets is not None:
        contract['method_support_budgets'] = {key: list(value) for key, value in method_support_budgets.items()}
    recovery_path = output / "evaluation-recovery.pkl"
    if resume:
        if json.loads((output / "contract.json").read_text()) != contract:
            raise ValueError("Evaluation resume contract differs")
        with recovery_path.open("rb") as stream:
            recovery = pickle.load(stream)
        if recovery["contract_sha256"] != content_hash(contract):
            raise ValueError("Evaluation recovery contract differs")
    else:
        if (output / "contract.json").exists():
            raise ValueError("Fresh evaluation requires a new output identity")
        write_json(output / "contract.json", contract)
        recovery = dict(contract_sha256=content_hash(contract), trial_index=0, next_episode=0, controller=None, records=[])
        _save_recovery(recovery_path, recovery)
    signatures = {key: _policy_signature(spec["policy"]) for key, spec in candidates.items()}
    motor_signature = _motor_signature(motor)
    trials = [(method, budget) for method in methods for budget in
        (support_budgets if method_support_budgets is None else method_support_budgets[method])]
    records = recovery["records"]
    started = time.monotonic()
    deadline = min(started + max_elapsed_seconds, external_deadline)
    qualification_pending = qualify is not None and not (output / "qualification.json").exists()
    for trial_index, (method, budget) in enumerate(trials):
        if trial_index < recovery["trial_index"]:
            continue
        controller = controller_factory(candidates[method]) if method in candidates else None
        start_episode = recovery["next_episode"] if trial_index == recovery["trial_index"] else 0
        if start_episode:
            _restore(controller, recovery["controller"])
        for position in range(start_episode, budget + 1):
            role, ordinal = ("support", position) if position < budget else ("query", 0)
            episode_id = f"trial-{trial_index:03d}-{method}-support-{budget}/{role}-{ordinal:03d}"
            hook = qualify if qualification_pending else None
            rows, receipt, sampler = rollout_episode(sim, motor, controller, case=case, method=method,
                role=role, ordinal=ordinal, episode_id=episode_id, ledger=ledger, reset_motor_rng=reset_motor_rng, qualify=hook, deadline=deadline)
            if records and receipt['initial_state_vector_sha256'] != records[0]['initial_state_vector_sha256']:
                raise RuntimeError('The declared support/query case changed its nominal physical state')
            if hook:
                qualification_pending = False
            trace = traces / (episode_id.replace("/", "-") + f"-{time.time_ns()}.pkl.gz")
            with gzip.open(trace, "xb", compresslevel=1) as stream:
                pickle.dump(rows, stream, protocol=5)
            receipt.update(method=method, support_budget=budget, trial_index=trial_index,
                           trace=str(trace.relative_to(output)), trace_sha256=file_sha256(trace))
            if _motor_signature(motor) != motor_signature or any(_policy_signature(spec["policy"]) != signatures[key] for key, spec in candidates.items()):
                raise RuntimeError("Frozen motor/controller parameters changed during evaluation")
            records.append(receipt)
            complete = position == budget
            recovery = dict(contract_sha256=content_hash(contract), trial_index=trial_index + int(complete),
                next_episode=0 if complete else position + 1, controller=None if complete else _snapshot(controller),
                motor_rng=motor_rng_state(), sampler=sampler, records=records,
                boundary="complete physical episode; partial later interactions remain separately charged in journal")
            _save_recovery(recovery_path, recovery)
            write_json(output / "progress.json", dict(completed_episodes=len(records), trial_index=recovery["trial_index"],
                next_episode=recovery["next_episode"], physical_costs=ledger.costs(), latest_recovery_sha256=file_sha256(recovery_path)))
    summary = []
    for trial_index, (method, budget) in enumerate(trials):
        episodes = [row for row in records if row["trial_index"] == trial_index]
        support, query = [row for row in episodes if row["role"] == "support"], [row for row in episodes if row["role"] == "query"]
        if len(support) != budget or len(query) != 1:
            raise RuntimeError("Support/query episode counts differ from the requested shot budget")
        summary.append(dict(method=method, support_budget=budget, query_success=query[0]["success"],
            support_episode_ids=[row["episode_id"] for row in support], query_episode_id=query[0]["episode_id"],
            support_primitive_steps=sum(row["primitive_steps"] for row in support), query_primitive_steps=query[0]["primitive_steps"],
            support_settling_steps=sum(row["settling_steps"] for row in support), query_settling_steps=query[0]["settling_steps"]))
    actual = ledger.costs()
    committed = {role: dict(primitive_steps=sum(row["primitive_steps"] for row in records if row["role"] == role),
        settling_steps=sum(row["settling_steps"] for row in records if row["role"] == role)) for role in ("support", "query")}
    interrupted = {role: dict(known_primitive_steps=actual[role]["primitive_steps"] - committed[role]["primitive_steps"],
        known_settling_steps=actual[role]["settling_steps"] - committed[role]["settling_steps"],
        uncertain_primitive_steps_ceiling=actual[role]["uncertain_primitive_steps_ceiling"],
        uncertain_settling_steps_ceiling=actual[role]["uncertain_settling_steps_ceiling"]) for role in ("support", "query")}
    result = dict(format=FORMAT, completed=True, case=asdict(case), trials=summary, episodes=records,
        physical_costs=actual, committed_episode_costs=committed, interrupted_physical_costs=interrupted,
        budget_matched_primary_eligible=not any(value for row in interrupted.values() for value in row.values()),
        full_support_grid=(set(support_budgets) == set(SUPPORT_BUDGETS) if method_support_budgets is None
            else all(set(value) == set(SUPPORT_BUDGETS) for value in method_support_budgets.values())),
        full_method_grid=set(methods) == set(METHODS),
        source_provenance=provenance, weights_unchanged=True, latest_recovery_sha256=file_sha256(recovery_path),
        interpretation="Single-case transfer only; query success is separate from support outcomes; no semantic-novelty assertion")
    write_json(output / "index.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "checkpoint", "openpi-root", "students-index", "assembly"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "expected-hostname", "students-index-sha256", "case-id", "suite"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--benchmark", choices=("libero90", "plus", "pro"), required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--initial-state", type=int, required=True)
    p.add_argument("--scene-seed", type=int, required=True)
    p.add_argument("--student-seed", type=int, required=True)
    p.add_argument("--max-primitive-steps", type=int, required=True)
    p.add_argument("--max-settling-steps", type=int, required=True)
    p.add_argument("--max-elapsed-seconds", type=float, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--history-device", default="cuda:0")
    p.add_argument("--support-budgets", type=int, nargs="+", default=SUPPORT_BUDGETS)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    p.add_argument("--allow-small-pilot", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        raise ValueError("Evaluate only on the explicitly identified allocated Linux host")
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed source commit")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    probe = json.loads(Path(os.environ["LYCO_BAIGE_GPU_MAP"]).read_text())
    if probe.get("status") != "passed" or len(probe.get("devices", [])) != 1:
        raise ValueError("Require a successful actual-allocation single-GPU probe")
    if not args.allow_small_pilot and (set(args.support_budgets) != set(SUPPORT_BUDGETS) or set(args.methods) != set(METHODS)):
        raise ValueError("A reduced technical grid requires --allow-small-pilot")
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("Resume requires the original evaluation directory")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    case = TargetCase(args.benchmark, args.case_id, args.suite, args.task_id, args.initial_state, args.scene_seed)
    sim = SimulatorClient(args.port)
    try:
        case.validate(sim.metadata)
        if file_sha256(args.checkpoint / "READY.json") != PI05_RECEIPT:
            raise ValueError("Require the original official OpenPI40 motor receipt before model loading")
        args.model_config, args.feature_interface = "pi05_libero", "vla-prefix"
        motor = FrozenOpenPI(args)
        candidates, provenance = load_candidates(args.students_index, args.students_index_sha256, args.assembly,
            motor, seed=args.student_seed, device=args.history_device, allow_small_pilot=args.allow_small_pilot)
        provenance["evaluation_source_commit"] = args.source_commit
        def reset_motor_rng(seed):
            import jax
            motor.policy._rng = jax.random.key(seed)
        def qualify(observation):
            noise = np.zeros((10, 32), np.float32)
            first, second = motor.infer(observation, sim.metadata["instruction"], noise), motor.infer(observation, sim.metadata["instruction"], noise.copy())
            delta = float(np.max(np.abs(first - second)))
            if not np.isfinite(delta) or delta > 1e-5:
                raise RuntimeError("Frozen motor same-noise parity changed")
            write_json(args.output / "qualification.json", dict(same_noise_maximum_delta=delta,
                features=motor.feature_encoder.qualify(observation, sim.metadata["instruction"])))
        evaluate_case(sim, motor, candidates, case=case, output=args.output, provenance=provenance,
            max_primitive_steps=args.max_primitive_steps, max_settling_steps=args.max_settling_steps,
            support_budgets=args.support_budgets, methods=args.methods, reset_motor_rng=reset_motor_rng,
            motor_rng_state=lambda: save_motor_rng(motor.policy._rng), qualify=qualify, resume=args.resume,
            max_elapsed_seconds=args.max_elapsed_seconds)
    except Exception as error:
        write_json(args.output / "failure.json", dict(error=repr(error),
            recovery="evaluation-recovery.pkl keeps the last complete episode; physical-journal charges partial/uncertain interactions"))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    main()
