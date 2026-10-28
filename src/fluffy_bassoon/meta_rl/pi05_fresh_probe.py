"""Three support attempts followed by independently branched attempt-four probes.

All branches retain K=4 clocks. A branch restores the complete learner boundary
state before each fresh probe, then permits ordinary recurrence inside it.
Probe replicas share a support history and are not independent statistical units.
This module owns no allocation, checkpoint loading or model-weight updates.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import math

import torch

from fluffy_bassoon.rl import maniskill_pi05 as ms
from fluffy_bassoon.rl.pi05_feature_bridge import clone_generator, public_chunk_feedback


ATTEMPTS, SUPPORT_ATTEMPTS, DENOISE_STEPS = 4, 3, 4
CHUNKS = ms.STEPS // ms.CHUNK
CONDITIONS = ("normal", "reset_memory", "opposite_history", "zero_bridge", "reward_blind")


def _copy_memory(memory, lanes, indices=None):
    if memory is None:
        return None
    if not isinstance(memory, dict) or not memory:
        raise ValueError("Require a complete dictionary of batched recurrent tensors")
    copied = {}
    for key, value in memory.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != lanes:
            raise ValueError("Every current CTM/Transformer-XL state tensor must have batch axis zero")
        copied[key] = (value.detach().clone() if indices is None else
                       value.detach().index_select(0, torch.tensor(indices, device=value.device)))
    return copied


@dataclass(frozen=True)
class ProbeBoundary:
    """Learner state just before an attempt reset; terminal feedback is still pending."""

    memory: dict | None
    previous_command: torch.Tensor
    pending_reward: torch.Tensor
    attempt_index: int
    history_ids: tuple[str, ...]
    attempts: int = ATTEMPTS
    chunk_index: int = 0
    attempt_end: bool = False
    noise_state: torch.Tensor | None = None

    def validate(self):
        lanes = len(self.history_ids)
        if (not lanes or len(set(self.history_ids)) != lanes
                or self.previous_command.shape != (lanes, 7)
                or not self.previous_command.is_floating_point()
                or not torch.isfinite(self.previous_command).all()
                or self.pending_reward.shape != (lanes,)
                or not ((self.pending_reward == 0) | (self.pending_reward == 1)).all()
                or self.attempts != ATTEMPTS or type(self.attempt_index) is not int
                or not 0 <= self.attempt_index <= ATTEMPTS or self.chunk_index != 0
                or self.attempt_end != (self.attempt_index > 0)
                or (self.noise_state is not None and (self.noise_state.ndim != 1
                    or self.noise_state.dtype != torch.uint8 or self.noise_state.device.type != "cpu"))):
            raise ValueError("Invalid complete K=4 attempt-boundary state")
        _copy_memory(self.memory, lanes)
        return lanes


def clone_boundary(state, *, indices=None, reset_memory=False):
    """Copy every state field, optionally transplant complete same-length histories."""
    lanes = state.validate()
    if indices is not None and (len(indices) != lanes or sorted(indices) != list(range(lanes))):
        raise ValueError("History transplant must be a complete lane permutation")
    def copy(value):
        return (value.detach().clone() if indices is None else
                value.detach().index_select(0, torch.tensor(indices, device=value.device)))
    return replace(state, memory=None if reset_memory else _copy_memory(state.memory, lanes, indices),
                   previous_command=copy(state.previous_command), pending_reward=copy(state.pending_reward),
                   history_ids=state.history_ids if indices is None else tuple(state.history_ids[i] for i in indices),
                   noise_state=None if state.noise_state is None else state.noise_state.detach().clone())


def _feedback(state, previous, pending_reward, chunk_index, reward_blind):
    return torch.cat([public_chunk_feedback(
        previous[lane:lane + 1], previous_reward=0. if reward_blind else float(pending_reward[lane]),
        attempt_end=state.attempt_end and chunk_index == 0, attempt_index=state.attempt_index,
        chunk_index=chunk_index, attempts=ATTEMPTS, chunks=CHUNKS)
        for lane in range(len(state.history_ids))])


def _tensor_digest(*tensors):
    digest = hashlib.sha256()
    for value in tensors:
        if not isinstance(value, torch.Tensor):
            return None
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _empty_counts():
    return dict(simulator_steps=0, action_chunks=0, native_prefix_queries=0,
                native_suffix_queries=0, bridge_context_updates=0)


def _add_counts(total, addition):
    for key in total:
        total[key] += addition[key]


@torch.no_grad()
def run_live_attempt(policy, motor, env, scenes, theta_degrees, reset_seed, state, *, generator,
                     reward_blind=False, preprocess=None, prepare_actions=None, on_public_chunk=None):
    """Run one 80-step attempt and return its complete next boundary and raw outcomes.

    Hidden angles and scene IDs are private environment arguments. Public action
    feedback always uses the command before the hidden actuator rotation.
    All native-query counts are lane-weighted, not just Python batch calls.
    """
    lanes = state.validate()
    if (state.attempt_index >= ATTEMPTS or len(scenes) != lanes or len(theta_degrees) != lanes
            or any(not math.isfinite(float(x)) for x in theta_degrees)
            or type(reset_seed) is not int or not 0 <= reset_seed < 2**32):
        raise ValueError("Require one valid scene/angle per lane and an unfinished K=4 trial")
    if preprocess is None:
        from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
        preprocess = lambda observation: preprocess_observation(observation, train=False)
    if prepare_actions is None:
        from rlinf.envs.action_utils import prepare_actions as native_prepare
        prepare_actions = lambda actions: native_prepare(
            raw_chunk_actions=actions, env_type="maniskill", model_type="openpi",
            num_action_chunks=ms.CHUNK, action_dim=7, policy="widowx_bridge")
    policy.eval()
    current = clone_boundary(state)
    reference = next(policy.parameters())
    previous = current.previous_command.to(reference)
    pending_reward, memory = current.pending_reward, current.memory
    ids = torch.tensor([scene["episode_id"] for scene in scenes], device=env.device)
    obs, _ = env.reset(seed=reset_seed, options={"episode_id": ids})
    if env.env.unwrapped.select_carrot_ids.cpu().tolist() != [scene["object"] for scene in scenes]:
        raise RuntimeError("Fresh-probe reset differs from its private scene plan")
    success_once = torch.zeros(lanes, dtype=torch.bool)
    success_at_end = torch.zeros_like(success_once)
    counts, first_prefix_digest, first_feedback = _empty_counts(), None, None
    for chunk in range(CHUNKS):
        feedback = _feedback(current, previous, pending_reward, chunk, reward_blind)
        observation = motor.env_obs_to_observation(obs)
        prepared = preprocess(observation)
        actions, memory, record = policy.sample_chunk(
            motor, prepared, feedback, memory, rng=generator, deterministic=True,
            reset=torch.zeros(lanes, device=reference.device, dtype=torch.bool), num_steps=DENOISE_STEPS)
        if on_public_chunk is not None:
            # Project before the callback: labels, private simulator plans and
            # the collector's recurrent features are never exported as inputs.
            on_public_chunk({key: record[key].detach().to("cpu", copy=True)
                             for key in ("kv_key", "kv_value", "prefix_mask", "feedback")})
        if record.get("recurrence_updates") != 1 or record.get("num_steps") != DENOISE_STEPS:
            raise RuntimeError("Fresh probes require one bridge update and four native suffix passes per chunk")
        if chunk == 0:
            first_prefix_digest = _tensor_digest(record.get("kv_key"), record.get("kv_value"),
                                                record.get("prefix_mask"))
            first_feedback = feedback.detach().cpu().tolist()
        decoded = motor.decode_actions(actions, observation.state)
        command = prepare_actions(decoded)
        if command.shape != (lanes, ms.CHUNK, 7) or not torch.isfinite(command).all():
            raise RuntimeError("Require a finite five-step native controller command")
        issued = ms.rotate_translation(command, theta_degrees)
        if not torch.equal(command[..., 2:], issued[..., 2:]):
            raise RuntimeError("Private rotation changed non-XY action coordinates")
        observations, _, _, _, infos = env.chunk_step(issued)
        if len(observations) != ms.CHUNK or len(infos) != ms.CHUNK:
            raise RuntimeError("Probe must execute all five steps of every action chunk")
        for info in infos:
            success_at_end = info["success"].detach().cpu().bool()
            if success_at_end.shape != (lanes,):
                raise RuntimeError("Require one simulator outcome per lane")
            success_once |= success_at_end
        previous = command[:, -1].detach().to(reference)
        pending_reward = success_once.float() if chunk == CHUNKS - 1 else torch.zeros(lanes)
        obs = observations[-1]
        counts["simulator_steps"] += lanes * ms.CHUNK
        counts["action_chunks"] += lanes
        counts["native_prefix_queries"] += lanes
        counts["native_suffix_queries"] += lanes * DENOISE_STEPS
        counts["bridge_context_updates"] += lanes
    boundary = ProbeBoundary(_copy_memory(memory, lanes), previous.detach().clone(),
                             pending_reward.detach().clone(), current.attempt_index + 1,
                             current.history_ids, attempt_end=True, noise_state=generator.get_state().clone())
    return dict(state=boundary, success_once=success_once.tolist(), success_at_end=success_at_end.tolist(),
                counts=counts, first_prefix_sha256=first_prefix_digest, first_feedback=first_feedback,
                final_noise_state_sha256=_tensor_digest(generator.get_state()))


def validate_probe_plans(support_plan, probe_plans, probe_seeds, opposite_indices=None):
    """Validate fresh scenes and matched opposite-angle/object cohorts before execution."""
    thetas = support_plan["theta_degrees"]
    lanes = len(thetas)
    support = support_plan["scenes"]
    if (not lanes or len(support) != SUPPORT_ATTEMPTS or len(support_plan["reset_seeds"]) != SUPPORT_ATTEMPTS
            or any(len(batch) != lanes for batch in support) or not probe_plans
            or len(probe_seeds) != len(probe_plans)
            or any(not math.isfinite(float(theta)) for theta in thetas)
            or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in probe_seeds)):
        raise ValueError("Require three support attempts and paired fresh K=4 probe plans/seeds")
    support_ids = {scene["episode_id"] for batch in support for scene in batch}
    for probe in probe_plans:
        if (len(probe["scenes"]) != lanes or type(probe["reset_seed"]) is not int
                or not 0 <= probe["reset_seed"] < 2**32
                or any(scene["episode_id"] in support_ids for scene in probe["scenes"])):
            raise ValueError("Every probe scene must be fresh relative to every support history")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in support_plan["reset_seeds"]):
        raise ValueError("Require valid support reset seeds")
    if opposite_indices is not None:
        if len(opposite_indices) != lanes or sorted(opposite_indices) != list(range(lanes)):
            raise ValueError("Opposite histories require a complete lane permutation")
        for lane, other in enumerate(opposite_indices):
            if (lane == other or not thetas[lane] or not math.isclose(thetas[lane], -thetas[other], abs_tol=1e-8)
                    or any(batch[lane]["object"] != batch[other]["object"] for batch in support)
                    or any(probe["scenes"][lane]["object"] != probe["scenes"][other]["object"] for probe in probe_plans)):
                raise ValueError("Opposite-history partners must have nonzero opposite angles and matching object sequences")
    return lanes


def _zero_bridge_policy(policy):
    """Copy only the small policy; preserve the original policy and frozen motor."""
    zero = deepcopy(policy).eval()
    with torch.no_grad():
        zero.bridge.residual_output.weight.zero_()
        zero.bridge.residual_output.bias.zero_()
    return zero


@torch.no_grad()
def evaluate_fresh_probes(policy, motor, env, support_plan, probe_plans, *, support_generator,
                          probe_seeds, history_batch_id, opposite_indices=None,
                          conditions=("normal", "reset_memory", "opposite_history", "zero_bridge"),
                          preprocess=None, prepare_actions=None, should_stop=None,
                          on_progress=None, on_boundary=None):
    """Evaluate paired fresh probes without allocation or model-loading changes.

    support_plan = {theta_degrees: [B], scenes: [3][B], reset_seeds: [3]}.
    Each probe plan = {scenes: [B], reset_seed: int}; probe_seeds supplies one
    native-noise seed per replicate, shared by every intervention condition.
    Opposite-history transplants include pending feedback and all recurrent state.
    reset_memory retains pending feedback, so it is a memory lesion rather than
    a completely history-free static adapter. Optional reward_blind recollects
    three supports with masked rewards and counts that extra work explicitly.

    should_stop() is checked before every complete attempt/condition. Progress
    callbacks receive independent JSON-safe snapshots. Optional on_boundary gets
    (detached cloned ProbeBoundary, JSON metadata) after each support attempt so
    the caller can save its full memory and motor RNG without tensors in results.
    """
    if (not isinstance(history_batch_id, str) or not history_batch_id or not conditions
            or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions)):
        raise ValueError("Require a unique support-batch identity and known distinct conditions")
    if "opposite_history" in conditions and opposite_indices is None:
        raise ValueError("Opposite-history evaluation requires declared matched lane partners")
    lanes = validate_probe_plans(support_plan, probe_plans, probe_seeds, opposite_indices)
    history_ids = tuple(f"{history_batch_id}/lane-{lane}" for lane in range(lanes))
    reference = next(policy.parameters())
    initial = ProbeBoundary(None, reference.new_zeros(lanes, 7), torch.zeros(lanes), 0, history_ids)
    hooks = dict(preprocess=preprocess, prepare_actions=prepare_actions)
    initial_support_rng = clone_generator(support_generator)
    support_counts, probe_counts = _empty_counts(), _empty_counts()
    support_counts_by_collection, probe_counts_by_condition = {}, {c: _empty_counts() for c in conditions}
    support_outcomes, states, support_rows, rows, pairing, branches = {}, {}, [], [], [], []
    support_conditions = ("normal", "reward_blind") if "reward_blind" in conditions else ("normal",)

    def snapshot(*, completed=False, stopped=False):
        total = {key: support_counts[key] + probe_counts[key] for key in support_counts}
        state_metadata = {condition: dict(attempt_index=state.attempt_index,
            attempt_end=state.attempt_end, chunk_index=state.chunk_index, attempts=state.attempts,
            history_ids=list(state.history_ids), pending_reward=state.pending_reward.cpu().tolist(),
            previous_command=state.previous_command.cpu().tolist(), memory_present=state.memory is not None,
            motor_noise_state_sha256=_tensor_digest(state.noise_state))
            for condition, state in states.items()}
        result = dict(protocol="pi05_three_support_fresh_probe_v1", history_batch_id=history_batch_id,
            completed=completed, stopped_at_attempt_boundary=stopped,
            cluster_unit="independent support-history batch; cloned probes and transplanted partners are grouped",
            independent_clusters=int(bool(branches)), support_history_lanes=lanes, conditions=list(conditions),
            support_outcomes=support_outcomes, support_rows=support_rows, rows=rows, pairing_checks=pairing,
            completed_probe_branches=branches, support_boundary_metadata=state_metadata,
            support_counts=support_counts, probe_counts=probe_counts,
            support_counts_by_collection=support_counts_by_collection,
            probe_counts_by_condition=probe_counts_by_condition, total_counts=total,
            simulator_steps=total["simulator_steps"], policy_chunks=total["action_chunks"],
            deployment_budget_per_replica=dict(support_attempts=3, probe_attempts=1, total_attempts=4),
            note="Each probe restores a fresh complete boundary; recurrence continues within it. Reset-memory retains pending public feedback.")
        # Detach Python containers too: callbacks cannot mutate future receipts.
        return json.loads(json.dumps(result, allow_nan=False))

    def emit(*, completed=False, stopped=False):
        result = snapshot(completed=completed, stopped=stopped)
        if on_progress is not None:
            on_progress(json.loads(json.dumps(result, allow_nan=False)))
        return result

    for condition in support_conditions:
        state = clone_boundary(initial)
        support_outcomes[condition] = []
        support_counts_by_collection[condition] = _empty_counts()
        generator = support_generator if condition == "normal" else clone_generator(initial_support_rng)
        for attempt in range(SUPPORT_ATTEMPTS):
            if should_stop is not None and should_stop():
                return emit(stopped=True)
            result = run_live_attempt(policy, motor, env, support_plan["scenes"][attempt],
                support_plan["theta_degrees"], int(support_plan["reset_seeds"][attempt]), state,
                generator=generator, reward_blind=condition == "reward_blind", **hooks)
            state = result["state"]
            states[condition] = clone_boundary(state)
            support_outcomes[condition].append(result["success_once"])
            _add_counts(support_counts, result["counts"])
            _add_counts(support_counts_by_collection[condition], result["counts"])
            for lane in range(lanes):
                support_rows.append(dict(cluster_id=history_batch_id, history_id=history_ids[lane],
                    collection_condition=condition, lane=lane, attempt_index=attempt,
                    theta_degrees=float(support_plan["theta_degrees"][lane]),
                    scene=dict(support_plan["scenes"][attempt][lane]),
                    reset_seed=int(support_plan["reset_seeds"][attempt]),
                    success_once=bool(result["success_once"][lane]),
                    success_at_end=bool(result["success_at_end"][lane])))
            if on_boundary is not None:
                on_boundary(clone_boundary(state), dict(history_batch_id=history_batch_id,
                    collection_condition=condition, completed_support_attempts=attempt + 1))
            emit()
    zero = None
    for probe_index, (plan, noise_seed) in enumerate(zip(probe_plans, probe_seeds, strict=True)):
        paired_start, paired_noise = {}, {}
        check = dict(probe_index=probe_index, first_prefix_sha256=paired_start,
                     final_noise_state_sha256=paired_noise, first_prefix_identical=None,
                     native_noise_streams_identical=None)
        pairing.append(check)
        for condition in conditions:
            if should_stop is not None and should_stop():
                return emit(stopped=True)
            state = states["reward_blind"] if condition == "reward_blind" else states["normal"]
            state = clone_boundary(state, indices=opposite_indices if condition == "opposite_history" else None,
                                   reset_memory=condition == "reset_memory")
            if condition == "zero_bridge" and zero is None:
                zero = _zero_bridge_policy(policy)
            branch = zero if condition == "zero_bridge" else policy
            generator = torch.Generator(device=support_generator.device).manual_seed(noise_seed)
            result = run_live_attempt(branch, motor, env, plan["scenes"], support_plan["theta_degrees"],
                int(plan["reset_seed"]), state, generator=generator,
                reward_blind=condition == "reward_blind", **hooks)
            _add_counts(probe_counts, result["counts"])
            _add_counts(probe_counts_by_condition[condition], result["counts"])
            paired_start[condition] = result["first_prefix_sha256"]
            paired_noise[condition] = result["final_noise_state_sha256"]
            for lane in range(lanes):
                rows.append(dict(cluster_id=history_batch_id, history_id=history_ids[lane],
                    donor_history_id=state.history_ids[lane], condition=condition, probe_index=probe_index,
                    lane=lane, theta_degrees=float(support_plan["theta_degrees"][lane]),
                    scene=dict(plan["scenes"][lane]), reset_seed=int(plan["reset_seed"]), noise_seed=noise_seed,
                    support_attempts=SUPPORT_ATTEMPTS, probe_attempt_index=SUPPORT_ATTEMPTS, attempts=ATTEMPTS,
                    success_once=bool(result["success_once"][lane]), success_at_end=bool(result["success_at_end"][lane]),
                    first_feedback=result["first_feedback"][lane]))
            branches.append(dict(probe_index=probe_index, condition=condition))
            if len(paired_start) == len(conditions):
                known = [value for value in paired_start.values() if value is not None]
                check["first_prefix_identical"] = len(known) == len(conditions) and len(set(known)) == 1
                check["native_noise_streams_identical"] = len(set(paired_noise.values())) == 1
            emit()
    return emit(completed=True)
