"""Private paired plans and admission contract for a bounded DPT→PPO comparison.

This module does not train, load weights, allocate resources or launch jobs.
The provider depends only on completed update count, never route, policy, wall
time or a mutable sampler. It can be used by a new trainer hook without changing
the contracts of existing runs.
"""

from copy import deepcopy
import re

from .pi05_balanced_plan import (
    canonical_sha256, make_balanced_history_manifest, plan_sha256,
)


PROTOCOL = "native_pi05_paired_dpt_ppo_v1"
PLAN_PROTOCOL = "native_pi05_paired_online_plan_v1"
GATE_PROTOCOL = "native_pi05_reviewed_interface_gate_v1"
MAX_UPDATES = 20
SIMULATOR_STEPS_PER_UPDATE = 6 * 4 * 80
POLICY_CHUNKS_PER_UPDATE = SIMULATOR_STEPS_PER_UPDATE // 5


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("Require an integer seed in [0,2**32)")
    return value


def _digest(value, length, name):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"Require the pinned {name}")
    return value


def _derived_seed(seed, update, purpose):
    return int(canonical_sha256([PLAN_PROTOCOL, seed, update, purpose])[:8], 16)


def online_budget(completed_updates):
    """Actual online exposure at complete update boundaries, excluding evaluation.

    optimizer_steps_max is a cap; KL stopping may take fewer steps and the
    realized optimizer count must be reported separately.
    """
    if type(completed_updates) is not int or not 0 <= completed_updates <= MAX_UPDATES:
        raise ValueError("Completed updates must be within the bounded 20-update phase")
    chunks = completed_updates * POLICY_CHUNKS_PER_UPDATE
    return dict(completed_updates=completed_updates, global_batch_trials=6,
                trials=6 * completed_updates, attempts=24 * completed_updates,
                simulator_steps=SIMULATOR_STEPS_PER_UPDATE * completed_updates,
                policy_chunks=chunks, native_prefix_queries=chunks,
                native_suffix_queries=4 * chunks, optimizer_steps_max=4 * completed_updates)


class PairedPPOPlanProvider:
    """Callable private plan provider: provider(iteration) uses zero-based iteration.

    Each update contains all six theta/object cells, with four 80-step attempts
    per lane. Both routes must consume the same returned plan and explicitly
    seed their motor/residual generator from motor_residual_rng_seed before
    collection. Returned dictionaries are copies so one caller cannot mutate
    another route's plan. Fresh observations still depend on each policy's own
    behavior; matched scene/noise plans do not imply identical trajectories.
    """

    def __init__(self, infos, *, seed=0, updates=MAX_UPDATES):
        self.seed = _seed(seed)
        if type(updates) is not int or not 1 <= updates <= MAX_UPDATES:
            raise ValueError("Initial paired phase permits 1..20 updates per route")
        self.updates = updates
        self._plans = []
        for iteration in range(updates):
            plan_seed = _derived_seed(seed, iteration, "scene_plan")
            manifest = make_balanced_history_manifest(infos, seed=plan_seed)
            plan = deepcopy(manifest["plans"][manifest["train_indices"][0]])
            # Retain the balanced planner's hash separately, then hash the full
            # online plan, including update/noise identity and budget metadata.
            parent_hash = plan.pop("plan_sha256")
            plan.update(format=PLAN_PROTOCOL, balanced_parent_plan_sha256=parent_hash,
                iteration=iteration, update=iteration + 1, scene_plan_seed=plan_seed,
                motor_residual_rng_seed=_derived_seed(seed, iteration, "motor_residual"),
                cumulative_online_budget=online_budget(iteration + 1),
                pairing="identical private plan and initial noise stream for direct and DPT routes")
            plan["plan_sha256"] = plan_sha256(plan)
            self._plans.append(plan)
        if len({plan["plan_sha256"] for plan in self._plans}) != updates:
            raise ValueError("Paired online plans must have distinct update identities")
        provenance = dict(protocol=PLAN_PROTOCOL, seed=seed, updates=updates,
            theta_degrees=[-45., 0., 45.], objects=[0, 1], global_batch_trials=6,
            attempts=4, episode_steps=80, action_chunk=5, denoise_steps=4,
            scene_contract=manifest["plans"][0]["scene_contract"],
            plan_sha256s=[plan["plan_sha256"] for plan in self._plans],
            planned_online_budget_per_route=online_budget(updates),
            repeat_rule="distinct scenes within an update; seeded scene reuse across updates is allowed",
            noise_rule="set explicit motor/residual generator per update; never seed from route",
            metadata_is_private=True)
        self._provenance = dict(provenance, provider_sha256=canonical_sha256(provenance))

    def __call__(self, iteration):
        if type(iteration) is not int or not 0 <= iteration < self.updates:
            raise ValueError("Plan iteration must be a remaining zero-based update index")
        return deepcopy(self._plans[iteration])

    def provenance(self):
        return deepcopy(self._provenance)


def require_live_competence_gate(receipt):
    """Require an explicit reviewed live decision, never infer promotion from loss.

    The run reviewer owns the scientific pass criterion and records it in
    decision. Evidence must contain actual bridge/zero/external-oracle live
    outcomes at all three training angles. This function checks provenance and
    interface identity; it does not invent a success threshold after results.
    """
    if not isinstance(receipt, dict):
        raise ValueError("A reviewed live same-interface competence receipt is required")
    expected = dict(protocol=GATE_PROTOCOL, status="passed", complete=True,
        evidence_kind="live_same_interface", sampler="StochasticPi05BridgePolicy.sample_chunk",
        deterministic=True, denoise_steps=4, action_chunk=5, active_dims=7, residual_scale=1.,
        core="persistent_ctm", width=64, ticks=4)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("PPO is gated on completed actual live four-Euler bridge competence")
    _digest(receipt.get("source_commit"), 40, "gate source commit")
    _digest(receipt.get("evidence_sha256"), 64, "live gate evidence SHA256")
    _digest(receipt.get("motor_sha256"), 64, "gate frozen motor SHA256")
    _digest(receipt.get("norm_stats_sha256"), 64, "gate normalization SHA256")
    if not isinstance(receipt.get("rlinf_sources"), dict) or not receipt["rlinf_sources"]:
        raise ValueError("Require the gate's pinned native runtime source hashes")
    for digest in receipt["rlinf_sources"].values():
        _digest(digest, 64, "gate native runtime source SHA256")
    if (receipt.get("evaluated_theta_degrees") != [-45., 0., 45.]
            or not isinstance(receipt.get("decision"), str) or not receipt["decision"].strip()):
        raise ValueError("Require all training-angle live comparisons and reviewed pass rationale")
    comparisons = receipt.get("comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {"-45.0", "0.0", "45.0"}:
        raise ValueError("Require live per-theta bridge, zero and external-oracle outcomes")
    for result in comparisons.values():
        if not isinstance(result, dict) or set(result) != {"bridge", "zero_bridge", "external_oracle"}:
            raise ValueError("Require every paired live reference condition")
        shapes = []
        for outcomes in result.values():
            if (not isinstance(outcomes, list) or not outcomes
                    or any(not isinstance(row, list) or not row for row in outcomes)):
                raise ValueError("Live outcomes must be nonempty attempt-by-lane arrays")
            shapes.append([len(row) for row in outcomes])
            if any(type(value) not in (bool, int, float) or value not in (0, 1)
                   for row in outcomes for value in row):
                raise ValueError("Require binary recorded live outcomes")
        if any(shape != shapes[0] for shape in shapes[1:]) or len(set(shapes[0])) != 1:
            raise ValueError("Live bridge/reference outcomes must have matched rectangular shapes")
    return deepcopy(receipt)


def make_paired_ppo_contract(provider, *, source_commit, warmstart_sha256,
                              warmstart_provenance, competence_gate):
    """Build a reviewable common contract; this is admission, not a launch command.

    The trainer must additionally pin native motor/norm/source/runtime hashes
    and validate its portable warm-start family/interface before loading. Saved
    resume states remain complete policy+optimizer+sampler+motor/global RNG.
    """
    _digest(source_commit, 40, "new phase source commit")
    _digest(warmstart_sha256, 64, "balanced DPT initialization SHA256")
    if not isinstance(warmstart_provenance, dict):
        raise ValueError("Require the portable balanced DPT parent manifest")
    parent = deepcopy(warmstart_provenance)
    interface = parent.get("compatibility", {})
    bridge, gaussian = interface.get("bridge", {}), interface.get("gaussian_policy", {})
    if (parent.get("family") != "native_pi05_history_v1"
            or type(parent.get("iteration")) is not int or parent["iteration"] < 1
            or interface.get("core") != "persistent_ctm"
            or any(bridge.get(key) != value for key, value in
                   dict(core="persistent_ctm", width=64, ticks=4, residual_scale=1.).items())
            or any(gaussian.get(key) != value for key, value in
                   dict(noise_scale=.05, active_dims=7, residual_scale=1.).items())
            or not isinstance(parent.get("data"), dict)
            or not isinstance(parent.get("pretraining_counters"), dict)
            or not parent["pretraining_counters"]
            or any(type(value) is not int or value < 0 for value in parent["pretraining_counters"].values())):
        raise ValueError("Require compatible balanced native DPT weights and separate pretraining costs")
    _digest(parent.get("source_commit"), 40, "DPT source commit")
    _digest(parent["data"].get("balanced_plan_sha256"), 64, "balanced DPT plan SHA256")
    gate = require_live_competence_gate(competence_gate)
    if any(gate[key] != interface.get(key) for key in ("motor_sha256", "norm_stats_sha256", "rlinf_sources")):
        raise ValueError("Live gate and DPT parent frozen motor/runtime identities differ")
    plans = provider.provenance()
    common = dict(protocol=PROTOCOL, source_commit=source_commit, plans=plans,
        core="persistent_ctm", bridge=dict(core="persistent_ctm", width=64, ticks=4, residual_scale=1.),
        gaussian_policy=dict(noise_scale=.05, active_dims=7, likelihood="summed joint auxiliary flow likelihood"),
        seed=provider.seed, global_batch_trials=6, attempts=4, episode_steps=80,
        action_chunk=5, denoise_steps=4,
        optimizer=dict(name="AdamW", learning_rate=1e-4, weight_decay=0., betas=[.9, .999], eps=1e-8),
        optimization=dict(epochs=4, clip=.2, value_coef=.01, reward_offset=0., entropy_coef=.01,
                          max_grad_norm=1., target_kl=.05, gamma=1., gae_lambda=1.),
        returns="trial termination only; equal attempt reward weights; no terminal bootstrap",
        initialization=dict(
            direct=dict(kind="fresh seeded CTM and zero residual head; zero critic"),
            dpt=dict(kind="portable balanced DPT weights", checkpoint_sha256=warmstart_sha256,
                     parent=parent, critic="zero"),
            shared="fresh optimizer and seeded global/motor RNG per route; no recurrent trial state transferred"),
        gate=gate, gate_sha256=canonical_sha256(gate),
        resource_proposal=dict(gpus=2, scheduler_wall_seconds_max=5400,
            driver_elapsed_seconds_max_per_route=3600, target_updates_per_route=provider.updates,
            status="pending allocation and measured batch6 throughput; no jobs launched by planner"),
        qualification=dict(first_updates=2, checks=["exact unchanged likelihood replay", "finite gradients",
            "frozen motor fingerprint", "global batch6", "same per-update plans/noise seeds", "update timing"]),
        comparison="report at common completed online budget; wall-capped partial routes are not 20-update completions",
        evaluation="held-out theta/scenes; matched own-history and common-history probes; separate evaluation costs",
        native_identity={key: deepcopy(interface[key])
                         for key in ("motor_sha256", "norm_stats_sha256", "rlinf_sources")})
    return dict(common, contract_sha256=canonical_sha256(common))
