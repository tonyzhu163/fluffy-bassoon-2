"""Small immutable 4D source prefixes for the existing Gaussian history learner.

This is an admission/projection adapter, not a collector, learner or scheduler.
Array/model operations and tests belong on allocated Linux. Metadata identities
are deliberately separate from the public observation/action/reward tensors.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sys


PROFILE = "frozen_vla_4d_history_profile_v1"
PREFIX = "frozen_vla_4d_source_prefix_v1"
ASSEMBLY = "frozen_vla_4d_history_assembly_v1"
HISTORY_LESIONS = ("intact", "reset", "reward-blind", "shuffle", "wrong-condition", "cross-task-transfer")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def reference(ref):
    if (not isinstance(ref, dict) or not {"path", "sha256"} <= ref.keys()
            or not ref.keys() <= {"path", "sha256", "root", "retention", "size_bytes"}
            or not isinstance(ref["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("Require an absolute immutable file reference with its actual SHA256")
    path = Path(ref["path"])
    if "root" in ref:
        root = Path(ref["root"])
        path = (root / path).resolve()
        if not root.is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("Canonical raw reference escapes its original absolute corpus root")
    if (not path.is_absolute() or digest(path) != ref["sha256"]
            or ("retention" in ref and ref["retention"] != "immutable")
            or ("size_bytes" in ref and path.stat().st_size != ref["size_bytes"])):
        raise ValueError("Reference changed, is rolling, or differs in canonical byte size")
    return path


def read_json(ref):
    path = reference(ref)
    value = json.loads(path.read_text())
    reference(ref)
    return value


def validate_profile(profile):
    interface = profile["interface"]
    if (profile.get("format") != PROFILE or profile.get("source_task") not in
            ("place_phone_stand", "place_object_stand")
            or not re.fullmatch("[0-9a-f]{64}", profile.get("scope_sha256", ""))
            or interface.get("coefficient_dim") != 4 or interface.get("native_horizon") != 32
            or interface.get("denoise_steps") != 5 or interface.get("arm") not in ("noise", "kv", "action")
            or not re.fullmatch("[0-9a-f]{64}", interface.get("motor_sha256", ""))
            or not re.fullmatch("[0-9a-f]{64}", interface.get("mapping_sha256", ""))
            or not re.fullmatch("[0-9a-f]{64}", interface.get("control_interface_sha256", ""))
            or not interface.get("feature_contract") or not interface.get("mapping_contract")
            or interface.get("label_coordinates") != "actual requested post-bound 4D coefficients before hidden actuator transform"
            or type(interface.get("observation_dim")) is not int or interface["observation_dim"] < 1
            or type(interface.get("native_observation_dim")) is not int or interface["native_observation_dim"] < 1
            or interface["observation_dim"] != interface["native_observation_dim"] + 14
            or interface.get("public_command_history") != "previous last-valid pre-condition 14D request; zero every physical reset"
            or type(interface.get("coefficient_limit")) not in (int, float)
            or interface["coefficient_limit"] != 1.0
            or interface.get("feature_contract", {}).get("public_only") is not True):
        raise ValueError("Require the qualified public-feature Motus H32/five-step exact 4D mapping")
    conditions = profile["conditions"]
    if len(conditions) != 2 or len(set(conditions)) != 2 or any(not x for x in conditions):
        raise ValueError("Require exactly two persistent conditions, excluded from student inputs")
    slots = profile["lineage_slots"]
    if (len(slots) != 6 or len({x["lineage_id"] for x in slots}) != 6
            or len({x["seed"] for x in slots}) != 6
            or any(type(x["seed"]) is not int or not 0 <= x["seed"] < 2**32 for x in slots)):
        raise ValueError("Predeclare the six exact collector method/lineage keys and distinct learner seeds")
    if (profile.get("max_support_episodes") != 8 or profile.get("query_k") != [0, 1, 2, 4]
            or not 3 <= profile.get("prefix_episodes", 0) <= 48
            or set(profile["teacher_lineage_ids"]) != set(conditions)):
        raise ValueError("Predeclare a complete source prefix and K0/1/2/4 query panel; 48 is a cap, not an entitlement")
    fit = profile["fit"]
    if (set(fit) != {"seed", "updates", "width", "layers", "heads", "context_length", "batch_size",
                     "eval_every", "validation_batches", "learning_rate", "max_seconds_per_fit"}
            or any(type(fit[k]) is not int or fit[k] < 1 for k in
                   ("updates", "width", "layers", "heads", "context_length", "batch_size",
                    "eval_every", "validation_batches"))
            or type(fit["seed"]) is not int or not 0 <= fit["seed"] < 2**32
            or fit["width"] % fit["heads"] or fit["width"] % 2
            or not all(math.isfinite(fit[k]) and fit[k] > 0 for k in
                       ("learning_rate", "max_seconds_per_fit"))):
        raise ValueError("Require one fixed matched Gaussian fit schedule")
    gate = profile["teacher_gate"]
    if (gate.get("paired_cases", 0) < 2 or not 0 < gate.get("minimum_success_rate", 0) <= 1
            or not 0 < gate.get("minimum_success_gain", 0) <= 1):
        raise ValueError("Predeclare nontrivial paired improvement and competence thresholds")
    if (type(profile.get("episode_action_limit")) is not int or profile["episode_action_limit"] < 1
            or type(profile.get("wrong_task_action_limit")) is not int or profile["wrong_task_action_limit"] < 1
            or type(profile.get("reset_policy_action_bound")) is not int or profile["reset_policy_action_bound"] < 0
            or len(profile["locked_query_case_ids"]) != 4 or len(set(profile["locked_query_case_ids"])) != 4
            or profile.get("wrong_task") not in ("place_phone_stand", "place_object_stand")
            or profile["wrong_task"] == profile["source_task"]):
        raise ValueError("Predeclare four locked case blocks, real horizons and the other source-candidate donor task")
    reservation = profile["shared_reservation"]
    if (not reservation.get("owner") or not reservation.get("pilot_id")
            or reservation.get("pilot_gpu_seconds") != 8 * 3600
            or reservation.get("pilot_actions") != 300000 or reservation.get("pilot_storage_bytes") != 64 * 1024**3
            or 4 * fit["max_seconds_per_fit"] > reservation["fit_gpu_seconds"]
            or reservation["fit_gpu_seconds"] > reservation["pilot_gpu_seconds"]
            or reservation.get("evaluation_bucket") != "student_evaluation"
            or not 0 < reservation["evaluation_actions"] <= 105000):
        raise ValueError("Fit/evaluation must be slices of the sole owner's shared pilot reservation")
    evaluation_plan(profile, validated=True)
    return profile


def evaluation_plan(profile, *, validated=False):
    if not validated:
        validate_profile(profile)
    def seed(tag):
        text = profile["shared_reservation"]["pilot_id"] + ":" + str(profile["fit"]["seed"]) + ":" + tag
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")

    cards, supports = [], []
    for condition_index, condition in enumerate(profile["conditions"]):
        for scene in range(2):
            block = profile["locked_query_case_ids"][condition_index * 2 + scene]
            # Reset seeds match across conditions. IDs and condition are audit metadata only.
            query_seed = seed(f"scene-{scene}:query")
            support_seeds = [seed(f"scene-{scene}:support-{i}") for i in range(4)]
            for objective in ("ad", "dpt"):
                policy = objective + "-full"
                support_id = block + ":" + policy
                supports.append(dict(id=support_id, case_id=block, condition_id=condition,
                    policy=policy, episodes=4, reset_seeds=support_seeds,
                    action_seed=seed(f"scene-{scene}:support-actions")))
                for k in profile["query_k"]:
                    cards.append(dict(case_id=block, condition_id=condition, policy=policy,
                        k=k, lesion="intact", support_id=support_id, reset_seed=query_seed,
                        action_seed=seed(f"scene-{scene}:query-actions")))
                opposite_index = 1-condition_index
                opposite_block = profile["locked_query_case_ids"][opposite_index*2+scene]
                for lesion in ("reset", "reward-blind", "shuffle", "wrong-condition"):
                    cards.append(dict(case_id=block, condition_id=condition, policy=policy,
                        k=4, lesion=lesion, support_id=(opposite_block+":"+policy if lesion == "wrong-condition" else support_id),
                        reset_seed=query_seed, action_seed=seed(f"scene-{scene}:query-actions")))
                cards.append(dict(case_id=block, condition_id=condition, policy=policy,k=2,
                    lesion="cross-task-transfer",support_id=objective+":cross-task:"+condition,
                    reset_seed=query_seed,action_seed=seed(f"scene-{scene}:query-actions")))
                cards.append(dict(case_id=block, condition_id=condition, policy=objective + "-static",
                    k=4, lesion="intact", support_id=support_id, context="static; matched K4 support discarded",
                    reset_seed=query_seed, action_seed=seed(f"scene-{scene}:query-actions")))
            supports.append(dict(id=block + ":stationary", case_id=block, condition_id=condition,
                policy="stationary-reference", episodes=4, reset_seeds=support_seeds,
                action_seed=seed(f"scene-{scene}:stationary-actions")))
            cards += [dict(case_id=block, condition_id=condition, policy="native", k=0,
                lesion="intact", support_id=None, reset_seed=query_seed, action_seed=seed(f"scene-{scene}:native-actions")),
                dict(case_id=block, condition_id=condition, policy="stationary-reference", k=4,
                lesion="stationary-retry", support_id=block + ":stationary", reset_seed=query_seed,
                action_seed=seed(f"scene-{scene}:query-actions"))]
    for objective in ("ad", "dpt"):
        for condition in profile["conditions"]:
            supports.append(dict(id=objective+":cross-task:"+condition,task=profile["wrong_task"],condition_id=condition,
                policy=objective+"-full",episodes=2,
                reset_seeds=[seed(f"cross-task:support-{i}") for i in range(2)],
                action_seed=seed("cross-task:support-actions")))
    # 56 support episodes + 88 independent reset queries = 144.
    episodes = sum(row["episodes"] for row in supports) + len(cards)
    cap = ((episodes - 8) * profile["episode_action_limit"] + 8 * profile["wrong_task_action_limit"]
           + episodes * profile["reset_policy_action_bound"])
    if cap > profile["shared_reservation"]["evaluation_actions"]:
        raise ValueError("All charged support/query episodes exceed the owner's evaluation slice")
    return dict(supports=supports, cards=cards, maximum_complete_episodes=episodes, maximum_actions=cap,
                query_snapshot_k=[0, 1, 2, 4], no_query_feedback_reuse=True, stochastic_students=True,
                cross_task_support="8 fresh donors total:2 per model/condition; K2 same-condition transfer, not a negative control",
                wrong_condition_support="K4 from actual opposite-condition same-model/scene public prefix; reconstructed, no cached-state relabeling",
                final_32_case_confirmation="separate, unopened, required before a reliability claim",
                scope="small paired pilot; no OOD, expert-data or ICRL competence claim")


def verify_control_qualification(ref, profile):
    """Map the actual A1 receipt fields; preserve references, never rerun episodes."""
    binding = read_json(ref)
    if binding.get("format") != "frozen_vla_4d_control_qualification_binding_v1":
        raise ValueError("Bind the original driver plan/interface/GPU receipt/completed episode references")
    interface = read_json(binding["control_interface"])
    plan = read_json(binding["driver_plan"])
    q = read_json(binding["driver_qualification"])
    episode = read_json(binding["completed_episode"])
    expected = profile["interface"]
    if (binding["control_interface"]["sha256"] != expected["control_interface_sha256"]
            or plan.get("interface_sha256") != expected["control_interface_sha256"]
            or plan.get("scope_sha256") != profile["scope_sha256"]
            or interface.get("motor_sha256") != expected["motor_sha256"]
            or interface.get("action_horizon") != 32 or interface.get("flow_steps") != 5
            or interface.get("public_dim") != expected["native_observation_dim"]
            or interface.get("coefficient_dim") != 4 or interface.get("coefficient_bound") != 1.0
            or q.get("status") != "passed" or not q.get("native_parity", {}).get("passed")
            or not all(q.get(k) is True for k in ("zero_kv_exact", "zero_fixed_kv_exact", "zero_action_exact"))
            or q.get("motor_gradients") != 0 or not q.get("kv_gradient_norm", 0) > 0
            or not re.fullmatch("[0-9a-f]{64}", q.get("frozen_before", ""))
            or q.get("frozen_before") != q.get("frozen_after")
            or q.get("complete_real_episode") != q.get("qualification_episode")
            or episode.get("complete") is not True
            or episode.get("spec", {}).get("id") != q.get("complete_real_episode")
            or not q.get("actual_actions", 0) > 0 or q["actual_actions"] != episode.get("primitive_steps")):
        raise ValueError("Original A1 receipt does not qualify this exact motor/interface and real completed episode")
    # Safetensors FILE hash above and actual frozen TENSOR fingerprint here are distinct identities.
    return dict(driver_qualification=binding["driver_qualification"], frozen_tensor_sha256=q["frozen_before"],
                motor_file_sha256=interface["motor_sha256"], actual_qualification_episode=q["complete_real_episode"])


def verify_condition_cross(profile, prefix, *, objective):
    from .steering_history_cross_admission import verify_condition_cross as verify
    return verify(profile, prefix, objective=objective)


def admit_prefix(profile, prefix, *, objective):
    """Split entire learning lineages and physical cases BEFORE making windows."""
    validate_profile(profile)
    if objective not in ("ad", "dpt"):
        raise ValueError("Admit AD and DPT independently with explicit objectives")
    if (prefix.get("format") != PREFIX or prefix.get("completed") is not True
            or prefix.get("profile_sha256") != content_hash(profile)
            or prefix.get("source_task") != profile["source_task"]
            or prefix.get("interface") != profile["interface"]
            or prefix.get("source_only") is not True):
        raise ValueError("Require a completed immutable source-only prefix, not a live mutable index")
    verify_control_qualification(prefix["control_qualification"], profile)
    lineages = prefix["lineages"]
    if len(lineages) != 6 or len({x["lineage_id"] for x in lineages}) != 6:
        raise ValueError("Require all two TRAIN plus one VAL lineage per condition without filtering")
    fields = ("lineage_id", "condition_id", "partition", "seed")
    if ({tuple(x[k] for k in fields) for x in lineages}
            != {tuple(x[k] for k in fields) for x in profile["lineage_slots"]}):
        raise ValueError("Prefix changed a predeclared collector lineage/method/partition/seed")
    cases, identities = {"train": set(), "validation": set()}, {"train": set(), "validation": set()}
    for condition in profile["conditions"]:
        rows = [x for x in lineages if x["condition_id"] == condition]
        if (sorted(x["partition"] for x in rows) != ["train", "train", "validation"]
                or len({x["seed"] for x in rows}) != 3):
            raise ValueError("Each condition needs two independent TRAIN seeds and one VAL seed")
    for lineage in lineages:
        if (lineage["condition_id"] not in profile["conditions"] or lineage.get("source_only") is not True
                or lineage.get("completed_episodes") != profile["prefix_episodes"]
                or len(lineage["episodes"]) != profile["prefix_episodes"]
                or lineage.get("initial_updates", -1) < 0
                or lineage.get("final_updates", -1) < lineage["initial_updates"]
                or (objective == "ad" and lineage["final_updates"] == lineage["initial_updates"])):
            raise ValueError("Require complete prefixes; AD additionally requires real source updates")
        for key in ("initial_recovery", "final_recovery"):
            reference(lineage[key])
        for episode in lineage["episodes"]:
            reference(episode["raw_trace"])
            if (not episode["case_id"] or not re.fullmatch("[0-9a-f]{64}", episode["initial_observation_sha256"])
                    or type(episode["actual_seed"]) is not int or not 0 <= episode["actual_seed"] < 2**32):
                raise ValueError("Require actual reset identities and initial observations for each complete episode")
            cases[lineage["partition"]].add(episode["case_id"])
            identities[lineage["partition"]].update(("seed:" + str(episode["actual_seed"]),
                                                      "obs:" + episode["initial_observation_sha256"]))
    if cases["train"] & cases["validation"] or identities["train"] & identities["validation"]:
        raise ValueError("Whole physical cases/reset seeds/initial observations overlap across partitions")
    reserved = set(profile["locked_query_case_ids"])
    if not reserved or (cases["train"] | cases["validation"]) & reserved:
        raise ValueError("Source histories must exclude the predeclared locked query cases")
    verify_condition_cross(profile, prefix, objective=objective)
    for condition in profile["conditions"]:
        gate_name = "learning_qualifications" if objective == "ad" else "teacher_qualifications"
        teacher = read_json(prefix[gate_name][condition])
        reference(teacher["evidence"])
        selected = next((x for x in lineages if x["lineage_id"] == profile["teacher_lineage_ids"][condition]), None)
        panel = teacher.get("paired_cases", [])
        if (teacher.get("condition_id") != condition
                or teacher.get("interface_sha256") != profile["interface"]["control_interface_sha256"]
                or len(panel) != profile["teacher_gate"]["paired_cases"]
                or len({x["case_id"] for x in panel}) != len(panel)):
            raise ValueError("Require the exact same-mapping condition qualification and fixed fresh panel")
        if objective == "ad" or teacher.get("teacher_kind") == "noise_sac":
            if (selected is None or selected["partition"] != "train" or selected["condition_id"] != condition
                    or (objective == "dpt" and teacher.get("teacher_lineage_id") != selected["lineage_id"])):
                raise ValueError("Require the predeclared TRAIN teacher lineage and the exact qualified snapshot")
            reference(teacher["teacher_recovery"])
        if objective == "ad":
            final_identity = equivalent_acting(teacher["teacher_recovery"], selected["final_recovery"],
                lineage=selected, profile=profile)
            baseline_identity = equivalent_acting(teacher["baseline_recovery"], selected["initial_recovery"],
                lineage=selected, profile=profile)
            if (teacher.get("label_sampling_mode") != final_identity["sampling_mode"]
                    or teacher.get("baseline_sampling_mode") != baseline_identity["sampling_mode"]):
                raise ValueError("AD validation must measure the actual uniform/learned sampled behavior convention")
        if objective == "dpt" and teacher.get("teacher_kind") not in ("noise_sac", "constant_4d_oracle"):
            raise ValueError("DPT requires a qualified same-mapping SAC actor or explicit 4D oracle")
        if objective == "dpt" and teacher.get("label_sampling_mode") not in (
                ("sampled_gaussian", "deterministic_mean") if teacher["teacher_kind"] == "noise_sac" else ("fixed_coefficients",)):
            raise ValueError("Bind the exact physically qualified stochastic-actor or fixed-oracle label convention")
        if objective == "dpt" and teacher["teacher_kind"] == "constant_4d_oracle":
            oracle = read_json(teacher["oracle"])
            if (oracle.get("interface_sha256") != profile["interface"]["control_interface_sha256"]
                    or oracle.get("condition_id") != condition or len(oracle.get("coefficients", [])) != 4
                    or not all(type(x) in (int, float) and math.isfinite(x)
                               and abs(x) <= profile["interface"]["coefficient_limit"] for x in oracle["coefficients"])):
                raise ValueError("A search/inverse oracle must declare actual bounded 4D requests and provenance")
            proof = read_json(oracle["provenance"])
            if (proof.get("mapping_sha256") != profile["interface"]["mapping_sha256"]
                    or proof.get("coefficient_dimension") != 4 or proof.get("physical_execution_verified") is not True
                    or proof.get("executed_coefficients") != oracle["coefficients"]):
                raise ValueError("An external decoded-action inverse is not proof of a realizable same-basis 4D oracle")
        panel_ids = {x["case_id"] for x in panel}
        if panel_ids & (cases["train"] | cases["validation"] | reserved):
            raise ValueError("Teacher qualification cases must be fresh and separate from source/query cases")
        if any(type(x[k]) is not bool for x in panel for k in ("baseline_success", "teacher_success")):
            raise ValueError("Qualification must contain actual paired boolean outcomes")
        baseline = sum(x["baseline_success"] for x in panel) / len(panel)
        final = sum(x["teacher_success"] for x in panel) / len(panel)
        if objective == "ad" and final - baseline < profile["teacher_gate"]["minimum_success_gain"]:
            raise ValueError("Uninformative behavior learning: AD gate failed")
        if objective == "dpt" and final < profile["teacher_gate"]["minimum_success_rate"]:
            raise ValueError("Poor final teacher/oracle: DPT gate failed, not expert data")
    return sorted(lineages, key=lambda x: x["lineage_id"])


def _linux(hostname):
    if sys.platform != "linux" or socket.gethostname() != hostname:
        raise ValueError("Run arrays/models only on the explicitly allocated Linux host")


def _source_state(ref, *, lineage, profile):
    from .warmstart import checked_load
    state = checked_load(reference(ref), ref["sha256"], weights_only=False)
    if state.get("format") == "source_coefficient_full_recovery_v1":
        import torch
        required_outer = {"collector", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "control", "event"}
        if (not required_outer <= state.keys() or ref.get("retention") != "immutable"
                or not isinstance(state["python_rng"], tuple) or not isinstance(state["numpy_rng"], tuple)
                or not isinstance(state["torch_rng"], torch.Tensor) or state["torch_rng"].dtype != torch.uint8
                or not isinstance(state["control"], dict) or not state["event"]):
            raise ValueError("Require the actual immutable full recovery with global/control RNG provenance")
        if state["cuda_rng"] is not None and (not isinstance(state["cuda_rng"], list)
                or any(not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 for x in state["cuda_rng"])):
            raise ValueError("Full recovery CUDA RNG provenance differs")
        state = state["collector"]
        if state.get("format") != "source_coefficient_teacher_v1":
            raise ValueError("Full recovery omitted its canonical inner collector state")
    if state.get("format") == "source_coefficient_teacher_v1":
        source, identity = state["profile"], state["lineage"]
        if (state.get("active_episode") is not None or state.get("pending") is not None
                or state.get("pending_update_cycles") != 0 or state.get("updating") is not False
                or identity.get("method") != lineage["lineage_id"] or identity.get("learner_seed") != lineage["seed"]
                or identity.get("partition") != lineage["partition"]
                or source.get("task") != profile["source_task"]
                or source.get("mapping_sha256") != profile["interface"]["mapping_sha256"]
                or source.get("interface_sha256") != profile["interface"]["control_interface_sha256"]
                or source.get("motor_sha256") != profile["interface"]["motor_sha256"]
                or source.get("scope_sha256") != profile["scope_sha256"]):
            raise ValueError("Require the exact canonical collector full recovery at a completed update boundary")
        state = state["learner"]
    required = {"contract", "actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer",
                "temperature_optimizer", "replay", "numpy_rng", "torch_rng", "train_torch_rng", "updates"}
    if not required <= state.keys() or state["replay"] is None:
        raise ValueError("Require a complete resumable NoiseSAC source state, not actor-only weights")
    return state


def _check_source(lineage, profile, *, require_learning):
    import torch
    initial, final = (_source_state(lineage[key], lineage=lineage, profile=profile) for key in ("initial_recovery", "final_recovery"))
    interface = profile["interface"]
    for state, updates in ((initial, lineage["initial_updates"]), (final, lineage["final_updates"])):
        contract = state["contract"]
        if (contract.get("format") != "frozen_vla_feature_noise_sac_v1"
                or contract.get("observation_dim") != interface["native_observation_dim"]
                or contract.get("action_dim") != 4 or contract.get("action_limit") != interface["coefficient_limit"]
                or contract.get("seed") != lineage["seed"] or state["updates"] != updates):
            raise ValueError("Source recovery dimension/bound/seed/update binding differs")
    if initial["contract"] != final["contract"] or initial["actor"].keys() != final["actor"].keys():
        raise ValueError("Source recovery contracts changed inside a lineage")
    if require_learning and all(torch.equal(initial["actor"][key], final["actor"][key]) for key in initial["actor"]):
        raise ValueError("AD requires changing actor weights, not an incremented collection clock")
    return final


def equivalent_acting(validation_ref, prefix_ref, *, lineage, profile):
    """Wrapper bytes can differ; actual actor/update/support must be identical."""
    if sys.platform != "linux":
        raise ValueError("Verify actual source checkpoint tensors only on allocated Linux")
    from .robotwin_history_corpus import parameter_digest
    a = _source_state(validation_ref, lineage=lineage, profile=profile)
    b = _source_state(prefix_ref, lineage=lineage, profile=profile)
    first, second = parameter_digest(a["actor"]), parameter_digest(b["actor"])
    if a["contract"] != b["contract"] or a["updates"] != b["updates"] or first != second:
        raise ValueError("Validation and published prefix have different actual acting weights or updates")
    return dict(validation_input=deepcopy(validation_ref), prefix_snapshot=deepcopy(prefix_ref),
        updates=int(a["updates"]), actor_parameter_sha256=first,
        sampling_mode="bounded_uniform" if a["updates"] == 0 else "sampled_gaussian")


def _teacher(state, *, seed, deterministic=False):
    """Load only the existing actor; use a separate labeling RNG, never source RNG."""
    import numpy as np
    import torch
    from .vla_noise_sac import SACConfig, SquashedGaussianActor
    from .warmstart import validate_weights
    contract = state["contract"]
    config = dict(contract["config"])
    config["hidden_dims"] = tuple(config["hidden_dims"])
    with torch.random.fork_rng(devices=[]):
        actor = SquashedGaussianActor(contract["observation_dim"], 4, SACConfig(**config), contract["action_limit"])
    validate_weights(actor, state["actor"])
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval().requires_grad_(False)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def label(observations):
        # The teacher remains on native public features, never student's previous-command suffix.
        result, _ = actor(torch.as_tensor(observations[:, :contract["observation_dim"]], dtype=torch.float32),
                          generator=rng, deterministic=deterministic)
        return result.numpy().astype(np.float32, copy=True)
    return label


def public_requested_command(reader, chunk, *, metadata=None, decision=None):
    """Select the last ACTUALLY VALID pre-condition command, never the proposal tail."""
    import numpy as np
    from .robotwin_history_corpus import _decode
    if decision is None:
        decision = reader.arrays(chunk["decision_arrays"], list(chunk["decision_arrays"]["arrays"]))
    if metadata is None:
        metadata = _decode(chunk["decision_metadata"], decision)
    residual = metadata.get("frozen_pi05_residual", {})
    if "public_agent_requested_actions" in residual:
        requested = np.asarray(residual["public_agent_requested_actions"])
    elif "audit_only_post_command_transform" in residual:
        raise ValueError("Hidden actuator commands cannot substitute for missing public requests")
    else:
        requested = np.asarray(decision["issued_motor_joint_targets"])
    requested = requested.reshape(-1, 14)
    mask = reader.arrays(chunk["native_result"]["arrays"], ["valid_action_mask"])["valid_action_mask"].reshape(-1)
    valid = np.flatnonzero(mask)
    if (not np.isin(mask, (0, 1)).all() or len(valid) != chunk["native_result"]["executed_primitive_steps"]
            or not len(valid) or valid[-1] >= len(requested) or not np.isfinite(requested).all()):
        raise ValueError("Require exact native validity and finite public requested commands")
    return requested[int(valid[-1])].astype(np.float32, copy=True)


def canonical_trial(lineage, profile, *, require_learning):
    """Read the existing raw corpus; never infer a KV edit from a noise seed.

    Compact decision metadata keys bind the coefficient to its actual
    native transition. Issued/post-calibration commands remain audit-only.
    """
    import numpy as np
    from .robotwin_history_corpus import CorpusReader, _decode
    reader = CorpusReader(lineage["corpus_root"])
    interface = profile["interface"]
    rows, actor_versions, reset_actions = [], set(), 0
    for episode_index, episode in enumerate(lineage["episodes"]):
        path = reference(episode["raw_trace"])
        manifest = reader.json(path, episode["raw_trace"]["sha256"])
        reset = reader.json(path.parent / manifest["reset"])
        refs = manifest.get("chunks", [])
        if (manifest.get("pending_decision") is not None or not refs
                or reset["initial_observation"].get("content_sha256") != episode["initial_observation_sha256"]
                or reset.get("reset_info", {}).get("actual_seed") != episode["actual_seed"]
                or reset.get("reset_info", {}).get("task") != profile["source_task"]):
            raise ValueError("Require an immutable completed canonical episode with its actual reset identity")
        reset_cost = reset["reset_info"].get("reset_policy_actions")
        if type(reset_cost) is not int or not 0 <= reset_cost <= profile["reset_policy_action_bound"]:
            raise ValueError("Actual reset policy actions exceed or lack the verified bound")
        reset_actions += reset_cost
        previous, actions, episode_updates = reset["initial_observation"], 0, set()
        previous_command = np.zeros(14, np.float32)
        for i, ref in enumerate(refs):
            chunk = reader.json(path.parent / ref["path"], ref["sha256"])
            arrays = reader.arrays(chunk["decision_arrays"], list(chunk["decision_arrays"]["arrays"]))
            meta = _decode(chunk["decision_metadata"], arrays)
            bindings = dict(steering_mapping_sha256=interface["mapping_sha256"],
                steering_interface_sha256=interface["control_interface_sha256"], steering_lineage_id=lineage["lineage_id"],
                steering_condition_id=lineage["condition_id"], steering_partition=lineage["partition"])
            if (chunk.get("status") != "complete" or chunk.get("episode_id") != manifest["episode_id"]
                    or chunk.get("chunk_index") != i + 1 or chunk["observation"] != previous
                    or any(meta.get(key) != value for key, value in bindings.items())):
                raise ValueError("Canonical steering transition identity, mapping or chronology differs")
            choice = np.asarray(meta["steering_coefficients"])
            update = meta.get("optimizer_updates")
            if choice.shape != (4,) or choice.dtype != np.float32 or type(update) is not int or update < 0:
                raise ValueError("Require actual requested float32 4D choices and pre-action optimizer identities")
            ctx = chunk["context"]
            if ctx.get("method") not in ("sac", lineage["lineage_id"]):
                raise ValueError("This source prefix must come from the declared NoiseSAC collector")
            acting = {key: meta.get(key, ctx.get(key)) for key in
                      ("policy_checkpoint", "policy_checkpoint_sha256", "policy_sha256")}
            if ctx.get("policy_version") == "bounded_uniform_warmup":
                if not meta.get("source_warmup") or acting["policy_checkpoint"] is not None:
                    raise ValueError("Warmup choice cannot be attributed to an actor checkpoint")
            else:
                version = re.fullmatch(r"sac:update:(\d+):([0-9a-f]{64})", ctx.get("policy_version", ""))
                if version is None or int(version[1]) != update or version[2] != acting["policy_sha256"]:
                    raise ValueError("Actual behavior actor version differs from the pre-action update")
                actor_path = Path(acting["policy_checkpoint"])
                if actor_path.is_absolute() and not actor_path.is_relative_to(reader.root):
                    from .robotwin_history_corpus import parameter_digest
                    from .warmstart import checked_load
                    saved = checked_load(actor_path, acting["policy_checkpoint_sha256"])
                    actual = parameter_digest(saved["state_dict"])
                    if saved.get("kind") != "sac_actor" or actual != saved.get("parameter_sha256") or actual != acting["policy_sha256"]:
                        raise ValueError("External immutable acting archive differs from the actual behavior version")
                    reader.inputs[str(actor_path)] = acting["policy_checkpoint_sha256"]
                else:
                    reader.acting_weights(acting, "sac")
                actor_versions.add(acting["policy_sha256"])
            episode_updates.add(update)
            native, outcome = chunk["native_result"], chunk["learning_result"]
            duration = native["executed_primitive_steps"]
            mask = reader.arrays(native["arrays"], ["valid_action_mask"])["valid_action_mask"]
            done = outcome["terminated"] or outcome["truncated"]
            if (type(duration) is not int or not 1 <= duration <= 32
                    or not np.isin(mask, (0, 1)).all() or int(mask.sum()) != duration
                    or type(outcome["terminated"]) is not bool or type(outcome["truncated"]) is not bool
                    or outcome.get("budget_truncated") is not False
                    or done != (i == len(refs) - 1)):
                raise ValueError("Only complete real native episodes with exact duration/mask may be projected")
            features = reader.arrays(chunk["observation"], ["features"])["features"]
            if features.shape != (interface["native_observation_dim"],) or features.dtype != np.float32:
                raise ValueError("Canonical public frozen feature dimension differs")
            logged_previous = np.asarray(meta["student_previous_agent_command"])
            if (logged_previous.dtype != np.float32 or logged_previous.shape != (14,)
                    or not np.array_equal(logged_previous, previous_command)):
                raise ValueError("Driver's public previous-command field differs from last valid request or reset zero")
            rows.append(dict(observation=np.concatenate((features, logged_previous)), choice=choice, reward=outcome["reward"],
                             done=done, start=i == 0, update=update, duration=duration))
            previous_command = public_requested_command(reader, chunk, metadata=meta, decision=arrays)
            actions += duration
            previous = chunk["next_observation"]
        if (len(episode_updates) != 1 or manifest["completed_chunks"] != len(refs)
                or manifest["executed_primitive_steps"] != actions):
            raise ValueError("Source updates must occur between complete episodes, with exact raw counters")
    for path, expected in reader.inputs.items():
        if digest(path) != expected:
            raise ValueError("Canonical source input changed during projection")
    if require_learning and len(actor_versions) < 2:
        raise ValueError("AD requires at least two actually acting source versions, not only warmup/random choices")
    return dict(observations=np.stack([x["observation"] for x in rows]),
        executed_noise=np.stack([x["choice"] for x in rows]),
        rewards=np.asarray([x["reward"] for x in rows], np.float32),
        dones=np.asarray([x["done"] for x in rows], np.bool_),
        episode_starts=np.asarray([x["start"] for x in rows], np.bool_),
        teacher_steps=np.asarray([x["update"] for x in rows], np.int64),
        primitive_steps=np.asarray([x["duration"] for x in rows], np.int64),
        reset_policy_actions=reset_actions)


def export_prefix(profile, prefix, output, *, objective, expected_hostname):
    """Project actual source coordinates; DPT changes ONLY target_noise at query t."""
    _linux(expected_hostname)
    import numpy as np
    from .vla_history import save_dataset, validate_trial
    lineages = admit_prefix(profile, prefix, objective=objective)
    for lineage in lineages:
        _check_source(lineage, profile, require_learning=objective == "ad")
    teachers, teacher_receipts = {}, {}
    if objective == "dpt":
        for i, condition in enumerate(profile["conditions"]):
            receipt = read_json(prefix["teacher_qualifications"][condition])
            teacher_receipts[condition] = receipt
            if receipt["teacher_kind"] == "noise_sac":
                from .robotwin_history_corpus import parameter_digest
                selected = next(x for x in lineages if x["lineage_id"] == profile["teacher_lineage_ids"][condition])
                state = _source_state(receipt["teacher_recovery"], lineage=selected, profile=profile)
                contract = state["contract"]
                if (contract.get("format") != "frozen_vla_feature_noise_sac_v1"
                        or contract.get("seed") != selected["seed"] or contract.get("action_dim") != 4
                        or contract.get("action_limit") != 1.0 or state["updates"] < 1
                        or contract.get("observation_dim") != profile["interface"]["native_observation_dim"]):
                    raise ValueError("Qualified DPT TRAIN snapshot differs from the exact 4D/bound1 source lineage")
                actor_digest = parameter_digest(state["actor"])
                if receipt.get("actor_parameter_sha256") != actor_digest:
                    raise ValueError("DPT qualification actor tensor digest differs from its full recovery")
                receipt = dict(receipt, labeling_seed=profile["fit"]["seed"] + 81000 + i,
                    labeling_order="sorted lineage IDs; chronological rows; TRAIN and VAL share this exact actor")
                teacher_receipts[condition] = receipt
                teachers[condition] = _teacher(state, seed=receipt["labeling_seed"],
                    deterministic=receipt["label_sampling_mode"] == "deterministic_mean")
            else:
                coefficients = np.asarray(read_json(receipt["oracle"])["coefficients"], np.float32)
                teachers[condition] = lambda observations, c=coefficients: np.tile(c, (len(observations), 1))
    data = {partition: [] for partition in ("train", "validation")}
    ids = {partition: [] for partition in ("train", "validation")}
    actions = 0
    for lineage in lineages:
        trial = canonical_trial(lineage, profile, require_learning=objective == "ad")
        actions += trial.pop("reset_policy_actions")
        primitive = trial.pop("primitive_steps")
        trial["target_noise"] = trial["executed_noise"].copy()
        length, features, coefficients = validate_trial(trial, objective=objective)
        if (features != profile["interface"]["observation_dim"] or coefficients != 4
                or int(trial["dones"].sum()) != profile["prefix_episodes"]
                or trial["teacher_steps"][0] < lineage["initial_updates"]
                or trial["teacher_steps"][-1] > lineage["final_updates"]
                or (np.abs(trial["executed_noise"]) > profile["interface"]["coefficient_limit"]).any()
                or primitive.dtype != np.int64 or primitive.shape != (length,) or (primitive < 1).any()):
            raise ValueError("Raw prefix dimension/bound/chronology/complete-episode/action count differs")
        ends = np.flatnonzero(trial["dones"]) + 1
        starts = np.r_[0, ends[:-1]]
        if max(int(primitive[a:b].sum()) for a, b in zip(starts, ends)) > profile["episode_action_limit"]:
            raise ValueError("Source episode exceeded its predeclared physical action limit")
        if max(int(ends[min(i + 8, len(ends) - 1)] - starts[i]) for i in range(len(starts))) > profile["fit"]["context_length"]:
            raise ValueError("A nine-complete-episode window exceeds Transformer memory")
        actions += int(primitive.sum())
        partition = lineage["partition"]
        ids[partition].append(lineage["lineage_id"])
        if objective == "dpt":
            # Only current public observations enter the final condition teacher.
            labels = teachers[lineage["condition_id"]](trial["observations"])
            if labels.shape != trial["executed_noise"].shape or not np.isfinite(labels).all():
                raise ValueError("DPT teacher returned invalid coefficient query labels")
            trial = dict(trial, target_noise=labels)
        data[partition].append(trial)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    metadata = dict(motor_sha256=profile["interface"]["motor_sha256"],
        feature_contract=profile["interface"]["feature_contract"], noise_contract=profile["interface"],
        profile_sha256=content_hash(profile), prefix_sha256=content_hash(prefix),
        teacher_rule="qualified per-condition actor or oracle; separate stochastic actor-labeling RNG",
        teacher_receipts=teacher_receipts,
        history_rule="actual behavior coefficients/rewards/boundaries; conditions/teacher IDs excluded",
        label_rule="AD actual executed 4D coefficients; DPT replaces current query target only",
        source_actions_count_once=actions, prefix_episodes=profile["prefix_episodes"])
    datasets[objective] = {}
    for partition in ids:
        path = output / f"{objective}-{partition}.npz"
        manifest = save_dataset(path, data[partition], objective=objective,
                                metadata=dict(metadata, trial_ids=ids[partition], partition=partition))
        datasets[objective][partition] = dict(path=str(path.resolve()), sha256=manifest["sha256"],
            manifest_sha256=digest(path.with_suffix(".json")))
    # Recheck every immutable source reference after projection and teacher calls.
    admit_prefix(profile, prefix, objective=objective)
    result = dict(format=ASSEMBLY, completed=True, objective=objective, profile=profile, prefix=prefix, datasets=datasets,
        source_actions_count_once=actions, additional_simulator_actions=0, additional_vla_queries=0,
        fit_admission="objective-specific source gate passed; student robot outcomes remain untested")
    if objective == "dpt":
        groups = {}
        for partition in data:
            for trial, lineage_id in zip(data[partition], ids[partition]):
                lineage = next(x for x in lineages if x["lineage_id"] == lineage_id)
                key = hashlib.sha256(trial["observations"][0].tobytes()).hexdigest()
                groups.setdefault(key, []).append(dict(condition=lineage["condition_id"],
                    label=trial["target_noise"][0].tolist()))
        result["k0_ambiguity"] = dict(exact_public_feature_groups=[dict(feature_sha256=k, rows=v)
            for k, v in groups.items() if len({x["condition"] for x in v}) > 1],
            interpretation="Different qualified condition labels at identical public K0 inputs imply a mixture; stochastic teacher draws also add variability. Gaussian means alone are not recovery.")
    (output / "assembly.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def fit_commands(assembly, *, output, source_commit, expected_hostname):
    validate_profile(assembly["profile"])
    if assembly.get("format") != ASSEMBLY or assembly.get("completed") is not True:
        raise ValueError("Require the actual admitted immutable prefix assembly")
    objective = assembly["objective"]
    admit_prefix(assembly["profile"], assembly["prefix"], objective=objective)
    fit = assembly["profile"]["fit"]
    commands = []
    for objective in (objective,):
        inputs = assembly["datasets"][objective]
        for partition, ref in inputs.items():
            reference({key: ref[key] for key in ("path", "sha256")})
            if digest(Path(ref["path"]).with_suffix(".json")) != ref["manifest_sha256"]:
                raise ValueError("Admitted dataset manifest changed")
        for mode in ("full", "static"):
            command = [sys.executable, "-m", "fluffy_bassoon.meta_rl.vla_history_train",
                "--mode", mode, "--output-dir", str(Path(output) / f"{objective}-{mode}"),
                "--source-commit", source_commit, "--expected-hostname", expected_hostname,
                "--device", "cuda:0", "--max-episodes-per-window", "9", "--checkpoint-selection", "final"]
            for partition in ("train", "validation"):
                command += ["--" + partition + "-data", inputs[partition]["path"],
                            "--" + partition + "-sha256", inputs[partition]["sha256"]]
            for key, value in fit.items():
                option = "max-elapsed-seconds" if key == "max_seconds_per_fit" else key.replace("_", "-")
                command += ["--" + option, str(value)]
            commands.append(command)
    return commands


def load_student(ref, assembly, *, mode, source_commit, device="cpu"):
    """Admit only the exact completed endpoint; a time-capped partial fit has no candidate."""
    if sys.platform != "linux":
        raise ValueError("Load student models only on allocated Linux")
    import torch
    from .vla_history import load_policy
    with torch.random.fork_rng(devices=[]):
        policy, manifest = load_policy(reference(ref), expected_sha256=ref["sha256"], device=device)
    contract, fit = manifest["contract"], assembly["profile"]["fit"]
    objective = assembly["objective"]
    expected_config = dict(observation_dim=assembly["profile"]["interface"]["observation_dim"], noise_dim=4,
        width=fit["width"], layers=fit["layers"], heads=fit["heads"], context_length=fit["context_length"], initial_std=.5)
    if (manifest.get("update") != fit["updates"] or contract.get("mode") != mode
            or contract.get("source_commit") != source_commit or contract.get("config") != expected_config
            or contract.get("objective") != objective or contract.get("seed") != fit["seed"]
            or contract.get("max_context") is not None or contract.get("batch_size") != fit["batch_size"]
            or contract.get("max_episodes_per_window") != 9
            or contract.get("learning_rate") != fit["learning_rate"]
            or contract.get("schedule") != dict(updates=fit["updates"], eval_every=fit["eval_every"],
                validation_batches=fit["validation_batches"], max_elapsed_seconds=fit["max_seconds_per_fit"], checkpoint_selection="final")
            or any(contract.get(key + "_sha256") != assembly["datasets"][objective][key]["sha256"]
                   for key in ("train", "validation"))):
        raise ValueError("Student candidate is partial or differs from the matched admitted source fit")
    for partition in ("train", "validation"):
        ref_data = assembly["datasets"][objective][partition]
        metadata = json.loads(Path(ref_data["path"]).with_suffix(".json").read_text())["metadata"]
        if contract.get(partition + "_metadata") != metadata:
            raise ValueError("Student source normalization/teacher/provenance metadata differs")
    return policy, dict(manifest, inference_checkpoint_sha256=ref["sha256"])


class FrozenCoefficientController:
    """Same generic frozen Gaussian policy, with explicit coefficient coordinates.

    The motor adapter receives these four coefficients and owns the exact fixed
    mapping. Replay only real, fresh support episodes; replayed labels/condition
    IDs never enter the controller. Query results cannot seed a later case.
    """
    def __init__(self, policy, manifest, *, interface, seed, case_id, lesion="intact"):
        import torch
        from .vla_history import FrozenHistoryController
        contract = manifest["contract"]
        if (contract["train_metadata"]["noise_contract"] != interface or lesion not in HISTORY_LESIONS
                or policy.config.noise_dim != 4 or contract["mode"] not in ("full", "static")
                or (contract["mode"] == "static" and lesion != "intact") or not isinstance(case_id, str) or not case_id):
            raise ValueError("Student motor/coefficient/mode or same-weight lesion contract differs")
        self.interface, self.lesion, self.seed = interface, lesion, seed
        self.case_id = case_id
        self.model_identity = manifest.get("inference_checkpoint_sha256", content_hash(manifest))
        rng = torch.Generator(device=next(policy.parameters()).device).manual_seed(seed)
        self.controller = FrozenHistoryController(policy, mode=contract["mode"], generator=rng)

    def act(self, observation):
        import numpy as np
        self.last_proposal=self.controller.act(observation, deterministic=False)
        action=np.clip(self.last_proposal,-self.interface["coefficient_limit"],self.interface["coefficient_limit"])
        self.last_clipped=action!=self.last_proposal
        return action

    def observe(self, executed_coefficients, reward, done):
        import numpy as np
        if (np.abs(executed_coefficients) > self.interface["coefficient_limit"]).any():
            raise ValueError("Observe the actual post-bound coefficients, not RNG or mapped motor commands")
        self.controller.observe(executed_coefficients, 0. if self.lesion == "reward-blind" else reward, done)

    def capture_state(self):
        if self.controller.pending or not self.controller.episode_start:
            raise ValueError("Capture query snapshots only before interaction or after a complete support episode")
        c = self.controller
        return dict(interface_sha256=content_hash(self.interface), mode=c.mode, seed=self.seed,
            case_id=self.case_id, model_identity=self.model_identity, lesion=self.lesion,
            state=deepcopy(c.state), previous_noise=c.previous_noise.copy(),
            previous_reward=c.previous_reward, previous_done=c.previous_done,
            episode_start=c.episode_start, rng=c.generator.get_state().clone())

    def restore_state(self, snapshot, *, query_action_seed):
        if (self.controller.pending or snapshot["interface_sha256"] != content_hash(self.interface)
                or snapshot["mode"] != self.controller.mode or snapshot["seed"] != self.seed):
            raise ValueError("Query memory/RNG snapshot belongs to a different case/interface/mode")
        if (snapshot["case_id"] != self.case_id or snapshot["model_identity"] != self.model_identity
                or (self.lesion != "reset" and snapshot["lesion"] != self.lesion)):
            raise ValueError("A query snapshot cannot cross cases, weights or reconstructed reward lesions")
        c = self.controller
        for key in ("state", "previous_noise", "previous_reward", "previous_done", "episode_start"):
            setattr(c, key, deepcopy(snapshot[key]))
        c.generator.set_state(snapshot["rng"])
        c.generator.manual_seed(query_action_seed)  # Pair query draws without changing support memory.
        if self.lesion == "reset":
            c.reset_trial()

    def prime_support(self, episodes):
        import numpy as np
        if self.controller.pending or self.controller.state is not None or len(episodes) > 8:
            raise ValueError("Prime at most eight fresh complete support episodes into an empty query case")
        order = list(range(len(episodes)))
        rewards = [row["reward"] for episode in episodes for row in episode]
        permutation = list(range(len(rewards)))
        if self.lesion == "shuffle":
            permutation = np.random.default_rng(self.seed + 91000).permutation(permutation).tolist()
            if len(permutation) > 1 and permutation == list(range(len(permutation))):
                permutation = permutation[1:] + permutation[:1]
        position = 0
        for index in order:
            episode = episodes[index]
            if not episode or not episode[-1]["done"] or any(row["done"] for row in episode[:-1]):
                raise ValueError("Shuffle/replay whole actual completed episodes only")
            for row in episode:
                self.controller.act(row["observation"], deterministic=True)  # Replay consumes no action RNG.
                reward = rewards[permutation[position]] if self.lesion == "shuffle" else row["reward"]
                self.observe(row["executed_coefficients"], reward, row["done"])
                position += 1
        if self.lesion == "reset":
            self.controller.reset_trial()  # Clears every memory and previous-feedback field.
        return permutation


class DriverCoefficientStudent:
    """The existing run_complete_episode controller callbacks, with causal 2076 input.

    No new simulator loop or command state. The qualified driver constructs the
    same typed public feature logged for offline projection. It resets that
    suffix at every physical episode boundary while full memory is retained.
    """
    def __init__(self, history, *, control, condition_id):
        self.history, self.control, self.condition_id = history, control, condition_id  # condition is audit-only.
        self.pending_features = self.pending_coefficients = None
        self._episode_active = False
        self._decisions = self._observations = self._steps = 0
        self._before_act = None

    def act(self, native_features):
        import numpy as np
        if self.pending_features is not None:
            raise RuntimeError("Resolve the actual native transition before another student decision")
        if not self._episode_active:
            self._episode_active = True
            self._decisions = self._observations = self._steps = 0
        elif self.history.controller.episode_start:
            raise RuntimeError("Reconcile the previous completed receipt before another physical episode")
        native = np.asarray(native_features, np.float32)
        if native.shape != (self.history.interface["native_observation_dim"],):
            raise ValueError("The existing motor exposes native features, not a pre-augmented or privileged state")
        # The qualified driver is the ONE owner of previous-command projection/reset state.
        augmented = np.asarray(self.control.student_observation(native), np.float32)
        c = self.history.controller
        self.action_rng_before = c.generator.get_state().cpu().numpy().copy()
        self._before_act = {key: deepcopy(getattr(c, key)) for key in
                            ("state", "previous_noise", "previous_reward", "previous_done", "episode_start")}
        choice = self.history.act(augmented)
        self._decisions += 1
        self.pending_features, self.pending_coefficients = native.copy(), choice.copy()
        self.last_augmented_observation = augmented.copy()
        return choice.copy()

    def decision_metadata(self):
        if self.pending_coefficients is None:
            raise RuntimeError("No actual frozen-student decision is pending")
        return dict(steering_coefficients=self.pending_coefficients.copy(),
            steering_lineage_id=self.history.case_id, steering_condition_id=self.condition_id,
            steering_partition="evaluation", optimizer_updates=0,
            policy_version="frozen-history:" + self.history.model_identity,
            student_input_sha256=hashlib.sha256(self.last_augmented_observation.tobytes()).hexdigest(),
            student_action_rng_before=self.action_rng_before.copy(),
            student_gaussian_proposal=self.history.last_proposal.copy(),student_proposal_clipped=self.history.last_clipped.copy())

    def observe(self, observations, coefficients, reward, following, terminated, truncated, primitive_steps, corpus_ref):
        import numpy as np
        if (self.pending_features is None or not np.array_equal(observations, self.pending_features)
                or not np.array_equal(coefficients, self.pending_coefficients) or primitive_steps < 1):
            raise ValueError("Frozen student observation must match its actually executed request")
        done = bool(terminated or truncated)
        self.history.observe(coefficients, reward, done)
        self._observations += 1
        self._steps += primitive_steps
        self._before_act = None
        self.pending_features = self.pending_coefficients = None

    def finish_episode(self, receipt):
        """Use on_episode=adapter.finish_episode; zero-valid proposals are not actions."""
        import numpy as np
        from .robotwin_history_corpus import CorpusReader
        if (not self._episode_active or receipt.get("complete") is not True
                or receipt.get("chunks") != self._decisions or receipt.get("primitive_steps") != self._steps
                or len(receipt.get("corpus_refs", [])) != self._decisions):
            raise ValueError("Reconcile only the exact complete canonical episode and actual valid-step sum")
        c = self.history.controller
        if self.pending_features is not None:
            if self._decisions != self._observations + 1 or self._before_act is None or not c.pending:
                raise ValueError("A pending valid/uncertain proposal cannot be discarded")
            ref = receipt["corpus_refs"][-1]
            reader = CorpusReader(ref["root"])
            chunk = reader.json(reader.path(ref["path"]), ref["sha256"])
            native, outcome = chunk["native_result"], chunk["learning_result"]
            mask = reader.arrays(native["arrays"], ["valid_action_mask"])["valid_action_mask"]
            if (chunk.get("status") != "complete" or native["executed_primitive_steps"] != 0
                    or chunk.get("context", {}).get("trial_id") != receipt.get("spec", {}).get("id")
                    or not np.isin(mask, (0, 1)).all() or mask.any()
                    or not (outcome["terminated"] or outcome["truncated"])):
                raise ValueError("Only the canonical zero-valid terminal call can close without observe")
            # Roll back the unexecuted decision token; retain the actually consumed
            # proposal RNG. Existing valid action/reward feedback is not overwritten.
            for key, value in self._before_act.items():
                setattr(c, key, value)
            c.pending = False
            c.previous_done, c.episode_start = True, True
            self.pending_features = self.pending_coefficients = None
            self._before_act = None
        elif self._decisions != self._observations or c.pending or not c.episode_start:
            raise ValueError("The complete episode has unresolved feedback or no actual boundary")
        self._episode_active = False

    def capture_state(self):
        if self._episode_active:
            raise RuntimeError("Reconcile the complete physical receipt before capturing a query snapshot")
        return self.history.capture_state()

    def restore_state(self, snapshot, *, query_action_seed):
        if self.pending_features is not None or self._episode_active:
            raise RuntimeError("An unresolved native action blocks a new query")
        self.history.restore_state(snapshot, query_action_seed=query_action_seed)


def query_nll(policy, manifest, *, interface, support, query, query_targets, seed, case_id, lesion="intact", primed_state=None):
    """Per-K query-only diagnostic on actual behavior; no teacher label is feedback.

    This is Gaussian coefficient surrogate NLL, not the clipped-action density,
    robot success or a claim that an ambiguous K0 action is recoverable.
    """
    import numpy as np
    import torch
    from torch.distributions import Normal
    controller = FrozenCoefficientController(policy, manifest, interface=interface, seed=seed, case_id=case_id, lesion=lesion)
    if primed_state is None:
        permutation = controller.prime_support(support)
    else:
        controller.restore_state(primed_state, query_action_seed=seed)
        permutation = []
    if len(query) != len(query_targets) or not query:
        raise ValueError("Require an actual query trace and one separate 4D label per query decision")
    losses = []
    with torch.no_grad():
        for row, label in zip(query, query_targets):
            mean = controller.controller.act(row["observation"], deterministic=True)
            std = policy.log_std.clamp(-5, 2).exp().cpu()
            label = np.asarray(label, np.float32)
            if label.shape != (4,) or not np.isfinite(label).all():
                raise ValueError("Invalid current query label")
            losses.append(float(-Normal(torch.as_tensor(mean), std).log_prob(torch.as_tensor(label)).sum()))
            controller.observe(row["executed_coefficients"], row["reward"], row["done"])
    return dict(k=len(support), lesion=lesion, query_decisions=len(query),
        mean_query_nll=float(np.mean(losses)), per_decision_query_nll=losses,
        support_reward_permutation=permutation, scope="conditional Gaussian coefficient surrogate only")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="operation", required=True)
    export = sub.add_parser("export")
    for name in ("profile", "prefix", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    for name in ("profile-sha256", "prefix-sha256", "expected-hostname"):
        export.add_argument("--" + name, required=True)
    export.add_argument("--objective", choices=("ad", "dpt"), required=True)
    fit = sub.add_parser("fit-commands")
    fit.add_argument("--assembly", type=Path, required=True)
    fit.add_argument("--assembly-sha256", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--source-commit", required=True)
    fit.add_argument("--expected-hostname", required=True)
    args = ap.parse_args(argv)
    if args.operation == "export":
        profile = read_json(dict(path=str(args.profile.resolve()), sha256=args.profile_sha256))
        prefix = read_json(dict(path=str(args.prefix.resolve()), sha256=args.prefix_sha256))
        result = export_prefix(profile, prefix, args.output, objective=args.objective, expected_hostname=args.expected_hostname)
    else:
        assembly = read_json(dict(path=str(args.assembly.resolve()), sha256=args.assembly_sha256))
        result = fit_commands(assembly, output=args.output, source_commit=args.source_commit,
                              expected_hostname=args.expected_hostname)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
