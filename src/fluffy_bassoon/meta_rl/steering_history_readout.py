"""Physical four-block student readout through the existing coefficient driver.

No learner, simulator loop, logger or allocator. The execution owner may call
run_readout inside its live qualified runtime/ledger, or use the CLI after
late-binding the same immutable artifacts and one remaining-clock ticket.
"""
from copy import deepcopy
from dataclasses import fields
import argparse
import hashlib
import json
import math
import signal
from pathlib import Path
import socket
import subprocess
import sys
import time

from .steering_history import (DriverCoefficientStudent, FrozenCoefficientController,
    _linux, _source_state, admit_prefix, content_hash, digest, evaluation_plan,
    load_student, query_nll, read_json, reference, validate_profile)


FORMAT = "frozen_vla_4d_physical_student_readout_v1"
PHASE = "student_evaluation"


class FrozenReference:
    """Read-only existing SAC actor or actual fixed oracle, never a learner."""
    def __init__(self, definition, *, seed):
        import numpy as np
        import torch
        self.definition, self.seed = definition, seed
        self.torch_rng = torch.Generator(device=definition["device"]).manual_seed(seed)
        self.numpy_rng = np.random.default_rng(seed)
        self.last = None

    def act(self, observations):
        import numpy as np
        import torch
        features = np.asarray(observations, np.float32)
        d = self.definition
        self.before_torch = self.torch_rng.get_state().cpu().numpy().copy()
        self.before_numpy = deepcopy(self.numpy_rng.bit_generator.state)
        if features.shape != (d["native_dim"],) or not np.isfinite(features).all():
            raise ValueError("Frozen reference only receives current native public features")
        if d["mode"] == "fixed_coefficients":
            action = d["coefficients"].copy()
        elif d["mode"] == "bounded_uniform":
            action = self.numpy_rng.uniform(-1., 1., 4).astype(np.float32)
        else:
            with torch.no_grad():
                value, _ = d["actor"](torch.as_tensor(features[None], device=d["device"]),
                    generator=self.torch_rng, deterministic=d["mode"] == "deterministic_mean")
            action = value[0].cpu().numpy().astype(np.float32, copy=True)
        if action.shape != (4,) or (np.abs(action) > 1.).any() or not np.isfinite(action).all():
            raise ValueError("Reference emitted a different coefficient support")
        self.last = action.copy()
        return action

    def decision_metadata(self):
        return dict(steering_coefficients=self.last.copy(), policy_version="frozen-reference:"+self.definition["identity"],
            steering_lineage_id=self.definition["lineage"], steering_condition_id=self.definition["condition"],
            steering_partition="evaluation", optimizer_updates=self.definition["updates"],
            reference_sampling_mode=self.definition["mode"], reference_artifact=self.definition["reference"],
            reference_action_rng_before=self.before_torch,reference_numpy_rng_before=self.before_numpy)

    def observe(self, *args):
        pass  # Frozen evaluation never observes into replay or updates any parameter.

    def state_dict(self):
        return dict(reference=self.definition["reference"], actor_identity=self.definition["identity"],
            torch_rng=self.torch_rng.get_state().cpu(), numpy_rng=deepcopy(self.numpy_rng.bit_generator.state), seed=self.seed)


def reference_factory(plan, assemblies, *, device):
    import numpy as np
    import torch
    from .robotwin_history_corpus import parameter_digest
    from .vla_noise_sac import SACConfig, SquashedGaussianActor
    from .warmstart import validate_weights
    profile = plan["profile"]
    prefix = next(iter(assemblies.values()))["prefix"]
    definitions = {}
    for condition, descriptor in plan["teachers"].items():
        if "dpt" in assemblies:
            actual = read_json(assemblies["dpt"]["prefix"]["teacher_qualifications"][condition])
            expected_ref = actual["teacher_recovery"] if actual["teacher_kind"] == "noise_sac" else actual["oracle"]
            if descriptor["reference"] != expected_ref or descriptor["sampling_mode"] != actual["label_sampling_mode"]:
                raise ValueError("Physical frozen reference changed the exact DPT label teacher or sampling convention")
        source_method=descriptor.get("source_method",profile["teacher_lineage_ids"][condition])
        slot = next(x for x in prefix["lineages"] if x["lineage_id"] == source_method)
        definition = dict(native_dim=profile["interface"]["native_observation_dim"], condition=condition,
            reference=descriptor["reference"], lineage=slot["lineage_id"], device=device, mode=descriptor["sampling_mode"])
        if descriptor["kind"] == "constant_4d_oracle":
            oracle = read_json(descriptor["reference"])
            if (oracle["interface_sha256"] != profile["interface"]["control_interface_sha256"]
                    or definition["mode"] != "fixed_coefficients"):
                raise ValueError("Physical oracle differs from the same-mapping qualified label teacher")
            definition.update(coefficients=np.asarray(oracle["coefficients"], np.float32),
                              identity=descriptor["reference"]["sha256"], updates=0)
        else:
            state = _source_state(descriptor["reference"], lineage=slot, profile=profile)
            contract = state["contract"]
            if (contract["action_dim"] != 4 or contract["action_limit"] != 1.
                    or contract["observation_dim"] != definition["native_dim"]):
                raise ValueError("Physical source reference differs from the exact 4D/bound1/native-feature policy")
            config = dict(contract["config"]); config["hidden_dims"] = tuple(config["hidden_dims"])
            with torch.random.fork_rng(devices=[]):
                actor = SquashedGaussianActor(definition["native_dim"], 4, SACConfig(**config), 1.)
            validate_weights(actor, state["actor"]); actor.load_state_dict(state["actor"], strict=True)
            definition.update(actor=actor.to(device).eval().requires_grad_(False),
                identity=parameter_digest(state["actor"]), updates=state["updates"])
        definitions[condition] = definition
    return lambda condition, seed: FrozenReference(definitions[condition], seed=seed)


def stationary_reference_factory(plan,assemblies,*,device):
    """Use ONE declared source-VAL-selected policy for both actual conditions."""
    from .steering_history import equivalent_acting
    from .steering_history_prefix import _file_ref
    selection=read_json(plan["blind_selection"])
    p=plan["profile"];prefix=next(iter(assemblies.values()))["prefix"]
    keys=[p["teacher_lineage_ids"][c] for c in p["conditions"]]
    if (selection.get("selected_from")!="source_validation" or selection.get("cross_cases_opened") is not False
            or set(selection["final_validation_successes"])!=set(keys)
            or selection["selected_source_method"]!=max(keys,key=lambda key:selection["final_validation_successes"][key])):
        raise ValueError("Stationary reference must be the single pre-query source-validation choice")
    chosen=selection["selected_source_method"];controller=selection["controller_ref"]
    slot=next(x for x in prefix["lineages"] if x["lineage_id"]==chosen)
    scores={}
    for ref in selection["source_validation_refs"]:
        record=read_json(_file_ref(ref));methods={r["spec"]["method"] for r in record["results"]}
        if record.get("complete") is not True or len(methods)!=1 or len(record["results"])!=p["teacher_gate"]["paired_cases"]:
            raise ValueError("Fixed blind selection changed its actual source-validation panel")
        key=methods.pop();scores[key]=sum(r["success"] for r in record["results"])
        if key==chosen:
            proof=equivalent_acting(record["source_checkpoint"],_file_ref(controller),lineage=slot,profile=p)
            if (proof["actor_parameter_sha256"]!=controller["policy_sha256"] or proof["updates"]!=controller["optimizer_updates"]
                    or proof["sampling_mode"]!=controller["sampling_mode"]):
                raise ValueError("Blind selection file differs from the actual validated acting state/mode")
    if scores!=selection["final_validation_successes"]:
        raise ValueError("Blind controller was not selected from the actual source outcomes")
    descriptor=dict(kind="noise_sac",sampling_mode=controller["sampling_mode"],reference=_file_ref(controller),source_method=chosen,
                    status="one fixed source-validation-selected blind controller")
    temporary=dict(plan,teachers={condition:deepcopy(descriptor) for condition in p["conditions"]})
    # Disable the privileged per-condition label-teacher equality check for this
    # separate control; the selected source actor is verified above instead.
    return reference_factory(temporary,{"ad":next(iter(assemblies.values()))},device=device)


def matrix_schedule(profile, objectives):
    objectives = tuple(objectives)
    if not objectives or len(set(objectives)) != len(objectives) or set(objectives) - {"ad", "dpt"}:
        raise ValueError("Predeclare the admitted objective subset, never substitute an unqualified model")
    schedule = evaluation_plan(profile)
    def include(row):
        prefix = row["policy"].split("-", 1)[0]
        return prefix not in ("ad", "dpt") or prefix in objectives
    schedule["supports"] = [r for r in schedule["supports"] if include(r)]
    schedule["cards"] = [r for r in schedule["cards"] if include(r)]
    episodes = sum(r["episodes"] for r in schedule["supports"]) + len(schedule["cards"])
    donors = sum(r["episodes"] for r in schedule["supports"] if "task" in r)
    schedule["maximum_complete_episodes"] = episodes
    schedule["maximum_actions"] = ((episodes-donors)*profile["episode_action_limit"]
        + donors*profile["wrong_task_action_limit"] + episodes*profile["reset_policy_action_bound"])
    schedule["objectives"] = list(objectives)
    return schedule


def validate_plan(plan):
    if plan.get("format") != FORMAT or plan.get("schedule") != matrix_schedule(plan["profile"], plan["objectives"]):
        raise ValueError("Only the exact four-block shared-prefix grid/paired seeds is admitted")
    keys = {objective+"-"+mode for objective in plan["objectives"] for mode in ("full", "static")}
    if set(plan["models"]) != keys or set(plan["assemblies"]) != set(plan["objectives"]):
        raise ValueError("Require both completed full/static models for every admitted objective")
    if set(plan["teachers"]) != set(plan["profile"]["conditions"]):
        raise ValueError("Bind one reference controller per condition before any query is opened")
    return plan


def load_inputs(plan, *, device):
    """Load only exact completed fits and their original admitted source artifacts."""
    validate_plan(plan)
    profile, students, assemblies = plan["profile"], {}, {}
    source_rows = None
    for objective in plan["objectives"]:
        assembly = read_json(plan["assemblies"][objective])
        if assembly["objective"] != objective or assembly["profile"] != profile:
            raise ValueError("Readout changed the actual fit/data profile")
        admit_prefix(profile, assembly["prefix"], objective=objective)
        rows = [(r["lineage_id"], r["initial_recovery"], r["final_recovery"], r["episodes"])
                for r in assembly["prefix"]["lineages"]]
        if source_rows is not None and rows != source_rows:
            raise ValueError("AD/DPT source contexts/exposure differ")
        source_rows = rows
        assemblies[objective] = assembly
        for mode in ("full", "static"):
            key = objective+"-"+mode
            students[key] = load_student(plan["models"][key], assembly, mode=mode,
                source_commit=plan["fit_commit"], device=device)
    return students, assemblies


def bind_plan(profile, *, assemblies, models, fit_commit, condition_contract, blind_selection=None, teachers=None):
    """Late-bind actual completed artifacts before opening the four query blocks."""
    validate_profile(profile)
    objectives = [x for x in ("ad", "dpt") if x in assemblies]
    values = {key: read_json(value) for key, value in assemblies.items()}
    prefix = next(iter(values.values()))["prefix"]
    blind_selection=blind_selection or prefix.get("blind_selection")
    if blind_selection is None:raise ValueError("Actual fixed source-validation blind selection is still pending")
    seeds, observations, feature_hashes = set(), set(), set()
    def exclude(reset):
        seeds.add(reset["actual_seed"])
        observations.add(reset["observation_sha256"])
        feature_hashes.add(reset["features_sha256"])
    for published in prefix["source_publications"].values():
        for reset in published["actual_resets"]:
            exclude(reset)
        from .steering_history_prefix import _file_ref
        for receipt in published["source_validation_receipts"].values():
            for row in read_json(_file_ref(receipt))["results"]:
                exclude(row["reset_info"])
    if "condition_cross" in prefix:
        from .steering_history_prefix import _file_ref
        for row in read_json(_file_ref(prefix["condition_cross"]["result"]))["records"]:
            exclude(row["reset_info"])
    if teachers is None:
        teachers = {}
        for condition in profile["conditions"]:
            if "dpt" in values:
                qualification = read_json(values["dpt"]["prefix"]["teacher_qualifications"][condition])
                kind = qualification["teacher_kind"]
                teachers[condition] = dict(kind=kind, sampling_mode=qualification["label_sampling_mode"],
                    reference=qualification["teacher_recovery"] if kind == "noise_sac" else qualification["oracle"],
                    status="qualified exact DPT label teacher")
            else:
                selected = next(r for r in prefix["lineages"] if r["lineage_id"] == profile["teacher_lineage_ids"][condition])
                teachers[condition] = dict(kind="noise_sac", sampling_mode="sampled_gaussian",
                    reference=selected["final_recovery"], status="frozen source candidate; no expert competence claim")
    result = dict(format=FORMAT, profile=deepcopy(profile), objectives=objectives,
        schedule=matrix_schedule(profile, objectives), assemblies=deepcopy(assemblies), models=deepcopy(models),
        fit_commit=fit_commit, condition_contract=deepcopy(condition_contract), teachers=deepcopy(teachers),
        blind_selection=deepcopy(blind_selection),
        source_actual_seeds=sorted(seeds), source_initial_observation_sha256=sorted(observations),
        source_initial_features_sha256=sorted(feature_hashes),
        locked32_confirmation="not opened by this mechanism pilot")
    return validate_plan(result)


def _new_history(students, model, profile, case_id, seed, lesion="intact"):
    policy, manifest = students[model]
    return FrozenCoefficientController(policy, manifest, interface=profile["interface"],
        seed=seed, case_id=case_id, lesion=lesion)


def _replay_rows(trace):
    rows = deepcopy(trace["rows"])
    # A verified zero-valid terminal is a real boundary after the last actual
    # action. Project that boundary only; no action/reward/observation is invented.
    if rows and not rows[-1]["done"]:
        if trace.get("zero_valid_terminal") is not True:
            raise ValueError("Support replay lacks its actual completed boundary")
        rows[-1]["done"] = True
    return rows


def prime_history(history, traces):
    """Rebuild actual public support, including an empty zero-valid boundary."""
    if any(not t["rows"] for t in traces):
        import numpy as np
        rewards=[r["reward"] for t in traces for r in t["rows"]]
        permutation=list(range(len(rewards)))
        if history.lesion=="shuffle":
            permutation=np.random.default_rng(history.seed+91000).permutation(permutation).tolist()
            if len(permutation)>1 and permutation==list(range(len(permutation))):permutation=permutation[1:]+permutation[:1]
        position=0
        for trace in traces:
            rows = _replay_rows(trace)
            if rows:
                # Prime one contiguous completed episode without resetting memory.
                for row in rows:
                    history.controller.act(row["observation"], deterministic=True)
                    reward=rewards[permutation[position]] if history.lesion=="shuffle" else row["reward"]
                    history.observe(row["executed_coefficients"], reward, row["done"]);position+=1
            elif trace.get("zero_valid_terminal") is True:
                history.controller.previous_done = history.controller.episode_start = True
            else:
                raise ValueError("An empty support must have a canonical zero-valid terminal receipt")
        if history.lesion == "reset":
            history.controller.reset_trial()
        return permutation
    return history.prime_support([_replay_rows(t) for t in traces])


def admit_readout_budget(budget, schedule):
    """Shared CLI/driver preflight against the original phase-wide allowance."""
    if budget.snapshot()["unresolved_decisions"]:
        raise ValueError("Unresolved prior physical work blocks a new readout runtime")
    # The canonical API requires an actual method even for phase-wide checks.
    # Evaluation has no per-method cap; all scheduled policies share this slice.
    method = schedule["supports"][0]["policy"]
    if budget.remaining(phase=PHASE, method=method) < schedule["maximum_actions"]:
        raise ValueError("The original shared phase ledger cannot admit this complete student grid")


def run_readout(plan, *, control, budget, students, teacher_factory, stationary_factory, select_condition,
                output, run_id, source_commit, guard, checkpoint, episode_runner=None):
    """Execute the bound grid using the SAME actual complete-episode runner.

    guard.admit_episode() reserves the existing remaining wall/cleanup budget;
    guard() checks it and shared storage between chunks. checkpoint is the
    owner's existing (event,payload,immutable=...) full recovery sink. No new
    ledger is created. Source collection must yield its existing serial ledger.
    """
    import numpy as np
    if episode_runner is None:
        from .robotwin_steering_screen import run_complete_episode
        episode_runner = run_complete_episode
    from .robotwin_steering_screen import paired_reset_fields, PHASE_CAPS
    validate_plan(plan)
    profile, schedule = plan["profile"], plan["schedule"]
    if (budget.header["total_actions"] != 300000 or budget.header["phase_caps"] != PHASE_CAPS
            or budget.header["campaign_id"] != profile["shared_reservation"]["pilot_id"]
            or set(budget.header["source_caps"]) != {r["lineage_id"] for r in profile["lineage_slots"]}
            or run_id != budget.run_id or source_commit != budget.source_commit
            or budget.header["contract"].get("scope_sha256") != profile["scope_sha256"]):
        raise ValueError("The original shared phase ledger cannot admit this complete student grid")
    admit_readout_budget(budget, schedule)
    if set(students) != set(plan["models"]):
        raise ValueError("Loaded students differ from the exact completed model bindings")
    for task, horizon in ((profile["source_task"], profile["episode_action_limit"]),
                          (profile["wrong_task"], profile["wrong_task_action_limit"])):
        if control.episode_limit_for_task(task) != horizon:
            raise ValueError("Real native task horizon differs from predeclared readout action accounting")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
    state = dict(format=FORMAT, plan=plan, completed_supports={}, completed_queries=[], pending=None,
                 source_commit=source_commit, query_feedback_reused=False)
    traces, snapshots, reset_signatures, paired_records = {}, {}, {}, {}
    source_ids = set(plan["source_actual_seeds"])
    source_obs = set(plan["source_initial_observation_sha256"])
    source_features = set(plan.get("source_initial_features_sha256", []))

    def persist(event):
        guard()
        # Public traces and detached memory/RNG snapshots are recovery material,
        # while all original motor commands/images remain in the canonical corpus.
        return checkpoint(event, dict(state=deepcopy(state), traces=deepcopy(traces),
            snapshots=deepcopy(snapshots), reset_signatures=deepcopy(reset_signatures),
            paired_records=deepcopy(paired_records)), immutable=False)

    def one(spec, adapter):
        guard.admit_episode()
        if control.active:
            raise RuntimeError("Never change a condition or reset during an active physical episode")
        select_condition(spec["condition_id"])
        pair_key = spec["task"]+":"+str(spec["reset_seed"])
        if pair_key in paired_records:
            # Only the original effective instruction/reset identity is reused;
            # no outcome or query memory is passed to another case/controller.
            spec = dict(spec, **paired_reset_fields(paired_records[pair_key]))
        rows = []
        state["pending"] = deepcopy(spec)
        persist("student_episode_intent")
        def before(current, actual_spec):
            info = current.last_reset_info
            if (info["actual_seed"] in source_ids or info["observation_sha256"] in source_obs
                    or info["features_sha256"] in source_features):
                raise ValueError("A student reset overlaps actual source, validation or A2 evidence")
            signature = {k: info[k] for k in ("actual_seed", "observation_sha256", "features_sha256")}
            key = actual_spec["task"]+":"+str(actual_spec["reset_seed"])
            if key in reset_signatures and reset_signatures[key] != signature:
                raise ValueError("Paired query/support/condition resets changed the actual initial public input")
            reset_signatures[key] = signature
            guard()
        def chunk(row):
            if isinstance(adapter, DriverCoefficientStudent):
                rows.append(dict(observation=adapter.last_augmented_observation.copy(),
                    executed_coefficients=row["executed_coefficients"].copy(), reward=float(row["reward"]),
                    done=bool(row["terminated"] or row["truncated"]),proposal_clipped=bool(adapter.history.last_clipped.any())))
                c=adapter.history.controller
                state["current_controller"]=dict(case_id=adapter.history.case_id,model=spec["method"],
                    state=deepcopy(c.state),previous_noise=c.previous_noise.copy(),previous_reward=c.previous_reward,
                    previous_done=c.previous_done,episode_start=c.episode_start,pending=c.pending,
                    action_rng=c.generator.get_state().cpu())
            elif adapter is not None and hasattr(adapter,"state_dict"):
                state["current_controller"]=adapter.state_dict()
            state["current_valid_rows"]=deepcopy(rows)
            guard()
        kwargs = dict(run_id=run_id, phase=PHASE, controller=adapter, before_step=before, on_chunk=chunk)
        if isinstance(adapter, DriverCoefficientStudent):
            kwargs["on_episode"] = adapter.finish_episode
        record = episode_runner(control, spec, budget, **kwargs)
        guard.maximum_episode_seconds = max(guard.maximum_episode_seconds, 1.5*record["wall_seconds"])
        compact = {key: value for key, value in record.items() if key != "control_recovery"}
        paired_records.setdefault(pair_key, deepcopy(compact))
        _atomic_json(output/(spec["id"]+".json"), compact)
        zero_terminal = bool(rows and not rows[-1]["done"]) or (not rows and compact["primitive_steps"] == 0)
        # DriverCoefficientStudent reconciles a zero-valid ending in on_episode.
        trace = dict(rows=rows, zero_valid_terminal=zero_terminal, episode=compact)
        state["pending"] = None
        state["last_completed_physical_episode"]=deepcopy(compact)
        state.pop("current_controller",None);state.pop("current_valid_rows",None)
        persist("student_episode_complete")
        return trace

    def spec_for(row, *, suffix, reset_seed, motor_seed, task=None):
        task = task or profile["source_task"]
        return dict(id=suffix, task=task, reset_seed=reset_seed, motor_seed=motor_seed,
            arm="native" if row["policy"] == "native" else profile["interface"]["arm"],
            coefficients=[0.]*4, method=row["policy"], condition_id=row["condition_id"],
            horizon=control.episode_limit_for_task(task), steering_partition="evaluation", phase=PHASE)

    try:
        for index, support in enumerate(schedule["supports"]):
            policy = support["policy"]
            condition = support.get("condition_id", profile["conditions"][0])
            support = dict(support, condition_id=condition)
            if policy.endswith("-full"):
                history = _new_history(students, policy, profile, support["id"], support["action_seed"])
                adapter = DriverCoefficientStudent(history, control=control, condition_id=condition)
                states = {0: adapter.capture_state()}
            else:
                adapter = stationary_factory(condition, support["action_seed"])
                states = {}
            episodes = []
            for i, reset_seed in enumerate(support["reset_seeds"]):
                spec = spec_for(support, suffix=f"support-{index:02d}-{i}", reset_seed=reset_seed,
                    motor_seed=(support["action_seed"]+i*997) % 2**32, task=support.get("task"))
                episodes.append(one(spec, adapter))
                if states is not None and isinstance(adapter, DriverCoefficientStudent):
                    states[i+1] = adapter.capture_state()
            traces[support["id"]] = episodes
            snapshots[support["id"]] = states
            state["completed_supports"][support["id"]] = [t["episode"]["spec"]["id"] for t in episodes]
            persist("student_support_prefix_complete")
        # ALL support is complete before any query. Query feedback never changes it.
        for index, card in enumerate(schedule["cards"]):
            key, k, lesion = card["policy"], card["k"], card["lesion"]
            support = traces.get(card["support_id"], [])[:k]
            if key in students:
                # Snapshot IDs bind to the actual full model/block. For lesions,
                # rebuild public history under the query case rather than alias weights.
                history_case = card["support_id"] if lesion == "intact" and key.endswith("-full") else card["case_id"]
                initial_seed = snapshots[card["support_id"]][k]["seed"] if lesion == "intact" and key.endswith("-full") else card["action_seed"]
                history = _new_history(students, key, profile, history_case, initial_seed,
                    lesion=lesion if key.endswith("-full") else "intact")
                if lesion == "intact" and key.endswith("-full"):
                    snapshot = snapshots[card["support_id"]][k]
                    # Support/query RNG seeds differ deliberately; preserve the
                    # original case seed binding while pairing only query draws.
                    history.restore_state(snapshot, query_action_seed=card["action_seed"])
                elif key.endswith("-full"):
                    prime_history(history, support)
                    history.controller.generator.manual_seed(card["action_seed"])
                adapter = DriverCoefficientStudent(history, control=control, condition_id=card["condition_id"])
            elif key == "native":
                adapter = None
            else:
                adapter = stationary_factory(card["condition_id"], card["action_seed"])
            spec = spec_for(card, suffix=f"query-{index:03d}", reset_seed=card["reset_seed"], motor_seed=card["action_seed"])
            trace = one(spec, adapter)
            result = dict(card=card, actual_outcome=trace["episode"], query_feedback_reused=False,
                controller_semantics="one fixed source-validation-selected blind reference" if key=="stationary-reference" else key,
                support_provenance=[dict(id=t["episode"]["spec"]["id"],task=t["episode"]["spec"]["task"],
                    condition_audit_only=t["episode"]["spec"]["condition_id"]) for t in support])
            if key in students and trace["rows"]:
                labeler = teacher_factory(card["condition_id"], (card["action_seed"]+310001) % 2**32)
                labels = [labeler.act(row["observation"][:profile["interface"]["native_observation_dim"]]) for row in trace["rows"]]
                result["first_query_diagnostic"]=dict(public_feature_sha256=hashlib.sha256(trace["rows"][0]["observation"].tobytes()).hexdigest(),
                    reference_label=labels[0].tolist(),sampling_mode=plan["teachers"][card["condition_id"]].get("sampling_mode","fixture"),
                    scope="first query decision; later within-query feedback may identify the condition")
                nll_history=_new_history(students,key,profile,history_case,card["action_seed"],lesion=lesion if key.endswith("-full") else "intact")
                used=support if key.endswith("-full") else []
                permutation=prime_history(nll_history,used)
                result["query_nll"] = query_nll(students[key][0], students[key][1], interface=profile["interface"],
                    support=[_replay_rows(t) for t in used],query=trace["rows"],query_targets=labels,
                    seed=card["action_seed"],case_id=history_case,lesion=lesion if key.endswith("-full") else "intact",
                    primed_state=nll_history.capture_state())
                result["query_nll"]["support_reward_permutation"]=permutation
                result["query_nll"]["nominal_query_k"]=k
                std=students[key][0].log_std.detach().clamp(-5,2).exp().cpu().numpy()
                result["student_sampling"]=dict(deterministic=False,standard_deviation=std.tolist(),
                    raw_gaussian_entropy=float(np.log(std*np.sqrt(2*np.pi*np.e)).sum()),
                    clipped_query_decisions=sum(r["proposal_clipped"] for r in trace["rows"]))
                if lesion=="shuffle":
                    rewards=[r["reward"] for t in support for r in t["rows"]]
                    result["reward_shuffle_changed_values"]=any(rewards[i]!=rewards[j] for i,j in enumerate(permutation))
            state["completed_queries"].append(result)
            _atomic_json(output/f"query-result-{index:03d}.json", result)
            persist("student_query_complete")
        state["status"] = "complete_four_block_mechanism_readout"
        groups={}
        for row in state["completed_queries"]:
            if row["card"]["k"]==0 and "first_query_diagnostic" in row:
                d=row["first_query_diagnostic"]
                groups.setdefault(d["public_feature_sha256"],[]).append(dict(condition_audit_only=row["card"]["condition_id"],
                    policy=row["card"]["policy"],label=d["reference_label"],sampling_mode=d["sampling_mode"]))
        state["k0_label_groups"]=groups
        state["ledger"] = budget.snapshot()
        state["scientific_scope"] = "four blocks only; independent32-case confirmation and seeds remain outstanding"
        checkpoint("student_readout_complete", state, immutable=True)
        _atomic_json(output/"result.json", state)
        return state
    except BaseException as error:
        state["status"] = "incomplete_preserved"
        state["failure"] = dict(type=type(error).__name__, message=str(error))
        # Do not clear pending episodes or replay unknown physical work.
        checkpoint("student_readout_interrupted", dict(state=state, traces=traces, snapshots=snapshots,
            reset_signatures=reset_signatures, paired_records=paired_records), immutable=True)
        _atomic_json(output/"result.json", {k: v for k, v in state.items()
            if k not in ("completed_supports","current_controller","current_valid_rows")})
        raise


def validate_allocation(plan, allocation, scope, *, source, hostname, output):
    """The existing owner supplies a slice of ONE shared8h/300k/64GiB phase."""
    profile = plan["profile"]
    if (allocation.get("phase") != PHASE or allocation.get("source_commit") != source
            or allocation.get("scope_sha256") != profile["scope_sha256"]
            or allocation.get("hostname") != hostname or allocation.get("gpu_count") != 1
            or Path(allocation["output"]).resolve() != Path(output).resolve()
            or scope.get("phase_id") != profile["shared_reservation"]["pilot_id"]
            or scope["phase_limits"]["cumulative_gpu_seconds"] != 28800
            or scope["phase_limits"]["all_in_simulator_actions"] != 300000
            or scope["phase_limits"]["outputs_gib"] != 64):
        raise ValueError("Actual readout ticket differs from the existing shared phase/source/host/output")
    counters = [allocation[key] for key in
        ("reserved_gpu_seconds", "prior_phase_gpu_seconds", "other_reserved_phase_gpu_seconds")]
    if any(type(x) is not int or x < 0 for x in counters) or counters[0] < 1 or sum(counters) > 28800:
        raise ValueError("Readout allocation would replenish or exceed the cumulative8h grant")
    if (not math.isfinite(allocation["allocation_started_unix"]) or not math.isfinite(allocation["driver_deadline_unix"])
            or allocation["driver_deadline_unix"]-allocation["allocation_started_unix"] > counters[0]
            or allocation["cleanup_reserve_seconds"] < 60 or allocation["maximum_episode_update_seconds"] != 0
            or not Path(output).resolve().is_relative_to(Path(allocation["phase_output_root"]).resolve())):
        raise ValueError("Use the original finite owner clock, cleanup reserve and shared output root")
    ledger = allocation["action_ledger"]
    from .robotwin_steering_screen import PHASE_CAPS, SOURCE_METHODS
    if (ledger["total_actions"] != 300000 or ledger["phase_caps"] != PHASE_CAPS
            or set(ledger["source_caps"]) != SOURCE_METHODS
            or ledger["contract"].get("scope_sha256") != profile["scope_sha256"]
            or not (Path(ledger["root"])/"events/0000000000.json").is_file()):
        raise ValueError("Never create/replenish a ledger for student evaluation")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "scope", "allocation", "joint-conventions"):
        ap.add_argument("--"+name, type=Path, required=True)
        ap.add_argument("--"+name+"-sha256", required=True)
    for name in ("output", "model-path", "norm-stats", "rlinf-root", "robotwin-root", "gpu-probe"):
        ap.add_argument("--"+name, type=Path, required=True)
    for name in ("source-commit", "run-id", "expected-hostname"):
        ap.add_argument("--"+name, required=True)
    args = ap.parse_args(argv)
    _linux(args.expected_hostname)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Run the exact clean qualified commit in the actual allocation")
    plan = read_json(dict(path=str(args.plan.resolve()), sha256=args.plan_sha256))
    scope = read_json(dict(path=str(args.scope.resolve()), sha256=args.scope_sha256))
    allocation = read_json(dict(path=str(args.allocation.resolve()), sha256=args.allocation_sha256))
    validate_plan(plan)
    if plan["profile"]["scope_sha256"] != args.scope_sha256:
        raise ValueError("Readout source profile did not bind the original shared scope")
    validate_allocation(plan, allocation, scope, source=source, hostname=socket.gethostname(), output=args.output)
    cpu = read_json(allocation["CPU_qualification_receipt"])
    if cpu.get("status") != "passed" or cpu.get("source_commit") != source:
        raise ValueError("Actual allocated CPU qualification of this exact readout entry point is required")
    probe = json.loads(args.gpu_probe.read_text())
    if probe.get("status") != "passed" or probe.get("hostname") != socket.gethostname() or len(probe.get("devices", [])) != 1:
        raise ValueError("Require the owner's actual one-GPU allocation CUDA probe")
    students, assemblies = load_inputs(plan, device="cpu")
    from .robotwin_steering_source import RuntimeGuard
    from .steering_condition_collection import BoundedTeacherRecovery
    from .robotwin_steering_screen import verify_source_conventions
    from fluffy_bassoon.rl.robotwin_steering import CoefficientSteering, validate_interface
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv, load_benchmark_episode_limits
    from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter, _atomic_json
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, CalibrationSpec
    binding = read_json(next(iter(assemblies.values()))["prefix"]["control_qualification"])
    interface = validate_interface(read_json(binding["control_interface"]))
    if (digest(args.model_path/"model.safetensors") != interface["motor_sha256"]
            or digest(args.norm_stats) != interface["normalization_sha256"]
            or interface["joint_convention_sha256"] != args.joint_conventions_sha256):
        raise ValueError("Motor FILE/normalization/joint conventions differ from the actual qualified interface")
    conventions = read_json(dict(path=str(args.joint_conventions.resolve()), sha256=args.joint_conventions_sha256))
    verify_source_conventions(conventions, interface, dict(robotwin=args.robotwin_root, rlinf=args.rlinf_root))
    guard = RuntimeGuard(allocation)
    guard()
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output/"admission.json", dict(source_commit=source, plan_sha256=args.plan_sha256,
        allocation_sha256=args.allocation_sha256, scope_sha256=args.scope_sha256))
    signal.signal(signal.SIGTERM, lambda *_: setattr(guard, "stopping", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(guard, "stopping", True))
    native_keys = {f.name for f in fields(CalibrationSpec)}
    distribution = CalibrationDistribution([CalibrationSpec(**{k: row[k] for k in native_keys})
        for row in plan["condition_contract"]["conditions"]])
    if content_hash(distribution.contract()) != content_hash(plan["condition_contract"]):
        raise ValueError("Hidden physical condition contract changed before runtime")
    import torch
    first = next(iter(assemblies.values()))["prefix"]["lineages"][0]
    outer = torch.load(reference(first["final_recovery"]), map_location="cpu", weights_only=False)
    if content_hash(outer["control"]["mapping"]["condition_audit_contract"]) != content_hash(distribution.contract()):
        raise ValueError("Readout hidden calibration differs from actual collected source conditions")
    limits = load_benchmark_episode_limits(args.robotwin_root, (plan["profile"]["source_task"], plan["profile"]["wrong_task"]))
    ledger, env = allocation["action_ledger"], None
    terminal=dict(source_commit=source,plan_sha256=args.plan_sha256,status="starting",cleanup=None)
    try:
        with CampaignBudget(ledger["root"], campaign_id=ledger["campaign_id"], total_actions=300000,
                source_caps=ledger["source_caps"], phase_caps=ledger["phase_caps"], contract=ledger["contract"],
                run_id=args.run_id, source_commit=source) as budget:
            admit_readout_budget(budget, plan["schedule"])
            for policy,_ in students.values():policy.to("cuda:0")
            env = RobotwinNoiseEnv(model_path=args.model_path, norm_stats=args.norm_stats,
                robotwin_root=args.robotwin_root, rlinf_root=args.rlinf_root, output_dir=args.output/"runtime",
                action_horizon=32, num_steps=5, task_episode_limits=limits)
            control = CoefficientSteering(env, interface, post_command_transform=distribution,
                condition_contract=distribution.contract(), interface_sha256=binding["control_interface"]["sha256"])
            control.set_coefficients(plan["profile"]["interface"]["arm"], [0.]*4)
            if control.mapping_sha256() != plan["profile"]["interface"]["mapping_sha256"]:
                raise ValueError("Physical readout selected arm/basis changed")
            writer = RobotwinCorpusWriter(args.output/"corpus", dict(schema=FORMAT, source_commit=source,
                plan_sha256=args.plan_sha256, role="source_four_block_frozen_student_mechanism"))
            control.attach_corpus(writer)
            qualified = read_json(binding["driver_qualification"])
            if env.frozen_fingerprint() != qualified["frozen_after"]:
                raise ValueError("Actual frozen tensor fingerprint differs from A1")
            sink = BoundedTeacherRecovery(args.output/"recovery", guard=guard, control=control)
            factory = reference_factory(plan, assemblies, device="cuda:0")
            stationary=stationary_reference_factory(plan,assemblies,device="cuda:0")
            signs={spec.condition_id:spec.sign for spec in distribution.specs.values()}
            if set(signs)!=set(plan["profile"]["conditions"]):raise ValueError("Readout condition identity changed")
            select = lambda condition: distribution.select(signs[condition])
            run_readout(plan, control=control, budget=budget, students=students, teacher_factory=factory,
                stationary_factory=stationary,select_condition=select, output=args.output/"matrix", run_id=args.run_id, source_commit=source,
                guard=guard, checkpoint=sink)
            if env.frozen_fingerprint() != qualified["frozen_after"]:
                raise ValueError("Frozen motor changed during the readout")
            terminal.update(status="complete",actions=budget.snapshot())
    except BaseException as error:
        terminal.update(status="failed",failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        cleanup_error=None
        if env is not None:
            try:env.close();terminal["cleanup"]="closed"
            except BaseException as error:
                cleanup_error=error;terminal.update(status="failed_cleanup",cleanup=dict(type=type(error).__name__,message=str(error)))
        else:terminal["cleanup"]="no_environment_constructed"
        _atomic_json(args.output/"runtime-result.json",terminal)
        if cleanup_error is not None:raise cleanup_error


if __name__ == "__main__":
    main()
