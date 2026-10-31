"""Deterministic actual collector publications -> immutable history PREFIX.

No collector or logger. Canonical raw episode/recovery files remain unchanged;
objective qualification stays separate from completion of a data prefix.
"""
from copy import deepcopy
import json
from pathlib import Path

from .steering_history import (PREFIX, _linux, _source_state, content_hash,
                               digest, equivalent_acting, read_json, reference, validate_profile)


def _file_ref(value):
    return {key: value[key] for key in ("path", "sha256", "root", "retention", "size_bytes") if key in value}


def _physical_case(record, acting):
    from .robotwin_history_corpus import CorpusReader, _decode, parameter_digest
    from .warmstart import checked_load
    path = reference(record["episode_manifest_ref"])
    manifest = read_json(record["episode_manifest_ref"])
    reset = json.loads((path.parent / manifest["reset"]).read_text())
    reader=CorpusReader(record["episode_manifest_ref"]["root"])
    info = reset["reset_info"]
    success, steps = False, 0
    for ref in manifest["chunks"]:
        chunk = read_json(dict(path=str(path.parent / ref["path"]), sha256=ref["sha256"]))
        if chunk["status"] != "complete":
            raise ValueError("Validation cannot qualify an unresolved physical chunk")
        arrays=reader.arrays(chunk["decision_arrays"],list(chunk["decision_arrays"]["arrays"]))
        meta=_decode(chunk["decision_metadata"],arrays)
        version=chunk["context"]["policy_version"]
        if acting["updates"]==0:
            if version!="bounded_uniform_warmup" or meta.get("source_warmup") is not True or meta.get("optimizer_updates")!=0:
                raise ValueError("Initial validation did not execute the actual bounded-uniform behavior")
        else:
            expected=f"sac:update:{acting['updates']}:{acting['actor_parameter_sha256']}"
            if version!=expected or meta.get("source_warmup") is not False or meta.get("optimizer_updates")!=acting["updates"]:
                raise ValueError("Validation executed a different actual source actor/version")
            archive=checked_load(meta["policy_checkpoint"],meta["policy_checkpoint_sha256"])
            if (archive.get("kind")!="sac_actor" or parameter_digest(archive["state_dict"])!=acting["actor_parameter_sha256"]):
                raise ValueError("Actual validation pure-actor archive differs from the executed input recovery")
        success |= chunk["native_result"]["success"]
        steps += chunk["native_result"]["executed_primitive_steps"]
    if success != record["success"] or steps != record["primitive_steps"]:
        raise ValueError("Physical validation outcome/cost differs from canonical raw chunks")
    return content_hash(dict(task=info["task"], actual_seed=info["actual_seed"],
                             initial_observation_sha256=reset["initial_observation"]["content_sha256"]))


def derive_source_qualifications(profile, publications, lineages, *, output):
    """Bind EXISTING fixed validation outcomes to actual acting state, not wrapper SHA aliases.

    Records are written even for failed science gates; admit_prefix computes
    improvement/competence. No additional episode, teacher or flag is invented.
    """
    from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
    result = dict(learning_qualifications={}, teacher_qualifications={})
    output = Path(output)
    for condition in profile["conditions"]:
        key = profile["teacher_lineage_ids"][condition]
        published = publications[key]
        rows = published["source_validation_receipts"]
        end = str(profile["prefix_episodes"])
        if "0" not in rows or end not in rows:
            continue  # Pending actual fixed evidence; not a qualified teacher.
        slot = next(x for x in lineages if x["lineage_id"] == key)
        before, after = (read_json(_file_ref(rows[index])) for index in ("0", end))
        for index, receipt in (("0", before), (end, after)):
            if (receipt.get("complete") is not True or receipt.get("phase") != "source_validation"
                    or receipt.get("source_checkpoint") != rows[index]["input_checkpoint"]
                    or len(receipt["results"]) != profile["teacher_gate"]["paired_cases"]
                    or receipt["episode_ids"] != [r["spec"]["id"] for r in receipt["results"]]
                    or receipt.get("behavior") != "bounded uniform iff completed optimizer updates0; otherwise frozen sampled actual actor"):
                raise ValueError("Require the actual fixed source-validation receipt and its executed input snapshot")
        initial_identity = equivalent_acting(rows["0"]["input_checkpoint"], slot["initial_recovery"], lineage=slot, profile=profile)
        final_identity = equivalent_acting(rows[end]["input_checkpoint"], slot["final_recovery"], lineage=slot, profile=profile)
        pairs = []
        for early, late in zip(before["results"], after["results"]):
            if (early["spec"]["method"] != key or late["spec"]["method"] != key
                    or early["spec"]["case"] != late["spec"]["case"]
                    or early["reset_identity"] != late["reset_identity"]
                    or early["spec"]["milestone"] != 0 or late["spec"]["milestone"] != profile["prefix_episodes"]
                    or early.get("complete") is not True or late.get("complete") is not True):
                raise ValueError("Fixed validation must pair the same actual case/reset and predeclared snapshots")
            first, second = _physical_case(early,initial_identity), _physical_case(late,final_identity)
            if first != second:
                raise ValueError("Validation changed the actual initial public state across milestones")
            pairs.append(dict(case_id=first, baseline_success=early["success"], teacher_success=late["success"],
                baseline_episode=deepcopy(early["episode_manifest_ref"]),
                teacher_episode=deepcopy(late["episode_manifest_ref"])))
        evidence = output / f"{condition}-acting-binding.json"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(evidence, dict(initial=initial_identity, final=final_identity,
            original_baseline_validation=_file_ref(rows["0"]), original_final_validation=_file_ref(rows[end])))
        q = dict(condition_id=condition, interface_sha256=profile["interface"]["control_interface_sha256"],
            teacher_kind="noise_sac", teacher_lineage_id=key,
            teacher_recovery=deepcopy(rows[end]["input_checkpoint"]),
            baseline_recovery=deepcopy(rows["0"]["input_checkpoint"]),
            prefix_initial_recovery=deepcopy(slot["initial_recovery"]), prefix_final_recovery=deepcopy(slot["final_recovery"]),
            actor_parameter_sha256=final_identity["actor_parameter_sha256"],
            label_sampling_mode=final_identity["sampling_mode"], baseline_sampling_mode=initial_identity["sampling_mode"],
            evidence=dict(path=str(evidence.resolve()), sha256=digest(evidence)), paired_cases=pairs,
            qualification_meaning="Actual measurements; objective-specific gates must still pass")
        path = output / f"{condition}-measured-qualification.json"
        _atomic_json(path, q)
        ref = dict(path=str(path.resolve()), sha256=digest(path))
        result["learning_qualifications"][condition] = ref
        result["teacher_qualifications"][condition] = deepcopy(ref)
    return result


def derive_blind_selection(profile, publications, lineages, *, output):
    """One source-validation-selected reference for BOTH conditions, including warmup0."""
    from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
    end=str(profile["prefix_episodes"])
    scores,refs,controllers={},[],{}
    keys=[profile["teacher_lineage_ids"][c] for c in profile["conditions"]]
    for key in keys:
        pub=publications[key]
        if end not in pub["source_validation_receipts"]:return None
        record=pub["source_validation_receipts"][end]
        panel=read_json(_file_ref(record))
        lineage=next(x for x in lineages if x["lineage_id"]==key)
        proof=equivalent_acting(record["input_checkpoint"],lineage["final_recovery"],lineage=lineage,profile=profile)
        if panel.get("complete") is not True or len(panel["results"])!=profile["teacher_gate"]["paired_cases"]:
            raise ValueError("Blind selection needs the predeclared actual source-validation panel")
        for row in panel["results"]:_physical_case(row,proof)
        scores[key]=sum(row["success"] for row in panel["results"])
        refs.append(_file_ref(record))
        controllers[key]=dict(lineage["final_recovery"],policy_sha256=proof["actor_parameter_sha256"],
            optimizer_updates=proof["updates"],source_method=key,sampling_mode=proof["sampling_mode"])
    selected=max(keys,key=lambda key:scores[key])
    result=dict(selected_method="source_validation_selected_blind",selected_source_method=selected,
        selected_from="source_validation",cross_cases_opened=False,controller_ref=controllers[selected],
        final_validation_successes=scores,source_validation_refs=refs,
        selection_rule="maximum actual final success count; predeclared condition order breaks ties")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);_atomic_json(output,result)
    return dict(path=str(output.resolve()),sha256=digest(output))


def build_prefix(profile, publications_by_lineage, *, control_qualification,
                 learning_qualifications=None, teacher_qualifications=None,
                 condition_cross=None, output=None, expected_hostname):
    """Consume the producer's ACTUAL publish_prefix dictionaries, including VAL.

    Publications retain the existing complete_episode_manifests, actual_resets,
    initial/final_full_state and source_validation_receipts. Initial update
    counts and case/frame identities are read from canonical artifacts, not
    guessed from lineage numbering or a requested reset. ``condition_cross`` is
    the original rich helper result reference, or an explicit ``{result, plan}``
    binding. Data completion remains separate from scientific admission.
    """
    _linux(expected_hostname)
    validate_profile(profile)
    slots = {row["lineage_id"]: row for row in profile["lineage_slots"]}
    if set(publications_by_lineage) != set(slots):
        raise ValueError("Wait for the same completed immutable cursor in all six TRAIN/VAL lineages")
    reference(control_qualification)
    # Reject nonportable incoming metadata before producing any output.
    publications = json.loads(json.dumps(publications_by_lineage, allow_nan=False))
    lineages = []
    for method in sorted(slots):
        slot, published = slots[method], publications[method]
        identity = published["lineage"]
        if (identity["method"] != method or identity["partition"] != slot["partition"]
                or identity["learner_seed"] != slot["seed"]
                or published["condition_id"] != slot["condition_id"]
                or published["mapping_sha256"] != profile["interface"]["mapping_sha256"]
                or published["interface_sha256"] != profile["interface"]["control_interface_sha256"]):
            raise ValueError("Actual collector publication differs from the bound lineage/mapping/seed")
        manifests, resets = published["complete_episode_manifests"], published["actual_resets"]
        if len(manifests) != profile["prefix_episodes"] or len(resets) != len(manifests):
            raise ValueError("All six publications must contain the exact complete common cursor")
        lineage = dict(slot, source_only=True, completed_episodes=len(manifests), episodes=[],
            initial_recovery=deepcopy(published["initial_full_state"]),
            final_recovery=deepcopy(published["final_full_state"]))
        initial = _source_state(lineage["initial_recovery"], lineage=lineage, profile=profile)
        final = _source_state(lineage["final_recovery"], lineage=lineage, profile=profile)
        lineage.update(initial_updates=int(initial["updates"]), final_updates=int(final["updates"]))
        if published["actual_optimizer_updates"] != lineage["final_updates"]:
            raise ValueError("Published optimizer cursor differs from actual complete full recovery")
        actions, roots = 0, set()
        for original_ref, actual in zip(manifests, resets):
            path = reference(original_ref)
            manifest = read_json(original_ref)
            root = Path(original_ref["root"]).resolve()
            roots.add(str(root))
            if manifest.get("pending_decision") is not None or not manifest.get("chunks"):
                raise ValueError("An unresolved/empty episode cannot enter an immutable prefix")
            reset_path = path.parent / manifest["reset"]
            reset = json.loads(reset_path.read_text())
            info = reset["reset_info"]
            if (info["task"] != profile["source_task"] or actual.get("actual_seed") != info["actual_seed"]
                    or any(actual.get(key) != info.get(key) for key in
                           ("requested_seed", "observation_sha256", "features_sha256", "reset_policy_actions"))):
                raise ValueError("Published actual reset differs from the canonical recorded reset")
            last_ref = manifest["chunks"][-1]
            last_path = path.parent / last_ref["path"]
            last = read_json(dict(path=str(last_path), sha256=last_ref["sha256"]))
            outcome = last["learning_result"]
            if (last.get("status") != "complete" or not (outcome["terminated"] or outcome["truncated"])
                    or outcome.get("budget_truncated") is not False
                    or manifest["completed_chunks"] != len(manifest["chunks"])):
                raise ValueError("The canonical episode is not a complete natural boundary")
            frame = reset["initial_observation"]["content_sha256"]
            physical_case = content_hash(dict(task=info["task"], actual_seed=info["actual_seed"],
                                              initial_observation_sha256=frame))
            lineage["episodes"].append(dict(case_id=physical_case, actual_seed=info["actual_seed"],
                initial_observation_sha256=frame, raw_trace=deepcopy(original_ref)))
            actions += int(manifest["executed_primitive_steps"])
        if len(roots) != 1 or actions != published["actual_primitive_steps"]:
            raise ValueError("Publication changed corpus root or actual source action counters")
        lineage["corpus_root"] = roots.pop()
        for row in published["source_validation_receipts"].values():
            # Preserve all producer fields; verify the original referenced files.
            reference(_file_ref(row))
            reference(row["input_checkpoint"])
        lineages.append(lineage)
    if output is not None and (learning_qualifications is None or teacher_qualifications is None):
        measured = derive_source_qualifications(profile, publications, lineages,
            output=Path(output).parent / (Path(output).stem + "-qualifications"))
        if learning_qualifications is None:
            learning_qualifications = measured["learning_qualifications"]
        if teacher_qualifications is None:
            teacher_qualifications = measured["teacher_qualifications"]
    result = dict(format=PREFIX, completed=True, source_only=True,
        profile_sha256=content_hash(profile), source_task=profile["source_task"],
        interface=deepcopy(profile["interface"]), control_qualification=deepcopy(control_qualification),
        lineages=lineages, learning_qualifications=deepcopy(learning_qualifications or {}),
        teacher_qualifications=deepcopy(teacher_qualifications or {}),
        source_publications=publications, source_publications_sha256=content_hash(publications),
        completion_meaning="Complete immutable data prefix; AD/DPT scientific admission is separate")
    if condition_cross is not None:
        if set(condition_cross) == {"result", "plan"}:
            binding = deepcopy(condition_cross)
        else:
            path = reference(_file_ref(condition_cross))
            plan = path.parent / "plan.json"
            binding = dict(result=deepcopy(condition_cross),
                           plan=dict(path=str(plan.resolve()), sha256=digest(plan)))
        reference(_file_ref(binding["result"]))
        reference(binding["plan"])
        result["condition_cross"] = binding
    if output is not None:
        from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
        output = Path(output)
        if output.exists():
            raise ValueError("Never overwrite a prefix admitted while collection continues")
        result["blind_selection"]=derive_blind_selection(profile,publications,lineages,
            output=output.parent/(output.stem+"-blind-selection.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(output, result)
    return result
