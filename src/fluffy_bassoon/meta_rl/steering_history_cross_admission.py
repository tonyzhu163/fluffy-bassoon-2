"""Recompute adopted A2 science gates from original full24-cell physical evidence."""
import json

from .steering_history import (_source_state,content_hash,equivalent_acting,read_json,reference)


def file_ref(value):
    return {k:value[k] for k in ("path","sha256","root","retention","size_bytes") if k in value}


def verify_condition_cross(profile,prefix,*,objective):
    binding=prefix.get("condition_cross")
    if not binding:
        raise ValueError("A2 condition-cross evidence is pending; complete source data alone cannot admit scientific fits")
    if profile["prefix_episodes"]!=24:
        raise ValueError("The one funded A2 grid qualifies the completed24 prefix; preserve12 as pending data")
    result=read_json(file_ref(binding["result"]))
    plan=read_json(binding["plan"])
    if (plan.get("format")!="source_only_frozen_condition_cross_v1"
            or result.get("format")!="source_only_frozen_condition_cross_v1" or result.get("status")!="complete"
            or result.get("pending") is not None or result.get("plan_sha256")!=content_hash(plan)
            or result.get("active_actor_rng") is not None or len(result.get("records",[]))!=24
            or plan.get("source_sgd")!=0 or plan.get("paired_gap")!=.25 or plan.get("aggregate_blind_gap")!=.125):
        raise ValueError("A2 requires the original complete24-cell result and its exact original plan")
    control=plan["profile"];interface=profile["interface"]
    for key,expected in dict(task=profile["source_task"],mapping_sha256=interface["mapping_sha256"],
        interface_sha256=interface["control_interface_sha256"],motor_sha256=interface["motor_sha256"],
        scope_sha256=profile["scope_sha256"],public_dim=interface["native_observation_dim"],native_horizon=32,
        episode_limit=profile["episode_action_limit"],arm=interface["arm"],
        teacher_label_selection=profile["teacher_lineage_ids"]).items():
        if control.get(key)!=expected:raise ValueError("A2 source task/mapping/motor/condition-teacher binding differs")
    conditions=profile["conditions"];primary=[profile["teacher_lineage_ids"][c] for c in conditions]
    blind=plan["condition_blind_method"];methods=(*primary,blind)
    refs=plan["controller_refs"]
    if set(refs)!=set(methods) or len(set(methods))!=3:
        raise ValueError("A2 needs both actual TRAIN1 actors and one fixed blind alias")
    import numpy as np
    from .robotwin_history_corpus import parameter_digest,CorpusReader,_decode
    from .steering_condition_collection import _state_digest
    from .steering_history_prefix import _physical_case
    from .warmstart import checked_load
    calibration=plan.get("condition_contract")
    if (not isinstance(calibration,dict)
            or [r.get("condition_id") for r in calibration.get("conditions",[])]!=conditions
            or [r.get("sign") for r in calibration["conditions"]]!=[-1,1]
            or any(r.get("mapping_sha256")!=interface["mapping_sha256"]
                   or r.get("motor_sha256")!=interface["motor_sha256"] for r in calibration["conditions"])):
        raise ValueError("A2 hidden calibration identities differ from the source conditions/mapping/motor")
    for slot in prefix["lineages"]:
        original=checked_load(reference(slot["final_recovery"]),slot["final_recovery"]["sha256"],weights_only=False)
        if (original.get("format")!="source_coefficient_full_recovery_v1"
                or original.get("control",{}).get("mapping",{}).get("condition_audit_contract")!=calibration):
            raise ValueError("A2 hidden calibration differs from the original source full-recovery contract")
    identities={}
    for method in primary:
        slot=next(x for x in prefix["lineages"] if x["lineage_id"]==method)
        state=_source_state(file_ref(refs[method]),lineage=slot,profile=profile)
        sha=parameter_digest(state["actor"])
        if (state["updates"]<1 or state["contract"]["action_dim"]!=4 or state["contract"]["action_limit"]!=1.
                or refs[method].get("policy_sha256")!=sha or refs[method].get("optimizer_updates")!=state["updates"]
                or refs[method].get("sampling_mode")!="sampled_gaussian"
                or plan["frozen"][method]["policy_sha256"]!=sha
                or plan["frozen"][method]["optimizer_updates"]!=state["updates"]
                or plan["frozen"][method]["full_state_sha256"]!=_state_digest(state)):
            raise ValueError("A2 did not execute the actual frozen source4D/bound1 actor/update/RNG state")
        equivalent_acting(file_ref(refs[method]),slot["final_recovery"],lineage=slot,profile=profile)
        identities[method]=(sha,state["updates"])
        if objective=="dpt":
            condition=slot["condition_id"]
            if condition not in prefix.get("teacher_qualifications",{}):
                raise ValueError("DPT teacher qualification is pending")
            q=read_json(prefix["teacher_qualifications"][condition])
            if q.get("teacher_kind")!="noise_sac" or q.get("label_sampling_mode")!="sampled_gaussian":
                raise ValueError("This original A2 evaluated SAC actors; an alternate oracle needs its own compatible evidence, not an alias")
            equivalent_acting(file_ref(refs[method]),q["teacher_recovery"],lineage=slot,profile=profile)
            if q["actor_parameter_sha256"]!=sha:raise ValueError("DPT labels changed the actual A2-qualified TRAIN actor")
    selection=read_json(file_ref(plan["blind_selection_ref"]))
    scores=selection["final_validation_successes"]
    if (set(scores)!=set(primary) or selection.get("selected_from")!="source_validation"
            or any(type(v) is not int or not 0<=v<=2 for v in scores.values())
            or selection.get("selected_method")!=blind
            or selection.get("cross_cases_opened") is not False
            or selection["selected_source_method"]!=max(primary,key=lambda k:scores[k])
            or selection["controller_ref"]!=refs[blind] or refs[blind]!=refs[selection["selected_source_method"]]):
        raise ValueError("A2 blind actor was not fixed using the predeclared source-validation rule")
    expected_validation_refs=[]
    for method in primary:
        publication=prefix["source_publications"][method]
        validation_ref=publication["source_validation_receipts"]["24"]
        panel=read_json(file_ref(validation_ref))
        slot=next(x for x in prefix["lineages"] if x["lineage_id"]==method)
        acting=equivalent_acting(panel["source_checkpoint"],slot["final_recovery"],lineage=slot,profile=profile)
        if panel.get("complete") is not True or panel.get("phase")!="source_validation" or len(panel["results"])!=2:
            raise ValueError("A2 blind selection requires the original fixed two-case source-validation panels")
        for record in panel["results"]:
            _physical_case(record,acting)
        if sum(record["success"] for record in panel["results"])!=scores[method]:
            raise ValueError("A2 blind selection scores differ from actual source-validation outcomes")
        expected_validation_refs.append(file_ref(validation_ref))
    if [file_ref(r) for r in selection["source_validation_refs"]]!=expected_validation_refs:
        raise ValueError("A2 blind selection refers to a different source-validation bank")
    identities[blind]=identities[selection["selected_source_method"]]
    if plan.get("actor_seeds_by_scene")!=[1930000000+997*i for i in range(4)]:
        raise ValueError("A2 changed paired private actor innovations")
    forbidden=[]
    for pub in prefix["source_publications"].values():
        forbidden.extend(pub["actual_resets"])
        for ref in pub["source_validation_receipts"].values():
            forbidden.extend(r["reset_info"] for r in read_json(file_ref(ref))["results"])
    cells={};scene_resets={};actions=0
    expected_cells={(c,i,m) for c in conditions for i in range(4) for m in methods}
    expected_specs={(r["condition_id"],r["case"],r["method"]):r for r in plan["specs"]}
    if len(plan["specs"])!=24 or set(expected_specs)!=expected_cells:
        raise ValueError("A2 plan is not the declared two-condition/four-scene/three-controller grid")
    for record in result["records"]:
        spec=record["spec"];cell=(spec["condition_id"],spec["case"],spec["method"])
        if (cell in cells or cell not in expected_cells or record.get("complete") is not True
                or type(record.get("success")) is not bool):
            raise ValueError("A2 cells are duplicated, missing, unplanned or incomplete")
        expected=expected_specs[cell]
        if (any(spec.get(k)!=v for k,v in expected.items()) or spec["task"]!=profile["source_task"]
                or spec["arm"]!=interface["arm"] or spec["horizon"]!=profile["episode_action_limit"]
                or spec["reset_seed"]!=2300000000+997*spec["case"] or spec["motor_seed"]!=1730000000+997*spec["case"]):
            raise ValueError("Actual A2 cell changed the fixed source task/arm/reset/noise grid")
        info=record["reset_info"];scene=spec["case"]
        signature=tuple(info[k] for k in ("actual_seed","observation_sha256","features_sha256"))
        if scene in scene_resets and scene_resets[scene]!=signature:raise ValueError("A2 initial public inputs differ across a paired scene")
        if scene not in scene_resets:
            if any(any(k in f and info[k]==f[k] for k in ("actual_seed","observation_sha256","features_sha256")) for f in forbidden):
                raise ValueError("A2 reused an actual source or source-validation case")
            if signature in scene_resets.values():raise ValueError("A2 repeated another actual scene")
            scene_resets[scene]=signature
        path=reference(record["episode_manifest_ref"]);manifest=read_json(record["episode_manifest_ref"])
        if (manifest["pending_decision"] is not None or not manifest["chunks"]
                or manifest["completed_chunks"]!=record["chunks"] or manifest["completed_chunks"]!=len(manifest["chunks"])):
            raise ValueError("A2 canonical physical episode remains incomplete")
        reset=json.loads((path.parent/manifest["reset"]).read_text())
        if any(reset["reset_info"].get(k)!=info.get(k) for k in
               ("task","actual_seed","observation_sha256","features_sha256","instruction")):
            raise ValueError("A2 reset identity differs from canonical raw reset")
        reader=CorpusReader(record["episode_manifest_ref"]["root"]);success=False;duration=0
        previous=reset["initial_observation"]
        for index,ref in enumerate(manifest["chunks"]):
            chunk=read_json(dict(path=str(path.parent/ref["path"]),sha256=ref["sha256"]))
            arrays=reader.arrays(chunk["decision_arrays"],list(chunk["decision_arrays"]["arrays"]))
            meta=_decode(chunk["decision_metadata"],arrays)
            if (chunk["status"]!="complete" or chunk["observation"]!=previous
                    or chunk["episode_id"]!=manifest["episode_id"] or chunk["chunk_index"]!=index+1
                    or meta.get("frozen_controller_ref")!=refs[spec["method"]]
                    or meta.get("cross_actor_sampling_mode")!="sampled_gaussian"
                    or meta.get("cross_actor_innovation_seed")!=1930000000+997*scene
                    or meta.get("steering_condition_id")!=spec["condition_id"]
                    or meta.get("steering_mapping_sha256")!=interface["mapping_sha256"]
                    or meta.get("steering_interface_sha256")!=interface["control_interface_sha256"]
                    or meta.get("coefficient_steering",{}).get("audit_only",{}).get("mapping",{}).get("condition_audit_contract")!=calibration
                    or meta.get("optimizer_updates")!=identities[spec["method"]][1]):
                raise ValueError("A2 raw records used another actor, counter or sampling convention")
            native=chunk["native_result"];outcome=chunk["learning_result"]
            mask=reader.arrays(native["arrays"],["valid_action_mask"])["valid_action_mask"]
            if (not np.isin(mask,(0,1)).all() or int(mask.sum())!=native["executed_primitive_steps"]
                    or not 0<=native["executed_primitive_steps"]<=32 or outcome.get("budget_truncated") is not False
                    or bool(outcome["terminated"] or outcome["truncated"])!=(index==len(manifest["chunks"])-1)):
                raise ValueError("A2 requires complete native episodes with exact valid-mask/action costs")
            success|=chunk["native_result"]["success"];duration+=chunk["native_result"]["executed_primitive_steps"]
            previous=chunk["next_observation"]
        for input_path,sha in reader.inputs.items():
            from .steering_history import digest
            if digest(input_path)!=sha:raise ValueError("A2 canonical arrays changed during admission")
        if success!=record["success"] or duration!=record["primitive_steps"] or duration!=manifest["executed_primitive_steps"]:
            raise ValueError("A2 claimed outcomes/costs differ from canonical native transitions")
        cells[cell]=record;actions+=duration
    if set(cells)!=expected_cells:raise ValueError("A2 full grid is missing physical cells")
    gaps={};matched=blind_success=0
    for i,condition in enumerate(conditions):
        counts={m:sum(cells[condition,j,m]["success"] for j in range(4)) for m in methods}
        gap=(counts[primary[i]]-counts[primary[1-i]])/4
        gaps[condition]=gap;matched+=counts[primary[i]];blind_success+=counts[blind]
    aggregate=(matched-blind_success)/8
    if min(gaps.values())<.25 or aggregate<.125:
        raise ValueError("A2 condition dependence failed: need .25 mismatch gap each condition and .125 pooled fixed-blind gap")
    return dict(original_result=binding["result"],original_plan=binding["plan"],matched_mismatch_gaps=gaps,
        pooled_blind_gap=aggregate,actual_actions=actions,verified_from_raw24_cells=True)
