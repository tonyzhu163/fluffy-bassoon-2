"""Explicit source/readout entry points for the new fitted+physical4D mapping.

The fitted endpoint is immutable execution context, never a SAC warmstart or
coefficient label. Old fixed-map A1 receipts cannot admit this interface.
"""
import sys

import numpy as np

from fluffy_bassoon.rl.robotwin_fitted_physical import create_control, validate_binding, load_frontend


SELECTION = "fitted_physical_source_selection_v1"
A1 = "fitted_physical_A1_v1"


def _complete_record(row, horizon, *, mapping_sha, interface_sha, frontend_sha):
    from .steering_history import reference, read_json
    path = reference(row["episode_manifest_ref"])
    manifest = read_json(row["episode_manifest_ref"])
    chunks = [read_json(dict(path=str(path.parent/ref["path"]), sha256=ref["sha256"]))
              for ref in manifest.get("chunks", [])]
    if (not chunks or manifest.get("pending_decision") is not None
            or manifest.get("completed_chunks") != len(chunks)
            or any(chunk.get("status") != "complete" for chunk in chunks)
            or not (chunks[-1]["learning_result"]["terminated"] or chunks[-1]["learning_result"]["truncated"])
            or chunks[-1]["learning_result"].get("budget_truncated") is not False
            or manifest.get("executed_primitive_steps") != row.get("primitive_steps")
            or sum(chunk["native_result"]["executed_primitive_steps"] for chunk in chunks) != row["primitive_steps"]
            or not 0 < row["primitive_steps"] <= horizon
            or bool(row["success"]) != any(bool(chunk["native_result"]["success"]) for chunk in chunks)):
        raise ValueError("A1 receipt differs from its complete canonical raw episode")
    for chunk in chunks:
        metadata = chunk["decision_metadata"]
        config = metadata.get("coefficient_steering", {}).get("audit_only", {}).get("mapping", {})
        if (metadata.get("steering_interface_sha256") != interface_sha
                or config.get("schema") != "frozen_fitted_frontend_physical4d_v1"
                or config.get("fitted_frontend", {}).get("checkpoint", {}).get("sha256") != frontend_sha
                or (row["spec"]["arm"] == "action" and metadata.get("steering_mapping_sha256") != mapping_sha)):
            raise ValueError("Canonical A1 chunks executed a different frontend/interface/mapping")


def validate_selection(selection, qualification, a1_result, profile, interface, *, interface_sha, selection_rule):
    binding = validate_binding(interface)
    if (selection.get("schema") != SELECTION or selection.get("status") != "selected_from_completed_A1"
            or selection.get("selection_basis") != "natural_A1_predeclared_criteria"
            or selection.get("condition_outcomes_used") is not False or profile.arm != "action"
            or selection.get("task") != profile.task or selection.get("arm") != "action"
            or selection.get("mapping_sha256") != profile.mapping_sha256
            or selection.get("interface_sha256") != interface_sha or profile.interface_sha256 != interface_sha
            or selection.get("motor_sha256") != profile.motor_sha256
            or profile.motor_sha256 != binding["motor_sha256"]
            or selection.get("episode_limit") != profile.episode_limit
            or selection.get("task_horizon_sha256") != profile.task_horizon_sha256
            or selection.get("frontend_checkpoint_sha256") != binding["checkpoint"]["sha256"]):
        raise ValueError("Require a new selected fitted+physical mapping; old fixed-map A1 is inadmissible")
    # Verify the small immutable endpoint on the actual allocated CPU before
    # any worker constructs a motor. This neither fits nor warms a SAC actor.
    load_frontend(binding, device="cpu")
    expected_rule = dict(format="fitted_physical_A1_rule_v1", confirmation_repeats=2,
        comparator="raw_native", baseline="frozen_fitted_frontend", candidate="frozen_fitted_frontend_plus_physical4d")
    if selection_rule != expected_rule:
        raise ValueError("Bind the explicit prospective composed A1 confirmation rule")
    if (qualification.get("format") != "fitted_physical_motor_qualification_v1"
            or qualification.get("status") != "passed"
            or qualification.get("mapping_sha256") != profile.mapping_sha256
            or qualification.get("interface_sha256") != interface_sha
            or qualification.get("frontend_checkpoint_sha256") != binding["checkpoint"]["sha256"]
            or qualification.get("frozen_frontend_before") != binding["head_state_sha256"]
            or qualification.get("frozen_frontend_after") != binding["head_state_sha256"]
            or not qualification.get("frozen_before") or qualification.get("frozen_before") != qualification.get("frozen_after")
            or qualification.get("motor_gradients") != 0 or qualification.get("frontend_gradients") != 0
            or any(qualification.get(key) is not True for key in ("fitted_baseline_zero_exact",
                "raw_native_control_exact", "physical_delta_bound_passed", "physical_nonzero_effect",
                "public_request_before_calibration"))):
        raise ValueError("Actual zero-to-fitted/raw-native/frozen/physical-bound/recording-order qualification required")
    if (a1_result.get("format") != A1 or a1_result.get("status") != "complete"
            or a1_result.get("mapping_sha256") != profile.mapping_sha256
            or a1_result.get("interface_sha256") != interface_sha
            or a1_result.get("frontend_checkpoint_sha256") != binding["checkpoint"]["sha256"]
            or a1_result.get("condition_outcomes_used") is not False):
        raise ValueError("Require completed A1 of this exact composed interface before source conditions")
    records, pairs = a1_result["records"], a1_result["confirmation_pairs"]
    if len(pairs) != 2 or len({pair["composed"] for pair in pairs}) != 2 or len({pair["raw_native"] for pair in pairs}) != 2:
        raise ValueError("Actual two independent composed/raw-native confirmation pairs required")
    seeds, cases, candidates = set(), set(), set()
    proof = dict(mapping_sha=profile.mapping_sha256, interface_sha=interface_sha,
                 frontend_sha=binding["checkpoint"]["sha256"])
    natural_failures = set()
    for key in a1_result["development_ids"]:
        development = records[key]
        if (development.get("complete") is not True or development["spec"]["arm"] != "native"
                or development["spec"]["task"] != profile.task):
            raise ValueError("Natural source development must use actual complete raw-native episodes")
        _complete_record(development, profile.episode_limit, **proof)
        if development.get("success") is False:
            natural_failures.add(development["spec"]["case"])
    for pair in pairs:
        winner, native = records[pair["composed"]], records[pair["raw_native"]]
        coefficients = np.asarray(winner["spec"]["coefficients"])
        if (winner.get("complete") is not True or native.get("complete") is not True
                or winner.get("success") is not True or native.get("success") is not False
                or winner["spec"]["arm"] != "action" or native["spec"]["arm"] != "native"
                or winner["spec"]["task"] != profile.task or native["spec"]["task"] != profile.task
                or winner["spec"]["case"] != native["spec"]["case"] or winner["spec"]["case"] not in natural_failures
                or winner["reset_identity"] != native["reset_identity"]
                or winner["reset_info"]["instruction"] != native["reset_info"]["instruction"]
                or winner["spec"]["motor_seed"] != native["spec"]["motor_seed"]
                or coefficients.shape != (4,) or not np.isfinite(coefficients).all() or (np.abs(coefficients) > 1).any()):
            raise ValueError("Composed A1 requires2/2 physical rescue against0/2 paired raw-native failures")
        for row in (winner, native):
            if not row.get("episode_manifest_ref") or not 0 < row["primitive_steps"] <= profile.episode_limit:
                raise ValueError("Canonical complete physical episode references/actions required")
            _complete_record(row, profile.episode_limit, **proof)
        seeds.add(winner["spec"]["motor_seed"])
        cases.add(winner["spec"]["case"])
        candidates.add(tuple(float(c) for c in coefficients))
    if len(seeds) != 2 or len(cases) != 1 or len(candidates) != 1:
        raise ValueError("Repeat one unchanged selected case/candidate with two independent paired motor draws")
    qrow = records.get(qualification.get("qualification_episode"), {})
    if (qrow.get("complete") is not True or qrow.get("primitive_steps") != qualification.get("actual_actions")
            or qualification.get("complete_real_episode") != qualification.get("qualification_episode")):
        raise ValueError("Qualification must cite its actual complete physical episode and charged action count")
    _complete_record(qrow, profile.episode_limit, **proof)


def verify_control_qualification(binding, profile):
    """New proof ABI for existing prefix admission; A2 is still checked later."""
    from .steering_history import read_json
    from .steering_condition_collection import SourceProfile
    interface = read_json(binding["control_interface"])
    source = SourceProfile(**read_json(binding["source_profile"]))
    selected = read_json(binding["selection"])
    q = read_json(binding["driver_qualification"])
    a1 = read_json(binding["a1_result"])
    plan, episode = read_json(binding["driver_plan"]), read_json(binding["completed_episode"])
    expected = profile["interface"]
    if (binding["control_interface"]["sha256"] != expected["control_interface_sha256"]
            or source.interface_sha256 != expected["control_interface_sha256"]
            or source.mapping_sha256 != expected["mapping_sha256"] or source.motor_sha256 != expected["motor_sha256"]
            or source.scope_sha256 != profile["scope_sha256"] or source.task != profile["source_task"]
            or source.source_selection_sha256 != binding["selection"]["sha256"]
            or source.public_dim != expected["native_observation_dim"] or expected["coefficient_dim"] != 4
            or expected["arm"] != "action" or source.arm != "action"
            or plan.get("interface_sha256") != source.interface_sha256
            or plan.get("scope_sha256") != source.scope_sha256
            or selected.get("qualification_receipt") != binding["driver_qualification"]
            or selected.get("a1_result") != binding["a1_result"]
            or episode.get("complete") is not True or episode.get("spec", {}).get("id") != q.get("complete_real_episode")
            or episode.get("primitive_steps") != q.get("actual_actions")):
        raise ValueError("Composed control proof differs from this exact collected source/student mapping")
    validate_selection(selected, q, a1, source, interface, interface_sha=source.interface_sha256,
        selection_rule=read_json(selected["selection_rule"]))
    _complete_record(episode, source.episode_limit, mapping_sha=source.mapping_sha256,
        interface_sha=source.interface_sha256, frontend_sha=interface["fitted_frontend"]["checkpoint"]["sha256"])
    return dict(driver_qualification=binding["driver_qualification"], frozen_tensor_sha256=q["frozen_before"],
        motor_file_sha256=interface["motor_sha256"], actual_qualification_episode=q["complete_real_episode"],
        frontend_checkpoint_sha256=interface["fitted_frontend"]["checkpoint"]["sha256"])


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in ("source", "readout"):
        raise ValueError("Select source or readout for the prospective fitted+physical4D interface")
    mode, arguments = arguments[0], arguments[1:]
    if mode == "source":
        from .robotwin_source_parallel import main as source_main
        return source_main(arguments, control_factory=create_control, selection_validator=validate_selection)
    from .steering_history_readout import main as readout_main
    return readout_main(arguments, control_factory=create_control)


if __name__ == "__main__":
    main()
