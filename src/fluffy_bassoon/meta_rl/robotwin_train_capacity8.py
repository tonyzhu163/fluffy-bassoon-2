"""Eight parent-funded TRAIN controls; no A1 selector, learner or allocator.

Pure validation helpers import only the standard library. Actual tensors and
physics are reachable only through a hash-bound, admitted allocated-Linux ticket.
"""
from copy import deepcopy
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

FORMAT = "phone_successful_TRAIN_capacity8_driver_v1"
PHASE = "frozen-pi05-train-capacity8-20261101-r1"
ACTION_PHASE = "train_capacity8"
MODES = ("raw_native", "frozen_fitted_frontend_zero4")
CHECKPOINT = "19ed635cbff5427a184c1969fa6955551f8f5f33a260c1747507617a93deb200"
ORIGIN = "aa44ce2d60c0352265c1138b078cba8106a86a69f875f7577e691a9705cee1c4"
MOTOR = "721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): result.update(block)
    return result.hexdigest()


def read_ref(ref):
    path = Path(ref["path"])
    if not path.is_absolute() or path.is_symlink() or path.stat().st_size > 4 << 20 or sha(path) != ref["sha256"]:
        raise ValueError("Bound immutable absolute JSON reference required")
    return json.loads(path.read_text())


def validate_parent_links(grant, request):
    """Grant paths are logical repo-relative artifact locations, never rewritten."""
    for key in ("authority", "protocol"):
        declared=grant[key];path=Path(declared["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Parent grant uses repository-relative immutable references")
        actual=Path(request[key]["path"])
        if (declared["sha256"]!=request[key]["sha256"] or not actual.is_absolute()
                or tuple(actual.parts[-len(path.parts):])!=path.parts):
            raise ValueError("Exact parent link SHA and logical repository-relative path suffix required")


def validate_origin(saved, old):
    """Actual completed547 readout recovery ABI, not a generic completion label."""
    state=saved["state"]
    if (state.get("status")!="complete_diagnostic_readout" or state.get("next_cell")!=15
            or len(state.get("records",[]))!=15 or any(r.get("complete") is not True for r in state["records"])
            or state["actions"]!=old or state.get("pending") is not None or state.get("control_active") is not False
            or saved["control"].get("active") is not False or saved.get("optimizer") is not None
            or not isinstance(saved["control"].get("motor_rng"),dict)
            or set(saved["global_rng"])!={"python","numpy","torch","cuda"} or saved["global_rng"]["cuda"] is None):
        raise ValueError("Actual complete_diagnostic_readout closed15/global/private recovery required")


def validate_protocol(protocol, grant, authority):
    if (protocol.get("format") != "phone_successful_TRAIN_capacity8_protocol_v1"
            or protocol.get("phase_id") != PHASE or protocol.get("canonical_TRAIN_records") != [f"phone-{i:02d}" for i in range(4)]
            or protocol.get("head_kind") != "action" or protocol.get("head_checkpoint", {}).get("sha256") != CHECKPOINT
            or protocol["head_checkpoint"].get("iteration") != 5000 or protocol.get("episodes") != 8
            or protocol.get("maximum_controller_actions") != 3200 or protocol.get("native_horizon") != 400
            or protocol.get("private_seed_rule") != "20261301+integer_record_id"
            or protocol.get("extra_qualification_episodes") != 0 or protocol.get("policy_settling_or_reset_commands") != 0
            or protocol.get("order") != "Case-major raw_native then frozen_fitted_frontend_zero4"
            or grant.get("phase_id") != PHASE or grant.get("maximum_GPU_seconds") != 1200
            or grant.get("maximum_CPU_worker_seconds") != 180 or grant.get("maximum_controller_actions") != 3200
            or grant.get("maximum_episodes") != 8 or grant.get("maximum_output_bytes") != 4 << 30
            or grant.get("GPU_resources") != dict(GPU=1,kind="RPBZZZ6",CPU=16,RAM_GiB=96,SHM_GiB=8,maximum_scheduler_minutes=19,automatic_retry=False)
            or authority.get("format") != "frozen_vla_bounded_followup_parent_authority_v1"
            or authority.get("phase_id") != PHASE or authority.get("only_first8_of_unfunded23_template") is not True
            or authority.get("no_automatic_source_or_student_admission") is not True
            or authority.get("sole_execution_owner") != "/root/finish_assembly"):
        raise ValueError("Only the actual fresh1200s/3200actions first8 TRAIN diagnostic grant")


def build_cells(protocol, inventory, descriptor, producers, manifests):
    """Resolve original event-teacher identities, never old DEV ordinal records."""
    provenance = descriptor["fit_provenance"]
    split = {f"phone-{i:02d}": "train" if i < 8 else "validation" for i in range(12)}
    if (descriptor.get("kind") != "action" or descriptor.get("iteration") != 5000
            or descriptor.get("checkpoint") != {k:protocol["head_checkpoint"][k] for k in ("path","sha256")}
            or provenance.get("source_eligible") is not False
            or provenance.get("diagnostic_purpose") != "fixed12_original_plan_command_capacity_diagnostic_only"
            or provenance["profile"].get("whole_split") != split
            or set(provenance["outcome_evidence"]) != set(split)
            or provenance["outcome_evidence"]["phone-04"].get("success") is not False):
        raise ValueError("Exact WIDE5000/full12 split/truthful diagnostic failed04 provenance required")
    result = []
    for i in range(4):
        ident = f"phone-{i:02d}";outcome = provenance["outcome_evidence"][ident]
        rows = [r for r in inventory["episodes"] if r["canonical_index"] == i]
        original = [r for r in producers[ident]["records"] if r["spec"]["id"] == outcome["record_id"]]
        if len(rows) != 1 or len(original) != 1: raise ValueError("Exactly one canonical whole episode and original event record")
        row, old = rows[0], original[0]
        reset = {k:old["reset_info"][k] for k in ("actual_seed","observation_sha256","features_sha256")}
        manifest = manifests[ident]
        if (row["reset_seed"] != i or not row.get("selected_instruction") or old.get("complete") is not True
                or old["spec"]["reset_seed"] != row["reset_seed"] or old["spec"]["paired_instruction"] != [row["selected_instruction"]]
                or old["reset_info"]["instruction"] != [row["selected_instruction"]]
                or reset != old["reset_identity"] or reset["actual_seed"] != row["reset_seed"]
                or old.get("success") is not True or outcome.get("success") is not True
                or any(old[k] != outcome[k] for k in ("primitive_steps","chunks","episode_manifest_ref"))
                or manifest.get("pending_decision") is not None or manifest.get("completed_chunks") != old["chunks"]
                or manifest.get("executed_primitive_steps") != old["primitive_steps"]):
            raise ValueError("Original successful TRAIN instruction/reset/canonical closed manifest mismatch")
        for mode in MODES:
            result.append(dict(id=f"train-capacity8-{ident}-{mode}",case=i,record_id=ident,task="place_phone_stand",
                stage="successful_TRAIN_capacity_diagnostic",role=mode,arm="native" if mode == MODES[0] else "action",
                method="train_capacity8_"+mode,coefficients=[0.]*4,horizon=400,reset_seed=row["reset_seed"],motor_seed=20261301+i,
                paired_instruction=[row["selected_instruction"]],expected_reset=deepcopy(reset),
                policy_version="raw-native" if mode == MODES[0] else CHECKPOINT,
                actual_acting_checkpoint_iteration=0 if mode == MODES[0] else 5000,new_optimizer_updates=0))
    return result


def validate_ticket(request, ticket, clock, *, source, request_sha, hostname, output, now):
    if (request.get("format") != FORMAT or request.get("status") != "owner_bound_request"
            or request.get("source_eligible") is not False
            or ticket.get("admitted") is not True or ticket.get("source_commit") != source
            or ticket.get("phase_id") != PHASE or ticket.get("request_sha256") != request_sha
            or ticket.get("hostname") != hostname or ticket.get("gpu_count") != 1
            or ticket.get("maximum_gpu_seconds") != 1200 or ticket.get("maximum_output_bytes") != 4 << 30
            or ticket.get("automatic_retry") is not False or ticket.get("scope") != "first8_TRAIN_controls_only"
            or ticket.get("resume") != request.get("resume")
            or Path(ticket["output"]).resolve() != Path(output).resolve()
            or not Path(output).resolve().is_relative_to(Path(ticket["aggregate_output_root"]).resolve())
            or ticket.get("clock_contract") != clock.get("contract") or clock.get("worker_seconds") != 1200):
        raise ValueError("Unbound/funded template is not actual allocation admission; no automatic replay")
    attempts=clock["attempts"];current=[r for r in attempts if r.get("status")=="started"]
    if len(current)!=1:raise ValueError("Exactly one current STARTED owner clock attempt")
    active=current[0];closed=[r for r in attempts if r is not active]
    if (active.get("path") != ticket.get("resource_attempt")
            or any(r.get("lineage")!=PHASE or r.get("lineage_seconds")!=1200 for r in attempts)
            or any(r.get("status")!="finished" or type(r.get("charged_seconds")) is not int
                or r["charged_seconds"]<0 or not r["started_unix"]<=r["ended_unix"]<=active["started_unix"] for r in closed)
            or len({r["path"] for r in attempts})!=len(attempts)
            or any(a["ended_unix"]>b["started_unix"] for a,b in zip(sorted(closed,key=lambda r:r["started_unix"]),sorted(closed,key=lambda r:r["started_unix"])[1:]))
            or type(active["reserved_seconds"]) is not int or not 0 < active["reserved_seconds"] <= 1200
            or sum(r["charged_seconds"] for r in closed)+active["reserved_seconds"]>1200
            or ticket.get("cleanup_seconds",0)<60 or not math.isfinite(ticket["driver_deadline_unix"])
            or not active["started_unix"]<=now<ticket["driver_deadline_unix"]<=active["started_unix"]+active["slice_seconds"]-ticket["cleanup_seconds"]
            or not 0<ticket.get("minimum_episode_admission_seconds",0)<active["slice_seconds"]):
        raise ValueError("Same finite1200 owner clock; closed charges plus reservation, no overlap/refill")
    if (closed or request.get("resume") is not None) and (ticket.get("continuation_admitted") is not True or not ticket.get("diagnosis_receipt")):
        raise ValueError("Diagnosed continuation requires explicit owner admission; never automatic retry")


def validate_resume(saved, head, specs, *, contract, ledger, output):
    """Resume only a current canonical closed prefix of this same fixed panel."""
    state=saved["state"];validate_prefix(state,specs);validate_prefix(head,specs)
    n=state["next_cell"];actions=head["actions"]
    logical=lambda value:{k:v for k,v in value.items() if k!="closed_recovery_ref"}
    expected={"schema":"robotwin_campaign_actions_v1",**{k:ledger[k] for k in ("campaign_id","total_actions","source_caps","phase_caps","contract")}}
    if (logical(state)!=logical(head) or state["format"]!=FORMAT or state["panel_contract"]!=contract or head["panel_contract"]!=contract
            or state["action_ledger"]!=ledger or head["action_ledger"]!=ledger
            or state["actions"]!=actions or actions["header"]!=expected or n>=8
            or head["next_cell"]!=n or head["records"]!=state["records"]
            or head.get("pending") is not None or head.get("control_active",False)
            or Path(state["output"]).resolve()!=Path(output).resolve()
            or actions["pending_actions"] or actions["uncertain_actions"] or actions["unresolved_decisions"]
            or actions["actual_actions"]!=actions["charged_actions"]
            or actions["actual_actions"]!=sum(r["primitive_steps"] for r in state["records"])
            or saved.get("optimizer") is not None or saved.get("control",{}).get("active") is not False
            or not isinstance(saved["control"].get("motor_rng"),dict)
            or set(saved["global_rng"])!={"python","numpy","torch","cuda"} or saved["global_rng"]["cuda"] is None
            or set(saved["case_rngs"])!={s["record_id"] for s in specs[:n]}
            or any(set(r)!={"python","numpy","torch","cuda"} or r["cuda"] is None for r in saved["case_rngs"].values())
            or len(state["closed_controls"])!=n or head["closed_controls"]!=state["closed_controls"]
            or (n and (state["qualification"].get("status")!="passed" or not saved["control"].get("mapping")))):
        raise ValueError("Same panel/source/head/ledger and actual closed physical prefix required")


def verify_prefix_files(state):
    """Keep the actual closed canonical/control artifacts immutable on continuation."""
    for ref in state["closed_controls"]:
        path=Path(ref["path"])
        if not path.is_absolute() or path.is_symlink() or sha(path)!=ref["sha256"]:
            raise ValueError("Closed typed control recovery changed")
    for record in state["records"]:
        for ref in [record["episode_manifest_ref"],*record["corpus_refs"]]:
            relative=Path(ref["path"]);path=Path(ref["root"])/relative
            if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or sha(path)!=ref["sha256"]:
                raise ValueError("Closed canonical prefix artifact changed")


def verify_resume_journal(head, ledger, run_id):
    """Read current header/tip before the sole owner opens its existing lock."""
    actions=head["actions"];count=actions["journal_events"];events=Path(ledger["root"])/"events"
    if (run_id in actions["runs"] or sorted(p.name for p in events.glob("*.json"))!=[f"{i:010d}.json" for i in range(count)]
            or json.loads((events/"0000000000.json").read_text())["header"]!=actions["header"]
            or json.loads((events/f"{count-1:010d}.json").read_text())["sha256"]!=actions["journal_sha256"]):
        raise ValueError("Actual current closed journal and fresh attempt run_id required")


def validate_prefix(state, specs):
    n = state["next_cell"]
    if (type(n) is not int or not 0 <= n <= 8 or len(state["records"]) != n
            or state.get("pending") is not None or state.get("control_active",False)
            or state.get("source_eligible") is not False or state.get("new_optimizer_updates") != 0
            or any(r.get("complete") is not True or r["spec"] != specs[i] for i,r in enumerate(state["records"]))):
        raise ValueError("Exact closed prefix only; never replay incomplete/uncertain physics")


def run_cells(state, specs, forks, episode, *, capture, restore, admit, persist):
    validate_prefix(state,specs)
    for i in range(state["next_cell"],8):
        if not admit(): state["status"]="closed_boundary_stop";persist(True);return
        spec=specs[i];key=spec["record_id"]
        if key not in forks: forks[key]=deepcopy(capture())
        restore(deepcopy(forks[key]));state["pending"]=deepcopy(spec);persist(False)
        record=episode(deepcopy(spec),qualify=(i==0))
        if record.get("complete") is not True: raise RuntimeError("Natural complete episode required")
        state["records"].append(record);state["next_cell"]=i+1;state["pending"]=None
        if sum(r["primitive_steps"] for r in state["records"]) > 3200: raise RuntimeError("Fresh action cap")
        persist(True)
    state["status"]="complete"


def qualify_first_raw(control, spec, output, interface, *, mapping_sha, interface_sha, probe_coefficients, guard):
    """Decode-only probes inside the first planned RAW episode; no extra step/reset."""
    import numpy as np
    import torch
    from .checkpoint import rng_state, restore_rng
    from .warmstart import atomic_save
    from fluffy_bassoon.rl.robotwin_fitted_physical import state_sha
    from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, calibration_pair
    backend = control.env.backend
    global_rng, private_rng = rng_state(), deepcopy(control.motor_rng.bit_generator.state)
    original_transform = backend.post_command_transform
    elapsed, previous = control.elapsed_steps, control.previous_agent_command.copy()
    cache = tuple(tuple(t.detach().clone() for t in pair) for pair in backend._kv_cache)
    motor_before, frontend_before = backend.frozen_fingerprint(), state_sha(control.frontend.state_dict())
    noise = np.random.default_rng(spec["motor_seed"]).standard_normal((32, 32)).astype(np.float32)
    bits = lambda a, b: np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
    try:
        guard();control.set_coefficients("native", [0.]*4)
        parity = control.env.check_noise_parity(noise)
        raw = backend.decode(noise)
        raw_hooks_off = backend.kv_adapter is None and backend.action_residual is None and backend.post_decode_residual is None
        raw_again = backend.decode(noise)
        control.set_coefficients("action", [0.]*4)
        backend.post_decode_residual = None
        fitted = backend.decode(noise)
        backend.post_decode_residual = control.action
        zero = backend.decode(noise)
        control.set_coefficients("action", probe_coefficients)
        physical = backend.decode(noise)
        record = backend._last_residual_metadata["post_decode_physical"]
        requested = record["requested_delta"].float().numpy();applied = record["applied_delta"].float().numpy()
        pair = calibration_pair(physical_basis=control.physical_basis(), physical_basis_sha256=control.action.basis_sha256,
            arm_joint_bound=interface["arm_joint_bound"], selected_arm="action", command_lower=interface["joint_lower"],
            command_upper=interface["joint_upper"], range_receipt_sha256=interface["joint_convention_sha256"],
            motor_sha256=interface["motor_sha256"], mapping_sha256=mapping_sha)
        transform = CalibrationDistribution(pair);transform.select(-1)
        backend.post_command_transform = transform
        calibrated = backend.decode(noise);audit = backend._last_residual_metadata
        public = audit["public_agent_requested_actions"].float().numpy()[0]
        actual = audit["audit_only_post_command_transform"]["actuator_commands"].float().numpy()[0]
        motor_after, frontend_after = backend.frozen_fingerprint(), state_sha(control.frontend.state_dict())
        control.assert_frontend_frozen();guard()
        cache_exact = all(torch.equal(t.contiguous().view(torch.uint8), saved.contiguous().view(torch.uint8))
            for row, old in zip(backend._kv_cache, cache) for t, saved in zip(row, old))
        reset_view=control.student_observation(np.zeros(2062,np.float32))
        flags = dict(reset_previous_agent_command_zero=bool(previous.shape==(14,) and bits(previous,np.zeros(14,np.float32))),
            public2076_reset_history_zero=bool(reset_view.shape==(2076,) and bits(reset_view[-14:],np.zeros(14,np.float32))),
            fitted_baseline_zero_exact=bits(fitted, zero), raw_native_control_exact=bool(raw_hooks_off and bits(raw, raw_again)
            and parity["passed"] and math.isfinite(parity["max_abs_error"]) and parity["max_abs_error"] <= parity["tolerance"]),
            physical_delta_bound_passed=bool(np.isfinite(requested).all() and np.isfinite(applied).all()
                and np.max(np.abs(requested)) <= interface["arm_joint_bound"]+1e-6
                and np.max(np.abs(applied)) <= interface["arm_joint_bound"]+1e-6 and bits(physical[:, [6, 13]], fitted[:, [6, 13]])
                and np.all(physical[:, [0,1,2,3,4,5,7,8,9,10,11,12]] >= np.asarray(interface["joint_lower"])[[0,1,2,3,4,5,7,8,9,10,11,12]])
                and np.all(physical[:, [0,1,2,3,4,5,7,8,9,10,11,12]] <= np.asarray(interface["joint_upper"])[[0,1,2,3,4,5,7,8,9,10,11,12]])),
            physical_nonzero_effect=bool(np.any(applied != 0)),
            public_request_before_calibration=bits(public, physical) and bits(actual, calibrated) and not bits(public, actual))
        gradients = dict(motor_gradients=sum(p.grad is not None for p in backend.model.parameters()),
            frontend_gradients=sum(p.grad is not None for p in control.frontend.parameters()))
        unchanged = (motor_before == motor_after and frontend_before == frontend_after == control.fitted_binding["head_state_sha256"]
            and cache_exact and control.elapsed_steps == elapsed and bits(previous, control.previous_agent_command))
        path = Path(output)/"qualification-probes.pt"
        evidence_sha = atomic_save(path, dict(noise=noise, raw_native=raw, fitted_frontend=fitted, zero_physical=zero,
            nonzero_physical=physical, calibrated_probe=calibrated, public_request=public, requested_delta=requested, applied_delta=applied,
            note="decode-only calibration arithmetic probe; no conditioned physics or outcome selection"))
        return dict(format="fitted_physical_motor_qualification_v1", status="observation_checks_passed_episode_pending" if unchanged and all(flags.values()) and not any(gradients.values()) else "failed_observation_checks",
            mapping_sha256=mapping_sha, interface_sha256=interface_sha, frontend_checkpoint_sha256=control.fitted_binding["checkpoint"]["sha256"],
            frozen_before=motor_before, frozen_after=motor_after, frozen_frontend_before=frontend_before, frozen_frontend_after=frontend_after,
            qualification_episode=spec["id"], actual_actions=0, native_parity=parity, cache_exact=cache_exact,
            extra_robot_actions=control.elapsed_steps-elapsed, extra_episodes=0, roundoff_tolerance=1e-6,
            evidence=dict(path=str(path), sha256=evidence_sha), **flags, **gradients)
    finally:
        backend.post_command_transform = original_transform
        control.set_coefficients(spec["arm"], spec["coefficients"])
        restore_rng(global_rng)
        if control.motor_rng.bit_generator.state != private_rng:
            raise RuntimeError("Decode qualification advanced the actual private acting RNG")


class FrozenZeroController:
    """Metadata adapter for the existing loop; no model, optimizer or learning."""
    def __init__(self,spec): self.spec=spec
    def act(self,features): return [0.]*4
    def decision_metadata(self):
        return dict(policy_version=self.spec['policy_version'],
            actual_acting_checkpoint_iteration=self.spec['actual_acting_checkpoint_iteration'],
            new_optimizer_updates=0,optimizer_updates=0,source_warmup=False,
            steering_lineage_id='TRAIN-capacity-diagnostic',steering_condition_id='natural',steering_partition='train')
    def observe(self,*_): pass


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','ticket','gpu-probe','model-path','norm-stats','robotwin-root','rlinf-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--request-sha256',required=True);parser.add_argument('--ticket-sha256',required=True)
    a=parser.parse_args(argv);request=read_ref(dict(path=str(a.request),sha256=a.request_sha256))
    ticket=read_ref(dict(path=str(a.ticket),sha256=a.ticket_sha256))
    if request.get('status')!='owner_bound_request' or ticket.get('admitted') is not True:
        raise ValueError('UNBOUND request/grant template is not runtime admission; no model or reset')
    if sys.platform!='linux': raise RuntimeError('Only actual allocated Linux resources may compute')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Clean exact parent-published source required')
    authority,protocol,grant=(read_ref(request[k]) for k in ('authority','protocol','grant'))
    validate_protocol(protocol,grant,authority)
    validate_parent_links(grant,request)
    clock=read_ref(ticket['current_worker_clock'])
    validate_ticket(request,ticket,clock,source=source,request_sha=a.request_sha256,
        hostname=socket.gethostname(),output=a.output,now=time.time())
    if ticket.get('continuation_admitted'):read_ref(ticket['diagnosis_receipt'])
    resume_head=read_ref(ticket['current_head']) if request.get('resume') is not None else None
    allocation=read_ref(ticket['actual_allocation'])
    if (allocation.get('job_id')!=ticket.get('job_id') or allocation.get('resources')!=grant['GPU_resources']
            or allocation.get('source_commit')!=source):
        raise ValueError('Actual scheduler allocation and exact qualified source receipt required')
    cpu=read_ref(ticket['focused_CPU_qualification']);read_ref(ticket['latest_main_review']);read_ref(ticket['storage_admission'])
    if cpu.get('source_commit')!=source or cpu.get('all_passed') is not True:
        raise ValueError('Actual focused exact-source CPU PASS required')
    terminal=read_ref(request['predecessor_terminal']);old=terminal['canonical_snapshot']
    if (terminal.get('all_checked_passed') is not True or old['actual_actions']!=29922 or old['charged_actions']!=29922
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions']
            or terminal['full_recovery']['matches'] is not True
            or terminal['full_recovery']['actual']['sha256']!=ORIGIN
            or {k:terminal['full_recovery']['actual'][k] for k in ('path','sha256')}!=request['origin_recovery']):
        raise ValueError('Exact closed15/29922/full recovery physical origin; old costs are not recharged')
    descriptor=terminal['frozen_heads']['action'];inventory=read_ref(request['inventory'])
    producers={};manifests={}
    for i in range(4):
        ident=f'phone-{i:02d}';evidence=descriptor['fit_provenance']['outcome_evidence'][ident]
        producers[ident]=read_ref(evidence['producer_result'])
        manifest=evidence['episode_manifest_ref'];relative=Path(manifest['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Canonical original manifest path')
        manifests[ident]=read_ref(dict(path=str(Path(manifest['root'])/relative),sha256=manifest['sha256']))
    specs=build_cells(protocol,inventory,descriptor,producers,manifests)
    # Tensor/runtime imports follow all real funding/origin/identity checks.
    import numpy as np
    import torch
    from fluffy_bassoon.rl.robotwin_fitted_physical import create_control,HeadroomPhysical4D,content_sha,validate_binding
    from fluffy_bassoon.rl.robotwin_steering import validate_interface,digest
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv,_RoboTwinBackend,load_benchmark_episode_limits
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter,_atomic_json,_fsync_directory
    from fluffy_bassoon.baige_gpu import verify_selected_cuda,selected_device
    from .robotwin_steering_screen import run_complete_episode,verify_source_conventions
    from .robotwin_reactive_capacity import validate_capacity_locked
    from .robotwin_event_replay import physical_audit,numpy_state
    from .robotwin_command_diagnostic_readout import initial_physical_identity,publish_recovery,prepare_output,checked_runtime_source
    from .robotwin_prelift_probe import observe_native_plans
    from .checkpoint import rng_state,restore_rng
    from .warmstart import atomic_save
    for obj in (main,create_control,HeadroomPhysical4D,_RoboTwinBackend,run_complete_episode,physical_audit,
                initial_physical_identity,publish_recovery,observe_native_plans,rng_state,atomic_save,CampaignBudget,RobotwinCorpusWriter,validate_capacity_locked):
        checked_runtime_source(obj,root,request['runtime_blob_sha256'])
    for relative,expected in request['runtime_blob_sha256'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(root/path)!=expected:raise ValueError('Runtime blob mismatch')
    base=validate_interface(read_ref(request['base_interface']))
    if request['base_interface']['sha256']!=descriptor['base_interface_sha256']:
        raise ValueError('Original fitted base interface bytes differ')
    module=root/'src/fluffy_bassoon/rl'
    binding=dict(descriptor,format='frozen_fitted_frontend_physical4d_v1',
        physical_basis_sha256=digest(HeadroomPhysical4D(base).basis.detach().cpu().numpy()),
        wrapper_sha256=sha(module/'robotwin_fitted_physical.py'),backend_sha256=sha(module/'robotwin_noise.py'),
        fixed_control_sha256=sha(module/'robotwin_steering.py'),frontend_implementation_sha256=sha(module/'robotwin_wide_action.py'),
        residual_primitives_sha256=sha(module/'pi05_residual.py'))
    interface=dict(base,fitted_frontend=binding);validate_binding(interface)
    interface_sha=content_sha(interface)
    if sha(a.model_path/'model.safetensors')!=base['motor_sha256'] or sha(a.norm_stats)!=base['normalization_sha256']:
        raise ValueError('Actual original motor/normalization files differ')
    conventions=read_ref(request['joint_conventions'])
    if request['joint_conventions']['sha256']!=base['joint_convention_sha256']:raise ValueError('Joint convention bytes')
    verify_source_conventions(conventions,interface,dict(robotwin=a.robotwin_root,rlinf=a.rlinf_root))
    limits=load_benchmark_episode_limits(a.robotwin_root,['place_phone_stand'])
    probe=read_ref(dict(path=str(a.gpu_probe),sha256=sha(a.gpu_probe)))
    if limits['place_phone_stand']!=400 or probe.get('status')!='passed' or probe.get('hostname')!=socket.gethostname() or len(probe.get('devices',[]))!=1:
        raise ValueError('Actual native400/one-GPU CUDA qualification required')
    if verify_selected_cuda(torch)!=selected_device()['uuid'] or selected_device()['cuda_uuid']!=probe['devices'][0]['cuda_uuid']:
        raise ValueError('Actual inference/renderer GPU selection differs')
    origin=Path(request['origin_recovery']['path'])
    if origin.is_symlink() or sha(origin)!=ORIGIN:raise ValueError('Actual closed origin recovery bytes differ')
    saved=torch.load(origin,map_location='cpu',weights_only=False)
    validate_origin(saved,old)
    origin_saved=saved
    ledger=ticket['action_ledger'];expected_contract=dict(phase_id=PHASE,authority_sha256=request['authority']['sha256'],
        protocol_sha256=request['protocol']['sha256'],grant_sha256=request['grant']['sha256'],
        prior_closed_actions=29922,origin_recovery_sha256=ORIGIN,source_eligible=False)
    if (ledger['campaign_id']!=PHASE or ledger['total_actions']!=3200 or ledger['source_caps']!={}
            or ledger['phase_caps']!={ACTION_PHASE:3200} or ledger['contract']!=expected_contract
            or (request.get('resume') is None and Path(ledger['root']).exists())
            or not Path(ledger['root']).resolve().is_relative_to(Path(ticket['aggregate_output_root']).resolve())):
        raise ValueError('Same scoped generic3200 ledger; only an initial start requires a fresh root')
    panel_contract=dict(protocol=request['protocol'],authority=request['authority'],grant=request['grant'],
        source_commit=source,interface_sha256=interface_sha,inventory=request['inventory'],origin_recovery=request['origin_recovery'])
    if request.get('resume') is not None:
        ref=request['resume'];path=Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or sha(path)!=ref['sha256']:raise ValueError('Actual full closed recovery bytes required')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        validate_resume(saved,resume_head,specs,contract=panel_contract,ledger=ledger,output=a.output)
        verify_resume_journal(resume_head,ledger,ticket['run_id']);verify_prefix_files(saved['state'])
        if not (a.output/'recoveries').is_dir():raise ValueError('Existing canonical output/recovery directory required')
    else:prepare_output(a.output)
    if os.statvfs(a.output).f_bavail*os.statvfs(a.output).f_frsize<4<<30:raise RuntimeError('Actual output capacity')
    write=a.output/'.write-probe'
    with write.open('xb') as stream:stream.write(b'capacity8\n');stream.flush();os.fsync(stream.fileno())
    write.unlink()
    if request.get('resume') is None:
        _atomic_json(a.output/'resolved-cells.json',specs);_atomic_json(a.output/'interface.json',interface)
    elif json.loads((a.output/'resolved-cells.json').read_text())!=specs or json.loads((a.output/'interface.json').read_text())!=interface:
        raise ValueError('Resume cannot replace original cell/interface binding files')
    state=dict(format=FORMAT,contract=dict(request_sha256=a.request_sha256,source_commit=source,interface_sha256=interface_sha),
        panel_contract=panel_contract,output=str(a.output.resolve()),status='running',next_cell=0,records=[],pending=None,control_active=False,closed_controls=[],physical_resets={},
        qualification={},case_audits=[],actions=None,action_ledger=deepcopy(ledger),previous_recovery=request['origin_recovery'],
        prior_closed_actions=29922,source_eligible=False,source_student_admitted=False,A1_selection=None,new_source_actions=0,
        actual_acting_checkpoint_iteration=5000,new_optimizer_updates=0,optimizer_updates=0,
        original_fit_provenance=deepcopy(descriptor['fit_provenance']))
    forks={}
    if request.get('resume') is not None:
        state=deepcopy(saved['state']);state['status']='running';forks=deepcopy(saved['case_rngs'])
    state.setdefault('attempt_requests',[]).append(dict(request_sha256=a.request_sha256,run_id=ticket['run_id'],resume=request.get('resume')))
    env=control=None;acting_ready=False;last_closed_control=deepcopy(saved['last_closed_control'])
    maximum_episode_seconds=state.get('maximum_episode_seconds',ticket['minimum_episode_admission_seconds'])
    def guard():
        if time.time()+ticket['cleanup_seconds']>=ticket['driver_deadline_unix']:raise TimeoutError('Owner clock; partial preserved, retry OFF')
        paths=list(Path(ticket['aggregate_output_root']).rglob('*'))
        if any(p.is_symlink() for p in paths) or sum(p.stat().st_size for p in paths if p.is_file())>4<<30:
            raise RuntimeError('Aggregate fresh4GiB bound')
    def persist(boundary=False):
        state['control_active']=False if control is None else control.active
        last=last_closed_control
        snapshot=deepcopy(saved['control']) if not acting_ready else dict(control.snapshot(),active=control.active,elapsed_steps=control.elapsed_steps,
            last_reset_info=deepcopy(getattr(control,'last_reset_info',None)))
        publish_recovery(a.output,state,dict(state=deepcopy(state),global_rng=rng_state() if acting_ready else deepcopy(saved['global_rng']),
            runtime_global_rng=rng_state(),acting_initialization_complete=acting_ready,case_rngs=deepcopy(forks),
            control=snapshot,last_closed_control=deepcopy(last),optimizer=None,
            origin_private_control=deepcopy(origin_saved['control'])),boundary=boundary)
    try:
        with CampaignBudget(**{k:ledger[k] for k in ('root','campaign_id','total_actions','source_caps','phase_caps','contract')},
                run_id=ticket['run_id'],source_commit=source) as budget:
            try:
                if request.get('resume') is not None:validate_capacity_locked(budget,resume_head)
                elif budget.snapshot()['charged_actions'] or budget.remaining(phase=ACTION_PHASE,method='train_capacity8')!=3200:
                    raise ValueError('Exactly fresh zero-charge3200 counter before initial runtime')
                if budget.remaining(phase=ACTION_PHASE,method='train_capacity8')<400*(8-state['next_cell']):
                    raise ValueError('Full remaining natural episodes must fit the same unrefilled ledger')
                guard();env=RobotwinNoiseEnv(model_path=a.model_path,norm_stats=a.norm_stats,robotwin_root=a.robotwin_root,
                    rlinf_root=a.rlinf_root,output_dir=a.output/'runtime',action_horizon=32,num_steps=5,task_episode_limits=limits)
                control=create_control(env,interface,interface_sha256=interface_sha)
                control.set_coefficients('action',[0.]*4);mapping=control.mapping_sha256()
                if state.get('mapping_sha256') not in (None,mapping):raise ValueError('Resume mapping drift')
                state['mapping_sha256']=mapping
                writer=RobotwinCorpusWriter(a.output/'corpus',dict(role=FORMAT,source_commit=source,contract=state['contract'],
                    source_eligible=False,new_optimizer_updates=0,actual_acting_checkpoint_iteration=5000))
                stats=writer.statistics()
                if (stats['unresolved_decisions'] or stats['episode_count']!=state['next_cell']
                        or stats['executed_primitive_steps']!=sum(r['primitive_steps'] for r in state['records'])):
                    raise ValueError('Canonical corpus must contain exactly the complete closed prefix')
                control.attach_corpus(writer)
                if env.frozen_fingerprint()!=MOTOR:raise ValueError('Actual frozen motor state differs')
                # Initialization may consume RNG; restore the actual previous ancestry AFTER it.
                prior=saved['control'].get('motor_rng')
                if not isinstance(prior,dict):raise ValueError('Original private motor RNG missing')
                control.motor_rng=np.random.default_rng();control.motor_rng.bit_generator.state=deepcopy(prior)
                if request.get('resume') is not None and saved['control'].get('mapping'):
                    previous=saved['control']
                    if previous['mapping']!=control.config():raise ValueError('Saved frozen control config drift')
                    control.set_coefficients(previous['arm'],previous['proposed_coefficients'])
                    control.previous_agent_command=np.asarray(previous['previous_agent_command'],np.float32).copy()
                    if control.previous_agent_command.shape!=(14,) or not np.isfinite(control.previous_agent_command).all():
                        raise ValueError('Closed public requested14 history must be restored exactly')
                    env.elapsed_steps=previous['elapsed_steps'];env.last_reset_info=deepcopy(previous['last_reset_info'])
                restore_rng(saved['global_rng']);acting_ready=True;state['actions']=budget.snapshot();persist(True)
                def episode(spec,*,qualify):
                    nonlocal maximum_episode_seconds,last_closed_control
                    guard();key=spec['record_id'];start=time.monotonic();audit_refs=[];before_chunk={}
                    with ExitStack() as stack:
                        def before(current,actual):
                            task=current.env.backend.env.venv.envs[0].task
                            physical=physical_audit(task);identity=initial_physical_identity(physical)
                            if key in state['physical_resets'] and state['physical_resets'][key]!=identity:
                                raise ValueError('Actual paired actor/measured/drive/gripper initial differs')
                            state['physical_resets'][key]=identity
                            _atomic_json(a.output/'reset-audits'/(spec['id']+'.json'),physical)
                            if qualify:
                                q=qualify_first_raw(current,actual,a.output,interface,mapping_sha=state['mapping_sha256'],
                                    interface_sha=interface_sha,probe_coefficients=[1.,0.,0.,0.],guard=guard)
                                q['initial_physics_unchanged']=identity==initial_physical_identity(physical_audit(task))
                                q['scope']='decode-only basis-unit integrity probe; no nonzero4D physics/search/selection'
                                state['qualification']=q
                                if q['status']!='observation_checks_passed_episode_pending' or not q['initial_physics_unchanged']:
                                    raise RuntimeError('Embedded first-raw qualification failed')
                            before_chunk['plans']=stack.enter_context(observe_native_plans(task))
                            persist(False)
                        class Metadata(FrozenZeroController):
                            def act(self,features):
                                before_chunk.update(global_numpy=numpy_state(),physical=physical_audit(control.env.backend.env.venv.envs[0].task))
                                persist(False);return super().act(features)
                        def chunk(transition):
                            guard();state['actions']=budget.snapshot();task=control.env.backend.env.venv.envs[0].task
                            audit=dict(cell=spec['id'],chunk=transition['chunk'],native_global_numpy_before=before_chunk['global_numpy'],
                                native_global_numpy_after=numpy_state(),physical_before=before_chunk['physical'],physical_after=physical_audit(task),
                                native_plan_observation=deepcopy(before_chunk['plans']),actual_acting_checkpoint_iteration=spec['actual_acting_checkpoint_iteration'],
                                new_optimizer_updates=0,public2076_previous_requested14=control.student_observation(transition['next_observations'])[-14:].tolist(),
                                full_submitted_plan_and_valid_mask='canonical corpus decision/chunk NPZ')
                            path=a.output/'chunk-audits'/spec['id']/(str(transition['chunk'])+'.json');_atomic_json(path,audit)
                            audit_refs.append(dict(path=str(path),sha256=sha(path)));persist(False)
                        result=run_complete_episode(control,spec,budget,run_id=ticket['run_id'],phase=ACTION_PHASE,
                            controller=Metadata(spec),before_step=before,on_chunk=chunk)
                    if qualify:
                        if result['primitive_steps']<=0:raise RuntimeError('Qualification needs actual charged complete first episode')
                        state['qualification'].update(status='passed',actual_actions=result['primitive_steps'],complete_real_episode=spec['id'])
                        _atomic_json(a.output/'motor-qualification.json',state['qualification'])
                    control.assert_frontend_frozen()
                    if any(p.grad is not None for p in control.frontend.parameters()):raise RuntimeError('Frozen frontend gradient appeared')
                    last_closed_control=result.pop('control_recovery')
                    path=a.output/'recoveries'/(spec['id']+'-closed-control.pt')
                    if path.exists():raise ValueError('A completed cell cannot be replayed or overwritten')
                    digest=atomic_save(path,dict(control=last_closed_control,global_rng=rng_state()))
                    state['closed_controls'].append(dict(path=str(path),sha256=digest))
                    result['native_chunk_audit_refs']=audit_refs
                    _atomic_json(a.output/'episodes'/(spec['id']+'.json'),result)
                    maximum_episode_seconds=max(maximum_episode_seconds,1.5*(time.monotonic()-start));state['maximum_episode_seconds']=maximum_episode_seconds
                    state['actions']=budget.snapshot();return result
                run_cells(state,specs,forks,episode,capture=rng_state,restore=restore_rng,
                    admit=lambda:time.time()+maximum_episode_seconds+ticket['cleanup_seconds']<ticket['driver_deadline_unix'],persist=persist)
                control.assert_frontend_frozen();state['final_frozen_motor']=env.frozen_fingerprint()
                if state['final_frozen_motor']!=MOTOR:raise RuntimeError('Frozen motor changed')
                persist(True)
            finally:state['actions']=budget.snapshot()
    except BaseException as error:
        state.update(status='blocked_failure',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        try:persist(state.get('pending') is None and (control is None or not control.active))
        finally:
            try:
                if env is not None:env.close()
            finally:_atomic_json(a.output/'result.json',state,replace=True)


if __name__=='__main__':main()
