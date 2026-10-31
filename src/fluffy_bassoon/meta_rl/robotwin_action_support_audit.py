"""Allocated-CPU support bounds for captured successful native action plans.

This inspects existing arrays only. It does not fit a policy or infer that an
interface cannot succeed using a different plan. No model or simulator imports.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

from .robotwin_replay_diagnosis import Audit, json_write, validate_reset_identity

PRODUCER = 'be247676c1217d9e046091c52cac2b7d0f2d3014'
RESULT_SHA = 'ba775b4fcbb3029120927ecd59461fabb1cccd7e95326f8d9f3a6f731ea77d3f'
TERMINAL_SHA = 'a352462ced9ce556498b1871d76fdeacc0a6837e22fe54d36a01f203a5db1bef'
INTERFACE_SHA = 'c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb'
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIP = [6, 13]
IDS = [f'event-teacher-phone-{i:02d}' for i in range(4)]
HEAD_SOURCES = {
    'src/fluffy_bassoon/meta_rl/robotwin_reactive_fit.py': 'e910770f0eceb314a410f01bbffd553fa3659e7acda8f36810373009dd6831f8',
    'src/fluffy_bassoon/meta_rl/robotwin_reactive_capacity.py': 'a501b515947f2e76a1c93fb8025fa6c405d68692248356161afa985d2de9672c',
    'src/fluffy_bassoon/rl/pi05_residual.py': 'dd647356594a9086cb006fbfedbce5715cde8f99a9d5f03d9c30a884744999ff',
}


def support_bounds(base, target, submitted, executed, joint_lower, joint_upper, *, arm_delta_bound):
    """Infimum of plan MSE under the recorded head's coordinate bounds.

    The head edits twelve arms by the specified bound, clips to joint headroom,
    and preserves raw grippers. This bound ignores network parameter coupling.
    """
    import numpy as np
    if type(arm_delta_bound) not in (int, float) or not math.isfinite(arm_delta_bound) or arm_delta_bound <= 0:
        raise ValueError('Positive finite arm correction bound required')
    arm_delta_bound = float(np.float32(arm_delta_bound))
    if not math.isfinite(arm_delta_bound) or arm_delta_bound <= 0:
        raise ValueError('Arm correction bound must be representable in Float32')
    base, target = np.asarray(base), np.asarray(target)
    # make_heads stores joint limits as Float32 buffers. Preserve those actual
    # endpoints before the audit's higher-precision geometric arithmetic.
    lower = np.asarray(joint_lower, dtype=np.float32).astype(np.float64)
    upper = np.asarray(joint_upper, dtype=np.float32).astype(np.float64)
    submitted, executed = np.asarray(submitted), np.asarray(executed)
    if (base.shape != (32, 14) or target.shape != base.shape
            or base.dtype.kind != 'f' or target.dtype.kind != 'f'
            or not np.isfinite(base).all() or not np.isfinite(target).all()
            or lower.shape != (14,) or upper.shape != (14,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or not (lower < upper).all()
            or submitted.shape != (32,) or executed.shape != (32,)
            or not np.isin(submitted, [0, 1]).all() or not np.isin(executed, [0, 1]).all()):
        raise ValueError('Finite H32 physical14 plans, limits and binary masks required')
    submitted, executed = submitted.astype(bool), executed.astype(bool)
    if (not submitted.any() or not executed.any()
            or not np.array_equal(submitted, np.arange(32) < submitted.sum())
            or not np.array_equal(executed, np.arange(32) < executed.sum())
            or np.any(executed & ~submitted)):
        raise ValueError('Nonempty executed prefix must be inside submitted prefix')
    if np.any(base[:, ARM] < lower[ARM]) or np.any(base[:, ARM] > upper[ARM]):
        raise ValueError('Recorded native ARM proposal violates the actual head admission')
    lo, hi = base.astype(np.float64).copy(), base.astype(np.float64).copy()
    # The head casts its Float32-bounded correction to the base dtype before
    # adding it. Native addition may round beyond a Float64 base +/- radius.
    radius = np.asarray(arm_delta_bound, dtype=base.dtype)
    lo[:, ARM] = np.maximum(np.subtract(base[:, ARM], radius, dtype=base.dtype).astype(np.float64), lower[ARM])
    hi[:, ARM] = np.minimum(np.add(base[:, ARM], radius, dtype=base.dtype).astype(np.float64), upper[ARM])
    difference = target.astype(np.float64) - np.maximum(lo, np.minimum(hi, target))
    requested_delta = target.astype(np.float64) - base
    result = {}
    for name, mask in [('submitted_plan', submitted), ('native_valid_prefix', executed)]:
        error = difference[mask]
        result[name] = dict(slots=int(mask.sum()), squared_error_sum14=(error ** 2).sum(0).tolist(),
            outside_support_counts14=(np.abs(error) > 1e-6).sum(0).tolist(),
            maximum_absolute_requested_delta14=np.abs(requested_delta[mask]).max(0).tolist(),
            MSE_floor_physical14=float((error ** 2).mean()),
            MSE_floor_arms12=float((error[:, ARM] ** 2).mean()),
            MSE_floor_grippers2=float((error[:, GRIP] ** 2).mean()))
    return result


def combine(rows, key):
    """Weight by actual label count, rather than averaging per-chunk MSE."""
    import numpy as np
    slots = sum(row[key]['slots'] for row in rows)
    squared = np.asarray([row[key]['squared_error_sum14'] for row in rows]).sum(0)
    return dict(slots=slots, squared_error_sum14=squared.tolist(),
        outside_support_counts14=np.asarray([row[key]['outside_support_counts14'] for row in rows]).sum(0).tolist(),
        MSE_floor_physical14=float(squared.sum() / (slots * 14)),
        MSE_floor_arms12=float(squared[ARM].sum() / (slots * 12)),
        MSE_floor_grippers2=float(squared[GRIP].sum() / (slots * 2)))


def physical_plan(arrays, metadata, name):
    import numpy as np
    ref = metadata[name]
    value = arrays[ref['numeric_array']]
    if value.shape != (1, 32, 14) or value.dtype.kind != 'f' or not np.isfinite(value).all():
        raise ValueError('Captured single-batch native H32 physical14 required')
    return value[0]


def inspect(packet, output, audit):
    import numpy as np
    if (packet.get('format') != 'robotwin_action_support_inputs_v1'
            or packet['result']['sha256'] != RESULT_SHA or packet['terminal']['sha256'] != TERMINAL_SHA
            or packet['interface']['sha256'] != INTERFACE_SHA
            or packet.get('episode_ids') != IDS or packet.get('arm_delta_bound') != .05
            or packet.get('counterfactual_arm_delta_bound') != 2.
            or packet.get('gripper_delta_bound') != 0.):
        raise ValueError('Exact producer, four preselected successful cases and head contract required')
    result, terminal, interface = (audit.json(packet[k]) for k in ('result', 'terminal', 'interface'))
    if interface['arm_joint_bound'] != packet['arm_delta_bound']:
        raise ValueError('Actual head bound differs from its sealed interface')
    if (result['source_commit'] != PRODUCER or result['status'] != 'blocked_failure'
            or result['stage'] != 'teacher_executability' or result['pending'] is not None
            or terminal['source_commit'] != PRODUCER or terminal['exit_code'] != 1
            or terminal['actual_new_actions'] != 1572 or terminal['cache_created']
            or terminal['kv_updates'] or terminal['action_updates']
            or result['actions']['actual_actions'] != 20171):
        raise ValueError('Only the exact closed no-fit event experiment is accepted')
    if ({ref['relative_path']: ref['sha256'] for ref in packet['head_source_refs']} != HEAD_SOURCES
            or len(packet['head_source_refs']) != 3):
        raise ValueError('Three exact producer source bindings required')
    for ref in packet['head_source_refs']:
        if not str(ref['path']).endswith('/'+ref['relative_path']):
            raise ValueError('Head source identity/path mismatch')
        audit.bound(ref['path'], ref['sha256'])
    records = {row['spec']['id']: row for row in result['records']}
    if len(records) != 6 or not set(IDS).issubset(records):
        raise ValueError('Complete original result with all selected records required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    all_chunks, episodes = [], []
    for identity in IDS:
        audit.guard(); row = records[identity]
        if (not row['success'] or not row['complete'] or row['primitive_steps'] != 193
                or row['spec']['stage'] != 'event_aligned_teacher_execution'
                or row['spec']['teacher_id'] != identity.removeprefix('event-teacher-')):
            raise ValueError('Only the preselected complete successful aligned plans are inspected')
        ref = row['episode_manifest_ref']; root = Path(ref['root'])
        path = audit.nested(root, ref)
        manifest = audit.json(dict(path=str(path), sha256=ref['sha256']))
        reset_path = path.with_name('reset.json')
        if reset_path.is_symlink() or not reset_path.is_file() or reset_path.stat().st_size > 4 << 20:
            raise ValueError('Small regular reset metadata required')
        audit.guard()
        reset_sha = hashlib.sha256(reset_path.read_bytes()).hexdigest()
        reset = audit.json(dict(path=str(reset_path), sha256=reset_sha))
        validate_reset_identity(reset, row['reset_info'], manifest)
        if (manifest['pending_decision'] is not None or len(manifest['chunks']) != row['chunks']
                or len(row['corpus_refs']) != len(manifest['chunks']) or manifest['executed_primitive_steps'] != 193):
            raise ValueError('Only closed bound canonical episodes are accepted')
        chunks = []; counted = 0
        for ordinal, ref in enumerate(manifest['chunks']):
            path_chunk = audit.nested(path.parent, ref)
            closed_ref = row['corpus_refs'][ordinal]
            if (closed_ref['sha256'] != ref['sha256']
                    or (Path(closed_ref['root'])/closed_ref['path']).resolve() != path_chunk.resolve()):
                raise ValueError('Closed record and manifest chunk identities differ')
            chunk = audit.json(dict(path=str(path_chunk), sha256=ref['sha256']))
            if (chunk['status'] != 'complete' or chunk['context']['trial_id'] != identity
                    or chunk['learning_result']['budget_truncated']):
                raise ValueError('Foreign, incomplete or budget-cut chunk')
            decisions = audit.arrays(root, chunk['decision_arrays'])
            native = audit.arrays(root, chunk['native_result']['arrays'])
            metadata = chunk['decision_metadata']['frozen_pi05_residual']
            base, target = (physical_plan(decisions, metadata, name) for name in ('base_actions', 'actions'))
            issued = decisions['issued_motor_joint_targets']; n = len(issued)
            valid = native['valid_action_mask'].reshape(-1)
            actual = chunk['native_result']['executed_primitive_steps']
            closed = bool(chunk['learning_result']['terminated'] or chunk['learning_result']['truncated'])
            if (issued.shape != (n, 14) or not 1 <= n <= 32 or not np.array_equal(issued, target[:n])
                    or valid.size != n or not np.array_equal(valid, np.arange(n) < actual)
                    or (not closed and (actual != 32 or n != 32))
                    or closed != (ordinal == len(manifest['chunks']) - 1)):
                raise ValueError('Captured issued plan, native mask or closed chain mismatch')
            submitted = np.arange(32) < n; executed = np.zeros(32, bool); executed[:n] = valid.astype(bool)
            support = support_bounds(base, target, submitted, executed, interface['joint_lower'], interface['joint_upper'],
                arm_delta_bound=packet['arm_delta_bound'])
            support['counterfactual_2rad'] = support_bounds(base, target, submitted, executed,
                interface['joint_lower'], interface['joint_upper'], arm_delta_bound=packet['counterfactual_arm_delta_bound'])
            support.update(ordinal=ordinal, canonical_chunk=dict(path=str(path_chunk), sha256=ref['sha256']),
                proposed_slots=32, submitted_slots=n, native_valid_slots=actual,
                terminal_unexecuted_submitted_slots=n-actual)
            chunks.append(support); counted += actual
        if counted != row['primitive_steps']:
            raise ValueError('Resolved native action total differs from canonical chunks')
        episodes.append(dict(id=identity, success=True, native_actual_actions=counted, chunks=chunks,
            submitted_plan=combine(chunks, 'submitted_plan'), native_valid_prefix=combine(chunks, 'native_valid_prefix'),
            counterfactual_2rad={key:combine([c['counterfactual_2rad'] for c in chunks], key)
                for key in ('submitted_plan', 'native_valid_prefix')}))
        all_chunks.extend(chunks)
    for path, ref in list(audit.files.items()): audit.bound(path, ref['sha256'])
    report = dict(format='robotwin_recorded_action_support_audit_v1', status='audit_complete', producer=PRODUCER,
        actual_arm_delta_bound=packet['arm_delta_bound'], counterfactual_arm_delta_bound=packet['counterfactual_arm_delta_bound'],
        episodes=episodes, submitted_plan=combine(all_chunks, 'submitted_plan'),
        native_valid_prefix=combine(all_chunks, 'native_valid_prefix'), input_files=audit.files,
        counterfactual_2rad={key:combine([c['counterfactual_2rad'] for c in all_chunks], key)
            for key in ('submitted_plan', 'native_valid_prefix')},
        interpretation=[
            'Coordinatewise MSE infimum for recorded bases and targets; network parameter coupling can make the actual fit worse.',
            'A nonzero floor proves only that this bounded head cannot exactly imitate these recorded successful plans; different bounded plans may still succeed.',
            'Native valid indices are cost/observation boundaries, not physical waypoint traces or a causal independence mask.',
            'Full known submitted plans are the imitation labels; native valid-prefix bounds are supplementary accounting diagnostics.',
            'Raw gripper overshoot is preserved; gripper targets are not silently clipped.',
            'Actual producer bound is0.05rad;2rad is an explicitly separate counterfactual, never a claim about the executed or fitted head.',
            'Coordinate intervals use Float32 buffer limits and bounds, correction casting and rounded addition in the captured base dtype; distances use Float64 arithmetic. Network coupling and discrete attainable-value gaps are ignored.',
            'Selected four known successes are a support diagnostic, not a qualified reduced training/validation bank or a reliability estimate.',
            'Recorded base proposals are SHA-bound captured native outputs, not newly recomputed or resampled motor predictions.'],
        new_robot_actions=0, model_forwards=0, SGD=0)
    if len(json.dumps(report, allow_nan=False).encode()) > 4 << 20:
        raise ValueError('Small metadata report exceeds4MiB')
    json_write(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'allocation', 'zero-gpu-proof'):
        parser.add_argument('--'+name, type=Path, required=True); parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--source-commit', required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True); args = parser.parse_args(argv)
    if sys.platform != 'linux' or not math.isfinite(args.deadline_unix):
        raise RuntimeError('Allocated Linux CPU and shared finite deadline required')
    audit = Audit(args.deadline_unix)
    allocation = audit.json(dict(path=str(args.allocation), sha256=args.allocation_sha256))
    proof = audit.json(dict(path=str(args.zero_gpu_proof), sha256=args.zero_gpu_proof_sha256))
    def git(*values):
        audit.guard(); return subprocess.check_output(['git', *values], text=True,
            timeout=max(.1, min(10, args.deadline_unix-time.time()))).strip()
    if (git('rev-parse', 'HEAD') != args.source_commit or git('status', '--porcelain')
            or allocation.get('source_commit') != args.source_commit or allocation.get('hostname') != socket.gethostname()
            or allocation.get('gpu_count') != 0 or allocation.get('deadline_unix') != args.deadline_unix
            or allocation.get('phase_id') != 'frozen-pi05-endtoend-20261031-r1'
            or allocation.get('prior_phase_cpu_charge', -1) < 978
            or allocation.get('maximum_phase_cpu_worker_seconds') != 1800
            or not 1 <= allocation['maximum_cpu_worker_seconds'] <= 180
            or allocation['prior_phase_cpu_charge']+allocation['maximum_cpu_worker_seconds'] > 1800
            or not allocation['started_unix'] <= time.time() < args.deadline_unix <= allocation['started_unix']+allocation['maximum_cpu_worker_seconds']
            or Path(allocation['inspection_output']).resolve() != args.output.resolve()
            or args.output.resolve() != Path(allocation['output']).resolve()/'inspection'
            or proof.get('passed') is not True or proof.get('hostname') != socket.gethostname()
            or proof.get('device_count_before_mask') != 0 or proof.get('cuda_initialized_before_mask') is not False
            or proof.get('actual_cpu_cores', 0) < 2 or proof.get('actual_memory_bytes', 0) < 16 << 30
            or allocation.get('zero_gpu_attestation_sha256') != args.zero_gpu_proof_sha256):
        raise ValueError('Actual clean bound0GPU/2CPU/16GiB allocation and inclusive clock required')
    report = inspect(audit.json(dict(path=str(args.inputs), sha256=args.inputs_sha256)), args.output, audit)
    if any(key.split('.')[0] in ('torch', 'jax', 'rlinf', 'openpi', 'sapien') for key in sys.modules):
        raise RuntimeError('Unexpected model/simulator import')
    print(json.dumps(dict(status=report['status'], episodes=4, new_robot_actions=0, model_forwards=0, SGD=0)))


if __name__ == '__main__':
    main()
