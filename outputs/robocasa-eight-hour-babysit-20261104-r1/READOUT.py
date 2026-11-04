"""Six-episode matched readout of one frozen, validation-selected joint learner."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--method', choices=('ad', 'dpt', 'dit', 'ic_iql', 'native_initial', 'native_final'), required=True)
    parser.add_argument('--split', choices=('ID', 'OOD'), required=True)
    parser.add_argument('--stage', choices=('qualification', 'final'), required=True)
    parser.add_argument('--count', type=int, choices=(2, 8, 50), required=True)
    parser.add_argument('--memory-mode', choices=('all', 'full', 'episode'), default='all')
    parser.add_argument('--shards', type=int, choices=(1,2), default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--snapshot-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--expected-hostname', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Require the exact allocated Linux container')
    if args.stage == 'final' and args.count != 50:
        raise ValueError('Final panels have fifty cases per task')
    if not 0 <= args.shard_index < args.shards:
        raise ValueError('Select a complete-bank shard within the bound partition')
    if args.snapshot_manifest and (args.stage!='qualification' or args.count!=2 or args.split!='ID' or args.shards!=1):
        raise ValueError('Interim snapshots qualify deployment on the two-case ID bank only')
    root = Path('/mowenjie/fluffy-bassoon-2')
    sys.path.insert(0, str(args.code / 'src'))
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, tensor_hash
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_policy
    checked_checkout(args.code, args.source_commit)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    import torch
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor, FlowReferenceMotor, ReferencePlaceholder
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions, _json, _RecordingEnvironment, native_video_writer
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend, case_key
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=False)
    fit = root / 'outputs/robocasa-three-task-meta-baseline-20261104-r2/joint'
    assembly = fit / 'preparation/ASSEMBLY.json'
    if file_sha256(assembly) != 'fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758':
        raise ValueError('The qualified pooled corpus changed')
    corpus = json.loads(assembly.read_text())
    selected = dict(kind=args.method)
    policy = None
    is_anchor = args.method in ('native_initial', 'native_final')
    if is_anchor and args.memory_mode == 'episode':
        raise ValueError('Native anchors use their single history-independent control')
    if not is_anchor:
        directory = fit / 'fits' / args.method
        if args.snapshot_manifest:
            snapshot = json.loads(args.snapshot_manifest.read_text())
            if (snapshot['method']!=args.method or snapshot['source_assembly_sha256']!=file_sha256(assembly)
                    or snapshot.get('qualification_only') is not True):
                raise ValueError('Require the explicit immutable diagnostic snapshot binding')
            source = Path(snapshot['candidate']['path'])
            if file_sha256(source)!=snapshot['candidate']['sha256']:
                raise ValueError('Frozen diagnostic actor changed')
            status = snapshot['training_status_at_freeze']
            selected.update(qualification_only=True,snapshot=snapshot)
        else:
            status = json.loads((directory / 'training-status.json').read_text())
            if status.get('converged') is not True or status.get('stop_reason') != 'validation_plateau':
                raise ValueError('Dispatch only this method\'s converged validation-selected candidate')
            source = directory / 'candidate.pt'
        digest = file_sha256(source)
        frozen = args.output / 'selected-candidate.pt'
        shutil.copyfile(source, frozen)
        if file_sha256(frozen) != digest or file_sha256(source) != digest:
            raise ValueError('Candidate changed while freezing evaluation selection')
        policy, manifest = load_policy(frozen, expected_sha256=digest, device='cuda:0')
        contract = manifest['contract']
        if (contract.get('objective', args.method) != args.method or contract['mode'] != 'full'
                or contract['train_sha256'] != corpus['datasets'][args.method + '_train']['sha256']
                or contract['validation_sha256'] != corpus['datasets'][args.method + '_validation']['sha256']
                or policy.config.observation_dim != 2073 or policy.config.noise_dim != 600
                or (policy.config.width, policy.config.layers, policy.config.heads, policy.config.context_length) != (256, 6, 8, 192)):
            raise ValueError('Selected actor does not match the live joint training contract')
        selected.update(sha256=digest, training_status=status, manifest=manifest)
    _json(args.output / 'SELECTION.json', selected)

    norm = root / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    model = root / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output / 'native'),
        ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    request = dict(model_directory=str(model), norm_path=str(norm), source_commit=args.source_commit,
        runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1')
    native, factory, signature = _load_native(request, args.code)
    if (signature['model_sha256'] != '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
            or signature['norm_sha256'] != '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'):
        raise ValueError('Original frozen feature motor or normalization changed')
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    codec = DirectActionCodec.from_stats(stats)
    for reference in corpus['source_assemblies'].values():
        path = Path(reference['path'])
        if file_sha256(path) != reference['sha256'] or json.loads(path.read_text())['codec'] != codec.record():
            raise ValueError('Source and evaluation physical-action encodings differ')
    if is_anchor:
        policy = ReferencePlaceholder().to('cuda:0')
        motor = FlowReferenceMotor(native, codec, native.model)
    else:
        motor = FlowDirectMotor(native, codec)
    policy.eval().requires_grad_(False)
    native_versions = tensor_hash(_versions(native.model))
    # This process-local entry extends routing, retaining the source H600 protocol.
    TASKS['CloseToasterOvenDoor'] = (600, 'atomic_seen', 'atomic', None)
    tasks = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug') if args.split == 'ID' else ('CloseFridge', 'TurnOnMicrowave')
    horizons = {task: TASKS[task][0] for task in tasks}
    modes = ('full',) if is_anchor else ('full', 'episode')
    if args.memory_mode != 'all':
        modes = tuple(mode for mode in modes if mode == args.memory_mode)
    base_seed = 63105000 if args.stage == 'qualification' else 64105000
    canonical = root / 'outputs/robocasa-eight-hour-babysit-20261104-r1/gpu' / ('canonical-' + args.stage + '-' + args.split)
    binding = dict(format='three_task_six_episode_readout_v1', source_commit=args.source_commit,
        selection=selected, signature=signature, source_assembly_sha256=file_sha256(assembly),
        stage=args.stage, split=args.split, cases_per_task=args.count,
        support_episodes=[0, 1, 2, 3, 4, 5], render_contract='bounded_two_intensity_levels',
        memory_modes=list(modes), shard_count=args.shards, shard_index=args.shard_index,
        sharding='whole fixed cohorts; (global task index + cohort index) modulo shard count',
        batch_contract='fixed_order_same_horizon_waves_v1', feedback='native sparse first-success per H50; simulator-supervised baseline')
    _json(args.output / 'BINDING.json', binding)
    results = []
    started = time.time()
    reference_actor = None
    prepared_hashes = dict(OpenDrawer='12055d1ca468333971884470a5ea411ea3e511907b8bc6a5dc7b78eeef3c66e8',
        CloseToasterOvenDoor='2a3df641c4e0b6cbb8123add201b17e16b874d3c1ffc49b2f002675350637659',
        CoffeeSetupMug='f9474e04c21f4a8708ae75e077b8f238dc95e90019266d600f4637d3da1a2842')
    for mode in modes:
        for task in tasks:
            task_binding = dict(binding)
            if args.method == 'native_final':
                # ID uses that task's fixed130 teacher; OOD always uses the
                # prospectively designated Drawer teacher, never an OOD fit.
                teacher_task = task if args.split == 'ID' else 'OpenDrawer'
                receipt_path = root / ('outputs/robocasa-flow-sde-10m-per-task-20261103-r1/'
                    + 'CORPUS-ARCHIVE-' + teacher_task + '-R1-PREPARED.json')
                if file_sha256(receipt_path) != prepared_hashes[teacher_task]:
                    raise ValueError('Final teacher preparation binding changed')
                receipt = json.loads(receipt_path.read_text())
                boundary = Path(receipt['checkpoint']['path'])
                boundary_record = boundary / 'pilot-boundary.json'
                if file_sha256(boundary_record) != receipt['checkpoint']['boundary_sha256']:
                    raise ValueError('Fixed130 teacher boundary changed')
                checkpoint = json.loads(boundary_record.read_text())
                if checkpoint.get('completed') is not True or checkpoint['global_step'] != 130:
                    raise ValueError('Require the complete fixed130 teacher')
                weights = boundary / 'actor/model_state_dict/full_weights.pt'
                weight_ref = next(item for item in checkpoint['actor_checkpoint_files']
                    if Path(item['path']) == weights)
                if file_sha256(weights) != weight_ref['sha256']:
                    raise ValueError('Final teacher actor bytes differ from the source boundary')
                teacher_identity = dict(task=teacher_task, step=130, weights=str(weights),
                    weights_sha256=weight_ref['sha256'], receipt_sha256=file_sha256(receipt_path))
                from hydra import compose, initialize_config_dir
                from rlinf.models.embodiment.openpi import get_model
                if reference_actor is not None:
                    del motor, reference_actor
                    torch.cuda.empty_cache()
                with initialize_config_dir(config_dir=str(args.code / 'configs/native_ppo'), version_base=None):
                    cfg = compose(config_name='robocasa365_pi05_eval')
                cfg.rollout.model.model_path = str(weights.parent)
                cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
                reference_actor = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
                motor = FlowReferenceMotor(native, codec, reference_actor)
                task_binding['final_teacher'] = teacher_identity
                _json(args.output / ('FINAL-TEACHER-' + task + '.json'), teacher_identity)
                reference_versions = tensor_hash(_versions(reference_actor))
            # The same task retains the same physical cases across methods/modes.
            task_number = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'CloseFridge', 'TurnOnMicrowave').index(task)
            for first in range(0, args.count, 8):
                if (task_number + first//8) % args.shards != args.shard_index:
                    continue
                if time.time() + 300 >= args.deadline:
                    raise TimeoutError('Preserve complete waves before the allocation boundary')
                if shutil.disk_usage(root).free < 220 * 2**30:
                    raise RuntimeError('Preserve the shared storage reserve before another wave')
                cases = [dict(case=dict(bank='joint_meta_' + args.stage + '_' + args.split,
                    task=task, split='target', native_seed=base_seed + task_number * 10000 + slot,
                    slot=slot, reset_ordinal=0, counted=True,
                    layout_source='native_target_constructor_seed_first_reset'), horizon=horizons[task])
                    for slot in range(first, min(args.count, first + 8))]
                directory = args.output / mode / task / f'bank-{first:03d}'
                journal = ActionJournal(directory / 'accounting', rank=0, world=1, attempt='six-episode',
                    maximum_actions=len(cases) * 6 * (horizons[task] + 128))
                backend = NativeAttemptBackend(output=directory / 'attempts', canonical_root=canonical,
                    motor=motor, spec=codec, native_factory=factory.paired, journal=journal,
                    writer_factory=native_video_writer, binding=task_binding, maximum_rgb_delta=2,
                    memory_mode=mode, actual_attempts=6, fixed_case_order=True)
                began = time.time()
                rows = backend.run_bank(policy, cases, lanes=len(cases), attempts=6, deadline_unix=args.deadline - 90)
                if tensor_hash(_versions(native.model)) != native_versions:
                    raise ValueError('Frozen native weights mutated')
                if reference_actor is not None and tensor_hash(_versions(reference_actor)) != reference_versions:
                    raise ValueError('Frozen final teacher weights mutated')
                expected_batches = 6 * (horizons[task] // 50)
                if backend.last_batch_sizes != [len(cases)] * expected_batches or len(rows) != 6 * len(cases):
                    raise ValueError('Actual native batch membership or episode coverage differs')
                # Each episode j is a query using j-1 complete support episodes.
                counts = {str(k): sum(row['success_once'] for row in rows if row['attempt'] == k + 1) for k in range(6)}
                result = dict(method=args.method, mode=mode, task=task, cases=cases,
                    first_success_counts_by_support_episodes=counts, denominators=len(cases),
                    episodes=rows, references=backend.references, actual_batch_sizes=backend.last_batch_sizes,
                    controls_charged=journal.charged(), elapsed_seconds=time.time() - began,
                    peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
                _json(directory / 'RESULT.json', result)
                results.append({key: value for key, value in result.items() if key not in ('episodes', 'references', 'cases', 'actual_batch_sizes')})
                _json(args.output / 'PROGRESS.json', dict(results=results, checked_unix=time.time()))
    _json(args.output / 'RESULT.json', dict(completed=True, binding=binding, results=results,
        elapsed_seconds=time.time() - started, deployment_optimizer_updates=0))


if __name__ == '__main__':
    main()
