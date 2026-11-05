"""Matched native ID/OOD six-episode readout of a converged frozen-expert bridge."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'fit', 'output', 'probe'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--interface', choices=('kv_bridge', 'velocity_bridge'), required=True)
    p.add_argument('--objective', choices=('ad', 'dpt', 'dit', 'ic_iql'), required=True)
    p.add_argument('--split', choices=('ID', 'OOD', 'qualification'), required=True)
    p.add_argument('--qualification-phase', choices=('deployment', 'reset'), default='deployment')
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--attempt', required=True)
    args = p.parse_args()
    minimum_free_gib = int(os.environ.get('STEERING_MIN_FREE_GIB', '220'))
    if minimum_free_gib < 96:
        raise ValueError('Retain at least96GiB shared hot-storage reserve')
    if sys.platform != 'linux' or not socket.gethostname().startswith('job-'):
        raise ValueError('Require the actual allocated Linux GPU container')
    sys.path[:0] = [str(args.code / 'src'), os.environ['RLINF_PATH']]
    import torch
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, tensor_hash, ActionJournal
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeCodec, FlowBridgeMotor, load_bridge_policy
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions, _json, native_video_writer
    from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    checked_checkout(args.code, args.commit)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    root = Path('/mowenjie/fluffy-bassoon-2')
    status = json.loads((args.fit / 'training-status.json').read_text())
    if not status['converged'] or status['stop_reason'] != 'validation_plateau':
        raise ValueError('Evaluate only the independently validation-selected converged bridge')
    source = args.fit / 'candidate.pt'
    digest = file_sha256(source)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_candidate = args.output / 'selected-candidate.pt'
    if not frozen_candidate.exists():
        shutil.copyfile(source, frozen_candidate)
    if file_sha256(frozen_candidate) != digest:
        raise ValueError('The immutable evaluation candidate changed')
    policy, manifest = load_bridge_policy(frozen_candidate, expected_sha256=digest, device='cuda:0')
    contract = manifest['contract']
    if (contract['interface'] != args.interface or contract['objective'] != args.objective
            or contract['qualification_only'] or contract['logical_batch'] != 8
            or (policy.config.width, policy.config.layers, policy.config.heads, policy.config.context_length) != (256, 6, 8, 192)
            or contract['assembly_sha256'] != 'fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758'):
        raise ValueError('The candidate does not match the eight-model frozen-expert contract')
    norm = root / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    model_path = root / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output / ('native-' + args.attempt)),
        ROBOCASA_PI05_TORCH_MODEL=str(model_path), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model_path), norm_path=str(norm),
        source_commit=args.commit, runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'), args.code)
    if (signature['model_sha256'] != contract['native_model_sha256']
            or signature['norm_sha256'] != contract['native_norm_sha256']):
        raise ValueError('The original frozen expert/normalization changed')
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    codec = FlowBridgeCodec.from_stats(stats, interface=args.interface)
    qualification = args.split == 'qualification'
    if args.qualification_phase != 'deployment' and not qualification:
        raise ValueError('Qualification phases do not change the main benchmark')
    if qualification:
        import importlib.util
        spec = importlib.util.spec_from_file_location('fitted_bridge_observer', Path(__file__).with_name('FITTED-CHECK.py'))
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        motor = observer.observed_motor(native, codec, policy)
    else:
        motor = FlowBridgeMotor(native, codec, policy)
    native_versions = tensor_hash(_versions(native.model))
    tasks = (('OpenDrawer', 'CloseFridge') if qualification else
             ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug') if args.split == 'ID' else
             ('CloseFridge', 'TurnOnMicrowave'))
    horizons = dict(OpenDrawer=600, CloseToasterOvenDoor=600, CoffeeSetupMug=600,
                    CloseFridge=900, TurnOnMicrowave=450)
    # Preserve the exact previously audited task horizons and case banks.
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    TASKS['CloseToasterOvenDoor'] = (600, 'atomic_seen', 'atomic', None)
    binding = dict(format='frozen_expert_eight_model_six_episode_readout_v1',
        interface=args.interface, objective=args.objective, source_commit=args.commit,
        selected_candidate_sha256=digest, trained_contract=contract,
        model_sha256=signature['model_sha256'], norm_sha256=signature['norm_sha256'],
        runtime_commit=signature['runtime_commit'], split=args.split, cases_per_task=50,
        support_episodes=[0, 1, 2, 3, 4, 5], memory_modes=['full', 'episode'],
        batch_contract='fixed_order_same_horizon_waves_v1',
        render_contract='bounded_two_intensity_levels', deployment_optimizer_updates=0)
    if qualification:
        waves = ([dict(task='CloseFridge', mode='episode', cases=1)] if args.qualification_phase == 'reset' else
                 [dict(task='OpenDrawer', mode='full', cases=8),
                  dict(task='CloseFridge', mode='full', cases=1),
                  dict(task='CloseFridge', mode='episode', cases=1)])
        binding.update(format='fitted_bridge_deployment_qualification_v1', cases_per_task=None,
            separate_qualification_output=True, maximum_allocated_GPU_hours=.25,
            qualification_phase=args.qualification_phase, waves=waves,
            purpose='Deployment, causal history and B8 throughput only; not the main ID/OOD benchmark')
    binding_path = args.output / 'BINDING.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('Do not mix immutable candidates or evaluation contracts when resuming')
    _json(binding_path, binding)
    binding_digest = file_sha256(binding_path)
    canonical = root / 'outputs/robocasa-eight-hour-babysit-20261104-r1/gpu' / ('canonical-final-' + args.split)
    results, started = [], time.time()
    for mode in ('full', 'episode'):
        for task in tasks:
            if qualification and args.qualification_phase == 'reset' and (mode != 'episode' or task != 'CloseFridge'):
                continue
            if qualification and task == 'OpenDrawer' and mode == 'episode':
                continue
            canonical_split = 'ID' if task in ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug') else 'OOD'
            canonical = root / 'outputs/robocasa-eight-hour-babysit-20261104-r1/gpu' / ('canonical-final-' + canonical_split)
            task_number = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'CloseFridge', 'TurnOnMicrowave').index(task)
            for first in range(0, 50, 8):
                if qualification and first:
                    break
                wave_size = 1 if qualification and task == 'CloseFridge' else 8
                cases = [dict(case=dict(bank='joint_meta_final_' + canonical_split, task=task, split='target',
                    native_seed=64105000 + task_number * 10000 + slot, slot=slot, reset_ordinal=0,
                    counted=True, layout_source='native_target_constructor_seed_first_reset'), horizon=horizons[task])
                    for slot in range(first, min(50, first + wave_size))]
                directory = args.output / mode / task / f'bank-{first:03d}'
                if (directory / 'RESULT.json').exists():
                    result = json.loads((directory / 'RESULT.json').read_text())
                    if result['binding_sha256'] != binding_digest or result['cases'] != cases:
                        raise ValueError('Completed evaluation wave has a different case/candidate binding')
                else:
                    if time.time() + 300 >= args.deadline:
                        _json(args.output / 'OPERATIONAL-CHECKPOINT.json', dict(completed=False,
                            completed_banks=len(results), next_bank=str(directory), binding_sha256=binding_digest))
                        return
                    if shutil.disk_usage(root).free < minimum_free_gib * 2**30:
                        _json(args.output / 'OPERATIONAL-CHECKPOINT.json', dict(completed=False,
                            reason='hot_storage_reserve', completed_banks=len(results),
                            next_bank=str(directory), minimum_free_GiB=minimum_free_gib,
                            binding_sha256=binding_digest))
                        return
                    journal = ActionJournal(directory / ('accounting-' + args.attempt), rank=0, world=1,
                        attempt='six-episode-' + args.attempt,
                        maximum_actions=len(cases) * 6 * (horizons[task] + 128))
                    backend = NativeAttemptBackend(output=directory / ('attempts-' + args.attempt),
                        canonical_root=canonical, motor=motor, spec=codec, native_factory=factory.paired,
                        journal=journal, writer_factory=native_video_writer, binding=binding,
                        maximum_rgb_delta=2, memory_mode=mode, actual_attempts=6, fixed_case_order=True)
                    began = time.time()
                    if qualification:
                        motor.start_wave(mode, horizons[task])
                    rows = backend.run_bank(policy, cases, lanes=len(cases), attempts=6,
                                            deadline_unix=args.deadline - 90)
                    if (tensor_hash(_versions(native.model)) != native_versions
                            or len(rows) != len(cases) * 6
                            or backend.last_batch_sizes != [len(cases)] * (6 * horizons[task] // 50)):
                        raise ValueError('Frozen identity, native wave membership or episode coverage changed')
                    result = dict(method=args.objective, interface=args.interface, mode=mode, task=task,
                        cases=cases, binding_sha256=binding_digest, denominators=len(cases),
                        first_success_counts_by_support_episodes={str(k): sum(row['success_once'] for row in rows if row['attempt'] == k + 1) for k in range(6)},
                        episodes=rows, references=backend.references,
                        controls_charged=journal.charged(), actual_batch_sizes=backend.last_batch_sizes,
                        elapsed_seconds=time.time() - began, optimizer_updates=0,
                        native_parameters_frozen=True)
                    if qualification:
                        result['fitted_deployment_check'] = motor.finish_wave(cases=len(cases),
                            horizon=horizons[task], output=directory / 'FITTED-DEPLOYMENT.json')
                        result['physical_controls_per_second'] = len(cases) * 6 * horizons[task] / result['elapsed_seconds']
                    _json(directory / 'RESULT.json', result)
                results.append({k: v for k, v in result.items()
                    if k not in ('episodes', 'references', 'cases', 'actual_batch_sizes')})
                _json(args.output / 'PROGRESS.json', dict(results=results, checked_unix=time.time()))
    _json(args.output / 'RESULT.json', dict(completed=True, binding=binding, results=results,
        deployment_optimizer_updates=0, elapsed_seconds=time.time() - started))


if __name__ == '__main__':
    main()
