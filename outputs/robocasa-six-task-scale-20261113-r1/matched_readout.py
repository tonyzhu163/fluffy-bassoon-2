"""Eight fixed-final KV policies on common physical cases and native draw streams."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

from labels import sha, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'CloseFridge', 'TurnOnMicrowave')
HORIZONS = (600, 600, 600, 900, 450)
METHODS = ('ad', 'dpt', 'dit', 'ic_iql')
MODELS = tuple(f'{size}/{method}' for size in (3, 6) for method in METHODS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cold', type=Path, required=True)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert 0 <= args.shard < args.shards and sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    assert set(spec['models']) == set(MODELS) and spec['cases_per_task'] == 40
    assert args.shards == spec['shards']
    # This endpoint shortcut must have an explicit migration, not silently
    # reinterpret the earlier 19,200-episode all-attempt reset contract.
    migration_ref = spec['evaluation_migration']
    assert sha(migration_ref['path']) == migration_ref['sha256']
    migration = json.loads(Path(migration_ref['path']).read_text())
    assert migration['full_attempts'] == [1, 2, 3, 4, 5, 6]
    assert migration['reset_physical_attempts'] == [1, 6]
    assert migration['cases_per_task'] == 40 and migration['physical_episodes'] == 12800
    qualification_ref = migration['native_reset_qualification']
    assert sha(qualification_ref['path']) == qualification_ref['sha256']
    qualification = json.loads(Path(qualification_ref['path']).read_text())
    assert qualification['completed'] and qualification['binding']['stage'] == 'qualification'
    assert {row['task'] for row in qualification['results']} == set(TASKS)
    sys.path[:0] = [str(CODE / 'src'), os.environ['RLINF_PATH']]
    import torch
    from hydra import compose, initialize_config_dir
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout, tensor_hash, ActionJournal
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeCodec, FlowBridgeMotor, load_bridge_policy
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native, _versions, native_video_writer
    from fluffy_bassoon.robocasa.steering_history_native import NativeAttemptBackend, case_key
    from fluffy_bassoon.robocasa.ppo import TASKS as ROUTES
    checked_checkout(CODE, args.pin)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    helper_path = HERE.parent / 'robocasa-dit-paired-confirmation-20261114-r1/readout.py'
    helper_spec = importlib.util.spec_from_file_location('paired_native_audit', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    assert sha(helper_path) == migration['qualified_audit_and_banking_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    args.cold.mkdir(parents=True, exist_ok=True)
    policies, contracts, identities = {}, {}, []
    for name in MODELS:
        size, objective = name.split('/')
        ref = spec['models'][name]
        assert sha(ref['complete']['path']) == ref['complete']['sha256']
        complete = json.loads(Path(ref['complete']['path']).read_text())
        assert complete['completed'] and complete['progress']['actor_update'] == 10000
        assert complete['candidate'] == ref['candidate'] and complete['native_parameters_unchanged']
        contract = complete['contract']
        assert contract['corpus_size'] == int(size) and contract['objective'] == objective
        assert contract['actor_updates'] == 10000 and contract['checkpoint_selection'] == 'fixed final'
        assert contract['value_updates'] == (1000 if objective in ('dit', 'ic_iql') else 0)
        assert contract['seed'] == 0 and contract['logical_batch'] == 8
        policy, manifest = load_bridge_policy(ref['candidate']['path'], expected_sha256=ref['candidate']['sha256'], device='cuda:0')
        assert manifest['update'] == 10000
        assert {k: v for k, v in manifest['contract'].items() if k != 'config'} == contract
        assert (policy.config.width, policy.config.layers, policy.config.heads, policy.config.context_length) == (256, 6, 8, 192)
        policies[name], contracts[name] = policy, contract
        identities.append(contract['native_identity'])
    assert all(identity == identities[0] for identity in identities)
    # Every method/corpus must refer to the same manifest, with exact original
    # task dataset references shared between the three- and six-task fits.
    assert len({contract['corpus_spec_sha256'] for contract in contracts.values()}) == 1
    for method in METHODS:
        small, large = contracts[f'3/{method}']['datasets'], contracts[f'6/{method}']['datasets']
        assert all(large[task] == refs for task, refs in small.items())
    model = ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    norm = ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output / 'native'),
        ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model), norm_path=str(norm),
        source_commit=args.pin, runtime_commit=identities[0]['runtime_commit']), CODE)
    assert signature['model_sha256'] == identities[0]['native_model_sha256']
    assert signature['norm_sha256'] == identities[0]['native_norm_sha256']
    stats = json.loads(norm.read_text())
    codec = FlowBridgeCodec.from_stats(stats.get('norm_stats', stats), interface='kv_bridge')
    with initialize_config_dir(config_dir=str(CODE / 'benchmarks/robocasa/configs'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    versions = tensor_hash(_versions(native.model))
    ROUTES['CloseToasterOvenDoor'] = (600, 'atomic_seen', 'atomic', None)
    binding = dict(format='matched_three_six_eight_KV_endpoint_v1', source_commit=args.pin,
        spec_sha256=args.spec_sha256, models=spec['models'], contracts=contracts, signature=signature,
        evaluation_migration=migration_ref, cases_per_task=40, shard=args.shard, shards=args.shards,
        full_attempts=[1, 2, 3, 4, 5, 6], reset_physical_attempts=[1, 6], deployment_optimizer_updates=0,
        pairing='All eight models share the same native physical canonical state and case/noise identities within each whole wave',
        recording='native_rgb_frame_digests_v1', render_contract='bounded_two_intensity_levels')
    write(args.cold / 'BINDING.json', binding)
    results = []
    for ti, task in enumerate(TASKS):
        for first in range(0, 40, 8):
            if (ti + first // 8) % args.shards != args.shard:
                continue
            if time.time() + 300 >= args.deadline:
                write(args.cold / 'INCOMPLETE.json', dict(next_task=task, next_first=first, completed_waves=results))
                return
            # One maximal 512-episode wave and its archive, derived from the
            # qualified measured bytes/control, need about17GiB at horizon900.
            assert shutil.disk_usage(args.output).free > 20 * 2**30
            wave = args.output / task / f'bank-{first:04d}'
            cases = [dict(case=dict(bank='matched_three_six_20261114', task=task, split='target',
                native_seed=73140000 + ti * 10000 + slot, slot=slot, reset_ordinal=0, counted=True,
                layout_source='native_target_constructor_seed_first_reset'), horizon=HORIZONS[ti])
                for slot in range(first, first + 8)]
            paired = {}
            for name in MODELS:
                policy = policies[name]
                motor = FlowBridgeMotor(native, codec, policy)
                arms = {}
                for arm, start, end, mode in [('full', 1, 6, 'full'),
                        ('reset_first', 1, 1, 'episode'), ('reset_sixth', 6, 6, 'episode')]:
                    directory = wave / name / arm
                    journal = ActionJournal(directory / 'accounting', rank=0, world=1, attempt='matched-final',
                        maximum_actions=len(cases) * (end - start + 1) * (HORIZONS[ti] + 128))
                    backend = NativeAttemptBackend(output=directory / 'attempts', canonical_root=wave / 'canonical',
                        motor=motor, spec=codec, native_factory=factory.paired, journal=journal,
                        writer_factory=native_video_writer, binding=dict(binding, model=name, task=task, arm=arm),
                        maximum_rgb_delta=2, memory_mode=mode, actual_attempts=end,
                        first_attempt=start, fixed_case_order=True, record_video=False)
                    began = time.time()
                    rows = backend.run_bank(policy, cases, lanes=8, attempts=end, deadline_unix=args.deadline - 120)
                    elapsed = time.time() - began
                    assert len(rows) == 8 * (end - start + 1)
                    assert backend.last_batch_sizes == [8] * ((end - start + 1) * HORIZONS[ti] // 50)
                    assert tensor_hash(_versions(native.model)) == versions
                    audited = helper.audit(backend.references, action_space=cfg.env.eval.action_space,
                        reset_policy=None if mode == 'full' else policy)
                    assert {(case_key(row['case']), row['attempt'], row['success_once']) for row in audited} == {
                        (case_key(row['case']), row['attempt'], row['success_once']) for row in rows}
                    endpoint = {case_key(row['case']): row for row in audited if row['attempt'] == end}
                    assert len(endpoint) == 8
                    arms[arm] = dict(episodes=rows, endpoint=endpoint, elapsed_seconds=elapsed,
                        physical_episodes=len(rows), physical_controls=len(rows) * HORIZONS[ti],
                        all_native_evidence_checked=True)
                    write(directory / 'RESULT.json', arms[arm])
                    write(args.cold / 'PROGRESS.json', dict(task=task, first=first, model=name, arm=arm,
                        completed_waves=len(results), elapsed_seconds=elapsed, checked_unix=time.time()))
                assert all(set(row['endpoint']) == set(arms['full']['endpoint']) for row in arms.values())
                paired[name] = arms
            compact = dict(task=task, first=first, cases=cases, models=paired, spec_sha256=args.spec_sha256)
            write(wave / 'PAIRED.json', compact)
            cold_wave = args.cold / task / f'bank-{first:04d}'
            receipt = helper.bank_wave(wave, cold_wave)
            write(cold_wave / 'PAIRED.json', compact)
            results.append(dict(task=task, first=first, archive=receipt,
                timing={name: {arm: row['elapsed_seconds'] for arm, row in arms.items()} for name, arms in paired.items()}))
            write(args.cold / 'PROGRESS.json', dict(completed_waves=len(results), results=results, checked_unix=time.time()))
    write(args.cold / 'COMPLETE.json', dict(completed=True, binding=binding, results=results,
        deployment_optimizer_updates=0, checked_unix=time.time()))


if __name__ == '__main__':
    main()
