"""Fresh paired native episodes for frozen DIT and its three controls."""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import sys
import tarfile
import time

TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'CloseFridge', 'TurnOnMicrowave')
HORIZONS = (600, 600, 600, 900, 450)
ARMS = ('DIT_full_history', 'DIT_episode_reset', 'original_SFT', 'fixed_PPO130')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def audit(references, *, reset_policy=None):
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.frozen_steering_history import checked_reference
    from fluffy_bassoon.robocasa.steering_history_native import attempt_noise
    from fluffy_bassoon.meta_rl.steering_history_eval import replay_physical_attempt
    from fluffy_bassoon.meta_rl.vla_history import FrozenHistoryController
    checked = []
    for reference in references:
        row = torch.load(checked_reference(reference), map_location='cpu', weights_only=False)
        checked_reference(row['initial'])
        truth = json.loads(checked_reference(row['truth']).read_text())
        video = json.loads(checked_reference(row['video']).read_text())
        frames = json.loads(checked_reference(video['frame_digests'][0]).read_text())
        assert row['completed'] and row['success_once'] == truth['simulator_success_once']
        assert frames['frame_count'] == row['horizon'] + 1 and not frames['playable_video']
        assert len(row['chunks']) * 50 == row['horizon'] and not video['videos']
        reward, chunks = 0., []
        for i, reference in enumerate(row['chunks']):
            with gzip.open(checked_reference(reference), 'rb') as stream:
                chunk = pickle.load(stream)
            noise, provenance = attempt_noise(row['case'], row['attempt'], i)
            assert chunk['case'] == row['case'] and chunk['attempt'] == row['attempt'] and chunk['chunk'] == i
            assert chunk['status'] == 'completed' and chunk['native_noise_provenance'] == provenance
            assert np.array_equal(chunk['native_noise'], noise)
            assert np.array_equal(chunk['requested_commands'], chunk['executed_commands'])
            assert np.array_equal(chunk['dispatched_commands'], chunk['executed_commands'])
            assert chunk['native_counts_after']['controls'] - chunk['native_counts_before']['controls'] == 50
            assert bool(chunk['done']) == (i == len(row['chunks'])-1)
            reward += chunk['reward']
            chunks.append(chunk)
        assert reward == float(row['success_once'])
        if reset_policy is not None:
            # Reset clears on act(), so pre-act diagnostic signatures can
            # retain stale state. Compare functional outputs from empty state.
            replay_physical_attempt(FrozenHistoryController(reset_policy, mode='episode'),
                                    chunks, horizon=row['horizon'])
        checked.append(dict(case=row['case'], attempt=row['attempt'], success_once=row['success_once'],
                            memory_entry=row['memory_entry'], memory_exit=row['memory_exit'],
                            reset_from_empty_replay_exact=reset_policy is not None))
    return checked


def bank_wave(directory, cold):
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    began = time.time()
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    inventory = {str(p.relative_to(directory)): dict(bytes=p.stat().st_size, sha256=file_sha256(p)) for p in files}
    archive = directory.with_suffix('.tar')
    with tarfile.open(archive, 'x') as bank:
        for path in files:
            bank.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    with tarfile.open(archive, 'r') as bank:
        import hashlib
        members = bank.getmembers()
        assert {m.name for m in members} == set(inventory)
        for member in members:
            with bank.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == inventory[member.name]['sha256']
    cold.mkdir(parents=True, exist_ok=False)
    target = cold / 'evidence.tar'
    with archive.open('rb') as reader, target.open('xb') as writer:
        shutil.copyfileobj(reader, writer, 16*2**20)
    digest = file_sha256(archive)
    assert file_sha256(target) == digest
    receipt = dict(archive=str(target), sha256=digest, bytes=target.stat().st_size,
                   inventory=inventory, every_member_verified=True, cold_full_readback=True,
                   elapsed_seconds=time.time()-began)
    write(cold / 'ARCHIVE.json', receipt)
    # Only this completed, read-back-verified wave's local copy is reclaimed.
    shutil.rmtree(directory)
    archive.unlink()
    return {k:v for k,v in receipt.items() if k != 'inventory'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--stage', choices=('qualification', 'confirmation'), required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    code = Path(__file__).parents[2]
    sys.path[:0] = [str(code/'src'), os.environ['RLINF_PATH']]
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout, tensor_hash, ActionJournal
    from fluffy_bassoon.meta_rl.flow_history import (FlowBridgeCodec, FlowBridgeMotor, load_bridge_policy,
        DirectActionCodec, FlowReferenceMotor, ReferencePlaceholder)
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native, _versions, native_video_writer
    from fluffy_bassoon.robocasa.steering_history_native import NativeAttemptBackend, case_key
    from fluffy_bassoon.robocasa.ppo import TASKS as ROUTES
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert 0 <= args.shard < args.shards and args.count > 0
    assert args.stage != 'qualification' or (args.count == 8 and args.shards == 1)
    checked_checkout(code, args.pin)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    args.cold.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((args.inputs/'RESTORED.json').read_text())
    assert inputs['completed']
    def input_file(name):
        record = inputs['inputs'][name]
        path = args.inputs/name
        assert file_sha256(path) == record['sha256']
        return path
    dit, manifest = load_bridge_policy(input_file('DIT/candidate.pt'),
        expected_sha256='7ccaae7048df38a90ced30530002469f1f73e1ee9cb37a17b07c8137cabb3721', device='cuda:0')
    contract = manifest['contract']
    assert contract['interface'] == 'kv_bridge' and contract['objective'] == 'dit'
    assert not contract['qualification_only'] and contract['logical_batch'] == 8
    assert (dit.config.width, dit.config.layers, dit.config.heads, dit.config.context_length) == (256,6,8,192)
    root = Path('/mowenjie/fluffy-bassoon-2')
    model = root/'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    norm = root/'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output/'native'),
                     ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model), norm_path=str(norm),
        source_commit=args.pin, runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'), code)
    assert signature['model_sha256'] == contract['native_model_sha256']
    assert signature['norm_sha256'] == contract['native_norm_sha256']
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    bridge_codec = FlowBridgeCodec.from_stats(stats, interface='kv_bridge')
    direct_codec = DirectActionCodec.from_stats(stats)
    placeholder = ReferencePlaceholder().to('cuda:0').eval().requires_grad_(False)
    bridge_motor = FlowBridgeMotor(native, bridge_codec, dit)
    initial_motor = FlowReferenceMotor(native, direct_codec, native.model)
    versions = tensor_hash(_versions(native.model))
    ROUTES['CloseToasterOvenDoor'] = (600, 'atomic_seen', 'atomic', None)
    binding = dict(format='paired_DIT_confirmation_v1', source_commit=args.pin,
        stage=args.stage, cases_per_task=args.count, shard=args.shard, shards=args.shards,
        source_inputs=inputs, signature=signature, selected_DIT_sha256=inputs['inputs']['DIT/candidate.pt']['sha256'],
        endpoint_attempt=6, render_contract='bounded_two_intensity_levels',
        sharding='whole paired waves: (task_index + wave_index) modulo shards',
        recording='native_rgb_frame_digests_v1', deployment_optimizer_updates=0)
    write(args.cold/'BINDING.json', binding)
    results, started = [], time.time()
    teacher = None
    for number, task in enumerate(TASKS):
        from hydra import compose, initialize_config_dir
        from rlinf.models.embodiment.openpi import get_model
        teacher_task = task if number < 3 else 'OpenDrawer'
        weights = input_file(f'PPO/{teacher_task}/full_weights.pt')
        if teacher is not None:
            del teacher_motor, teacher
            torch.cuda.empty_cache()
        with initialize_config_dir(config_dir=str(code/'benchmarks/robocasa/configs'), version_base=None):
            cfg = compose(config_name='robocasa365_pi05_eval')
        cfg.rollout.model.model_path = str(weights.parent)
        cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
        teacher = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
        teacher_motor = FlowReferenceMotor(native, direct_codec, teacher)
        teacher_versions = tensor_hash(_versions(teacher))
        for first in range(0, args.count, 8):
            if (number + first//8) % args.shards != args.shard:
                continue
            if time.time()+300 >= args.deadline:
                write(args.cold/'INCOMPLETE.json', dict(next_task=task, next_first=first, reason='deadline', results=results))
                return
            assert shutil.disk_usage(args.output).free > 40*2**30, 'Retain bounded wave and archive scratch'
            wave = args.output/task/f'bank-{first:04d}'
            cases = [dict(case=dict(bank='paired_DIT_'+args.stage+'_20261114', task=task,
                split='target', native_seed=(71140000 if args.stage=='qualification' else 72140000)+number*10000+slot,
                slot=slot, reset_ordinal=0, counted=True,
                layout_source='native_target_constructor_seed_first_reset'), horizon=HORIZONS[number])
                for slot in range(first, min(args.count, first+8))]
            paired = {}
            arms = ARMS + (('DIT_reset_serial',) if args.stage=='qualification' and number==0 else ())
            for arm in arms:
                policy, codec, motor = ((dit,bridge_codec,bridge_motor) if arm.startswith('DIT') else
                    (placeholder,direct_codec,initial_motor if arm=='original_SFT' else teacher_motor))
                full = arm == 'DIT_full_history'
                first_attempt = 1 if full or arm=='DIT_reset_serial' else 6
                episodes = 7-first_attempt
                directory = wave/arm
                journal = ActionJournal(directory/'accounting', rank=0, world=1, attempt=args.stage,
                    maximum_actions=len(cases)*episodes*(HORIZONS[number]+128))
                backend = NativeAttemptBackend(output=directory/'attempts', canonical_root=wave/'canonical',
                    motor=motor, spec=codec, native_factory=factory.paired, journal=journal,
                    writer_factory=native_video_writer, binding=dict(binding,arm=arm,task=task,teacher_task=teacher_task),
                    maximum_rgb_delta=2, memory_mode='full' if full else 'episode',
                    actual_attempts=6, first_attempt=first_attempt, fixed_case_order=True, record_video=False)
                began=time.time()
                rows=backend.run_bank(policy,cases,lanes=len(cases),attempts=6,deadline_unix=args.deadline-120)
                elapsed=time.time()-began
                assert len(rows)==len(cases)*episodes
                assert backend.last_batch_sizes==[len(cases)]*(episodes*HORIZONS[number]//50)
                assert tensor_hash(_versions(native.model))==versions and tensor_hash(_versions(teacher))==teacher_versions
                audited=audit(backend.references, reset_policy=None if full else policy)
                endpoint={case_key(x['case']):x for x in audited if x['attempt']==6}
                assert len(endpoint)==len(cases)
                paired[arm]=dict(episodes=rows,endpoint=endpoint,elapsed_seconds=elapsed,
                    physical_episodes=len(rows),physical_controls=len(rows)*HORIZONS[number],
                    references=backend.references,all_native_evidence_checked=True)
                write(directory/'RESULT.json', paired[arm])
                write(args.cold/'PROGRESS.json',dict(checked_unix=time.time(),task=task,first=first,arm=arm,
                    completed_waves=len(results),latest_physical_episodes=len(rows),elapsed_seconds=elapsed))
            keys=set(paired[ARMS[0]]['endpoint'])
            assert all(set(paired[arm]['endpoint'])==keys for arm in arms)
            serial = None
            if 'DIT_reset_serial' in paired:
                serial=dict(cases=len(keys),pre_act_signature_exact=all(
                    paired['DIT_reset_serial']['endpoint'][k]['memory_entry']==paired['DIT_episode_reset']['endpoint'][k]['memory_entry'] for k in keys),
                    functional_reset_replay_exact=all(paired[arm]['endpoint'][k]['reset_from_empty_replay_exact']
                        for arm in ('DIT_reset_serial','DIT_episode_reset') for k in keys),
                    outcome_discordances=sum(paired['DIT_reset_serial']['endpoint'][k]['success_once'] != paired['DIT_episode_reset']['endpoint'][k]['success_once'] for k in keys),
                    interpretation='Every recorded control replays exactly from a fresh episode controller on its recorded features and feedback; identical episode6 draw law. Pre-act stale signatures are cleared by act(). Render differences can change trajectories.')
                assert serial['functional_reset_replay_exact']
            compact=dict(task=task,first=first,cases=cases,arms={arm:{k:v for k,v in row.items() if k!='references'} for arm,row in paired.items()},reset_qualification=serial)
            write(wave/'PAIRED.json',compact)
            cold_wave=args.cold/task/f'bank-{first:04d}'
            receipt=bank_wave(wave,cold_wave)
            write(cold_wave/'PAIRED.json',compact)
            results.append(dict(task=task,first=first,archive=receipt,timing={arm:row['elapsed_seconds'] for arm,row in paired.items()}))
            write(args.cold/'PROGRESS.json',dict(checked_unix=time.time(),completed_waves=len(results),results=results))
    write(args.cold/'COMPLETE.json',dict(completed=True,binding=binding,results=results,
        elapsed_seconds=time.time()-started,optimizer_updates=0))


if __name__=='__main__':
    main()
