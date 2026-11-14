"""Fixed step30 DPT labels and frozen public features, banking whole streams."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_payload(payload, item, task):
    import numpy as np
    from fluffy_bassoon.rlinf.native_ppo import tensor_hash
    obs = payload['current_observation']
    assert tensor_hash(obs) == payload['current_observation_sha256']
    assert payload['phase'] == 'training' and payload['iteration'] == item['version'] and payload['round'] == 0
    states = np.asarray(obs['states'], np.float32)
    actions = np.asarray(payload['decoded_actions'], np.float32)
    rewards = np.asarray(payload['rewards'], np.float32)
    terms, truncs = (np.asarray(payload[k], bool) for k in ('terminations', 'truncations'))
    seeds = np.asarray(payload['seeds'], np.int64)
    assert states.shape == (16, 25) and np.isfinite(states).all()
    assert actions.shape == (16, 50, 12) and np.isfinite(actions).all()
    # These source versions saved already-mapped physical controls. Fail if
    # an unmapped score ever appears; never silently reinterpret behavior.
    assert np.isin(actions[..., [6, 11]], [-1., 1.]).all()
    assert rewards.shape == terms.shape == truncs.shape == (16, 50)
    assert np.isin(rewards, [0., 1.]).all() and (rewards.sum(-1) <= 1).all()
    expected_truncs = np.zeros((16, 50), bool)
    expected_truncs[:, -1] = item['chunk'] == 12
    assert np.array_equal(truncs, expected_truncs)
    assert seeds.shape == (16,) and len(set(seeds.tolist())) == 16
    assert payload['episode_ids'] == [f'{task}/{s}' for s in seeds]
    assert (payload['terminal_observation'] is not None) == (item['chunk'] == 12)
    assert set(('main_images', 'wrist_images', 'extra_view_images', 'states', 'task_descriptions')) <= obs.keys()
    return states, actions, rewards, terms, truncs, seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pin', required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--probe', type=Path)
    p.add_argument('--deadline', type=float)
    p.add_argument('--cpu-audit', action='store_true')
    args = p.parse_args()
    code = Path(__file__).resolve().parents[2]
    root = Path('/mowenjie/fluffy-bassoon-2')
    runtime = root/'sources/critic-selected-a1229239-r1'
    sys.path[:0] = [str(code/'src'), str(runtime)]
    assert sys.platform == 'linux'
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout, tensor_hash
    checked_checkout(code, args.pin)
    assert sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    source = spec['tasks'][args.task]
    packed_path = Path(source['packed']['path'])
    assert sha(packed_path) == source['packed']['sha256']
    packed = json.loads(packed_path.read_text())
    assert packed['completed'] and packed['updates'] == 30 and packed['query_decisions'] == 46080
    assert sha(packed['index_path']) == packed['index_sha256']
    index = json.loads(Path(packed['index_path']).read_text())
    assert len(index) == 2880 and {(x['rank'], x['version'], x['chunk']) for x in index} == {
        (r, v, c) for r in range(8) for v in range(30) for c in range(1, 13)}
    archive = Path(packed['archive'])
    assert archive.stat().st_size == packed['archive_bytes']
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor, teacher_noise
    from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
    from fluffy_bassoon.robocasa.residual import public_observation
    reader = FlowSourceQueries({args.task: dict(index_path=packed['index_path'],
        index_sha256=packed['index_sha256'], archive=str(archive))}, None, cached_members=1)
    if args.cpu_audit:
        selected = [x for x in index if (x['version'], x['chunk']) in ((0, 1), (29, 12))]
        for item in selected:
            validate_payload(reader.read_member(args.task, item['rank'], item['version'], item['chunk']), item, args.task)
        reader.close()
        write(args.cold/'CPU-AUDIT.json', dict(passed=True, task=args.task, source_commit=args.pin,
            checked_unix=time.time(), packed_sha256=source['packed']['sha256'],
            sampled_members=len(selected), total_members=len(index),
            coverage='First and last member of each rank; complete payload validation remains in label pass'))
        return
    assert socket.gethostname().startswith('job-') and args.teacher and args.probe and args.deadline
    assert sha(args.teacher) == source['teacher']['sha256']
    assert sha(archive) == packed['archive_sha256']
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    device = selected_device()
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(device['allocation_ordinal'])
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    args.cold.mkdir(parents=True, exist_ok=False)
    model, norm = Path(spec['native_model']['path']), Path(spec['native_norm']['path'])
    os.environ.update(RLINF_PATH=str(runtime), ROBOCASA_NATIVE_OUTPUT=str(args.output/'native'),
        ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native, _versions
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.envs.sim.robocasa365.robocasa365_env import map_released_control_scores
    native, _, signature = _load_native(dict(model_directory=str(model), norm_path=str(norm),
        source_commit=args.pin, runtime_commit=spec['runtime_commit']), code)
    assert signature['model_sha256'] == spec['native_model']['sha256']
    assert signature['norm_sha256'] == spec['native_norm']['sha256']
    with initialize_config_dir(config_dir=str(code/'benchmarks/robocasa/configs'), version_base=None):
        cfg = compose(config_name='robocasa365_pi05_eval')
    cfg.rollout.model.model_path = str(args.teacher.parent)
    cfg.rollout.model.openpi_data.norm_stats_path = str(norm)
    teacher = get_model(cfg.rollout.model, torch_dtype=torch.bfloat16).to('cuda:0').eval().requires_grad_(False)
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    codec = DirectActionCodec.from_stats(stats)
    motor = FlowDirectMotor(native, codec)
    frozen = (tensor_hash(_versions(native.model)), tensor_hash(_versions(teacher)))
    began, completed, shards = time.time(), 0, []
    try:
        for shard in range(4):
            start_rank = 2*shard
            slots, steps = 32, 360
            states = np.zeros((slots, steps, 25), np.float32)
            actions = np.zeros((slots, steps, 50, 12), np.float32)
            rewards = np.zeros((slots, steps, 50), np.float32)
            terms = np.zeros((slots, steps, 50), bool); truncs = terms.copy()
            seeds = np.zeros((slots, steps), np.int64)
            features = np.zeros((slots, steps, 2073), np.float32)
            labels = np.zeros_like(actions); done = np.zeros((slots, steps), bool)
            members = [x for x in index if start_rank <= x['rank'] < start_rank+2]
            for item in members:
                assert time.time() < args.deadline-120, 'Preserve completed shards within allocation limit'
                data = reader.read_member(args.task, item['rank'], item['version'], item['chunk'])
                s, a, r, term, trunc, seed = validate_payload(data, item, args.task)
                with torch.no_grad():
                    feat, prepared = motor.prepare_batch(public_observation(data['current_observation']))
                    noise = np.stack([teacher_noise(item['sha256'], row) for row in range(16)])
                    original = teacher.build_prefix_cache
                    try:
                        teacher.build_prefix_cache = lambda *a, **kw: prepared['prefix']
                        prediction, _ = teacher.predict_action_batch(prepared['observation'], mode='eval',
                            noise=torch.as_tensor(noise, device=teacher.device))
                        mapped = map_released_control_scores(prediction.detach().float().cpu().numpy(), cfg.env.eval.action_space)
                    finally:
                        teacher.build_prefix_cache = original
                        native._prepared = None
                assert feat.shape == (16, 2073) and np.array_equal(feat[:, :25], s)
                assert np.isfinite(feat).all() and mapped.shape == (16, 50, 12) and np.isfinite(mapped).all()
                lane = slice((item['rank']-start_rank)*16, (item['rank']-start_rank+1)*16)
                t = item['version']*12+item['chunk']-1
                assert not done[lane, t].any()
                states[lane,t], actions[lane,t], rewards[lane,t] = s, a, r
                terms[lane,t], truncs[lane,t], seeds[lane,t] = term, trunc, seed
                features[lane,t], labels[lane,t], done[lane,t] = feat, mapped, True
                completed += 16
                if completed % 768 == 0:
                    write(args.cold/'PROGRESS.json', dict(task=args.task, checked_unix=time.time(),
                        completed_decisions=completed, expected_decisions=46080,
                        banked_shards=len(shards), elapsed_seconds=time.time()-began,
                        inference_batch=16, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            assert done.all()
            assert (rewards.reshape(slots, 30, 12, 50).sum((2,3)) <= 1).all()
            assert (seeds.reshape(slots,30,12) == seeds.reshape(slots,30,12)[:,:,:1]).all()
            assert len(set(seeds[:,::12].flatten().tolist())) == slots*30
            directory = args.output/f'shard-{shard:02d}'; directory.mkdir()
            np.savez(directory/'RAW-PUBLIC-CORPUS.npz', public_states=states, physical_actions=actions,
                primitive_rewards=rewards, native_seeds=seeds, native_terminations=terms, native_truncations=truncs,
                producer_versions=np.broadcast_to(np.repeat(np.arange(30),12),(slots,steps)),
                episode_starts=np.broadcast_to(np.arange(steps)%12==0,(slots,steps)),
                physical_episode_ends=np.broadcast_to(np.arange(steps)%12==11,(slots,steps)))
            np.savez(directory/'FEATURES-TEACHER-30.npz', features=features, mapped_teacher_actions=labels)
            target = args.cold/directory.name; target.mkdir()
            files = {}
            for path in directory.iterdir():
                digest = sha(path); destination = target/path.name
                with path.open('rb') as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src,dst,16*2**20)
                assert sha(destination) == digest
                files[path.name] = dict(path=str(destination),sha256=digest,bytes=destination.stat().st_size)
            assert frozen == (tensor_hash(_versions(native.model)),tensor_hash(_versions(teacher)))
            receipt = dict(completed=True,task=args.task,shard_index=shard,global_slots=list(range(start_rank*16,(start_rank+2)*16)),
                source_commit=args.pin,spec_sha256=args.spec_sha256,files=files,codec=codec.record(),
                provenance=dict(raw_archive=packed,teacher=source['teacher'],signature=signature,
                    label_sampling='teacher_noise(raw_member_SHA,row), native eval ODE10, exact native mapped12 controls'),
                teacher_queries=int(done.sum()),all_payloads_validated=True,all_files_cold_readback=True,
                elapsed_seconds=time.time()-began,scheduler_uuid=device['uuid'])
            write(target/'SHARD.json', receipt); shards.append(dict(path=str(target/'SHARD.json'),sha256=sha(target/'SHARD.json')))
            shutil.rmtree(directory)
        write(args.cold/'COMPLETE.json',dict(completed=True,task=args.task,source_commit=args.pin,
            spec_sha256=args.spec_sha256,shards=shards,total_teacher_queries=completed,
            elapsed_seconds=time.time()-began,optimizer_updates=0))
    except BaseException as error:
        write(args.cold/'ERROR.json',dict(error=repr(error),completed_decisions=completed,
            banked_shards=shards,checked_unix=time.time(),partial_shard_accepted=False))
        raise
    finally:
        reader.close()


if __name__ == '__main__':
    main()
