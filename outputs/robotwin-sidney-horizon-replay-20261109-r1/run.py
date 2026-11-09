"""Rerun the clean native Sidney phone panel with a 1,200-target action limit."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'native_sft', OUTPUTS/'robotwin-sidney-native-sft-20261109-r1/run.py')
native_sft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_sft)
# The same evaluator helpers, model worker and source pins as the native panel.
from evaluate import NATIVE_COMMIT, environment_args, panel, save, sha
from qualify import LEROBOT, LEROBOT_COMMIT, MODEL_PYTHON, wait_for

NATIVE_LIMIT = 400
HORIZON = 1200


def episode(task, config, native, case, worker, scratch, directory):
    seed, index, prompt = case['seed'], case['index'], case['instruction']
    task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **config)
    assert task.step_lim == NATIVE_LIMIT and task.take_action_cnt == 0
    # The only change from the native panel.
    task.step_lim = HORIZON
    actors = native_sft.bind_instruction(task, native, prompt)
    directory.mkdir()
    obs = task.get_obs()
    panel(obs, directory/'initial.jpg')
    chunks = []
    success_at = None
    while not task.eval_success and task.take_action_cnt < HORIZON:
        images, state = native_sft.read_inputs(obs)
        if not chunks:
            np.savez_compressed(directory/'initial.npz', images=images, state=state, instruction=prompt)
        source = scratch/'observation.npz'
        target = scratch/f'{directory.name}-{len(chunks)}.npz'
        np.savez(source, images=images, state=state, instruction=prompt)
        noise_seed = 1000 + index*100 + len(chunks)
        assert len(chunks) < 100
        worker.stdin.write(json.dumps(dict(observation=str(source), actions=str(target), noise_seed=noise_seed))+'\n')
        worker.stdin.flush()
        receipt = target.with_suffix('.json')
        wait_for(receipt, worker)
        detail = json.loads(receipt.read_text())
        assert detail['frozen_unchanged']
        with np.load(target, allow_pickle=False) as result:
            actions = result['actions'].copy()
        assert actions.shape == (50, 14) and np.isfinite(actions).all()
        np.save(directory/f'actions-{len(chunks):02d}.npy', actions)
        before = task.take_action_cnt
        for action in actions:
            task.take_action(action)
            obs = task.get_obs()
            if task.eval_success or task.take_action_cnt == HORIZON:
                break
        if task.eval_success and success_at is None:
            success_at = task.take_action_cnt
        chunks.append(dict(noise_seed=noise_seed, delivered=task.take_action_cnt-before,
                           actions_sha256=detail['actions_sha256']))
        for temp in (target, target.with_suffix('.inputs.npz'), receipt):
            temp.unlink()
    assert task.eval_success or task.take_action_cnt == HORIZON
    success = bool(task.eval_success)
    result = dict(index=index, seed=seed, instruction=prompt, actors=actors, chunks=chunks,
                  success=success, targets=task.take_action_cnt, success_at=success_at,
                  success_by_400=success and success_at <= NATIVE_LIMIT,
                  success_by_1200=success)
    save(directory/'episode.json', result)
    task.close_env(clear_cache=((index+1) % config['clear_cache_freq'] == 0))
    return result


def run(out, count):
    import torch
    from envs.place_phone_stand import place_phone_stand
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer
    assert torch.cuda.device_count() == 1 and os.uname().nodename.startswith('job-')
    torch.cuda.set_device(0)
    _install_allocated_renderer(out/'renderers')
    native = Path(os.environ['NATIVE_SOURCE'])
    native_sft.verify_source(native, NATIVE_COMMIT)
    native_sft.verify_source(LEROBOT, LEROBOT_COMMIT)
    bundle = json.loads((OUTPUTS/'robotwin-sft-initializer-selection-20261109-r1/SIDNEY-ORIGINAL-BUNDLE.json').read_text())
    for item in bundle['files']:
        assert sha(Path(bundle['path'])/item['path']) == item['sha256']
    rows = json.loads((OUTPUTS/'robotwin-motus-release-baseline-20261109-r1/PHONE100-CASES.json').read_text())['settings']['demo_clean']
    scratch = Path(os.environ['TASK_LOCAL'])
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], model_revision=bundle['revision'],
                  native_commit=NATIVE_COMMIT, lerobot_commit=LEROBOT_COMMIT, optimizer_updates=0,
                  native_limit=NATIVE_LIMIT, horizon=HORIZON, episodes=count, settings={})
    save(out/'RESULT.json', report)
    with (out/'model.log').open('w') as log:
        proc = subprocess.Popen([MODEL_PYTHON, str(native_sft.QUAL/'model_worker.py'), '--model', bundle['path'],
                                 '--output', str(out)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                text=True, env=dict(os.environ, PYTHONPATH=str(LEROBOT/'src')))
        try:
            wait_for(out/'READY.json', proc)
            ready = json.loads((out/'READY.json').read_text())
            assert ready['strict_weight_load'] and ready['chunk_size'] == 50 and ready['flow_steps'] == 10
            report['status'] = 'running'
            config = environment_args(native, 'demo_clean')
            config.update(policy_name='sidney_lerobot', ckpt_setting='sidney_release')
            task = place_phone_stand()
            task.suc = task.test_num = 0
            episodes = report['settings']['demo_clean'] = []
            for case in rows[:count]:
                directory = out/f"demo_clean-{case['index']:03d}-seed-{case['seed']}"
                episodes.append(episode(task, config, native, case, proc, scratch, directory))
                report['success_by_400'] = sum(e['success_by_400'] for e in episodes)
                report['success_by_1200'] = sum(e['success_by_1200'] for e in episodes)
                save(out/'RESULT.json', report)
            proc.stdin.close()
            assert proc.wait(timeout=60) == 0
            report['status'] = 'complete'
            save(out/'RESULT.json', report)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--output', type=Path, required=True)
    args.add_argument('--episodes', type=int, choices=(2, 100), required=True)
    parsed = args.parse_args()
    run(parsed.output, parsed.episodes)
