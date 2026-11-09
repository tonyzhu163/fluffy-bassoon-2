"""Evaluate the original LeRobot policy on the frozen native phone panel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

OUTPUTS = Path(__file__).resolve().parents[1]
REFERENCE = OUTPUTS / 'robotwin-motus-release-baseline-20261109-r1'
QUAL = OUTPUTS / 'robotwin-sidney-qualification-20261105-r1'
sys.path[:0] = [str(REFERENCE), str(QUAL)]
from evaluate import NATIVE_COMMIT, environment_args, panel, save, sha
from qualify import LEROBOT, LEROBOT_COMMIT, MODEL_PYTHON, wait_for


def verify_source(path, commit):
    assert subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', path, 'status', '--porcelain', '-uno'])


def read_inputs(obs):
    images = np.stack([obs['observation'][name]['rgb'] for name in ('head_camera', 'left_camera', 'right_camera')])
    state = np.asarray(obs['joint_action']['vector'], dtype=np.float32)
    assert images.dtype == np.uint8 and images.shape == (3, 240, 320, 3)
    assert all(v.std() > 1 for v in images)
    assert state.shape == (14,) and np.isfinite(state).all()
    return images, state


def bind_instruction(task, native, prompt):
    description = native / 'description'
    names = [f'077_phone/base{task.phone_id}', f'078_phonestand/base{task.stand_id}']
    labels = []
    for name in names:
        terms = json.loads((description/'objects_description'/f'{name}.json').read_text())
        labels.append(terms.get('unseen') or terms['seen'])
    templates = json.loads((description/'task_instruction/place_phone_stand.json').read_text())['unseen']
    assert any(prompt == template.replace('{A}', 'the '+phone).replace('{B}', 'the '+stand)
               for template in templates if '{A}' in template and '{B}' in template
               for phone in labels[0] for stand in labels[1]), (names, prompt)
    task.set_instruction(instruction=prompt)
    assert task.get_instruction() == prompt
    return names


def episode(task, config, native, case, worker, scratch, directory):
    seed, index, prompt = case['seed'], case['index'], case['instruction']
    task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **config)
    assert task.step_lim == 400 and task.take_action_cnt == 0
    actors = bind_instruction(task, native, prompt)
    directory.mkdir()
    obs = task.get_obs()
    panel(obs, directory/'initial.jpg')
    chunks = []
    while not task.eval_success and task.take_action_cnt < task.step_lim:
        images, state = read_inputs(obs)
        if not chunks:
            np.savez_compressed(directory/'initial.npz', images=images, state=state, instruction=prompt)
        source = scratch/'observation.npz'
        # Unique output per request prevents picking up a previous response.
        target = scratch/f'{directory.name}-{len(chunks)}.npz'
        np.savez(source, images=images, state=state, instruction=prompt)
        noise_seed = 1000 + index*100 + len(chunks)
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
            if task.eval_success or task.take_action_cnt == task.step_lim:
                break
        chunks.append(dict(noise_seed=noise_seed, delivered=task.take_action_cnt-before,
                           actions_sha256=detail['actions_sha256']))
        for temp in (target, target.with_suffix('.inputs.npz'), receipt):
            temp.unlink()
    assert task.eval_success or task.take_action_cnt == 400
    result = dict(index=index, seed=seed, instruction=prompt, actors=actors, chunks=chunks,
                  success=bool(task.eval_success), targets=task.take_action_cnt)
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
    verify_source(native, NATIVE_COMMIT)
    verify_source(LEROBOT, LEROBOT_COMMIT)
    bundle = json.loads((OUTPUTS/'robotwin-sft-initializer-selection-20261109-r1/SIDNEY-ORIGINAL-BUNDLE.json').read_text())
    for item in bundle['files']:
        assert sha(Path(bundle['path'])/item['path']) == item['sha256']
    cases = json.loads((REFERENCE/'PHONE100-CASES.json').read_text())['settings']
    scratch = Path(os.environ['TASK_LOCAL'])
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'], model_revision=bundle['revision'],
                  native_commit=NATIVE_COMMIT, lerobot_commit=LEROBOT_COMMIT, optimizer_updates=0,
                  episodes_per_domain=count, settings={})
    save(out/'RESULT.json', report)
    with (out/'model.log').open('w') as log:
        proc = subprocess.Popen([MODEL_PYTHON, str(QUAL/'model_worker.py'), '--model', bundle['path'],
                                 '--output', str(out)], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                text=True, env=dict(os.environ, PYTHONPATH=str(LEROBOT/'src')))
        try:
            wait_for(out/'READY.json', proc)
            ready = json.loads((out/'READY.json').read_text())
            assert ready['strict_weight_load'] and ready['chunk_size'] == 50 and ready['flow_steps'] == 10
            report['status'] = 'running'
            for setting, rows in cases.items():
                config = environment_args(native, setting)
                config.update(policy_name='sidney_lerobot', ckpt_setting='sidney_release')
                task = place_phone_stand()
                task.suc = task.test_num = 0
                episodes = report['settings'][setting] = []
                for case in rows[:count]:
                    directory = out/f"{setting}-{case['index']:03d}-seed-{case['seed']}"
                    episodes.append(episode(task, config, native, case, proc, scratch, directory))
                    save(out/'RESULT.json', report)
                del task
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
