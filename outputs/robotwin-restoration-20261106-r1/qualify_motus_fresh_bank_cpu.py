"""Run-scoped Linux qualification, using the actual pinned native function.

Tasks are controlled doubles. This tests reference control flow and ordered
selection; it does not qualify physics, planner CUDA support or fresh scenes.
"""
import argparse
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from types import SimpleNamespace

from motus_fresh_bank import ROOT, SCAN, sha, write, declaration, accepted, select, collect, check_initial_observation

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def rejected(fn):
    try: fn()
    except (AssertionError, ValueError): return
    raise AssertionError('Invalid scan or row accepted')


class Task:
    def __init__(self, mode): self.mode = mode
    def setup_demo(self, *, now_ep_num, seed, **options):
        assert now_ep_num == seed and options == {'step_lim': 192}
        if self.mode == 'setup_error': raise RuntimeError('controlled setup rejection')
        self.ep_num = seed + (1 if self.mode == 'identity_drift' else 0)
        self.plan_success = self.mode != 'planner_failure'
    def get_obs(self): return None
    def play_once(self):
        if self.mode == 'play_error': raise RuntimeError('controlled planner rejection')
    def check_success(self): return self.mode != 'task_failure'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True)
    parser.add_argument('--contract', choices=('MOTUS-FRESH-SCENES02-CONTRACT.json',), required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'cpumotusfreshbank[0-9]{2}', args.identity)
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    source = os.environ['TASK_SOURCE']; code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == source
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    out = BASE/args.identity; out.mkdir()
    report = dict(status='checking', source_commit=source, GPU_seconds=0,
        actual_simulators_or_models_constructed=False, physically_qualified_scenes=0,
        actual_scene_collection_still_required=True, new_GPU_allocation_admitted=False)
    write(out/'RESULT.json', report)
    try:
        plan, candidates = declaration()
        native = Path(os.environ['ROBOTWIN_PATH'])/'robotwin/envs/vector_env.py'
        assert sha(native) == plan['native_vector_env_sha256']
        native_source = native.read_text()
        cls = next(n for n in ast.parse(native_source).body if isinstance(n, ast.ClassDef) and n.name == 'SubEnv')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'check_seed')
        namespace = dict(time=time, logging=logging)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(native), 'exec'), namespace)
        checks = []
        for mode in ('pass', 'setup_error', 'planner_failure', 'task_failure', 'play_error', 'identity_drift'):
            task = Task(mode)
            sub = SimpleNamespace(task=task, global_lock=threading.Lock(), lock=threading.Lock(),
                args={'step_lim': 192}, env_id=0)
            value = namespace['check_seed'](sub, candidates[0])
            expected = mode in ('pass', 'identity_drift')
            assert value['play_once_success'] is expected
            assert value['setup_demo_success'] is (mode != 'setup_error')
            row = dict(seed=candidates[0], **value, expert_physical_seed=getattr(task, 'ep_num', -1),
                initial_reset_exact=expected, initial_observation_sha256='a'*64 if expected else None)
            assert accepted(row) is (mode == 'pass')
            checks.append(dict(mode=mode, native_result=value, accepted=accepted(row)))
        rows = []
        for index, seed in enumerate(candidates):
            valid = index % 3 != 0
            rows.append(dict(seed=seed, setup_demo_success=True, play_once_success=valid, cost_time=0,
                expert_physical_seed=seed, initial_reset_exact=valid,
                initial_observation_sha256='b'*64 if valid else None))
            if sum(accepted(row) for row in rows) == 64: break
        selected = select(rows, candidates)
        assert len(rows) == 96 and selected == [seed for i, seed in enumerate(candidates[:96]) if i % 3 != 0]
        rejected(lambda: select(rows[:-1], candidates))
        rejected(lambda: select(rows[1:], candidates))
        rejected(lambda: select(rows[1:2]+rows[:1]+rows[2:], candidates))
        rejected(lambda: select(rows+[dict(rows[-1], seed=candidates[96])], candidates))
        rejected(lambda: accepted(dict(rows[1], SFT_success=True)))
        assert not accepted(dict(rows[1], expert_physical_seed=rows[1]['seed']+1))
        assert not accepted(dict(rows[1], initial_reset_exact=False, initial_observation_sha256=None))
        import torch
        assert not torch.cuda.is_initialized()
        image = torch.arange(48, dtype=torch.uint8).reshape(1, 4, 4, 3)
        observation = dict(states=torch.zeros(1, 14), main_images=image,
            wrist_images=torch.stack([image, image], dim=1), task_descriptions=['place phone'])
        check_initial_observation(observation)
        black_wrist = observation['wrist_images'].clone(); black_wrist[0, 1] = 0
        rejected(lambda: check_initial_observation(dict(observation, wrist_images=black_wrist)))
        rejected(lambda: check_initial_observation(dict(observation, states=torch.full((1, 14), float('nan')))))
        assert not torch.cuda.is_initialized()
        seeds = Path(os.environ['REPO_PATH'])/'rlinf/envs/sim/robotwin/seeds/eval_seeds.json'
        assert sha(seeds) == plan['full_eval_source_sha256']
        actual = json.loads(seeds.read_bytes())[plan['task']]['success_seeds']
        assert actual == plan['excluded_native_eval_seeds']
        contract = ROOT/args.contract
        admission = json.loads(contract.read_bytes())
        for name, digest in admission['source_files'].items(): assert sha(code/name) == digest
        subprocess.run(['bash', '-n', str(ROOT/'motus_fresh_bank.sh')], check=True)
        assert admission['closed_conservative_gpu_seconds']+admission['reserved_gpu_seconds'] == admission['all_worst_gpu_seconds'] == 359859
        assert admission['all_worst_gpu_seconds'] <= admission['approved_cap_gpu_seconds'] == 360000
        assert admission['reserved_gpu_seconds'] == 3600 and admission['worker_timeout_seconds'] == 3100
        # Exercise the real shell environment without importing a device library.
        shared = code/'outputs/robotwin-sidney-matched-baselines-20261106-r1/runtime_environment.sh'
        wrapper = (ROOT/'motus_fresh_bank.sh').read_text()
        guard = 'if [[ ${CUDA_VISIBLE_DEVICES+x} != x ]]; then export CUDA_VISIBLE_DEVICES=0; fi'
        assert guard in wrapper and wrapper.index(guard) < wrapper.index('source "$TASK_RUN/runtime_environment.sh"')
        for previous, expected in ((None, '0'), ('scheduler-provided', 'scheduler-provided')):
            command = ('unset CUDA_VISIBLE_DEVICES\n' if previous is None else 'export CUDA_VISIBLE_DEVICES=scheduler-provided\n')
            command += guard+'\nsource "$1"\nprintf "%s" "$CUDA_VISIBLE_DEVICES"'
            assert subprocess.check_output(['bash', '-c', command, 'fresh-scene-CVD', str(shared)], text=True) == expected
        rejected(lambda: collect(contract))
        original = Path(admission['original_configuration'])
        cfg = json.loads(original.read_bytes())['env']['eval']
        assert cfg['task_config']['task_name'] == plan['task'] and cfg['max_episode_steps'] == 192
        report.update(status='passed_fresh_bank_selection_and_native_check_seed_CPU_reference',
            scan_sha256=sha(SCAN), native_vector_env_sha256=sha(native), native_check_seed_cases=checks,
            deterministic_candidate_count=len(candidates), excluded_native_eval_seed_count=len(actual),
            synthetic_first64_complete_prefix=96, missing_reordered_extra_or_policy_conditioned_rows_rejected=True,
            physical_seed_substitution_rejected=True, failed_initial_reset_rejected=True,
            original_configuration_sha256=sha(original), prepared_GPU_collection_rejected=True,
            native_uint8_initial_packet_checks_passed=True, each_camera_required_nonblank=True,
            bounded_one_GPU_launcher_Linux_bash_syntax_passed=True, current100h_cap_reservation_arithmetic_passed=True,
            actual_shared_shell_unset_visibility_falls_back_to_one_and_explicit_visibility_preserved=True,
            original_sampler_inventory_not_edited=True, native_function_sha256=hashlib.sha256(ast.get_source_segment(native_source, method).encode()).hexdigest())
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally: write(out/'RESULT.json', report)
    print('FRESH_BANK_CPU_REFERENCE', str(out/'RESULT.json'), sha(out/'RESULT.json'), flush=True)


if __name__ == '__main__': main()
