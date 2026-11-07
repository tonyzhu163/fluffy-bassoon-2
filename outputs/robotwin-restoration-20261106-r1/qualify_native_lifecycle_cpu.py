"""Run-local Linux CPU checks before admitting the lifecycle GPU fixture."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.rl.robotwin_batch import accounted_controller_source
import qualify_native_lifecycle as lifecycle
import qualify_scene_setup as reference


ROOT = Path(os.environ['TASK_DRIVER'])
CODE = Path(os.environ['TASK_CODE'])
BASE = reference.BASE
RUN_ID = os.environ.get('TASK_CPU_CHECK_ID', 'cpulifecycle01')
assert re.fullmatch(r'cpulifecycle[0-9]{2}', RUN_ID)
LOCAL = Path('/tmp'/Path('rt-'+RUN_ID))
OUT = BASE/RUN_ID


HOOK = r'''
import ast,json,inspect,os,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
import motus_teacher_entry as teacher
import evaluate_motus_dev as evaluation
import fluffy_bassoon.rl.robotwin_noise as noise
import fluffy_bassoon.rl.robotwin_native_setup as setup
flag=json.loads(sys.argv[1]);out=Path(sys.argv[2])
assert evaluation.configure_environment is teacher.configure_environment
for cls in (teacher.MotusEnvironment,evaluation.EvaluationEnvironment):
 source=inspect.getsource(cls.init_worker)
 assert 'root = configure_environment(self)' in source
with patch.object(setup,'worker_setup',lambda *args:None),patch.object(noise,'_install_allocated_renderer',lambda path:None):
 for mode in ('train','eval'):
  worker=SimpleNamespace(_rank=0,cfg=OmegaConf.create(dict(native_pilot=dict(
   output=str(out/mode),defer_scene_setup=flag,recycle_scene_executor=flag))))
  teacher.configure_environment(worker)
  cfg,original,effective=evaluation.configuration(out/mode,defer_scene_setup=flag,recycle_scene_executor=flag)
  evaluation.validate_evaluation(cfg,original,effective)
  assert cfg.native_pilot.defer_scene_setup==cfg.native_pilot.recycle_scene_executor==flag
from robotwin.envs.vector_env import SubEnv,VectorEnv
assert SubEnv._fb2_defer_setup_scene==flag
assert bool(getattr(VectorEnv,'_fb2_recycles_executor',False))==flag
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='passed',optimized_setup=flag,shared_train_eval_hook=True,
 evaluation_recipe_preserved=True,hardware_edges_mocked=True,cuda_initialized=False)))
'''


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    OUT.mkdir(); LOCAL.mkdir()
    report = dict(status='checking', runtime_source_commit=os.environ['TASK_SOURCE'],
        cpu_driver_source_commit=os.environ['TASK_CPU_DRIVER_SOURCE'], gpu_seconds=0)
    try:
        assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
        assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
        controller = accounted_controller_source(os.environ['ROBOTWIN_PATH'], LOCAL/'controller')
        assert controller['controller_sha256'] == json.loads((reference.REFERENCE/'controller.json').read_bytes())['controller_sha256']
        os.environ.update(TASK_CONTROLLER=controller['controller_source'],
            TASK_CONTROLLER_SHA=controller['controller_sha256'], TASK_LOCAL=str(LOCAL),
            TASK_COLD=str(OUT/'unused-cold'))
        cfg = reference.inputs(LOCAL, plan=lifecycle.contract())
        assert cfg['env']['eval']['total_num_envs'] == 16
        for index in range(6):
            row = read_tensor(LOCAL/'state/traces/rank-00'/f'{index:05d}.pt.gz')
            assert tuple(row['proposed_motor_actions'].shape) == (4, 32, 14)
            assert len(row['outputs']) == 5
        subprocess.run(['bash', '-n', str(ROOT/'qualify_native_lifecycle.sh')], check=True)
        for path in (ROOT/'qualify_native_lifecycle.py', CODE/'src/fluffy_bassoon/rl/robotwin_noise.py'):
            ast.parse(path.read_bytes())
        tests = OUT/'pytest.txt'
        with tests.open('x') as stream:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                str(CODE/'tests/test_robotwin_noise.py'), str(CODE/'tests/test_robotwin_execution.py')],
                stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
        passed = int(re.search(r'(\d+) passed', tests.read_text()).group(1))
        schemas, hooks = [], []
        for flag in (False, True):
            path = OUT/f'configs-{int(flag)}.json'
            with path.open('x') as stream:
                subprocess.run([sys.executable, str(ROOT/'motus_configuration.py'), 'validate_schema',
                    *(['--optimized-setup'] if flag else [])], stdout=stream, stderr=subprocess.STDOUT,
                    check=True, timeout=240)
            value = json.loads(path.read_text().strip().splitlines()[-1])
            assert value['status'] == 'complete' and len(value['configs']) == 2
            assert all(all(row['simulator_setup'][key] == flag for key in
                ('defer_scene_setup', 'recycle_scene_executor')) for row in value['configs'])
            schemas.append(value['configs'])
            path = OUT/f'hooks-{int(flag)}.json'
            with path.open('x') as stream:
                subprocess.run([sys.executable, '-c', HOOK, json.dumps(flag), str(OUT/f'profile-{int(flag)}')],
                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=240)
            value = json.loads(path.read_text().strip().splitlines()[-1])
            assert value['status'] == 'passed'
            hooks.append(value)
        for old, new in zip(*schemas, strict=True):
            assert {k:v for k,v in old.items() if k != 'simulator_setup'} == {k:v for k,v in new.items() if k != 'simulator_setup'}
        assert not torch.cuda.is_initialized()
        report.update(status='passed', core_tests_passed=passed, new_core_cases=4,
            four_native_configurations=True, shared_train_eval_hooks=hooks,
            unchanged_numerical_recipe=True, controller_sha256=controller['controller_sha256'],
            source_traces_cold_staged=24, rank0_traces_deserialized=6,
            model_or_simulator_constructed=False, cuda_initialized=False,
            files={p.name:sha(p) for p in OUT.glob('*') if p.is_file()})
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(OUT/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
