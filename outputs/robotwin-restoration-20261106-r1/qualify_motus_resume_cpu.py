"""Run the actual resume staging body on Linux CPU; leave GPU admission prepared."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    assert sys.platform == 'linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    plan = json.loads((ROOT/'MOTUSRESUME01-CONTRACT.json').read_bytes())
    assert plan['status'] == 'prepared' and not plan['budget_extension_approved']
    assert plan['cpu_boundary_qualification_passed']
    assert sha(Path(plan['cpu_boundary_result'])) == plan['cpu_boundary_sha256']
    assert json.loads(Path(plan['cpu_boundary_result']).read_bytes())['status'] == 'passed'
    output = BASE/'cpumotusresume01'; output.mkdir()
    started = time.monotonic()
    launcher = ROOT/'resume_motus01.sh'
    assert sha(launcher) == plan['launcher_sha256']
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", launcher.read_text(), re.S)
    assert len(blocks) == 4
    for block in blocks: ast.parse(block)
    subprocess.run(['bash', '-n', str(launcher)], check=True)
    local = Path('/tmp/rtr1-cpu-staging01')
    cold = BASE/'cpumotusresume01-stage'
    os.environ.update(TASK_LOCAL=str(local), TASK_COLD=str(cold), TASK_OUT=str(output),
        NATIVE_PPO_OUTPUT=str(cold/'state'), NATIVE_PPO_ACCOUNTING_ROOT=str(cold/'action-accounting'),
        NATIVE_PPO_DEADLINE=str(int(time.time())+9000), NATIVE_PPO_ATTEMPT='ppo-resume01',
        NATIVE_PPO_WORLD='8', NATIVE_PPO_PLACEMENT='0-7')
    # Real10GB checkpoint/model hashes and preserved journals, without allocating CUDA.
    subprocess.run([sys.executable, '-c', blocks[1], str(ROOT)], check=True)
    from fluffy_bassoon.rl.robotwin_batch import accounted_controller_source
    controller = accounted_controller_source(os.environ['ROBOTWIN_PATH'], local/'controller')
    os.environ.update(TASK_CONTROLLER=controller['controller_source'], TASK_CONTROLLER_SHA=controller['controller_sha256'])
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import rlinf.config as native
    import motus_resume_entry
    from motus_configuration import overrides, validate
    with initialize_config_dir(version_base='1.1', config_dir=str(ROOT.parents[1]/'configs/native_ppo')):
        cfg = compose(config_name='motus_phone_two', overrides=overrides('ppo', optimized_setup=True)+[
            'native_pilot.stop_after_iteration=2', f'runner.resume_dir={local}/global_step_1'])
    recipe = validate(cfg)
    class Placement:
        def get_world_size(self, role):
            assert role in ('actor', 'rollout', 'env')
            return 8
    with patch.object(native, 'Cluster', lambda *args, **kwargs: None), \
            patch.object(native, 'HybridComponentPlacement', lambda *args, **kwargs: Placement()):
        cfg = native.validate_cfg(cfg)
    assert recipe['world'] == 8 and recipe['accumulation'] == 1
    assert cfg.runner.resume_dir == str(local/'global_step_1')
    assert not torch.cuda.is_initialized()
    assert json.loads((ROOT/'MOTUSRESUME01-CONTRACT.json').read_bytes()) == plan
    write(output/'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
    write(output/'RESULT.json', dict(status='passed', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, native_models_or_simulator_constructed=False,
        exact_launcher_staging_body_executed=True, full_checkpoint_staged_and_hashed=True,
        old_action_escrow_preserved=True, actual_controller_source=controller,
        full_native_schema_with_declared_eight_rank_placement=True,
        entry_import_passed=True, recipe=recipe, cpu_boundary_sha256=plan['cpu_boundary_sha256'],
        entry_sha256=plan['entry_sha256'], launcher_sha256=plan['launcher_sha256'],
        gpu_restore_still_required=True, gpu_launch_admitted=False, seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
