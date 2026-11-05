"""Allocated Linux import/config checks; no models, simulators or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from hydra import compose, initialize_config_dir

code = Path(os.environ['TASK_CODE'])
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
assert sys.platform == 'linux' and torch.cuda.device_count() == 0
with (out/'tests.log').open('x') as log:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q',
        str(code/'tests/test_rlinf_native_ppo.py'), str(code/'tests/test_robotwin_noise.py')],
        stdout=log, stderr=subprocess.STDOUT, check=False)
assert result.returncode == 0, 'See allocated core-test log'
spec = importlib.util.spec_from_file_location('source_pilot_entry', Path(__file__).with_name('train_entry.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
model = os.environ['NATIVE_PPO_MODEL']
with initialize_config_dir(version_base='1.1', config_dir=str(code/'configs/native_ppo')):
    cfg = compose(config_name='adjust_bottle_published_batch_two', overrides=[
        'env.train.task_config.task_name=place_phone_stand',
        'env.eval.task_config.task_name=place_phone_stand',
        f'++actor.model.openpi_data.norm_stats_path={model}/assets/pi0.5_clean_randomize_joint_training/norm_stats.json',
        '++actor.model.openpi_data.adapt_to_pi=true',
        '++actor.model.openpi_data.extra_delta_transform=true'])
contract = entry.validate_source(cfg)
assert cfg.env.train.task_config.task_name == cfg.env.eval.task_config.task_name == 'place_phone_stand'
(out/'RESULT.json').write_text(json.dumps(dict(passed=True, source_commit=os.environ['TASK_SOURCE'],
    core_test_exit=result.returncode, actual_config=contract, cuda_count=0,
    model_constructed=False, simulator_constructed=False, optimizer_updates=0), indent=2)+'\n')
print('Allocated CPU checks passed', flush=True)
