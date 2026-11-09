"""Compare observation packaging against LeRobot's eval path on saved real inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config

bundle, root = map(Path, sys.argv[1:3])
tokenizer = '/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c'
camera = dict(head_camera='cam_high', left_camera='cam_left_wrist', right_camera='cam_right_wrist')
rename = {f'observation.images.{k}': f'observation.images.{v}' for k,v in camera.items()}
cfg = PreTrainedConfig.from_pretrained(str(bundle))
assert isinstance(cfg, PI05Config)
cfg.device = 'cpu'
common = dict(device_processor={'device':'cpu'}, tokenizer_processor={'tokenizer_name':tokenizer})
current, _ = make_pre_post_processors(cfg, pretrained_path=str(bundle), preprocessor_overrides=common.copy())
public, _ = make_pre_post_processors(cfg, pretrained_path=str(bundle), preprocessor_overrides={
    **common, 'rename_observations_processor': {'rename_map':rename}})
files = sorted(root.glob('demo_clean-*/initial.npz'))[:16]
assert len(files) == 16
for file in files:
    with np.load(file, allow_pickle=False) as arr:
        images, state, prompt = arr['images'], arr['state'], arr['instruction'].item()
    manual = {'observation.state':torch.from_numpy(state.copy()), 'task':prompt}
    gym = dict(pixels={}, agent_pos=state[None].copy())
    for i,(source,dest) in enumerate(camera.items()):
        manual[f'observation.images.{dest}'] = torch.from_numpy(images[i].copy()).permute(2,0,1).float()/255
        gym['pixels'][source] = images[i][None].copy()
    gym = preprocess_observation(gym)
    gym['task'] = [prompt]
    a, b = current(manual), public(gym)
    tens = {k for k,v in a.items() if torch.is_tensor(v)}
    assert set(rename.values()) <= tens
    assert {'observation.state','observation.language.tokens','observation.language.attention_mask'} <= tens
    assert tens == {k for k,v in b.items() if torch.is_tensor(v)}
    for key in tens:
        assert a[key].dtype == b[key].dtype and torch.equal(a[key],b[key]), (file,key)
print(json.dumps(dict(status='passed',cases=[p.parent.name for p in files],exact_tensor_keys=sorted(tens),
                      comparison='identical prompt; all processor outputs compared; CPU; no model constructed'),indent=2))
