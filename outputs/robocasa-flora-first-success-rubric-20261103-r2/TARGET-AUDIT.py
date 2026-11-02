"""CPU scene-reset audit of the geometric uppermost-target wording; no policy steps."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

assert platform.system() == 'Linux'
root = Path('/mowenjie/fluffy-bassoon-2')
out = root/'outputs/robocasa-flora-first-success-rubric-20261103-r2'
old = root/'outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1'
new = root/'outputs/robocasa-flora-first-success-rubric-20261103-r1'
profile_source = subprocess.check_output(['git', '-C', str(root/'repo'), 'show',
    '17755d7a2afecf8eccae00e301933f9bc216a726:outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1/REPLAY-SOURCE.json'])
profile = json.loads(profile_source)['native_profile']
os.environ.update(profile)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = profile['PYTHONPATH'].split(':')
import numpy as np
import torch
assert not torch.cuda.is_available()
from hydra import compose, initialize_config_dir
import robosuite
import robocasa
from robocasa.utils.env_utils import create_env
from robocasa.models.fixtures.cabinets import Drawer

code = root/'worktrees/drawer-physics-51b254c2'
with initialize_config_dir(config_dir=str(code/'configs/native_ppo'), version_base=None):
    cfg = compose(config_name='robocasa365_pi05_eval')
c = cfg.env.eval
native_make = robosuite.make
def no_render(**kwargs):
    kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    return native_make(**kwargs)
robosuite.make = no_render
def canonical(value, key=None):
    if isinstance(value, dict):
        return {k: canonical(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return sorted(value) if key == 'groups_containing_sampled_obj' else [canonical(v) for v in value]
    return value
episodes = json.loads((old/'SOURCE-SURVEY.json').read_text())['episodes']
episodes += json.loads((new/'SOURCE-SURVEY.json').read_text())['episodes']
selected = json.loads((out/'PUBLIC-JOBS.json').read_text())
wanted = {item['id'].split('/')[-2] for item in selected}
byid = {item['id']: item for item in episodes if item['id'] in wanted}
assert len(byid) == 6
rows = []
for eid, item in byid.items():
    saved = item['monitor']['native_ep_meta']
    env = create_env('OpenDrawer', robots=c.robot_name, camera_names=list(c.camera_names),
        camera_widths=c.init_params.camera_widths, camera_heights=c.init_params.camera_heights,
        seed=item['monitor']['case']['native_seed'], split='target', render_onscreen=False,
        camera_depths=False, translucent_robot=bool(c.get('translucent_robot', False)),
        randomize_cameras=bool(c.get('randomize_cameras', False)),
        generative_textures=c.get('generative_textures', None))
    try:
        env.reset()
        meta = json.loads(json.dumps(env.get_ep_meta()))
        equal = {k: canonical(meta.get(k)) == canonical(v) for k, v in saved.items()}
        assert all(equal.values()), (eid, equal)
        target = env.drawer
        drawers = [f for f in env.fixtures.values() if isinstance(f, Drawer)]
        stack = [f for f in drawers if np.linalg.norm(np.asarray(f.pos)[:2]-np.asarray(target.pos)[:2]) < 1e-4]
        rows.append(dict(id=eid, target=target.name, target_pos=np.asarray(target.pos).tolist(),
            native_instruction=meta['lang'], metadata_equal=equal,
            same_stack=[dict(name=f.name, pos=np.asarray(f.pos).tolist(),
                meets_top_drawer_height=bool(.7 <= f.pos[2] <= .9)) for f in stack],
            drawers_above_target=[f.name for f in stack if f.pos[2] > target.pos[2]+1e-5]))
        print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
result = dict(completed=True, episodes=rows, policy_controls=0, motor_calls=0,
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
    limitations='Scene-reset geometry audit only; full retained-action replay qualification is recorded separately.')
with (out/'TARGET-AUDIT.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
