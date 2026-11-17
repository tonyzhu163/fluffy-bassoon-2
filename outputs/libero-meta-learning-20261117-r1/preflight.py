"""CPU import and input-contract checks; does not qualify GPU or simulation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from job import CODE, PRO, SETUP, environment


def main():
    assert sys.platform == 'linux'
    os.environ.update(environment())
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = os.environ['PYTHONPATH'].split(':')
    import numpy as np
    import torch
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.models.embodiment.openpi import get_model
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig
    from native import public

    expected = dict(robosuite='1.4.0',mujoco='3.2.3',numpy='1.26.4')
    actual = {name:importlib.metadata.version(name) for name in expected}
    assert actual == expected, actual
    setup = json.loads((SETUP/'preparation.json').read_text())
    for row in setup['python_sources']:
        path = Path(setup['simulator_package'])/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    from libero.libero.envs import nonphysical_perturbations
    assert Path(nonphysical_perturbations.__file__).resolve().is_relative_to(SETUP)
    manifest = TaskManifest.load(str(PRO/'assets/task_manifest.json'))
    splits = json.loads((CODE/'outputs/libero-meta-learning-20261117-r1/SPLITS.json').read_text())
    assert manifest.sha256 == splits['manifest_sha256']
    for case in splits['qualification']:
        for variant in case['variants']:
            manifest.require(case['suite'],case['task_id'],variant)
    # Equal-size images isolate the official 180-degree orientation contract.
    image = np.arange(224*224*3,dtype=np.uint8).reshape(224,224,3)
    raw = dict(agentview_image=image,robot0_eye_in_hand_image=image,
        robot0_eef_pos=np.array([.1,.2,.3]),robot0_eef_quat_xyzw=np.array([0.,0.,0.,1.]),
        robot0_gripper_qpos=np.array([.02,-.02]))
    item = public(raw,'qualification')
    assert np.array_equal(item['main_images'][0],image[::-1,::-1])
    assert np.array_equal(item['wrist_images'][0],image[::-1,::-1])
    assert np.allclose(item['states'][0],[.1,.2,.3,0,0,0,.02,-.02])
    cfg = FlowBridgeConfig(observation_dim=2056,history_action_dim=320,width=256,
        layers=6,heads=8,context_length=160,kv_heads=1,head_dim=256,kv_rank=4,
        kv_layers=tuple(range(18)),scale=1.,action_horizon=10,
        model_action_dim=32,active_action_dim=7)
    result = dict(versions=actual,torch=torch.__version__,manifest=manifest.sha256,
        image_orientation_and_state=True,source_copy_hashes=True,
        runtime_imports=True,model_loaded=False,gpu_qualified=False,
        simulation_episode_run=False,history_action_dim=cfg.history_action_dim)
    (SETUP/'cpu-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
