"""Allocated-GPU diagnosis of repeated native reset identity; no policy actions."""
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
CORE = ROOT/'worktrees/opendrawer-ad-dpt-20261102-r1'
OUT = Path(os.environ.get('TASK_PROBE_OUTPUT',str(ROOT/'outputs/opendrawer-reset-pairing-20261102-r1')))


def digest(value):
    import numpy as np
    if isinstance(value, dict):
        return {key:digest(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [digest(item) for item in value]
    if isinstance(value, (str,int,float,bool)) or value is None:
        return value
    array=np.asarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def captured_factory(fn,seed,seed_globals):
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo_env import counted_native_factory
    original=counted_native_factory(fn,physics_tick_limit=(900+128)*128)
    def create():
        import numpy as np
        if seed_globals:
            random.seed(seed);np.random.seed(seed)
        env=original();reset=env.reset
        def capture():
            result=reset()
            env._pairing_probe=dict(xml_sha256=hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest(),
                state=digest(env.sim.get_state().flatten()),raw=digest(result),counts=dict(env._flora_accounting))
            return result
        env.reset=capture
        return env
    return create


def main():
    import numpy as np
    import torch
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env
    from rlinf.envs.sim.robocasa.venv import RobocasaSubprocEnv
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    assert sys.platform=='linux' and torch.cuda.device_count()==1
    OUT.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B',str(CORE/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(OUT/'gpu-probe.json')],check=True)
    with initialize_config_dir(config_dir=str(CORE/'configs/native_ppo'),version_base=None):
        cfg=compose(config_name='robocasa365_pi05_eval')
    journal=ActionJournal(OUT/'accounting',rank=0,world=1,attempt='reset-probe-r1')
    rows=[]
    for seeded in (False,True):
        for task,index in [('CloseFridge',0),('CloseFridge',4),('CloseFridge',6),('OpenDrawer',0)]:
            seed=(880210000 if task=='CloseFridge' else 880110000)+index
            for repeat in range(3):
                class ProbeNative(Robocasa365Env):
                    def _init_env(self):
                        self._refresh_task_context()
                        self.env=RobocasaSubprocEnv([captured_factory(fn,seed,seeded) for fn in self.get_env_fns()])
                config=OmegaConf.create(OmegaConf.to_container(cfg.env.eval,resolve=True))
                config.seed=seed;config.seed_strategy='same'
                config.auto_reset=False;config.ignore_terminations=True;config.task_filter=[task]
                horizon,soup,mode,_=TASKS[task]
                config.task_soup=soup;config.task_mode=mode;config.split='target'
                config.max_episode_steps=config.max_steps_per_rollout_epoch=horizon
                if seeded:
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                intent=journal.begin(128,phase='repeated_native_reset_qualification')
                env=ProbeNative(config,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
                try:
                    obs,_=env.reset();capture=env.env.get_env_attr('_pairing_probe')[0]
                    journal.finish(intent,capture['counts']['reset_controls'])
                    row=dict(task=task,index=index,seed=seed,seed_globals=seeded,repeat=repeat,
                        capture=capture,public=digest(obs),charged_actions=journal.charged())
                    rows.append(row)
                    (OUT/'resets.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps(dict(task=task,index=index,seed_globals=seeded,repeat=repeat,
                        xml=capture['xml_sha256'],state=capture['state'],public=digest(obs))),flush=True)
                finally:env.close()
    result=dict(completed=True,hostname=socket.gethostname(),reset_count=len(rows),
        policy_actions=0,charged_actions=journal.charged(),finished_unix=time.time())
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
