"""Compare fork/spawn native rendering and the unchanged loaded motor."""
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2')
CORE=ROOT/'worktrees/opendrawer-ad-dpt-20261102-r1'
OUT=Path(os.environ['TASK_PROBE_OUTPUT'])


def digest(value):
    import numpy as np
    array=np.asarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def captured_factory(fn,label):
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo_env import counted_native_factory
    original=counted_native_factory(fn,physics_tick_limit=(900+128)*128)
    def create():
        import numpy as np
        env=original();reset=env.reset
        def capture():
            result=reset()
            model=env.sim.model
            arrays={key:digest(getattr(model,key)) for key in
                ('tex_rgb','tex_data','mat_rgba','geom_rgba','geom_pos','geom_quat','body_pos','body_quat',
                 'light_pos','cam_pos','cam_quat','mesh_vert','mesh_normal','mesh_face','mesh_texcoord') if hasattr(model,key)}
            xml=model.get_xml();xml_sha=hashlib.sha256(xml.encode()).hexdigest()
            xml_path=OUT/(xml_sha+'.xml')
            if not xml_path.exists():xml_path.write_text(xml)
            images={key:np.asarray(value) for key,value in result.items() if key.endswith('_image')}
            np.savez_compressed(OUT/(label+'-images.npz'),**images)
            env._pairing_probe=dict(xml_sha256=xml_sha,state=digest(env.sim.get_state().flatten()),
                compiled=arrays,camera_position=digest(env.sim.data.cam_xpos),camera_matrix=digest(env.sim.data.cam_xmat),
                images={key:digest(value) for key,value in images.items()},counts=dict(env._flora_accounting))
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
    from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor
    from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import TASKS
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
    assert sys.platform=='linux' and torch.cuda.device_count()==1
    OUT.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B',str(CORE/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(OUT/'gpu-probe.json')],check=True)
    with initialize_config_dir(config_dir=str(CORE/'configs/native_ppo'),version_base=None):
        cfg=compose(config_name='robocasa365_pi05_eval')
    assembly=json.loads((ROOT/'outputs/opendrawer-ppo-ad-dpt-20261102-r1/data/assembly.json').read_text())
    initializer=assembly['contract']['initializer']
    cfg.rollout.model.model_path=initializer['model_directory']
    cfg.rollout.model.openpi_data.norm_stats_path=initializer['norm_path']
    from rlinf.models.embodiment.openpi import get_model
    model=get_model(cfg.rollout.model,torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    stats=json.loads(Path(initializer['norm_path']).read_text())
    motor=FrozenRoboCasaMotor(model,stats.get('norm_stats',stats),seed=991)
    journal=ActionJournal(OUT/'accounting',rank=0,world=1,attempt='loaded-motor-render-probe-r4')
    rows=[]
    for start_method in ('fork','spawn'):
        mp.set_start_method(start_method,force=True)
        for task,index in [('CloseFridge',0),('CloseFridge',4),('CloseFridge',6),('OpenDrawer',0)]:
            seed=(880210000 if task=='CloseFridge' else 880110000)+index
            noise_seed=int(np.random.default_rng((881310000 if task=='CloseFridge' else 881210000)+index).integers(0,2**63-1))
            for repeat in range(3):
                label=f'{start_method}-{task}-{index}-{repeat}'
                class ProbeNative(Robocasa365Env):
                    def _init_env(self):
                        self._refresh_task_context()
                        self.env=RobocasaSubprocEnv([captured_factory(fn,label) for fn in self.get_env_fns()])
                config=OmegaConf.create(OmegaConf.to_container(cfg.env.eval,resolve=True))
                config.seed=seed;config.seed_strategy='same'
                config.auto_reset=False;config.ignore_terminations=True;config.task_filter=[task]
                horizon,soup,mode,_=TASKS[task]
                config.task_soup=soup;config.task_mode=mode;config.split='target'
                config.max_episode_steps=config.max_steps_per_rollout_epoch=horizon
                intent=journal.begin(128,phase='loaded_motor_repeated_native_reset_qualification')
                env=ProbeNative(config,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
                try:
                    obs,_=env.reset();capture=env.env.get_env_attr('_pairing_probe')[0]
                    journal.finish(intent,capture['counts']['reset_controls'])
                    base,features,_=motor.infer(obs,noise_seed=noise_seed)
                    again,features_again,_=motor.infer(obs,noise_seed=noise_seed)
                    assert np.array_equal(base,again) and np.array_equal(features,features_again)
                    row=dict(task=task,index=index,seed=seed,start_method=start_method,repeat=repeat,capture=capture,
                        base=digest(base),features=digest(features),same_observation_motor_replay_exact=True,
                        charged_actions=journal.charged())
                    rows.append(row);(OUT/'resets.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(json.dumps(dict(task=task,index=index,start_method=start_method,repeat=repeat,
                        base=row['base'],features=row['features'])),flush=True)
                finally:env.close()
    (OUT/'RESULT.json').write_text(json.dumps(dict(completed=True,hostname=socket.gethostname(),reset_count=len(rows),
        policy_actions=0,motor_inferences=2*len(rows),charged_actions=journal.charged(),finished_unix=time.time()),indent=2)+'\n')


if __name__=='__main__':main()
