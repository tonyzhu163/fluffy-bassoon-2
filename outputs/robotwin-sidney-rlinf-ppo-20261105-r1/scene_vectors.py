"""Capture three source observations and audit language against actual objects."""
import argparse
import json
import os
from pathlib import Path


def main():
    import numpy as np
    from PIL import Image
    from fluffy_bassoon.rl.robotwin_noise import (
        environment_config,_install_allocated_renderer,_install_reset_retry_limit,
        _install_synchronous_vector_executor,
    )
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert os.uname().nodename.startswith('job-')
    rlinf=Path(os.environ['REPO_PATH']);robotwin=Path(os.environ['ROBOTWIN_PATH'])
    os.chdir(robotwin)
    _install_allocated_renderer(args.output/'renderer');_install_reset_retry_limit()
    rows=[]
    for task,seeds in [('place_a2b_left',[100100001,100100004]),('place_phone_stand',[100100002])]:
        cfg=environment_config(rlinf,robotwin,task,seeds[0],400,final_observation_only=False)
        cfg.task_config.camera.head_camera_type='Large_D435'
        cfg.task_config.camera.wrist_camera_type='Large_D435'
        env=RoboTwinEnv(cfg,1,seed_offset=0,total_num_processes=1,worker_info=None)
        _install_synchronous_vector_executor(env.venv)
        try:
            for seed in seeds:
                obs,_=env.reset(env_seeds=[seed]);sub=env.venv.envs[0]
                actual=sub.task.get_info()
                assert sub.task.ep_num==seed
                assert sub.episode_info_list==[actual]
                assert sub.task.get_instruction()==sub.instruction==obs['task_descriptions'][0]
                images=np.stack([obs['main_images'][0].cpu().numpy(),
                    obs['wrist_images'][0,0].cpu().numpy(),obs['wrist_images'][0,1].cpu().numpy()])
                assert images.shape==(3,480,640,3) and all(v.std()>1 for v in images)
                stem=f'{len(rows):02d}-{task}-{seed}'
                np.savez_compressed(args.output/(stem+'.npz'),images=images,
                    state=obs['states'][0].cpu().numpy().astype(np.float32),
                    instruction=np.asarray(obs['task_descriptions'][0]))
                Image.fromarray(np.concatenate(images,axis=1)).save(args.output/(stem+'.png'))
                row=dict(file=stem+'.npz',task=task,seed=seed,actual_scene_info=actual,
                    instruction=obs['task_descriptions'][0],metadata_matches=True,physical_controls=0)
                rows.append(row);print(json.dumps(row),flush=True)
        finally:
            env.venv.close()
    (args.output/'SCENES.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
