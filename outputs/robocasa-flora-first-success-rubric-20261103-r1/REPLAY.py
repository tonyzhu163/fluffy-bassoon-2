"""CPU-only replay of four retained action traces; qualify against saved states."""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

ROOT=Path('/mowenjie/fluffy-bassoon-2')
RUN='robocasa-flora-first-success-rubric-20261103-r1'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def main():
    assert platform.system()=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    import numpy as np
    import torch
    assert not torch.cuda.is_available()
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    import robosuite
    import robocasa
    from robocasa.utils.env_utils import create_env
    from robocasa.models.fixtures.cabinets import Drawer
    from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env,map_released_control_scores
    p=argparse.ArgumentParser();p.add_argument('--code',required=True,type=Path);a=p.parse_args()
    out=ROOT/'outputs'/RUN;dest=out/'physics-replay';dest.mkdir(exist_ok=False)
    with initialize_config_dir(config_dir=str(a.code/'configs/native_ppo'),version_base=None):cfg=compose(config_name='robocasa365_pi05_eval')
    c=cfg.env.eval
    state_cfg=OmegaConf.to_container(c.observation,resolve=True)
    extractor=SimpleNamespace(observation_cfg=state_cfg)
    native_make=robosuite.make
    def no_render(**kwargs):
        kwargs.update(has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
        return native_make(**kwargs)
    robosuite.make=no_render
    norm=read(ROOT/'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json')
    norm=norm.get('norm_stats',norm)['state'];mean=np.asarray(norm['mean'])[:16];std=np.asarray(norm['std'])[:16]
    def canonical(v,key=None):
        if isinstance(v,dict):return {k:canonical(x,k) for k,x in v.items()}
        if isinstance(v,list):return sorted(v) if key=='groups_containing_sampled_obj' else [canonical(x) for x in v]
        return v
    survey=read(out/'SOURCE-SURVEY.json');results=[]
    for item in survey['episodes']:
        start=time.time();eid=item['id'];trace=torch.load(out/(eid+'.pt'),map_location='cpu',weights_only=False)
        seed=trace['case']['native_seed'];saved=item['monitor']['native_ep_meta']
        env=create_env('OpenDrawer',robots=c.robot_name,camera_names=list(c.camera_names),
            camera_widths=c.init_params.camera_widths,camera_heights=c.init_params.camera_heights,
            seed=seed,split='target',render_onscreen=False,camera_depths=False,
            translucent_robot=bool(c.get('translucent_robot',False)),
            randomize_cameras=bool(c.get('randomize_cameras',False)),generative_textures=c.get('generative_textures',None))
        try:
            obs=env.reset();meta=json.loads(json.dumps(env.get_ep_meta()))
            meta_equal={k:canonical(meta.get(k))==canonical(v) for k,v in saved.items()}
            state_errors=[];states=[];controls=0
            def state_error(record):
                raw=Robocasa365Env._extract_state_vector(extractor,obs)
                state=raw[[*range(11,25),7,8]]
                actual=((state-mean)/(std+1e-6)).astype(np.float32)
                expected=np.asarray(record['features'])[:16]
                return float(np.abs(actual-expected).max())
            first_error=state_error(trace['records'][0])
            if not all(meta_equal.values()) or first_error>1e-4:
                row=dict(id=eid,qualified=False,reason='Initial scene/state mismatch; no policy controls replayed',
                    metadata_equal=meta_equal,first_state_max_error=first_error,controls=0,metadata=meta)
                write(dest/(eid+'.json'),row);results.append(row);print(json.dumps({k:row[k] for k in ('id','qualified','reason','first_state_max_error','controls')}),flush=True);continue
            drawers={name:f for name,f in env.fixtures.items() if isinstance(f,Drawer)}
            def snapshot():
                return dict(control=controls,target=env.drawer.name,
                    success=bool(env._check_success()),
                    normalized_opening={n:{k:float(v) for k,v in f.get_door_state(env).items()} for n,f in drawers.items()},
                    target_depth=float(env.drawer.size[1]))
            states.append(snapshot())
            for record in trace['records']:
                state_errors.append(state_error(record))
                for action in np.asarray(record['executed_chunk']):
                    physical=map_released_control_scores(action,OmegaConf.to_container(c.action_space,resolve=True))
                    obs,_,_,_=env.step(physical);controls+=1;states.append(snapshot())
            first=next((s['control'] for s in states if s['success']),None)
            final=states[-1]['success'];truth=item['monitor']['simulator_final_success']
            max_error=max(state_errors)
            qualified=all(meta_equal.values()) and max_error<1e-4 and final==truth and controls==750
            row=dict(id=eid,qualified=qualified,metadata_equal=meta_equal,first_state_max_error=first_error,
                chunk_state_max_errors=state_errors,max_state_error=max_error,controls=controls,
                first_success_control=first,first_success_seconds=None if first is None else first/20,
                final_success=final,retained_final_success=truth,states=states,seconds=time.time()-start,
                limitations='No saved full MuJoCo state; identity checked against full native metadata and all15 saved public16D states.')
            write(dest/(eid+'.json'),row);results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('states','metadata_equal','chunk_state_max_errors')}),flush=True)
        finally:env.close()
    summary=dict(completed=True,episodes=len(results),qualified=sum(r['qualified'] for r in results),
        replay_controls=sum(r['controls'] for r in results),newly_sampled_actions=0,motor_calls=0,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],results=[{k:v for k,v in r.items() if k!='states'} for r in results])
    write(dest/'SUMMARY.json',summary)
    export={p.name:read(p) for p in dest.glob('*.json')}
    Path('/root/workspace/drawer-first-success-physics.txt').write_text(base64.b64encode(gzip.compress(json.dumps(export).encode())).decode())
    print('REPLAY_EXPORT_READY',flush=True)
if __name__=='__main__':main()
