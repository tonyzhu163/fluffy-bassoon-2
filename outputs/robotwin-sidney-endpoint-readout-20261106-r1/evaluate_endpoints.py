"""Frozen SFT/PPO40/GRPO40 comparison under the initial RLinf protocol."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

BASE=Path('/mowenjie/fluffy-bassoon-2')
RUN=BASE/'outputs/robotwin-sidney-endpoint-readout-20261106-r1'
INITIAL=BASE/'outputs/robotwin-sidney-initial-performance-20261106-r1/eval01'
CONVERTED=BASE/'cache/robotwin-sidney-rlinf-20261105-r1'
MODES=('sft','ppo40','grpo40')


def save(path,value):
    tmp=path.with_name(path.name+'.writing')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def exposed(task):
    return [dict(qpos=a.get_qpos().tolist(),qvel=a.get_qvel().tolist(),root_p=a.get_root_pose().p.tolist(),
        root_q=a.get_root_pose().q.tolist(),root_linear_velocity=a.get_root_linear_velocity().tolist(),
        root_angular_velocity=a.get_root_angular_velocity().tolist()) for a in task.scene.get_all_articulations()]


def main():
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from fluffy_bassoon.rl.sidney import deterministic_reference,install_rlinf_transforms,model_config
    from fluffy_bassoon.rl.robotwin_noise import (environment_config,_install_allocated_renderer,
        _install_reset_retry_limit,_install_synchronous_vector_executor,_observation_hash)
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import snapshot_rng,restore_rng
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    import rlinf.models.embodiment.openpi as factory
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--exports',type=Path,required=True)
    p.add_argument('--storage-contract',type=Path);args=p.parse_args()
    assert sys.platform=='linux' and os.uname().nodename.startswith('job-') and torch.cuda.device_count()==1
    torch.cuda.set_device(0);deterministic_reference();install_rlinf_transforms()
    root=args.output;assert root.is_dir() and not (root/'RESULT.json').exists()
    assert json.loads((args.exports/'RESULT.json').read_text())['plain_inference_weights_qualified']
    assert json.loads((args.exports/'EXIT.json').read_text())['exit_code']==0
    manifest=json.loads(args.manifest.read_text());cases=manifest['cases']
    bank=Path(manifest['bank_path']);assert sha(bank)==manifest['bank_sha256']
    assert len(cases)==16 and cases==json.loads(bank.read_text())['place_phone_stand']['success_seeds'][:16]
    assert sha(CONVERTED/'model.safetensors')=='c71114641cf0819900f8b1d7b0568158a12e631b2ca08b2b19bc2d5b761b69b3'
    assert sha(CONVERTED/'sidney-processors.json')=='38ffadf70cf85cff5f8a0a4a6bf23e4886b33081b93f8bb2b20469b6cd571f5b'
    rlinf=Path(os.environ['REPO_PATH']);robotwin=Path(os.environ['ROBOTWIN_PATH'])
    versions={}
    for folder,expected in ((rlinf,'c605b9d55ed93a180119b099e6f836abff293b08'),(robotwin,'dca9ec682688821c42944eacd7f5dd9bfb15396f')):
        actual=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip();assert actual==expected
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain','--untracked-files=no'],text=True).strip()
        versions[str(folder)]=actual
    report=dict(status='starting',source_commit=os.environ['TASK_SOURCE'],runtime_commits=versions,
        modes=MODES,cases=cases,episodes=[],optimizer_updates=0,horizon=400,chunk=50,flow_steps=5,
        noise=.3,batch_size=1,controller='released fast gen_sparse_reward_data',prompt='generated seen',
        requested_controls=0,measured_delivered_controls=None,physical_duration_measured=False,
        final_trained_policies=True,qualified_exports=str(args.exports),started_unix=time.time(),model_calibrations={})
    save(root/'RESULT.json',report);models={};envs={};param_versions={};requested=0

    # Default retains the earlier operational guard. The explicit migration
    # closes completed diagnostic reservations; acting/eval science is unchanged.
    storage=None if args.storage_contract is None else json.loads(args.storage_contract.read_text())
    def required_free(used):
        floor=310 if storage is None else storage['other_and_floor_gib']+storage['concurrent_execution_qualification_cap_gib']
        return floor*2**30+max(0,12*2**30-used)

    def deadline():assert time.time()<float(os.environ['TASK_DEADLINE'])-90,'Allocated readout deadline'

    def rgb(raw):return np.stack([raw['main_images'][0].cpu().numpy(),
        raw['wrist_images'][0,0].cpu().numpy(),raw['wrist_images'][0,1].cpu().numpy()])

    def sample(model,raw,seed):
        state=snapshot_rng()
        try:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                noise=torch.randn((1,50,32),device='cuda',dtype=torch.float32)
                actions,detail=model.predict_action_batch(raw,mode='train',compute_values=False,noise=noise)
            return actions.float().cpu().numpy(),detail['prev_logprobs'].float().cpu().numpy(),detail['forward_inputs']['denoise_inds'].cpu().numpy()
        finally:restore_rng(state)

    try:
        with np.load(INITIAL/'case-00/sde5_fast/chunk-00-observation.npz',allow_pickle=False) as data:
            images=data['images'].copy();calib_raw=dict(main_images=torch.from_numpy(images[0:1]),
                wrist_images=torch.from_numpy(images[None,1:]),states=torch.from_numpy(data['state'].copy()[None]),
                task_descriptions=[str(data['instruction'].item())])
        for mode in MODES:
            deadline()
            if mode=='sft':
                cfg=model_config(rlinf,CONVERTED,flow_steps=5,task='rl')
                cfg.add_value_head=False;cfg.openpi.add_value_head=False
                cfg.openpi.noise_method='flow_sde';cfg.openpi.noise_level=.3;cfg.openpi.train_expert_only=True
            else:
                arm=args.exports/('ppo' if mode=='ppo40' else 'grpo')
                cfg=OmegaConf.create(json.loads((arm/'resolved-native-config.json').read_text())['actor']['model'])
                receipt=json.loads((arm/'EXPORT.json').read_text());assert receipt['status']=='qualified'
                assert sha(arm/'policy.pt')==receipt['sha256']
            model=factory.get_model(cfg).cuda()
            if mode!='sft':
                weights=torch.load(arm/'policy.pt',map_location='cpu',weights_only=True)
                current=model.state_dict();assert current.keys()==weights.keys()
                assert all(current[k].shape==v.shape and current[k].dtype==v.dtype for k,v in weights.items())
                loaded=model.load_state_dict(weights,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
                del weights,current;gc.collect()
            model.eval().requires_grad_(False);models[mode]=model
            param_versions[mode]={n:v._version for n,v in model.named_parameters()}
            actions,logprobs,indices=sample(model,calib_raw,1000)
            assert actions.shape==(1,50,14) and np.isfinite(actions).all() and np.isfinite(logprobs).all()
            if mode=='sft':
                reference=np.load(INITIAL/'case-00/sde5_fast/chunk-00-actions.npy',allow_pickle=False)
                np.testing.assert_allclose(actions,reference,atol=1e-5,rtol=1e-5)
                calibration=dict(actions_max_abs=float(np.max(np.abs(actions-reference))),original_SFT_reference=True)
            else:
                with np.load(arm/'calibration.npz',allow_pickle=False) as reference:
                    np.testing.assert_allclose(actions,reference['actions'],atol=1e-5,rtol=1e-5)
                    np.testing.assert_allclose(logprobs,reference['logprobs'],atol=1e-3,rtol=1e-5)
                    assert np.array_equal(indices,reference['denoise_indices'])
                    calibration=dict(actions_max_abs=float(np.max(np.abs(actions-reference['actions']))),
                        logprobs_max_abs=float(np.max(np.abs(logprobs-reference['logprobs']))),strict_load=True,
                        export_sha256=receipt['sha256'],source_checkpoint=json.loads((arm/'RESULT.json').read_text())['source_checkpoint'])
            report['model_calibrations'][mode]=calibration;save(root/'RESULT.json',report)
        _install_allocated_renderer(root/'renderers');_install_reset_retry_limit();os.chdir(robotwin)
        for mode in MODES:
            cfg=environment_config(rlinf,robotwin,'place_phone_stand',int(cases[0]),400,final_observation_only=False)
            cfg.skip_intermediate_renders=True;cfg.malloc_trim_reset_interval=1
            cfg.task_config.camera.head_camera_type='Large_D435';cfg.task_config.camera.wrist_camera_type='Large_D435'
            envs[mode]=RoboTwinEnv(cfg,1,seed_offset=0,total_num_processes=1,worker_info=None)
            _install_synchronous_vector_executor(envs[mode].venv)
        report['models_loaded']=True;save(root/'RESULT.json',report)
        for ci,seed in enumerate(cases):
            used=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file())
            assert used<=12*2**30 and shutil.disk_usage(root).free>=required_free(used)
            order=MODES[ci%3:]+MODES[:ci%3];paired=None
            for mode in order:
                deadline();directory=root/f'case-{ci:02d}'/mode;directory.mkdir(parents=True)
                env=envs[mode];raw,_=env.reset(env_seeds=[int(seed)]);sub=env.venv.envs[0];task=sub.task
                assert int(task.ep_num)==int(sub.env_seed)==int(seed) and sub.episode_info_list==[task.get_info()]
                assert raw['task_descriptions']==[sub.instruction]==[task.get_instruction()]
                identity=dict(seed=int(seed),scene=task.get_info(),instruction=raw['task_descriptions'],
                    public_sha256=_observation_hash(raw),exposed_state=exposed(task))
                save(directory/'initial.json',identity)
                if paired is None:paired=identity
                assert identity==paired,'Initial scene/instruction/public state mismatch'
                np.savez_compressed(directory/'initial.npz',images=rgb(raw),state=raw['states'][0].cpu().numpy(),instruction=np.asarray(raw['task_descriptions'][0]))
                Image.fromarray(np.concatenate(rgb(raw),axis=1)).save(directory/'initial.jpg')
                episode=dict(case_index=ci,seed=int(seed),mode=mode,initial=identity,chunks=[],complete=False,
                    success=False,requested_controls=0,measured_executed_controls=None)
                report['episodes'].append(episode)
                for chunk in range(8):
                    deadline();noise_seed=1000+ci*100+chunk
                    np.savez_compressed(directory/f'chunk-{chunk:02d}-observation.npz',images=rgb(raw),
                        state=raw['states'][0].cpu().numpy(),instruction=np.asarray(raw['task_descriptions'][0]))
                    started=time.monotonic();actions,logprobs,indices=sample(models[mode],raw,noise_seed)
                    assert actions.shape==(1,50,14) and np.isfinite(actions).all() and np.isfinite(logprobs).all()
                    np.save(directory/f'chunk-{chunk:02d}-actions.npy',actions)
                    inference_seconds=time.monotonic()-started;requested+=50;assert requested<=19200
                    observations,rewards,terms,truncs,infos=env.chunk_step(torch.from_numpy(actions.copy()))
                    success=bool(torch.as_tensor(infos[-1]['episode']['success_once']).any())
                    terminated=bool(torch.as_tensor(terms).any());truncated=bool(torch.as_tensor(truncs).any())
                    raw=observations[-1];episode['success']|=success;episode['requested_controls']+=50
                    episode['chunks'].append(dict(chunk=chunk,noise_seed=noise_seed,inference_seconds=inference_seconds,
                        success=success,terminated=terminated,truncated=truncated,reward=float(torch.as_tensor(rewards).sum()),
                        sampling=dict(denoise_indices=indices.tolist()),public_state_after=raw['states'][0].cpu().tolist(),
                        exposed_state_after=exposed(task),measured_executed_controls=None))
                    report.update(status='running',requested_controls=requested)
                    save(directory/'episode.json',episode);save(root/'RESULT.json',report)
                    print(json.dumps(dict(case=ci,mode=mode,chunk=chunk,success=success)),flush=True)
                    if terminated or truncated:break
                episode['complete']=True
                Image.fromarray(np.concatenate(rgb(raw),axis=1)).save(directory/'final.jpg')
                save(directory/'episode.json',episode);save(root/'RESULT.json',report)
            gc.collect()
        assert len(report['episodes'])==48 and all(e['complete'] for e in report['episodes'])
        for mode,model in models.items():assert param_versions[mode]=={n:p._version for n,p in model.named_parameters()}
        report.update(status='complete',completed_episodes=48,frozen_parameter_versions_unchanged=True,
            summaries={m:dict(episodes=16,successes=sum(e['success'] for e in report['episodes'] if e['mode']==m)) for m in MODES})
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        for env in envs.values():env.venv.close()
        report['ended_unix']=time.time();save(root/'RESULT.json',report)


if __name__=='__main__':main()
