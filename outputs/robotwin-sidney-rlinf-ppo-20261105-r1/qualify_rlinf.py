"""Checkpoint-port replay, sampler replay/capacity and one real source episode."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import random


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    import numpy as np
    import torch
    from fluffy_bassoon.rl.sidney import install_rlinf_transforms,model_config,deterministic_reference
    import rlinf.models.embodiment.openpi as factory
    parser=argparse.ArgumentParser();parser.add_argument('--vectors',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count()==1
    torch.cuda.set_device(0);deterministic_reference();install_rlinf_transforms()
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)
    report=dict(status='starting',optimizer_updates=0,source_commit=os.environ['TASK_SOURCE'],
        chunk_size=50,native_flow_steps=10,ppo_flow_steps=5,rows=[])
    result=args.output/'RESULT.json';save(result,report)
    def env_obs(path,batch=1):
        with np.load(path,allow_pickle=False) as data:
            cameras=torch.from_numpy(data['images'].copy())
            return dict(main_images=cameras[0][None].expand(batch,-1,-1,-1).contiguous(),
                wrist_images=cameras[1:][None].expand(batch,-1,-1,-1,-1).contiguous(),
                states=torch.from_numpy(data['state'].copy())[None].expand(batch,-1).contiguous(),
                task_descriptions=[str(data['instruction'].item())]*batch)
    cfg=model_config(os.environ['REPO_PATH'],args.model)
    model=factory.get_model(cfg).cuda().eval().requires_grad_(False)
    def replay(reference,precision):
        rows=[]
        with torch.inference_mode():
            for source in sorted(args.vectors.glob('*.npz')):
                stem=source.stem+'-actions'
                with np.load(reference/(stem+'.npz'),allow_pickle=False) as data:
                    noise=torch.from_numpy(data['noise'].copy()).cuda()
                    expected=torch.from_numpy(data['actions'].copy())[None]
                    expected_norm=torch.from_numpy(data['normalized'].copy())
                obs=env_obs(source);prepared=model.env_obs_to_observation(obs)
                with np.load(reference/(stem+'.inputs.npz'),allow_pickle=False) as native:
                    assert np.array_equal(prepared.tokenized_prompt.cpu().numpy(),native['tokens'])
                    assert np.array_equal(prepared.tokenized_prompt_mask.cpu().numpy(),native['token_mask'])
                    state_error=float(np.abs(prepared.state.cpu().numpy()-native['state']).max())
                    pixels=np.stack([v.cpu().numpy() for v in prepared.images.values()],axis=1)
                    pixel_error=float(np.abs(pixels-native['images']).max())
                    assert state_error<=1e-6 and pixel_error<=1e-6,(state_error,pixel_error)
                started=time.monotonic()
                actions,info=model.predict_action_batch(obs,mode='eval',noise=noise)
                normalized=info['model_actions'][...,:14].float().cpu()
                decoded=float((actions.float().cpu()-expected).abs().max())
                norm_error=float((normalized-expected_norm).abs().max())
                assert torch.isfinite(actions).all()
                row=dict(vector=source.name,precision=precision,tokens_equal=True,
                    state_max_abs=state_error,pixels_max_abs=pixel_error,
                    normalized_max_abs=norm_error,decoded_max_abs=decoded,seconds=time.monotonic()-started)
                rows.append(row);print(json.dumps(row),flush=True)
        return rows
    try:
        mixed=replay(args.reference,'native mixed BF16/full FP32 vision')
        report['rows']=mixed;save(result,report)
        precise=all(r['decoded_max_abs']<=5e-4 and r['normalized_max_abs']<=1e-3 for r in mixed)
        if not precise:
            reference=args.output/'fp32-reference'
            subprocess.run([sys.executable,str(Path(__file__).with_name('native_replay.py')),
                '--vectors',str(args.vectors),'--output',str(reference),'--precision','float32'],check=True)
            factory._apply_openpi_param_dtypes(model,torch.float32)
            rows=replay(reference,'FP32 diagnostic only')
            report['fp32_rows']=rows;save(result,report)
            assert all(r['decoded_max_abs']<=5e-4 and r['normalized_max_abs']<=1e-3 for r in rows),rows
            factory._apply_openpi_param_dtypes(model,None)
        report.update(status='conversion_parity_passed',parity_basis='mixed BF16' if precise else 'FP32 reference; mixed BF16 drift reported')
        save(result,report)
        del model;gc.collect();torch.cuda.empty_cache()
        # Explicit VLRLA sampler migration. No SFT success-rate estimation.
        cfg=model_config(os.environ['REPO_PATH'],args.model,flow_steps=5,task='rl')
        cfg.openpi.noise_level=.3;cfg.openpi.detach_critic_input=True
        model=factory.get_model(cfg).cuda().eval()
        assert all(not p.requires_grad for n,p in model.named_parameters() if n.startswith(('img.','llm.embedder.')))
        assert any(p.requires_grad for n,p in model.named_parameters() if n.startswith('llm.'))
        assert model.freeze_vlm()==0
        source=sorted(args.vectors.glob('*.npz'))[-1]
        capacity=[]
        for batch in (32,16,8):
            torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            try:
                obs=env_obs(source,batch)
                with torch.no_grad():
                    actions,sampled=model.predict_action_batch(obs,mode='train',compute_values=True)
                    forward=model(forward_inputs=sampled['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=True,use_cache=False)
                errors={}
                for out,old in [('logprobs','prev_logprobs'),('values','prev_values')]:
                    before,after=sampled[old].float(),forward[out].float()
                    assert before.shape==after.shape and torch.isfinite(after).all()
                    errors[out]=float((after-before).abs().max())
                row=dict(batch=batch,seconds=time.monotonic()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),max_abs_error=errors)
                capacity.append(row);print(json.dumps(row),flush=True)
                assert errors['logprobs']<=1e-3 and errors['values']<=1e-4,errors
                break
            except torch.OutOfMemoryError:
                capacity.append(dict(batch=batch,out_of_memory=True));gc.collect();torch.cuda.empty_cache()
        else:raise RuntimeError('No qualified execution batch')
        report.update(status='sampler_replay_passed',capacity=capacity,execution_microbatch=batch)
        save(result,report)
        from fluffy_bassoon.rl.robotwin_noise import (
            environment_config,_install_allocated_renderer,_install_reset_retry_limit,
            _install_synchronous_vector_executor,
        )
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
        robotwin=Path(os.environ['ROBOTWIN_PATH']);os.chdir(robotwin)
        _install_allocated_renderer(args.output/'renderer');_install_reset_retry_limit()
        cfg=environment_config(Path(os.environ['REPO_PATH']),robotwin,'place_phone_stand',100100002,400,final_observation_only=False)
        cfg.task_config.camera.head_camera_type='Large_D435';cfg.task_config.camera.wrist_camera_type='Large_D435'
        env=RoboTwinEnv(cfg,1,seed_offset=0,total_num_processes=1,worker_info=None)
        _install_synchronous_vector_executor(env.venv)
        try:
            obs,_=env.reset(env_seeds=[100100002]);sub=env.venv.envs[0]
            assert sub.task.ep_num==100100002 and sub.episode_info_list==[sub.task.get_info()]
            episode=dict(task='place_phone_stand',seed=100100002,scene_info=sub.task.get_info(),
                instruction=obs['task_descriptions'],controls=0,chunks=[],flow_steps=5)
            while episode['controls']<400:
                actions,_=model.predict_action_batch(obs,mode='eval')
                raw,rewards,terms,truncs,infos=env.chunk_step(actions)
                info=infos[-1];count=int(info['valid_action_mask'].sum())
                assert count>0;episode['controls']+=count
                row=dict(controls=episode['controls'],success=bool(info['episode']['success_once'].any()))
                episode['chunks'].append(row);report['episode']=episode;save(result,report);print(json.dumps(row),flush=True)
                obs=raw[-1]
                if bool(terms.any()) or bool(truncs.any()):break
            episode['complete']=True
        finally:env.venv.close()
        report.update(status='complete',optimizer_updates=0)
        save(result,report)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save(result,report);raise


if __name__=='__main__':main()
