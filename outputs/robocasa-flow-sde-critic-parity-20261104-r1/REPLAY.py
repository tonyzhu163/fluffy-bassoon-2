"""Stage matching weights and measure actual saved-chain critic discrepancies."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def source_helpers():
    path=Path(__file__).resolve().parents[1]/'robocasa-flow-sde-10m-per-task-20261103-r1/OFFLINE-INITIAL-RATIO.py'
    spec=importlib.util.spec_from_file_location('critic_source_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def stage(args):
    data=json.loads(args.prepared.read_text());assert data['completed']
    args.output.mkdir(parents=True,exist_ok=False);base=source_helpers();receipts=[]
    for step in [0,1,10,120]:
        matches=[r['checkpoint'] for r in data['cohorts'] if r['policy_step']==step]
        assert matches and len({r['sha256'] for r in matches})==1
        checkpoint=matches[0];directory=args.output/f'{step:03d}';directory.mkdir()
        path,temporary,storage=base.obtain_weights(checkpoint,directory)
        ready=directory/'READY.json'
        write(ready,dict(completed=True,step=step,checkpoint=checkpoint,path=str(path),
            temporary=temporary,storage=storage,script_sha256=sha(__file__),
            prepared_sha256=sha(args.prepared)))
        consumed=args.result/f'CONSUMED-{step:03d}.json';deadline=time.monotonic()+5400
        while not consumed.exists():
            if time.monotonic()>deadline:raise TimeoutError('Consumer absent; private weights retained')
            if (args.result/'clock.json').exists():
                assert json.loads((args.result/'clock.json').read_text())['exit_code']==0,'Consumer failed; preserve private weights'
            time.sleep(5)
        receipt=json.loads(consumed.read_text())
        assert receipt['step']==step and receipt['model_sha256']==checkpoint['sha256']
        assert receipt['ready_sha256']==sha(ready) and receipt['strict_load_passed']
        if temporary:
            assert path.parent==directory and sha(path)==checkpoint['sha256'];path.unlink()
        receipts.append(dict(step=step,consumed_sha256=sha(consumed),private_copy_reclaimed=temporary))
        write(args.output/'PROGRESS.json',dict(completed=False,receipts=receipts))
    write(args.output/'PROGRESS.json',dict(completed=True,receipts=receipts))


def run(args):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.sampling import rl_sampler
    from fluffy_bassoon.rl.rlinf_checkpoint_compat import (
        _convert_official_openpi_keys,_normalize_state_dict,as_state_dict)
    from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import (
        flow_logprob_parity,keep_frozen_image_encoder_in_eval)
    import rlinf.algorithms.losses
    from rlinf.algorithms.registry import policy_loss

    data=json.loads(args.prepared.read_text());assert data['completed'] and len(data['cohorts'])==16
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'tensors').mkdir()
    cfg_path=Path(data['cohorts'][0]['config']);assert sha(cfg_path)==data['cohorts'][0]['config_sha256']
    cfg=OmegaConf.create(json.loads(cfg_path.read_text()))
    assert not cfg.actor.model.openpi.value_after_vlm and cfg.actor.model.openpi.detach_critic_input
    model=get_model(cfg.actor.model,torch_dtype=torch.bfloat16).to('cuda:0')
    assert hasattr(model,'default_forward') and model.rl_cfg.joint_logprob is False
    model.train();keep_frozen_image_encoder_in_eval(model)
    rows=[]

    def gradient_profile(features,reference,fixture,new_logprobs):
        parameters=list(model.value_head.parameters())
        values=model.value_head(features)[:,0].float()
        loss,metrics=policy_loss(task_type='embodied',loss_type='actor_critic',
            reward_type='chunk_level',logprob_type='chunk_level',single_action_dim=12,
            logprobs=new_logprobs.detach(),old_logprobs=fixture['old_logprobs'].to('cuda:0'),
            advantages=torch.zeros((16,1),device='cuda:0'),values=values,
            prev_values=reference,returns=fixture['returns'].to('cuda:0'),
            loss_mask=fixture['loss_mask'].to('cuda:0'),
            loss_mask_sum=None if fixture['loss_mask_sum'] is None else fixture['loss_mask_sum'].to('cuda:0'),
            max_episode_steps=600,clip_ratio_low=.2,clip_ratio_high=.2,value_clip=.2,huber_delta=10.)
        gradients=torch.autograd.grad(loss,[values,*parameters])
        return dict(loss=float(metrics['critic/value_loss']),clip_ratio=float(metrics['critic/value_clip_ratio']),
            prediction_gradient=gradients[0].detach(),head_gradient=torch.cat([x.detach().reshape(-1) for x in gradients[1:]]))

    for step in [0,1,10,120]:
        ready_path=args.staging/f'{step:03d}'/'READY.json'
        deadline=time.monotonic()+2400
        while not ready_path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Verified model staging absent')
            time.sleep(5)
        ready=json.loads(ready_path.read_text());checkpoint=ready['checkpoint']
        assert ready['step']==step and ready['prepared_sha256']==sha(args.prepared)
        assert ready['script_sha256']==sha(__file__) and sha(checkpoint['marker'])==checkpoint['marker_sha256']
        path=Path(ready['path']);assert sha(path)==checkpoint['sha256']
        state=_convert_official_openpi_keys(_normalize_state_dict(as_state_dict(
            torch.load(path,map_location='cpu',weights_only=False))),path)
        model.load_state_dict(state,strict=True);del state;gc.collect()
        model.set_global_step(step)
        write(args.output/f'CONSUMED-{step:03d}.json',dict(step=step,model_sha256=checkpoint['sha256'],
            ready_sha256=sha(ready_path),strict_load_passed=True))
        for cohort in [r for r in data['cohorts'] if r['policy_step']==step]:
            p=Path(cohort['fixture']);assert sha(p)==cohort['fixture_sha256']
            fixture=torch.load(p,map_location='cpu',weights_only=False)
            fi={k:v.to('cuda:0') for k,v in fixture['forward_inputs'].items()}
            selected=fi['denoise_inds'][:,0].long();assert selected.shape==(16,)
            assert len(torch.unique(selected))==1
            rng_before=torch.cuda.get_rng_state().clone()
            captured=[]
            hook=model.value_head.register_forward_pre_hook(lambda m,inputs:captured.append(inputs[0].detach()))
            with torch.no_grad():
                native=model.default_forward(fi,compute_values=True)
                parity=flow_logprob_parity(native['logprobs'].cpu(),fixture['old_logprobs'])
                assert parity['passed'],parity
                assert len(captured)==1;selected_features=captured[0];captured.clear()
                observation=model._observation_from_forward_inputs(fi)
                _,prefix_mask,cache=model._build_prefix_cache_for_actor(observation)
                timesteps=rl_sampler.get_timesteps(model.num_steps,model.device)
                values=[]
                for index in range(model.num_steps):
                    t=timesteps[index].expand(16).float()
                    suffix=model.run_suffix(observation,fi['chains'][:,index],t,cache,prefix_mask)
                    values.append(rl_sampler.value_from_suffix(model.value_head,suffix,
                        action_chunk=model.action_chunk,chunk_critic_input=model.rl_cfg.chunk_critic_input,
                        detach_critic_input=True))
                all_values=torch.stack(values,dim=1)
                mean=all_values.mean(1);picked=all_values[torch.arange(16,device='cuda:0'),selected]
                old=fixture['old_values'].to('cuda:0').flatten()
                mean_error=float((mean-old).abs().max())
                selected_error=float((picked-native['values'].flatten()).abs().max())
                assert mean_error<=1e-4,(step,'Saved rollout mean not reproduced',mean_error)
                assert selected_error<=1e-5,(step,'Native selected value not reproduced',selected_error)
            hook.remove()
            original=gradient_profile(selected_features,old[:,None],fixture,native['logprobs'])
            consistent=gradient_profile(selected_features,picked.detach()[:,None],fixture,native['logprobs'])
            assert torch.equal(rng_before,torch.cuda.get_rng_state()),'Replay consumed sampler RNG'
            gap=picked-old;delta=original['head_gradient']-consistent['head_gradient']
            normal=consistent['head_gradient'].norm();original_norm=original['head_gradient'].norm()
            label=f'policy-{step:03d}-actor-{cohort["actor_rank"]:02d}-chunk-{cohort["time_index"]:02d}'
            tensor_path=args.output/'tensors'/f'{label}.pt'
            torch.save(dict(all_suffix_values=all_values.cpu(),saved_old_values=old.cpu(),
                selected_values=picked.cpu(),returns=fixture['returns'],loss_mask=fixture['loss_mask'],
                native_prediction_gradient=original['prediction_gradient'].cpu(),
                consistent_prediction_gradient=consistent['prediction_gradient'].cpu()),tensor_path)
            row=dict(policy_step=step,actor_rank=cohort['actor_rank'],time_index=cohort['time_index'],
                denoise_index=int(selected[0]),samples=16,fixture_sha256=cohort['fixture_sha256'],
                model_sha256=checkpoint['sha256'],saved_mean_max_abs_error=mean_error,
                native_selected_max_abs_error=selected_error,logprob_parity=parity,
                value_gap_max_abs=float(gap.abs().max()),value_gap_mean_abs=float(gap.abs().mean()),
                value_gap_rms=float(gap.square().mean().sqrt()),
                initial_value_clip_fraction=float((gap.abs()>.2).float().mean()),
                native_critic_loss=original['loss'],consistent_reference_critic_loss=consistent['loss'],
                native_head_gradient_norm=float(original_norm),consistent_head_gradient_norm=float(normal),
                head_gradient_delta_norm=float(delta.norm()),
                head_gradient_relative_delta=float(delta.norm()/normal) if float(normal)>1e-12 else None,
                head_gradient_cosine=float(torch.dot(original['head_gradient'],consistent['head_gradient'])/(normal*original_norm))
                    if float(normal*original_norm)>1e-12 else None,
                suppressed_prediction_gradient_fraction=float(((original['prediction_gradient']==0)&
                    (consistent['prediction_gradient']!=0)).float().mean()),
                tensors=str(tensor_path),tensors_sha256=sha(tensor_path),sampler_rng_unchanged=True)
            rows.append(row);print(json.dumps(row),flush=True)
            write(args.output/'RESULT.json',dict(completed=False,rows=rows))
            del fi,fixture,captured,all_values,original,consistent;gc.collect()
    write(args.output/'RESULT.json',dict(completed=True,rows=rows,cohorts=16,decisions=256,
        prepared_sha256=sha(args.prepared),script_sha256=sha(__file__),new_controls=0,SGD=0,
        scope='Matching frozen production checkpoints; gradient comparison holds historical return targets fixed and isolates clipping-reference inconsistency, not counterfactual GAE or learning.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('stage');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--staging',type=Path,required=True)
    args=parser.parse_args();stage(args) if args.command=='stage' else run(args)
