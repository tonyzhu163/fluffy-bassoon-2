"""Replay an actual64-scene rollout in smaller actor execution batches."""
import gzip
import json
import os
from pathlib import Path
import time


def main():
    import torch
    from fluffy_bassoon.rl.sidney import deterministic_reference,install_rlinf_transforms,model_config
    from rlinf.models.embodiment.openpi import get_model
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import write
    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count()==1
    torch.cuda.set_device(0);deterministic_reference();install_rlinf_transforms()
    root=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-rlinf-ppo-20261105-r1/train03')
    initial=torch.load(root/'initial-states/source-rank-00-0000.pt',map_location='cpu',weights_only=False)
    with gzip.open(root/'sampling/rank-00/00000.pt.gz','rb') as handle:
        sampled=torch.load(handle,map_location='cpu',weights_only=False)
    assert sampled['producing_policy_iteration']==0 and sampled['sampling_record']=='full'
    cfg=model_config(os.environ['REPO_PATH'],Path('/mowenjie/fluffy-bassoon-2/cache/robotwin-sidney-rlinf-20261105-r1'),flow_steps=5,task='rl')
    cfg.openpi.noise_level=.3;cfg.openpi.detach_critic_input=True
    model=get_model(cfg).cuda().eval()
    obs=model.env_obs_to_observation(initial['initial_observation'])
    assert torch.equal(obs.state.cpu(),sampled['processed_model_state'])
    fields=sampled['forward_sampling_fields']
    inputs=dict(chains=fields['chains'],denoise_inds=fields['denoise_inds'],
        obs_state=obs.state,tokenized_prompt=obs.tokenized_prompt,
        tokenized_prompt_mask=obs.tokenized_prompt_mask)
    inputs.update({'obs_image__'+k:v for k,v in obs.images.items()})
    inputs.update({'obs_image_mask__'+k:v for k,v in obs.image_masks.items()})
    inputs={k:v.cuda() for k,v in inputs.items()}
    old=sampled['old_logprobs'].float().cuda();assert old.shape[0]==64
    report=dict(status='running',optimizer_updates=0,rows=[],source_commit=os.environ['TASK_SOURCE'])
    output=Path(os.environ['TASK_OUT'])/'RESULT.json';write(output,report)
    with torch.no_grad():
        for batch in (64,32,16,8):
            started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            pieces=[]
            for start in range(0,64,batch):
                part={k:v[start:start+batch] for k,v in inputs.items()}
                pieces.append(model(forward_inputs=part,compute_values=False)['logprobs'])
            actual=torch.cat(pieces).float();assert actual.shape==old.shape
            assert torch.isfinite(actual).all() and torch.isfinite(old).all()
            error=float((actual-old).abs().max())
            report['rows'].append(dict(batch=batch,max_abs_logprob_error=error,
                seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                passed=error<=1e-3))
            write(output,report);print(json.dumps(report['rows'][-1]),flush=True)
    # Batch partition drift is measured, not hidden with a larger tolerance.
    # Qualify matching rollout/actor64 including a real backward graph; no SGD.
    assert report['rows'][0]['passed']
    model.train();model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    forward=model(forward_inputs=inputs,compute_values=True)
    error=float((forward['logprobs'].float()-old).abs().max())
    assert error<=1e-3
    loss=forward['logprobs'].float().mean()+forward['values'].float().square().mean()
    assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
    groups={'policy':[],'critic':[]}
    for name,param in model.named_parameters():
        if not param.requires_grad:assert param.grad is None
        elif param.grad is not None:
            assert torch.isfinite(param.grad).all(),name
            groups['critic' if name.startswith('value_head.') else 'policy'].append(float(param.grad.float().norm()))
    assert all(any(value>0 for value in norms) for norms in groups.values())
    report.update(status='complete',execution_microbatch=64,
        backward=dict(batch=64,seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),logprob_max_abs_error=error,
            policy_gradient_tensors=len(groups['policy']),critic_gradient_tensors=len(groups['critic']),
            frozen_vlm_has_no_gradients=True,optimizer_updates=0))
    write(output,report)


if __name__=='__main__':main()
