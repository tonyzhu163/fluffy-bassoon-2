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
    model=get_model(cfg).cuda().eval().requires_grad_(False)
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
    report['status']='complete' if all(r['passed'] for r in report['rows']) else 'failed'
    write(output,report);assert report['status']=='complete'


if __name__=='__main__':main()
