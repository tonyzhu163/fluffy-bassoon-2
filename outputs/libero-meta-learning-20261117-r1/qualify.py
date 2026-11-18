"""Finite native qualification on development tasks only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from native import Motor, array_hash, make_adapter, public, qualify_gradient, write
from simulator import Simulator


def rollout(motor, case, output, *, zero_adapter=False, gradient=False,
            role='development runtime qualification; not held-out adaptation'):
    output.mkdir(parents=True,exist_ok=False)
    policy=make_adapter().eval() if zero_adapter else None
    began=time.monotonic()
    rows=[]; states=[]; images=[]; features=[]; proposals=[]; executed=[]; masks=[]
    previous=np.zeros(322,np.float32); memory=None
    action_hash=hashlib.sha256()
    env=Simulator()
    try:
        spec,first=env.call('create',case)
        max_steps=spec['max_steps']
        replan = motor.replan_steps
        queries = (max_steps + replan - 1) // replan
        noises=np.random.default_rng(case['seed']+817117).standard_normal(
            (queries,1,10,32)).astype(np.float32)
        raw=first['observation']; prompt=first['language_instruction']
        first_hash={k:array_hash(v) for k,v in raw.items()}
        write(output/'initial.json',dict(spec=spec,public_hashes=first_hash,
            environment=first['environment_reproducibility'],schema=first['action_schema']))
        controls=0; success=False; terminated=False; truncated=False
        while controls<max_steps and not (terminated or truncated):
            item=public(raw,prompt)
            prepared=motor.prepare(item)
            if gradient and not rows:
                qualify_gradient(motor,prepared,noises[0],output/'gradient.json')
            context=None
            if policy is not None:
                token=torch.cat((prepared['features'],torch.from_numpy(previous[None]).to('cuda'),
                    torch.tensor([[not rows]],dtype=torch.float32,device='cuda')),-1)
                with torch.no_grad():
                    context,memory=policy.encode_step(token,memory)
            t=time.monotonic()
            chunk=motor.actions(prepared,noises[len(rows)],policy=policy,context=context)
            inference_seconds=time.monotonic()-t
            delivered=np.zeros((10,7),np.float32); mask=np.zeros((10,7),bool)
            reward=0.
            for k,action in enumerate(chunk[:min(replan,max_steps-controls)]):
                transition=env.call('step',action)
                delivered[k]=action; mask[k]=True; controls+=1
                action_hash.update(action.tobytes())
                raw=transition['observation']; reward+=transition['reward']
                success=success or transition['success']
                terminated=transition['terminated']; truncated=transition['truncated']
                if terminated or truncated:
                    break
            states.append(item['states'][0]); images.append(np.stack((item['main_images'][0],item['wrist_images'][0])))
            features.append(prepared['features'][0].cpu().numpy()); proposals.append(chunk)
            executed.append(delivered); masks.append(mask)
            rows.append(dict(controls=controls,reward=reward,success=success,
                terminated=terminated,truncated=truncated,inference_seconds=inference_seconds))
            padded=np.zeros((10,32),np.float32); padded[:,:7]=delivered
            previous=np.concatenate((padded.reshape(-1),[reward,terminated or truncated])).astype(np.float32)
            if len(rows)%10==0:
                print(json.dumps(dict(case=case,controls=controls,success=success)),flush=True)
        np.savez_compressed(output/'episode.npz',states=states,images=images,features=features,
            proposed_actions=proposals,issued_actions=executed,action_masks=masks,
            noise=noises[:len(rows)],rewards=np.array([r['reward'] for r in rows],np.float32),
            dones=np.array([r['terminated'] or r['truncated'] for r in rows],bool))
        result=dict(case=case,prompt=prompt,zero_adapter=zero_adapter,success=success,controls=controls,
            initial_hashes=first_hash,initial_state=first['environment_reproducibility']['initial_state_fingerprint'],
            action_sha256=action_hash.hexdigest(),seconds=time.monotonic()-began,rows=rows,
            npz_bytes=(output/'episode.npz').stat().st_size,
            action_record='public commands issued to env.step; hidden delay/filter effects are not student inputs',
            role=role)
        write(output/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','initial_hashes')}),flush=True)
        return result
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--checkpoint',choices=('openpi','rlinf'),default='rlinf')
    p.add_argument('--reference-cadence',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    motor=Motor(args.checkpoint, reference_cadence=args.reference_cadence)
    write(args.output/'motor-config.json',dict(motor.config, executed_before_replan=motor.replan_steps))
    # Both are predeclared development tasks, not official test tasks.
    suite=('libero_spatial','libero_goal')[args.rank]
    case=dict(suite=suite,task_id=1,init_state_id=0,seed=171117+args.rank,variant='BASE')
    results=[]
    for ordinal,variant in enumerate(('BASE','BASE','S15','S20')):
        if time.time()>=args.deadline:
            raise TimeoutError('Finite qualification deadline reached')
        result=rollout(motor,dict(case,variant=variant),args.output/f'episode-{ordinal}',
            zero_adapter=ordinal==1,gradient=ordinal==0)
        results.append(result)
        if ordinal==1:
            a,b=results[:2]
            assert a['initial_hashes']==b['initial_hashes'] and a['initial_state']==b['initial_state'], 'Reset pairing differs'
            assert a['action_sha256']==b['action_sha256'], 'Zero-adapter native action trace differs'
    write(args.output/'summary.json',dict(complete=True,rank=args.rank,episodes=[
        {k:r[k] for k in ('case','success','controls','seconds','npz_bytes','zero_adapter')} for r in results],
        paired_native_trace_exact=True,peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
