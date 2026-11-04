"""Diagnose ordered actor tensors against actual native env lanes and sampled noise."""
from pathlib import Path
import hashlib, json
import torch
import gzip
import sys
from collections import Counter
ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs')
old=ROOT/'robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01'
new=ROOT/'robocasa-opendrawer-ppo-ablation-3m-20261104-r1/E/attempt-01'
def digest(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def actor(root):
    rows=[]
    for rank in range(2):
        p=root/'training-rollouts'/f'iteration-0001-policy-0000-rank-{rank:02d}.pt'
        b=torch.load(p, mmap=True, map_location='cpu', weights_only=False)['rollout_batch']
        f=b['forward_inputs']
        for i in range(64):
            rows.append(dict(rank=rank,lane=i,noise=[digest(z) for z in f['chains'][:,i,0]],
                indices=f['denoise_inds'][:,i,0].tolist(),state=digest(f['observation/state'][0,i]),
                rgb=[digest(f[k][0,i]) for k in ('observation/image','observation/wrist_image','observation/extra_view_image')],
                first_actions=b['actions'][0,i].clone(),reward=float(b['rewards'][:,i].sum())))
    return rows
def native(root,rank,chunk):
    p=root/'environment'/f'rank-{rank:02d}'/f'training-i000-round00-chunk{chunk:02d}.pt.gz'
    with gzip.open(p,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)
a,b=actor(old),actor(new)
lookup={tuple(r['noise'][:1]):r for r in a}
assert len(lookup)==128
matched=0; allnoise=0; inds=0;state_eq=0; rgb_eq=0; first_exact=0; first_exact_equal_rgb=0; reward_equal=0; shifts=Counter(); unmatched=[]; perm=[]
for r in b:
    orig=lookup.get(tuple(r['noise'][:1]))
    if orig is None:
        unmatched.append([r['rank'],r['lane']]);continue
    matched+=1;allnoise+=r['noise']==orig['noise'];inds+=r['indices']==orig['indices'];state_eq+=r['state']==orig['state']
    same=r['rgb']==orig['rgb'];rgb_eq+=same
    exact=torch.equal(r['first_actions'],orig['first_actions']);first_exact+=exact;first_exact_equal_rgb+=same and exact
    reward_equal+=r['reward']==orig['reward'];shifts[str((r['rank'],orig['rank'],orig['lane']-r['lane']))]+=1
    perm.append([r['rank'],r['lane'],orig['rank'],orig['lane']])
env=[]
for rank in range(8):
    left,right=(native(root,rank,1) for root in (old,new))
    assert left['seeds']==right['seeds']
    x=torch.as_tensor(left['decoded_actions']);y=torch.as_tensor(right['decoded_actions'])
    env.append(dict(rank=rank,first_exact_episodes=int((x==y).flatten(1).all(1).sum()),max_gap=float((x-y).abs().max())))
out=dict(noise_joined=matched,all_noises_equal=allnoise,all_indices_equal=inds,state_equal=state_eq,rgb_equal=rgb_eq,
    first_actions_exact=first_exact,first_actions_exact_on_equal_rgb=first_exact_equal_rgb,
    reward_equal=reward_equal,shift_counts=shifts,unmatched=unmatched,permutation=perm,native=env)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k not in ('native','permutation','unmatched','shift_counts')},flush=True)
print(env,flush=True)
