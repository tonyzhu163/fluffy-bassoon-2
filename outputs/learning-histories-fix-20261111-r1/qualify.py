"""CUDA numerical qualification and capacity timing for the new learner recipes."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

code, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(code/'exploration/learning-histories'))
import sessions as S
from train import Learner, dense_logits

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
s = S.sample(np.random.default_rng(91), 16, 6, 'train')
c, o = S.behave(np.random.default_rng(92), s, 6)
errors=[]
for config in [dict(), dict(fourier=4), dict(d=256,layers=6,heads=8,fourier=4)]:
    torch.manual_seed(93)
    model=Learner(8,**config).cuda().eval()
    for cond in S.CONDS:
        with torch.no_grad():
            full=dense_logits(model,'cuda',s,cond,c,o,6,list(range(8)))
            separate=torch.stack([dense_logits(model,'cuda',s,cond,c,o,t-1,[t])[:,0] for t in range(8)],1)
            torch.testing.assert_close(full,separate,atol=3e-6,rtol=3e-5)
            assert torch.equal(full.argmax(-1),separate.argmax(-1))
            x,_=S.encode(s,cond,c,o,6,7)
            ne=1 if cond=='demo' else 7
            e=torch.from_numpy(x[:,:ne]).cuda()
            qx=torch.zeros((16,8,S.N,S.F),device='cuda')
            qx[...,:S.A]=torch.from_numpy(s.scenes).cuda()
            qx[...,S.A+4]=1
            qt=torch.tensor([min(t,ne) for t in range(8)],device='cuda')
            for t in range(ne):
                changed=e.clone(); changed[:,t:]=torch.randn_like(changed[:,t:])
                altered=model.dense(changed,qx,qt)
                torch.testing.assert_close(full[:,t],altered[:,t],atol=3e-6,rtol=3e-5)
            changed=qx.clone(); changed[:,:3]=torch.randn_like(changed[:,:3]); changed[:,4:]=torch.randn_like(changed[:,4:])
            torch.testing.assert_close(full[:,3],model.dense(e,changed,qt)[:,3],atol=3e-6,rtol=3e-5)
        errors.append(dict(config=config,cond=cond,max_abs_error=float((full-separate).abs().max())))
    # The shared pass must give the same objective gradient as independent prefixes.
    model.train(); model.zero_grad()
    targets=torch.from_numpy(s.true).cuda()
    full=dense_logits(model,'cuda',s,'corrections',c,o,6,list(range(8)))
    torch.nn.functional.cross_entropy(full.flatten(0,1),targets.flatten()).backward()
    full_grads={n:p.grad.detach().clone() for n,p in model.named_parameters()}
    model.zero_grad()
    for t in range(8):
        separate=dense_logits(model,'cuda',s,'corrections',c,o,t-1,[t])[:,0]
        (torch.nn.functional.cross_entropy(separate,targets[:,t])/8).backward()
    max_grad=0.
    for n,p in model.named_parameters():
        torch.testing.assert_close(p.grad,full_grads[n],atol=5e-6,rtol=5e-4)
        max_grad=max(max_grad,float((p.grad-full_grads[n]).abs().max()))
    errors.append(dict(config=config,gradient_max_abs_error=max_grad))
    del model,full_grads
    torch.cuda.empty_cache()
print('DENSE NUMERICS PASSED',flush=True)


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def run(args,log):
    with open(log,'w') as f:
        subprocess.run([sys.executable,'-Bu',str(code/'exploration/learning-histories/train.py'),*args],check=True,stdout=f,stderr=subprocess.STDOUT)

for name,flags in [('legacy',[]),('dense',['--dense','--fourier','4'])]:
    folder=out/name; folder.mkdir()
    common=['--cond','corrections','--K','6','--steps','40','--batch','16','--eval-n','32','--eval-batch','8','--device','cuda',*flags]
    run(common+['--out',str(folder/'whole.json')],folder/'whole.log')
    run(common+['--out',str(folder/'partial.json'),'--stop-after','20'],folder/'partial.log')
    resume=folder/'partial.pt'
    if name=='legacy':
        old=torch.load(resume,map_location='cpu',weights_only=False)
        for key in ['dense','fourier','d','layers','heads','eval_seed']: old['args'].pop(key,None)
        resume=folder/'legacy-schema.pt'; torch.save(old,resume)
    run(common+['--out',str(folder/'resumed.json'),'--resume',str(resume)],folder/'resumed.log')
    a=torch.load(folder/'whole.pt',map_location='cpu',weights_only=False)
    b=torch.load(folder/'resumed.pt',map_location='cpu',weights_only=False)
    for key in ['step','model','optimizer','scheduler','numpy_rng','torch_rng','cuda_rng']: equal(a[key],b[key])
    a,b=(json.loads((folder/n).read_text()) for n in ['whole.json','resumed.json'])
    for split in ['train','heldout']: assert a[split]==b[split]
    print('RESUME',name,'bitwise exact',flush=True)

# Real old checkpoint verifies historical compatibility, not merely a synthetic schema.
old_grid=Path('/mowenjie/fluffy-bassoon-2/outputs/learning-histories-20261111-r1/grid01/attempts-s0')
run(['--cond','attempts','--K','6','--seed','0','--steps','20000','--resume',str(old_grid/'attempts.pt'),'--out',str(out/'original-replay.json'),'--device','cuda'],out/'original-replay.log')
a,b=(json.loads(p.read_text()) for p in [old_grid/'attempts.json',out/'original-replay.json'])
for split in ['train','heldout']: assert a[split]==b[split]
print('ORIGINAL CURVES EXACT',flush=True)

timings=[]
for name,flags in [('base',['--lr','1e-3']),('dense',['--dense','--fourier','4','--lr','1e-3']),('wide',['--dense','--fourier','4','--lr','5e-4','--d','256','--layers','6','--heads','8'])]:
    folder=out/f'timing-{name}'; folder.mkdir()
    run(['--cond','corrections','--K','6','--steps','20000','--stop-after','200','--out',str(folder/'result.json'),'--device','cuda',*flags],folder/'fit.log')
    state=torch.load(folder/'result.pt',map_location='cpu',weights_only=False)
    seconds=state['elapsed_seconds']/200
    row=dict(recipe=name,batch=256,steps=200,seconds_per_update=seconds,projected_train_minutes=seconds*20000/60,peak_allocated_bytes=state['cuda_peak_memory_bytes'])
    timings.append(row); print('TIMING',json.dumps(row),flush=True)
receipt=dict(status='passed',torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(),dense_numerics=errors,resumption={'legacy_schema':'bitwise_exact','dense':'bitwise_exact'},original_curves='exact',timings=timings,grid_90min_admissible=all(t['projected_train_minutes']<=80 for t in timings))
(out/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('QUALIFICATION PASSED',json.dumps(receipt['timings']),flush=True)
