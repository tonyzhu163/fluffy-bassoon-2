"""Audit context identity, first-success suffixes and exact actor exposure on CPU."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE/'src'))
TASKS = ('OpenDrawer','CloseToasterOvenDoor','CoffeeSetupMug',
         'PickPlaceCounterToStove','TurnOffStove','TurnOnSinkFaucet')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert sys.platform=='linux'
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==a.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from fluffy_bassoon.meta_rl.vla_history import load_dataset
    a.output.mkdir(parents=True,exist_ok=False)
    began=time.time()
    result=dict(pin=a.pin,script_sha256=sha(__file__),started_unix=began,GPU_hours=0,
                task_order=TASKS,sources={},sampler={},probe={},presuccess={})
    data={};probe_train=[];probe_val=[]
    for task in TASKS:
        assembly=json.loads((a.source/'datasets01'/task/'ASSEMBLY.json').read_text())
        ref=assembly['preparation'];assert sha(ref['path'])==ref['sha256']
        complete=json.loads(Path(ref['path']).read_text())
        seed_parts=[];control_reward=[];rawrefs=[]
        for shardref in complete['shards']:
            assert sha(shardref['path'])==shardref['sha256']
            shard=json.loads(Path(shardref['path']).read_text())
            raw=shard['files']['RAW-PUBLIC-CORPUS.npz']
            assert sha(raw['path'])==raw['sha256']
            with np.load(raw['path'],allow_pickle=False) as f:
                seeds=f['native_seeds'].reshape(32,30,12)
                assert (seeds==seeds[:,:,:1]).all()
                seed_parts.append(seeds[:,:,0])
                control_reward.append(f['primitive_rewards'].reshape(32,30,600))
            rawrefs.append(raw)
        seeds=np.concatenate(seed_parts);primitive=np.concatenate(control_reward)
        assert seeds.shape==(128,30) and primitive.shape==(128,30,600)
        assert set(np.unique(primitive))<={0.,1.} and (primitive.sum(-1)<=1).all()
        success=primitive.sum(-1)>0
        after_control=np.cumsum(primitive,axis=-1)-primitive>0
        chunk_rewards=primitive.reshape(128,30,12,50).sum(-1)
        after_query=np.cumsum(chunk_rewards,axis=-1)-chunk_rewards>0
        result['sources'][task]=dict(raw_refs=rawrefs,episodes=int(seeds.size),
            unique_native_seeds=int(np.unique(seeds).size),
            distinct_seeds_per_stream=np.unique([len(set(s.tolist())) for s in seeds]).tolist(),
            adjacent_repeated_seeds=int((seeds[:,1:]==seeds[:,:-1]).sum()),
            successful_episodes=int(success.sum()),post_success_control_fraction=float(after_control.mean()),
            post_success_query_fraction=float(after_query.mean()),splits={})
        data[task]={}
        for split,count,offset in [('train',112,0),('validation',16,112)]:
            d=assembly['datasets']['dit_'+split]
            trials,manifest=load_dataset(d['path'],expected_sha256=d['sha256'])
            assert len(trials)==count
            rewards=np.stack([t['rewards'].reshape(30,12) for t in trials])
            assert np.array_equal(rewards,chunk_rewards[offset:offset+count])
            after=np.cumsum(rewards,axis=-1)-rewards>0
            data[task][split]=dict(rewards=rewards,after=after)
            # Probe only episode-start observations: no feedback/current action input.
            initial=np.stack([t['observations'][::12] for t in trials]).reshape(-1,2073)
            (probe_train if split=='train' else probe_val).append(initial)
            result['sources'][task]['splits'][split]=dict(episodes=count*30,queries=count*360,
                post_success_queries=int(after.sum()),post_success_query_fraction=float(after.mean()),
                initial_query_feature_unique=int(len({x.tobytes() for x in initial})),
                dataset_sha256=d['sha256'])
            del trials
        print('SOURCE',task,flush=True)

    # Exact actor RNG sequence: uniform task/stream, end episode, window count, query.
    for size in (3,6):
        rng=np.random.default_rng(0);seen=np.zeros((size,112,360),dtype=np.int32)
        context_counts=np.zeros(12,dtype=np.int64);prior_counts=np.zeros(12,dtype=np.int64)
        for update in range(10000):
            tasks=rng.integers(size,size=8);slots=rng.integers(112,size=8)
            for task,slot in zip(tasks,slots):
                end=int(rng.integers(1,31));count=int(rng.integers(1,min(12,end)+1))
                rel=int(rng.integers(0,count*12));query=(end-count)*12+rel
                seen[task,slot,query]+=1;context_counts[count-1]+=1;prior_counts[rel//12]+=1
        total=int(seen.sum());assert total==80000
        post=np.stack([data[t]['train']['after'].reshape(112,360) for t in TASKS[:size]])
        result['sampler'][str(size)]=dict(actor_updates=10000,logical_batch=8,total_queries=total,
            available_training_queries=int(seen.size),unique_queries=int((seen>0).sum()),
            fraction_unique_covered=float((seen>0).mean()),
            sampled_post_success_queries=int(seen[post].sum()),sampled_post_success_fraction=float(seen[post].sum()/total),
            queries_by_teacher_version=seen.reshape(size,112,30,12).sum((0,1,3)).tolist(),
            queries_by_task=seen.sum((1,2)).tolist(),sampled_window_episodes=context_counts.tolist(),
            preceding_episodes_at_actual_query=prior_counts.tolist(),
            fraction_queries_without_previous_episode=float(prior_counts[0]/total))
        bindings={}
        for method in ('ad','dpt','dit','ic_iql'):
            auditdir='final-fit-audit01' if method in ('ad','dpt') else 'final-fit-audit02'
            audit=json.loads((a.source/auditdir/f'{size}_{method}.json').read_text())
            ref=audit['checkpoint'];assert audit['passed'] and sha(ref['path'])==ref['sha256']
            saved=torch.load(ref['path'],map_location='cpu',weights_only=False)
            assert saved['progress']['actor_update']==10000
            assert saved['rng']['sampler']==rng.bit_generator.state, (size,method,'sampler diverged')
            bindings[method]=ref
            del saved
        result['sampler'][str(size)].update(final_rng_matches_all_four_saved_fits=True,checkpoints=bindings)

    # Fixed nearest-centroid probe, no model or hyperparameter selection.
    x=np.concatenate(probe_train).astype(np.float64);v=np.concatenate(probe_val).astype(np.float64)
    y=np.repeat(np.arange(6),112*30);yv=np.repeat(np.arange(6),16*30)
    mean=x.mean(0);std=x.std(0).clip(1e-6)
    x=(x-mean)/std;v=(v-mean)/std
    for name,cols in [('public_state',slice(0,25)),('pooled_prefix',slice(25,None)),('combined',slice(None))]:
        centers=np.stack([x[y==t,cols].mean(0) for t in range(6)])
        val=v[:,cols]
        distances=(val*val).sum(1)[:,None]+(centers*centers).sum(1)[None,:]-2*val@centers.T
        pred=distances.argmin(1);conf=np.zeros((6,6),dtype=np.int64)
        np.add.at(conf,(yv,pred),1)
        result['probe'][name]=dict(accuracy=float((pred==yv).mean()),confusion_matrix=conf.tolist(),
            train_queries=len(x),validation_queries=len(v),prior_accuracy=1/6,
            interpretation='Current-observation source-task identification only; not evidence of policy quality or absence of useful hidden variables')
    del x,v,probe_train,probe_val

    # Correct the proxy interpretation: reward is spent after the first success.
    state=json.loads((a.diagnostic/'inference01/STATE.json').read_text())
    assert state['completed']
    rng=np.random.default_rng(20261116)
    for pth in sorted((a.diagnostic/'inference01').glob('*-queries.jsonl.gz')):
        assert sha(pth)==state['files'][pth.name]
        with gzip.open(pth,'rt') as f:rows=[json.loads(line) for line in f]
        model=rows[0]['model'];result['presuccess'][model]={}
        for task in sorted({r['task'] for r in rows}):
            rs=[r for r in rows if r['task']==task]
            after=data[task]['validation']['after']
            before=[r for r in rs if not after[r['stream']-112,r['update'],r['query']]]
            post=[r for r in rs if after[r['stream']-112,r['update'],r['query']]]
            assert all(r['return_']==0 and r['success']==1 for r in post)
            counts=rng.multinomial(16,np.full(16,1/16),size=4000)
            num=np.zeros((16,16));den=np.zeros((16,16))
            for version in (0,9,19,29):
                for query in range(12):
                    group=[r for r in before if r['update']==version and r['query']==query]
                    for pos in [r for r in group if r['success']]:
                        for neg in [r for r in group if not r['success']]:
                            i,j=pos['stream']-112,neg['stream']-112
                            pa,na=pos['q']-pos['v'],neg['q']-neg['v']
                            den[i,j]+=1;num[i,j]+=int(pa>na)+.5*int(pa==na)
            bn=np.einsum('bi,ij,bj->b',counts,num,counts);bd=np.einsum('bi,ij,bj->b',counts,den,counts)
            bs=np.divide(bn,bd,out=np.full(4000,np.nan),where=bd>0)
            w=np.array([r['weight'] for r in before]);success=np.array([r['success'] for r in before])
            result['presuccess'][model][task]=dict(original_queries=len(rs),before_or_at_first_success_queries=len(before),
                post_success_queries=len(post),post_success_zero_return_verified=True,
                presuccess_advantage_auc=float(num.sum()/den.sum()) if den.sum() else None,
                presuccess_auc_bootstrap95=np.nanquantile(bs,[.025,.975]).tolist(),
                comparable_pairs=int(den.sum()),
                presuccess_success_share_shift_pp=float(100*((w*success).sum()/w.sum()-success.mean())),
                presuccess_ESS_fraction=float(w.sum()**2/(len(w)*(w*w).sum())),
                limitation='Outcome-associated remaining success, still not causal action advantage')
    result.update(completed=True,elapsed_seconds=time.time()-began)
    write(a.output/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('sampler','probe','presuccess')},indent=2),flush=True)


if __name__=='__main__':main()
