"""Assemble one task once; AD/DIT/IC-IQL share the same behavior-target data."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from labels import sha, write


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pin',required=True)
    p.add_argument('--complete',type=Path,required=True)
    p.add_argument('--complete-sha256',required=True)
    p.add_argument('--task',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cold',type=Path,required=True)
    a=p.parse_args()
    assert sys.platform=='linux'
    code=Path(__file__).resolve().parents[2];sys.path.insert(0,str(code/'src'))
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec
    from fluffy_bassoon.meta_rl.vla_history import save_dataset,load_dataset
    import numpy as np
    checked_checkout(code,a.pin)
    assert sha(a.complete)==a.complete_sha256
    complete=json.loads(a.complete.read_text())
    assert complete['completed'] and complete['task']==a.task and complete['total_teacher_queries']==46080
    assert len(complete['shards'])==4
    raw,features,labels,reference={},[],[],None
    for index,ref in enumerate(complete['shards']):
        assert sha(ref['path'])==ref['sha256']
        shard=json.loads(Path(ref['path']).read_text())
        assert shard['completed'] and shard['task']==a.task and shard['shard_index']==index
        assert shard['global_slots']==list(range(index*32,(index+1)*32))
        assert shard['teacher_queries']==11520 and shard['all_payloads_validated'] and shard['all_files_cold_readback']
        assert shard['spec_sha256']==complete['spec_sha256'] and shard['source_commit']==complete['source_commit']
        identity=dict(provenance=shard['provenance'],codec=shard['codec'])
        if reference is None:reference=identity
        assert identity==reference
        for name,item in shard['files'].items():
            assert Path(item['path']).name==name and sha(item['path'])==item['sha256']
        with np.load(shard['files']['RAW-PUBLIC-CORPUS.npz']['path'],allow_pickle=False) as f:
            for key in f.files:raw.setdefault(key,[]).append(f[key])
        with np.load(shard['files']['FEATURES-TEACHER-30.npz']['path'],allow_pickle=False) as f:
            features.append(f['features']);labels.append(f['mapped_teacher_actions'])
    raw={k:np.concatenate(v) for k,v in raw.items()}
    features,labels=np.concatenate(features),np.concatenate(labels)
    assert features.shape==(128,360,2073) and labels.shape==(128,360,50,12)
    assert np.array_equal(features[:,:,:25],raw['public_states']) and np.isfinite(features).all()
    assert raw['physical_actions'].shape==labels.shape
    starts=np.broadcast_to(np.arange(360)%12==0,(128,360))
    ends=np.broadcast_to(np.arange(360)%12==11,(128,360))
    versions=np.broadcast_to(np.repeat(np.arange(30,dtype=np.int64),12),(128,360))
    assert np.array_equal(raw['episode_starts'],starts) and np.array_equal(raw['physical_episode_ends'],ends)
    assert np.array_equal(raw['producer_versions'],versions)
    rewards=np.asarray(raw['primitive_rewards'].sum(-1),np.float32)
    assert np.isin(rewards,[0.,1.]).all() and (rewards.reshape(128,30,12).sum(-1)<=1).all()
    codec=DirectActionCodec(tuple(reference['codec']['mean']),tuple(reference['codec']['std']))
    behavior,teacher=codec.encode(raw['physical_actions']),codec.encode(labels)
    provenance=reference['provenance'];native=provenance['signature']
    source=provenance['raw_archive']
    metadata=dict(task=a.task,motor_sha256=native['model_sha256'],norm_sha256=native['norm_sha256'],
        runtime_commit=native['runtime_commit'],source_commit=a.pin,label_source_commit=complete['source_commit'],
        preparation=dict(path=str(a.complete),sha256=a.complete_sha256),teacher=provenance['teacher'],
        raw_archive=dict(path=source['archive'],sha256=source['archive_sha256']),
        schedule=dict(streams=128,policy_versions=30,chunks_per_episode=12),
        feature_contract=dict(dimension=2073,meaning='original frozen prefix2048 RMS plus raw public state25'),
        noise_contract=codec.record(),value_contract=dict(discount=.99,boundary='physical_done',
            rewards='actual native first-success once',terminal_bootstrap=0,
            next_observation='following saved query within episode; no invented terminal RGB'),
        validation='Held-out collection slots112..127 per task; PPO source saw all collection observations')
    a.output.mkdir(parents=True,exist_ok=False);a.cold.mkdir(parents=True,exist_ok=False)
    datasets={}
    for split,slots in [('train',range(112)),('validation',range(112,128))]:
        ids=[f'{a.task}/flow-source-slot-{s:03d}' for s in slots]
        for kind,objective,target in [('behavior','ad',behavior),('teacher30','dpt',teacher)]:
            trials=[dict(observations=features[s],executed_noise=behavior[s],target_noise=target[s],
                rewards=rewards[s],dones=ends[s].copy(),episode_starts=starts[s].copy(),teacher_steps=versions[s].copy()) for s in slots]
            path=a.output/f'{kind}-{split}.npz'
            manifest=save_dataset(path,trials,objective=objective,metadata=dict(metadata,trial_ids=ids,
                shared_objectives=['ad','dit','ic_iql'] if kind=='behavior' else ['dpt']))
            # Numerical readback protects the source arrays and exact stream split.
            restored,_=load_dataset(path,expected_sha256=manifest['sha256'])
            assert len(restored)==len(trials)
            assert all(np.array_equal(x[k],y[k]) for x,y in zip(trials,restored,strict=True) for k in x)
            for source_path in [path,path.with_suffix('.json')]:
                destination=a.cold/source_path.name
                with source_path.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst,16*2**20)
                assert sha(destination)==sha(source_path)
            item=dict(path=str(a.cold/path.name),sha256=manifest['sha256'],
                manifest_sha256=sha(a.cold/path.with_suffix('.json').name),trials=len(trials),trial_ids=ids)
            for method in (['ad','dit','ic_iql'] if kind=='behavior' else ['dpt']):datasets[f'{method}_{split}']=item
    result=dict(completed=True,student_fit_ready=True,task=a.task,source_commit=a.pin,
        preparation=dict(path=str(a.complete),sha256=a.complete_sha256),datasets=datasets,
        sources={a.task:dict(index_path=source['index_path'],index_sha256=source['index_sha256'],
            archive=source['archive'],archive_sha256=source['archive_sha256'])},
        codec=codec.record(),native_model_sha256=native['model_sha256'],native_norm_sha256=native['norm_sha256'],
        runtime_commit=native['runtime_commit'],train_streams=112,validation_streams=16,
        teacher_queries=46080,physical_episodes=3840,all_arrays_roundtrip_exact=True,
        storage='Four canonical files per task. Behavior data shared by AD/DIT/IC-IQL; no pooled3/6-task copies or copied RGB.')
    write(a.cold/'ASSEMBLY.json',result)
    # Only this freshly exported and fully read-back-verified local staging.
    shutil.rmtree(a.output)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
