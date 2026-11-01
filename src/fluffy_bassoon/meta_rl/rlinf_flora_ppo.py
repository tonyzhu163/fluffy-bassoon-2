"""Matched LIBERO PPO reward boundary; official actor/critic/GAE stay upstream.

Metadata and reward-only operations. Flora inference runs through its pinned
terminal CLI in a separate environment, never inside the PPO Python process.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

RLINF='c605b9d55ed93a180119b099e6f836abff293b08'
FLORA_CODE='129a2ada74c526ee4692f3d1b111bea6ca19d5b7'
FLORA_MODEL='031299e46e452fc0cedb4e0975eb9fa95fc37783'
INITIALIZER_REPO='RLinf/RLinf-Pi05-LIBERO-SFT'
INITIALIZER_REVISION='45ccfcc4e28634f1576ebf78cab0fbe2fd82432d'
INITIALIZER_WEIGHT_SHA='4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7'
INITIALIZER_WEIGHT_BYTES=7473091464
NORM_SHA='dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1'
FORMAT='rlinf_libero44_flora_terminal_ppo_v1'
TASK='turn on the stove'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def initializer_manifest(receipt,model_directory):
    """Published40-task Torch release, prospectively distinct from old C565."""
    root=Path(model_directory).resolve()
    if (receipt.get('completed') is not True or receipt.get('repository_id')!=INITIALIZER_REPO
            or receipt.get('revision')!=INITIALIZER_REVISION
            or Path(receipt['model_directory']).resolve()!=root
            or receipt.get('normalization_sha256')!=NORM_SHA
            or receipt.get('model_action_dim')!=32 or receipt.get('action_horizon')!=10):
        raise ValueError('Exact published RLinf40 PyTorch/H10/padded32 release receipt required')
    files={}
    for row in receipt['files']:
        path=Path(row['path']).resolve()
        if not path.is_relative_to(root) or str(path) in files:raise ValueError('Invalid/duplicate initializer artifact path')
        files[str(path)]=row
    expected={str(root/'model.safetensors'):(INITIALIZER_WEIGHT_SHA,INITIALIZER_WEIGHT_BYTES),
        str(root/'physical-intelligence/libero/norm_stats.json'):(NORM_SHA,4507)}
    for path,(checksum,size) in expected.items():
        if path not in files or files[path].get('sha256')!=checksum or files[path].get('bytes')!=size:
            raise ValueError('Published weight/bundled normalization identity differs')
    return list(files.values())


def baseline_validation(rows):
    """Paired scorer validation only; never fit a training-reward threshold."""
    if len(rows)!=64 or len({r['id'] for r in rows})!=64:raise ValueError('Exactly64 unique fresh baseline episodes required')
    scores=[];truth=[]
    for row in rows:
        score=row['expected_score'];label=row['simulator_final_success']
        if type(label) is not bool or isinstance(score,bool) or not math.isfinite(score) or not 1<=score<=5:
            raise ValueError('Finite official expected-score and separate binary calibration truth required')
        scores.append(score);truth.append(label)
    positives=[s for s,y in zip(scores,truth) if y];negatives=[s for s,y in zip(scores,truth) if not y]
    rate=len(positives)/64
    auc=None if not positives or not negatives else sum(float(p>n)+.5*float(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives))
    return dict(format=FORMAT,baseline_sha256=digest(rows),score_field='expected_score',
        training_reward_rule='(expected_score-1)/4; no threshold or fitted transform',count=64,
        initial_success_rate=rate,AUROC=auc,go=bool(.05<=rate<=.70 and auc is not None and auc>=.7),
        calibration_only=True,policy_optimizer_updates=0)


def validate_video(episode):
    fields={'id','task','camera_order','info_on_video','extra_info_on_video','raw_rgb_only',
        'success_termination_disabled','horizon','completed','video','frame_clock','rank','round','local_env'}
    if (set(episode)!=fields or episode['task']!=TASK
            or episode.get('camera_order')!=['agentview'] or episode.get('info_on_video') is not False
            or episode.get('extra_info_on_video')!=[] or episode.get('raw_rgb_only') is not True
            or episode.get('success_termination_disabled') is not True
            or episode.get('horizon')!=400 or episode.get('completed') is not True):
        raise ValueError('Only completed full-horizon clean agentview episodes are scoreable')
    video=episode['video'];clock=episode['frame_clock']
    if (set(video)!={'path','sha256'} or set(clock)!={'frame_count','native_fps','encoded_fps','resampled',
            'physical_steps','initial_frame_included','timestamp_sha256'}
            or type(clock['frame_count']) is not int or clock['frame_count']!=401
            or not math.isfinite(clock['native_fps']) or clock['native_fps']<=0
            or clock['encoded_fps']!=clock['native_fps'] or clock.get('resampled') is not False
            or clock['physical_steps']!=400 or clock.get('initial_frame_included') is not True
            or not isinstance(clock.get('timestamp_sha256'),str) or len(clock['timestamp_sha256'])!=64
            or len(video['sha256'])!=64 or not Path(video['path']).is_absolute()):
        raise ValueError('Actual synchronized native frameclock/video identity required; no default30FPS')
    return dict(id=episode['id'],task=episode['task'],videos=[video['path']])


def flora_jobs(episodes):
    if not episodes or len({e['id'] for e in episodes})!=len(episodes):raise ValueError('Unique fresh episode IDs required')
    # No labels, rewards, success, episode length annotations or score enter the prompt.
    return [validate_video(e) for e in episodes]


def terminal_command(python,model_directory,jobs,output,*,seed=0):
    return [str(python),'-B','-m','rewarder.batch_utils.terminal_reward','--model',str(model_directory),
        '--jobs',str(jobs),'--output',str(output),'--seed',str(seed),
        '--tensor-parallel-size','1','--max-num-seqs','1','--max-model-len','16384',
        '--gpu-memory-utilization','0.78']


def checked_labels(episodes,outputs,validation,receipt):
    if (validation.get('go') is not True or validation.get('score_field')!='expected_score'
            or receipt.get('code_commit')!=FLORA_CODE or receipt.get('model_revision')!=FLORA_MODEL
            or receipt.get('completed') is not True or receipt.get('jobs_sha256')!=digest(flora_jobs(episodes))
            or receipt.get('video_refs_sha256')!=digest([e['video'] for e in episodes])):
        raise ValueError('Successful frozen baseline validation and pinned completed Flora batch required')
    return parse_outputs(episodes,outputs)


def parse_outputs(episodes,outputs):
    """Official terminal output validation, also usable before the baseline gate."""
    flora_jobs(episodes)
    expected={e['id']:e for e in episodes}
    if len(expected)!=len(episodes) or len(outputs)!=len(episodes) or len({r['id'] for r in outputs})!=len(outputs):
        raise ValueError('All fresh batch labels required; no missing, duplicate or dropped episode')
    labels={}
    for row in outputs:
        if row['id'] not in expected or row['task']!=expected[row['id']]['task']:raise ValueError('Flora label task/batch identity differs')
        score=row['expected_score'];prob=row['probabilities']
        if (isinstance(score,bool) or not math.isfinite(score) or not 1<=score<=5
                or set(prob)!=set('12345') or any(not math.isfinite(v) or not 0<=v<=1 for v in prob.values())
                or abs(sum(prob.values())-1)>0.0006 or abs(sum(int(k)*v for k,v in prob.items())-score)>.002
                or not math.isfinite(row['progress']) or abs(row['progress']-(score-1)/4)>.0002):
            raise ValueError('Malformed official terminal expected-score/probability output')
        labels[row['id']]=(score-1.)/4.
    return labels


def relabel_full_horizon(raw_rewards,valid_mask,episode_ids,labels):
    """One terminal scalar per column BEFORE standard chunk-level GAE."""
    import numpy as np
    raw=np.asarray(raw_rewards);mask=np.asarray(valid_mask)
    if (raw.ndim!=3 or not np.issubdtype(raw.dtype,np.floating) or raw.shape[0]*raw.shape[2]!=400 or mask.shape!=raw.shape or mask.dtype!=np.bool_
            or not mask.all() or len(episode_ids)!=raw.shape[1] or len(set(episode_ids))!=len(episode_ids)
            or set(labels)!=set(episode_ids) or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in labels.values())
            or not np.isfinite(raw).all()):raise ValueError('Complete matched full400-step batch/mask/normalized labels required')
    result=np.zeros_like(raw)
    for column,key in enumerate(episode_ids):result[-1,column,-1]=labels[key]
    return result


def freeze_batch(path,*,arm,iteration,episodes,rollout_ref,recovery_refs,rank_pending_receipts,profile_sha256,policy_sha256):
    if arm not in ('S','F') or type(iteration) is not int or iteration<1 or len(episodes)!=128:
        raise ValueError('Matched32env×4rounds completed on-policy batch required')
    jobs=flora_jobs(episodes)
    if (not recovery_refs or any(not r.get('complete') or not r.get('optimizer_sampler_RNG') for r in recovery_refs)
            or rollout_ref.get('producer_iteration')!=iteration-1):raise ValueError('Full pending-batch recovery and producing policy version required')
    value=dict(format=FORMAT,arm=arm,iteration=iteration,producer_iteration=iteration-1,episodes=deepcopy(episodes),
        jobs_sha256=digest(jobs),rollout=rollout_ref,recovery=recovery_refs,profile_sha256=profile_sha256,
        policy_sha256=policy_sha256,status='WAITING_LABELS',optimizer_updates_for_batch=0)
    keys=[(e['rank'],e['round'],e['local_env']) for e in episodes]
    if len(set(keys))!=128:raise ValueError('Native rank/round/env layout must identify every fresh episode')
    value['episode_ids_by_rank']={str(rank):[e['id'] for e in sorted(episodes,key=lambda x:(x['round'],x['local_env'])) if e['rank']==rank]
        for rank in sorted({e['rank'] for e in episodes})}
    if (set(rank_pending_receipts)!=set(value['episode_ids_by_rank'])
            or {r.get('role') for r in recovery_refs}!={'actor','rollout','env'}):
        raise ValueError('All actor/rollout/env pending-batch states and rank-local batches must be bound')
    value.update(rank_pending_receipts=rank_pending_receipts,pending_recovery_complete=True)
    write_new(path,value);return value


def commit_labels(path,batch,labels,*,label_receipt,baseline_validation_sha256):
    ids={e['id'] for e in batch['episodes']}
    if (batch['status']!='WAITING_LABELS' or batch['optimizer_updates_for_batch']!=0 or set(labels)!=ids
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in labels.values())
            or batch['arm']=='S' and any(v not in (0,1) for v in labels.values())):
        raise ValueError('Incomplete batch remains immutable/held; no fallback or partial SGD')
    if (label_receipt.get('completed') is not True or label_receipt.get('input_batch_sha256')!=digest(batch)
            or (batch['arm']=='F' and (label_receipt.get('code_commit')!=FLORA_CODE
                or label_receipt.get('model_revision')!=FLORA_MODEL or not baseline_validation_sha256))):
        raise ValueError('Exact fresh on-policy batch/label completion proof required')
    ready=dict(batch_sha256=digest(batch),format=FORMAT,arm=batch['arm'],iteration=batch['iteration'],
        labels=labels,label_receipt=label_receipt,baseline_validation_sha256=baseline_validation_sha256,status='READY_FOR_GAE')
    write_new(path,ready);return ready


def ready_rewards(batch,ready,raw_rewards,valid_mask,episode_ids):
    if (ready.get('status')!='READY_FOR_GAE' or ready['batch_sha256']!=digest(batch)
            or ready['iteration']!=batch['iteration'] or ready['arm']!=batch['arm']
            or batch['optimizer_updates_for_batch']!=0
            or set(ready['labels'])!={e['id'] for e in batch['episodes']}
            or not set(episode_ids).issubset(ready['labels'])):raise ValueError('Stale/partial/updated batch cannot enter GAE')
    if batch['arm']=='F':
        import numpy as np
        if np.any(np.asarray(raw_rewards)!=0):raise ValueError('Strip native rewards in environment transport before Flora actor receives the batch')
    return relabel_full_horizon(raw_rewards,valid_mask,episode_ids,{key:ready['labels'][key] for key in episode_ids})
