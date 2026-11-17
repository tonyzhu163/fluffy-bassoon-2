"""Fit a full native KV adapter; preserve the fixed logical batch and query order."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from native import load, rollout, write
from bridge import Corpus, BalancedQueries, Queries, batch, commands, Student
from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig, HistoryFlowSteeringPolicy, export_bridge_policy
from fluffy_bassoon.meta_rl.flow_bridge_train import bridge_update, save_resume, restore_resume, query_context
from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _versions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('privileged','full','current'),required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--updates',type=int,default=4096)
    p.add_argument('--scale',type=float,default=.1)
    p.add_argument('--execution-batch',type=int,default=2)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--native-eval',action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    manifest = json.loads(a.manifest.read_text())
    contract = dict(mode=a.mode, seed=a.seed, updates=a.updates, scale=a.scale,
                    manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                    logical_batch=8, learning_rate=.0003, weight_decay=0., source=a.pin)
    write(a.output/'contract.json',contract)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    def log(kind, **data):
        row = dict(kind=kind,time=time.time(),**data)
        with (a.output/'events.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    native,motor,codec,stats,identity = load(a.pin)
    write(a.output/'motor.json',identity)
    corpus = Corpus(manifest['trials'],privileged=a.mode=='privileged')
    reader = Queries(corpus,native,motor)
    first = corpus.strata[0][0]
    prepared = reader.prepare([first]); cache = prepared['prefix'][2]
    config = FlowBridgeConfig(observation_dim=2076 if a.mode=='privileged' else 2073,
        width=256,layers=6,heads=8,context_length=360,kv_rank=4,scale=a.scale,
        kv_heads=cache[0][0].shape[2],head_dim=cache[0][0].shape[3],kv_layers=tuple(range(len(cache))))
    native._prepared = None; del cache,prepared
    torch.manual_seed(a.seed)
    policy = HistoryFlowSteeringPolicy(config).to('cuda:0')
    policy.fit_observation_normalization([dict(observations=e['features']) for e in corpus.episodes])
    optimizer = torch.optim.AdamW(policy.parameters(),lr=.0003,weight_decay=0.)
    sampler = BalancedQueries(corpus,a.seed)
    generator = torch.Generator(device='cuda:0').manual_seed(771105+a.seed)
    frozen = _versions(native.model)
    progress = dict(update=0)
    if a.resume:
        progress = restore_resume(a.resume,contract=contract,policy=policy,optimizer=optimizer,
                                  sampler=sampler.rng,generator=generator)
        sampler.restore(progress['sampler'])
    fixed_indices = []
    for stratum in corpus.strata:
        active = [i for i in stratum if corpus.episode(i)[0]['meta']['rows'][i[2]]['active']]
        quiet = [i for i in stratum if not corpus.episode(i)[0]['meta']['rows'][i[2]]['active']]
        fixed_indices.extend([group[len(group)//2] for group in (active,quiet) if group])
    diagnostic_rng = torch.Generator(device='cuda:0').manual_seed(871105)

    @torch.no_grad()
    def diagnostic():
        policy.eval(); rows = []
        for idx in fixed_indices:
            ep,q = corpus.episode(idx)
            b = batch(corpus,[idx],policy,codec,stats,a.mode,diagnostic_rng)
            context = query_context(policy,b)
            prepared = reader.prepare([idx])
            from native import base_prediction
            base = codec.mapped_physical(base_prediction(native,prepared,ep['noise'][q]))
            predicted = commands(native,policy,prepared,context,ep['noise'][q][None])[0]
            n = int(ep['action_masks'][q,:,0].sum())
            target = ep['teacher_actions'][q]
            rows.append(dict(index=idx,active=ep['meta']['rows'][q]['active'],
                native_xyz_mse=float(np.mean((base[:n,:3]-target[:n,:3])**2)),
                adapter_xyz_mse=float(np.mean((predicted[:n,:3]-target[:n,:3])**2)),
                adapter_all_mse=float(np.mean((predicted[:n]-target[:n])**2)),
                correction_xyz=(predicted[:n,:3]-base[:n,:3]).mean(0).tolist(),
                target_correction_xyz=(target[:n,:3]-base[:n,:3]).mean(0).tolist()))
            if progress['update'] == 0 and a.resume is None:
                np.testing.assert_array_equal(predicted,base)
        write(a.output/f'decoded-{progress["update"]:05d}.json',dict(rows=rows,
            note='Training/development command diagnostic, not held-out policy performance'))
        log('decoded',update=progress['update'],active_queries=sum(r['active'] for r in rows),
            active_native_mse=float(np.mean([r['native_xyz_mse'] for r in rows if r['active']])),
            active_adapter_mse=float(np.mean([r['adapter_xyz_mse'] for r in rows if r['active']])),
            inactive_adapter_mse=float(np.mean([r['adapter_xyz_mse'] for r in rows if not r['active']])))

    def save():
        assert _versions(native.model) == frozen
        progress['sampler'] = sampler.state()
        latest, previous = a.output/'latest.pt', a.output/'previous.pt'
        if latest.exists(): latest.replace(previous)
        save_resume(latest,contract=contract,policy=policy,optimizer=optimizer,sampler=sampler.rng,
                    generator=generator,progress=progress)

    export_bridge_policy(a.output/'initial.pt',policy,manifest=contract)
    diagnostic()
    timings = []
    while progress['update'] < a.updates and not stopped and time.time() < a.deadline:
        start = time.monotonic()
        indices = sampler.draw(8)
        b = batch(corpus,indices,policy,codec,stats,a.mode,generator)
        metrics = bridge_update(policy,native.model,optimizer,b,reader,execution_batch_size=a.execution_batch)
        torch.cuda.synchronize(); timings.append(time.monotonic()-start)
        progress['update'] += 1
        if progress['update'] == 1 or progress['update'] % 32 == 0:
            log('update',update=progress['update'],mean_seconds=float(np.mean(timings)),
                decoder_gradient_l2=float(torch.stack([x.grad.square().sum() for x in policy.core.parameters()
                    if x.grad is not None]).sum().sqrt()),
                adapter_gradient_l2=float(torch.stack([x.grad.square().sum() for x in policy.kv_adapter.parameters()
                    if x.grad is not None]).sum().sqrt()),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),**metrics)
            timings.clear()
        if progress['update'] % 128 == 0:
            save()
        if progress['update'] in (128,512,1024,2048,4096):
            diagnostic()
            export_bridge_policy(a.output/f'policy-{progress["update"]:05d}.pt',policy,
                                  manifest=dict(contract,update=progress['update']))
    save()
    complete = progress['update'] == a.updates
    if complete:
        export_bridge_policy(a.output/'candidate.pt',policy,manifest=dict(contract,update=progress['update']))
    write(a.output/'summary.json',dict(complete=complete,update=progress['update'],config=asdict(config),
        frozen_motor_unchanged=_versions(native.model)==frozen,
        coverage=[dict(queries=len(c),seen=int((c>0).sum()),minimum=int(c.min()),maximum=int(c.max())) for c in sampler.coverage]))
    if complete and a.native_eval:
        assert a.mode == 'privileged'
        student = Student(native,policy,'privileged')
        seen,results=set(),[]
        for ep in corpus.episodes:
            meta=ep['meta']; key=(meta['scene'],tuple(meta['offset']))
            if key in seen: continue
            seen.add(key)
            for name,fraction,actor in (('sft',0.,None),('teacher',1.,None),('privileged',0.,student)):
                if time.time() >= a.deadline:
                    write(a.output/'native-eval.json',dict(complete=False,results=results)); return
                if actor is not None: actor.reset()
                result=rollout(native,motor,codec,directory=a.output/f'eval-{len(seen):02d}-{name}',
                    scene=meta['scene'],offset=meta['offset'],fraction=fraction,
                    target_start=meta['start_signature'],attempt=2,student=actor,save_images=False,log=log)
                results.append(dict(arm=name,scene=meta['scene'],offset=meta['offset'],
                    success=result['success'],controls=result['controls']))
                write(a.output/'native-eval.json',dict(complete=False,results=results))
        write(a.output/'native-eval.json',dict(complete=True,results=results,
            note='Privileged interface engineering test on development scenes; no public-history claim'))


if __name__ == '__main__':
    main()
