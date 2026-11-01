#!/usr/bin/env python
"""Run the pinned official terminal scorer in its separate allocated environment.

One smoke/baseline invocation, or a finite13-batch training service. No simulator
truth path is accepted for training. Existing partial output is never overwritten.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import (
    FLORA_CODE,FLORA_MODEL,baseline_validation,checked_labels,commit_labels,digest,
    flora_jobs,parse_outputs,sha,terminal_command,write_new,
)
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_monitor import ProgressLog


def score(episodes,output,model,code,model_receipt):
    jobs=flora_jobs(episodes)
    for episode in episodes:
        if sha(episode['video']['path'])!=episode['video']['sha256']:raise ValueError('Video bytes changed')
    jobs_path=output.with_suffix('.jobs.jsonl')
    with jobs_path.open('x') as stream:
        for row in jobs:stream.write(json.dumps(row)+'\n')
    command=terminal_command(sys.executable,model,jobs_path,output)
    log=ProgressLog(output.parent/(output.stem+'-progress'),'F',contract_sha256=digest(dict(jobs=jobs,model_receipt=model_receipt))).start()
    try:
        log.emit(phase='official_Flora_terminal_inference',episodes_complete=len(episodes))
        subprocess.run(command,cwd=code,check=True)
        log.emit(phase='labels_written',labels_complete=len(episodes))
    except BaseException as error:
        log.emit(phase='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:log.close()
    outputs=[json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    labels=parse_outputs(episodes,outputs)
    receipt=dict(code_commit=FLORA_CODE,model_revision=FLORA_MODEL,model_receipt=model_receipt,
        completed=True,jobs_sha256=digest(jobs),video_refs_sha256=digest([e['video'] for e in episodes]),
        output=dict(path=str(output),sha256=sha(output)),command=command,training_reward_rule='(expected_score-1)/4')
    write_new(output.with_suffix('.receipt.json'),receipt)
    return outputs,labels,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['smoke','baseline','train'])
    for key in ('code','model','model-receipt','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--episodes');p.add_argument('--baseline-monitor');p.add_argument('--exchange-root')
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--max-batches',type=int,default=13)
    a=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('Model computation only on allocated Linux')
    code=Path(a.code).resolve();model=Path(a.model).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    actual=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    receipt_path=Path(a.model_receipt)
    staged=json.loads(receipt_path.read_text())
    if (actual!=FLORA_CODE or staged.get('model_revision')!=FLORA_MODEL or staged.get('completed') is not True
            or Path(staged['model_directory']).resolve()!=model or not staged.get('files')):
        raise ValueError('Official code and actual pinned local HF snapshot receipt required')
    bound_model=dict(path=str(receipt_path),sha256=sha(receipt_path))
    if a.mode!='train':
        episodes=json.loads(Path(a.episodes).read_text())
        if len(episodes)!=(1 if a.mode=='smoke' else 64):raise ValueError('Exact fresh smoke/baseline count required')
        outputs,_,scorer=score(episodes,out/'terminal.jsonl',model,code,bound_model)
        if a.mode=='baseline':
            monitor=[r for path in sorted(Path(a.baseline_monitor).glob('rank-*-round-*.json')) for r in json.loads(path.read_text())]
            truth={r['id']:r['simulator_final_success'] for r in monitor if r['case']['counted']}
            if set(truth)!={e['id'] for e in episodes}:raise ValueError('Paired native baseline truth differs')
            rows=[dict(id=r['id'],expected_score=r['expected_score'],simulator_final_success=truth[r['id']]) for r in outputs]
            validation=baseline_validation(rows)
            validation.update(completed=True,scorer_receipt=scorer,policy_optimizer_updates=0)
            write_new(out/'baseline-validation.json',validation)
        return
    if a.baseline_monitor:raise ValueError('Training scorer cannot receive simulator truth')
    if not 1<=a.max_batches<=13 or not a.exchange_root:raise ValueError('Finite admitted training service required')
    seen=set()
    while time.time()<a.deadline_unix and len(seen)<a.max_batches:
        for path in sorted(Path(a.exchange_root).glob('F/train/iteration-*/label-request.json')):
            if str(path) in seen:continue
            request=json.loads(path.read_text());batch=read_ref(request['batch']);validation=read_ref(request['baseline_validation'])
            if batch['arm']!='F':raise ValueError('Flora service only labels its own fresh F batch')
            target=out/f'iteration-{batch["iteration"]:04d}.jsonl'
            outputs,_,receipt=score(batch['episodes'],target,model,code,bound_model)
            labels=checked_labels(batch['episodes'],outputs,validation,receipt)
            receipt['input_batch_sha256']=digest(batch)
            commit_labels(path.parent/'ready.json',batch,labels,label_receipt=receipt,
                baseline_validation_sha256=request['baseline_validation']['sha256'])
            seen.add(str(path))
        if (Path(a.exchange_root)/'F'/'SERVICE-CLOSED.json').exists():break
        time.sleep(1)


if __name__=='__main__':main()
