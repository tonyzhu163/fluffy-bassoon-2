"""Mixed-horizon reward attribution: all eighty grades before joint GAE."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl.flora_terminal_shards import joint_episode_shard, gather_joint_terminal_outputs
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest, flora_jobs, parse_outputs, sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS, protocol
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS


def joint_bank():
    episodes,rows,protocols=[],[],{}
    for i,task in enumerate(list(PPO_TASKS)*16):
        key=f'joint-{i:03d}'
        proto=protocol(task,baseline_count=50);horizon=proto['horizon'];lower=1+i%4
        episode=dict(id=key,task=f'Native instruction for {task}.',camera_order=list(REWARD_CAMERAS),
            info_on_video=False,extra_info_on_video=[],raw_rgb_only=True,
            success_termination_disabled=True,horizon=horizon,completed=True,
            videos=[dict(camera=c,path=f'/qualified/{key}-{c}.mp4',sha256='a'*64) for c in REWARD_CAMERAS],
            frame_clock=dict(frame_count=horizon+1,native_fps=20.,encoded_fps=20.,resampled=False,
                physical_steps=horizon,initial_frame_included=True,timestamp_sha256='b'*64),
            rank=i//10,round=0,local_env=i%10)
        episodes.append(episode);protocols[key]=proto
        rows.append(dict(id=key,task=episode['task'],score=lower,expected_score=lower+.5,
            progress=(lower-.5)/4,probabilities={str(k):.5 if k in (lower,lower+1) else 0. for k in range(1,6)}))
    return episodes,rows,protocols


def test_joint80_mixed_horizons_preserve_all_grades_and_reject_partial_or_wrong_tasks():
    episodes,rows,protocols=joint_bank()
    shards={r:list(reversed(rows[r*10:(r+1)*10])) for r in range(8)}
    for rank in range(8):
        subset,mapping=joint_episode_shard(episodes,rank,protocols)
        assert subset==episodes[rank*10:(rank+1)*10] and len(mapping)==10
        assert [mapping[e['id']]['task'] for e in subset]==list(PPO_TASKS)*2
    ordered,labels=gather_joint_terminal_outputs(episodes,shards,protocols)
    assert ordered==rows and labels==parse_outputs(episodes,rows,episode_protocols=protocols)
    assert all(labels[row['id']]==(row['expected_score']-1)/4 for row in rows)
    for kind in ('missing_rank','missing_case','duplicate','wrong_rank'):
        bad=deepcopy(shards)
        if kind=='missing_rank':bad.pop(7)
        elif kind=='missing_case':bad[7].pop()
        elif kind=='duplicate':bad[7][-1]=bad[7][0]
        else:bad[0],bad[1]=bad[1],bad[0]
        with pytest.raises(ValueError):gather_joint_terminal_outputs(episodes,bad,protocols)
    for kind in ('missing','OOD','wrong_horizon','wrong_order'):
        mapping=deepcopy(protocols)
        if kind=='missing':mapping.pop(episodes[0]['id'])
        elif kind=='OOD':mapping[episodes[0]['id']]=protocol('CloseFridge',baseline_count=50)
        elif kind=='wrong_horizon':mapping[episodes[0]['id']]['horizon']=750
        else:mapping[episodes[0]['id']],mapping[episodes[1]['id']]=mapping[episodes[1]['id']],mapping[episodes[0]['id']]
        with pytest.raises(ValueError):joint_episode_shard(episodes,0,mapping)


def test_joint80_service_binds_task_map_pending_source_and_waits_for_all_official_rows(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[1]/'scripts/diagnostics/robocasa_residual_score.py'
    spec=importlib.util.spec_from_file_location('joint80_score_under_test',source)
    service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
    original_sha=service.sha
    monkeypatch.setattr(service,'sha',lambda p:'a'*64 if str(p).startswith('/qualified/') else original_sha(p))
    episodes,rows,protocols=joint_bank();by_id={r['id']:r for r in rows}
    class Worker:
        model_load_count=1;completed_batches=0;batch_size=8;gpu_memory_utilization=.78
        def score_jobs(self,jobs,path):
            assert all(set(job)=={'id','task','videos'} for job in jobs)
            result=[by_id[job['id']] for job in jobs]
            path.write_text(''.join(json.dumps(r)+'\n' for r in result))
            self.completed_batches+=1
            return result
    worker=Worker();pending=tmp_path/'complete-pending-fixture.json';pending.write_text('{}\n')
    request=dict(episodes=episodes,episode_protocols=protocols,episode_protocols_sha256=digest(protocols),
        pending_checkpoint=dict(path=str(pending),sha256=sha(pending)),source_commit='c'*40,
        jobs=flora_jobs(episodes,episode_protocols=protocols))
    request['jobs_sha256']=digest(request['jobs'])
    path=tmp_path/'labels-001/request.json';write_new(path,request)
    base=dict(task='joint_five',source_commit='c'*40,model_receipt=dict(path='/same/model/READY.json',sha256='d'*64),
        deadline_unix=time.time()+30,closure_seconds=1)
    progress=SimpleNamespace(emit=lambda **kw:None)
    for rank in list(range(1,8))+[0]:
        assert not (path.parent/'ready.json').exists()
        out=tmp_path/f'rank-{rank}';out.mkdir()
        service.serve_shard(dict(base,annotation_rank=rank),worker,out,None,progress,path,request)
    ready=json.loads((path.parent/'ready.json').read_text())
    assert ready['pending_checkpoint_sha256']==sha(pending)
    assert ready['episode_protocols_sha256']==digest(protocols) and ready['source_commit']=='c'*40
    assert ready['jobs_sha256']==digest(request['jobs']) and len(ready['shard_receipts'])==8
    assert json.loads(Path(ready['outputs']['path']).read_text())==rows
    assert worker.model_load_count==1 and worker.completed_batches==8
    for field in ('episode_protocols_sha256','source_commit'):
        bad=deepcopy(request);bad[field]='0'*40
        with pytest.raises(ValueError):
            service.serve_shard(dict(base,annotation_rank=0),worker,tmp_path,None,progress,path,bad)
    assert worker.completed_batches==8


def test_streamed_grades_are_reused_only_for_identical_jobs_and_finish_at_request(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[1]/'scripts/diagnostics/robocasa_residual_score.py'
    spec=importlib.util.spec_from_file_location('streamed_score_under_test',source)
    service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
    original_sha=service.sha
    monkeypatch.setattr(service,'sha',lambda p:'a'*64 if str(p).startswith('/qualified/') else original_sha(p))
    episodes,rows,protocols=joint_bank();by_id={r['id']:r for r in rows}
    scored=[]
    class Worker:
        model_load_count=1;completed_batches=0;batch_size=8;gpu_memory_utilization=.78
        def score_jobs(self,jobs,path):
            scored.extend(job['id'] for job in jobs)
            result=[by_id[job['id']] for job in jobs]
            path.write_text(''.join(json.dumps(r)+'\n' for r in result))
            self.completed_batches+=1
            return result
    worker=Worker();policy=tmp_path/'policy'
    videos=policy/'train/videos/parallel-000000/rank-00/videos';videos.mkdir(parents=True)
    for episode in episodes[:10]:
        write_new(videos/(episode['id']+'.json'),episode)
    out=tmp_path/'scorer-00';out.mkdir()
    profile=dict(task='joint_five',source_commit='c'*40,annotation_rank=0,F_output=str(policy),stream_scoring=True,
        model_receipt=dict(path='/same/model/READY.json',sha256='d'*64),deadline_unix=time.time()+30,closure_seconds=1)
    progress=SimpleNamespace(emit=lambda **kw:None)
    service.prescore(profile,worker,out,progress)
    assert scored==[e['id'] for e in episodes[:8]]  # One engine batch while the bank still runs.
    stale=out/'prescore'/(episodes[1]['id']+'.json');value=json.loads(stale.read_text())
    value['video_refs'][0]['sha256']='e'*64;stale.write_text(json.dumps(value))
    subset,mapping=joint_episode_shard(episodes,0,protocols)
    result,labels,receipt=service.score(worker,subset,out/'labels-001.jsonl',profile['model_receipt'],
        episode_protocols=mapping,cache=out/'prescore')
    assert result==rows[:10] and labels==parse_outputs(subset,rows[:10],episode_protocols=mapping)
    assert scored[8:]==[episodes[1]['id'],episodes[8]['id'],episodes[9]['id']]
    assert receipt['streamed']['prescored']==7 and receipt['streamed']['scored_at_request']==3
    assert receipt['episode_protocols_sha256']==digest(mapping)
