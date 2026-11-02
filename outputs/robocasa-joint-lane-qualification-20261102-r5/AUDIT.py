"""Read-only CPU audit of the completed finite colocation qualification."""
import json
import os
from pathlib import Path
import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new

assert not torch.cuda.is_initialized()
root=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-joint-lane-qualification-20261102-r5')
out=root/'gpu'
read=lambda p:json.loads(p.read_text())
result=read(out/'RESULT.json');ready=read(out/'engine-ready.json')
assert result['completed'] and result['child_exit_codes']==[0,0]
assert result['source_commit']=='6de858c3c77d1805120c196e2941c046f48f03bd'
assert result['optimizer_updates']==0 and result['selected_lanes']==4
assert result['policy_controls']==6000 and result['records']==120 and result['episodes']==10
assert ready['gpu_memory_utilization']==.75 and ready['batch_size']==8 and ready['warmup_videos']==8
assert not result['headroom'][0]['admitted'] and result['headroom'][1]['admitted']
directory=out/'lanes-4/train/videos/parallel-000000/rank-00'
pending=torch.load(directory/'result.pt',map_location='cpu',weights_only=False)
request=torch.load(out/'request-lanes-4.pt',map_location='cpu',weights_only=False)
episodes,records=pending['episodes'],pending['records']
assert len(episodes)==len(request['cases'])==10 and len(records)==120
assert sum(r['done'] for r in records)==10 and all(r['reward']==0 for r in records)
assert all(e['horizon']==e['frame_clock']['physical_steps']==600 and e['frame_clock']['frame_count']==601
           and e['frame_clock']['initial_frame_included'] and not e['frame_clock']['resampled']
           and len(e['videos'])==3 for e in episodes)
assert [c['task'] for c in request['cases']]==['OpenDrawer','PickPlaceCounterToCabinet',
    'PickPlaceDrawerToCounter','TurnOffStove','CoffeeSetupMug']*2
sidecars=[];monitors=[]
for episode in episodes:
    path=directory/'videos'/(episode['id']+'.json')
    assert read(path)==episode
    sidecars.append(path)
    for ref in episode['videos']:
        assert sha(ref['path'])==ref['sha256']
    monitor=read(out/'monitoring'/(episode['id']+'.json'))
    assert monitor['native_counts']['controls']-monitor['native_counts']['reset_controls']==600
    assert all(1<=step<=600 for step in monitor['success_steps'])
    monitors.append(monitor)
labels=[json.loads(line) for line in (out/'scorer-4/final.jsonl').read_text().splitlines()]
assert [row['id'] for row in labels]==[e['id'] for e in episodes]
streamed=[read(p) for p in (out/'scorer-4/prescore').glob('*.json')]
assert len(streamed)==10 and {row['row']['id'] for row in streamed}=={e['id'] for e in episodes}
assert result['scoring']['receipt']['streamed']['prescored']==10
assert result['scoring']['receipt']['streamed']['scored_at_request']==0
assert result['scoring']['model_load_count']==1
calls=[]
for p in (out/'actions-lanes-4').rglob('*.json'):
    value=read(p)
    if value.get('format')=='rlinf_native_ppo_actions_v1':calls.extend(value['calls'])
assert calls and all(c['resolved'] for c in calls)
charged=sum(c['executed_or_conservative'] for c in calls)
assert charged==sum(m['native_counts']['controls'] for m in monitors)
first=min(c['started_unix'] for c in calls)
last=max(p.stat().st_mtime for p in sidecars)
seconds=last-first
samples=[json.loads(line) for line in (out/'memory.jsonl').read_text().splitlines()]
write_new(root/'PARENT-REVIEW.json',dict(completed=True,source_commit=result['source_commit'],job_id='job-q7okv1fl1gda',
    GPU_count=1,episodes=10,policy_controls=6000,all_in_controls=charged,records=120,optimizer_updates=0,
    streamed_grades=10,late_grades=0,model_load_count=1,unresolved_intents=0,
    first_constructor_intent_unix=first,last_complete_video_sidecar_unix=last,
    collection_and_video_seconds=seconds,collection_controls_per_second=6000/seconds,
    native_load_and_collect_seconds=result['inclusive_native_load_collect_seconds'],
    native_load_and_collect_controls_per_second=result['inclusive_controls_per_second'],
    collection_and_scoring_seconds=result['combined_collect_score_seconds'],
    sampled_peak_GPU_MiB=result['sampled_peak_used_mib'],sampled_peak_GPU_GiB=result['sampled_peak_used_mib']/1024,
    minimum_free_GPU_MiB=result['minimum_free_mib'],NVML_samples=len(samples),NVML_sample_errors=sum('error' in r for r in samples),
    all_video_hashes_checked=True,pending_checkpoint_sha256=sha(directory/'result.pt'),
    scope='OneGPU ten-case600-control collection with live official Flora; no8GPU joint iteration or SGD qualification',
    production_headroom=result['headroom'],production_memory_guard_passed=True,
    production_four_lane_headroom_margin_MiB=ready['memory_mib'][1]-21973,
    streamed_scoring_call_sizes=[len(p.read_text().splitlines()) for p in sorted((out/'scorer-4/prescore').glob('*.jsonl'))],
    steady_engine_batch_limit=8,gpu_memory_utilization=.75,
    cold_engine_including_first8_scores_seconds=ready['cold_seconds']))
print('Full episodes, frame clocks, video hashes, ten streamed grades and resolved physical journal verified.')
