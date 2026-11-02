"""Prepare new causal-prefix scoring jobs from the retained video manifest.

Preparation only. No simulator replay, model inference or production change.
Truth stays in the original sidecar, never in the scoring jobs or prompt.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--horizon',type=int,default=400)
    p.add_argument('--limit',type=int,default=128);args=p.parse_args()
    if args.horizon<50 or args.horizon%50 or args.limit<1:
        raise ValueError('Positive complete-chunk horizon and video bound required')
    tasks=('OpenDrawer','PickPlaceDrawerToCounter','CoffeeSetupMug','TurnOffStove')
    items=[v for v in json.loads(args.videos.read_text()) if v['task_key'] in tasks and v['horizon']>=args.horizon]
    # Equal per-task quotas, independent of simulator labels or score extremes.
    per_task={task:sorted((v for v in items if v['task_key']==task),
                key=lambda v:hashlib.sha256(v['id'].encode()).hexdigest()) for task in tasks}
    selected=[]
    while len(selected)<args.limit and any(per_task.values()):
        for task in tasks:
            if per_task[task] and len(selected)<args.limit:
                selected.append(per_task[task].pop(0))
    jobs=[]
    for video in selected:
        if [v['camera'] for v in video['videos']]!=['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']:
            raise ValueError('Require original synchronized three-view recordings')
        for step in range(50,args.horizon+1,50):
            jobs.append(dict(id=video['id']+f'/prefix-{step:04d}',task=video['native_instruction'],
                videos=[v['path'] for v in video['videos']],prefix_controls=step,initial_frame_included=True))
    output=dict(format='flora_causal_prefix_pilot_v1',production_admitted=False,
        source=dict(path=str(args.videos),sha256=hashlib.sha256(args.videos.read_bytes()).hexdigest()),
        selected_video_refs=[dict(id=v['id'],task_key=v['task_key'],videos=v['videos']) for v in selected],
        horizon=args.horizon,stride=50,video_count=len(selected),label_count=len(jobs),jobs=jobs,
        truth_policy='Native first-success sidecars are an independent offline join; missing traces are not terminal-success proxies.',
        execution='Prepared only; qualify causal sampling and bind a bounded GPU allocation before scoring.')
    with args.output.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(videos=len(selected),labels=len(jobs),output=str(args.output))))


if __name__=='__main__':main()
