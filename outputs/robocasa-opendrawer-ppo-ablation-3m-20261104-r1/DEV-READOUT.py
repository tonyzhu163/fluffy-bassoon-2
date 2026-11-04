"""Descriptive same-case DEV reads. The primary hypothesis tests use TEST512."""
import json
import statistics
import sys
from pathlib import Path

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs')
RUN=ROOT/'robocasa-opendrawer-ppo-ablation-3m-20261104-r1'
STAGE=ROOT/'robocasa-ppo-grpo-meta-stage1-20261104-r1/ppo/attempt-01'
ARMS={'stage1':STAGE,'E':RUN/'E/attempt-02','C':RUN/'C/attempt-01','D':RUN/'D/attempt-01'}


def read(path):return json.loads(path.read_text())


def panel(folder, step):
    result={}
    paths=sorted(folder.glob(f'environment/rank-*/eval_step_{step:03d}-i{step:03d}-round00-episodes.json'))
    if not paths and step==40:
        paths=sorted(folder.glob('environment/rank-*/after-i040-round00-episodes.json'))
    assert len(paths)==8, (folder,step,len(paths))
    for p in paths:
        j=read(p)
        assert j['completed'] and len(j['rows'])==8
        initial=read(p.with_name(p.name.replace('-episodes.json','-initial.json')))
        report=initial['report']
        assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
        canonical=read(Path(initial['canonical']))
        orig=read(STAGE/'canonical-eval'/p.parent.name/'complete.json')
        assert canonical['snapshot_sha256']==orig['snapshot_sha256']
        for row in j['rows']:
            assert row['native_counts']['controls']-row['native_counts']['reset_controls']==600
            assert row['seed'] not in result
            result[row['seed']]=int(row['success_once'])
    assert set(result)==set(range(610100,610164))
    return result


def main(steps):
    out={}
    for step in steps:
        controls=panel(STAGE,step)
        rows={}
        for name,folder in ARMS.items():
            outcomes=panel(folder,step)
            up=[read(folder/f'updates/iteration-{i:03d}-rank-00.json') for i in range(1,step+1)]
            replay=[read(folder/f'logprob-parity/iteration-{i:03d}-rank-{r:02d}.json') for i in range(1,step+1) for r in range(2)]
            assert all(x['passed'] and x['value']['passed'] for x in replay)
            rows[name]=dict(first_successes=sum(outcomes.values()),cases=64,wins_vs_stage1=sum(outcomes[s]>controls[s] for s in controls),
                losses_vs_stage1=sum(outcomes[s]<controls[s] for s in controls),
                mean_clip_fraction=statistics.mean(u['metrics']['actor/clip_fraction'] for u in up),
                mean_logged_approx_KL=statistics.mean(u['metrics']['actor/approx_kl'] for u in up),
                optimizer_steps=up[-1]['optimizer_steps_after'],passed_replays=len(replay))
        out[step]=rows
    print(json.dumps(dict(descriptive=True,common_physical_public_reset_and_case_ids=True,results=out),indent=2))


if __name__=='__main__':
    main(list(map(int,sys.argv[1:])))
