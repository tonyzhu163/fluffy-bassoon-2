"""Independently recount the experiment from native, per-episode receipts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from scipy.stats import binomtest

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-ppo4-grpo40-test512-20261105-r1')
FOLDER=ROOT/'evaluation/attempt-01'
LABELS=('SFT','PPO4','GRPO')
FIRST=85411000


def read(p):return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    m=read(ROOT/'MANIFEST.json')
    hashed=sha(ROOT/'MANIFEST.json')
    combined=read(FOLDER/'RESULT.json')
    assert combined['completed'] and hashed==combined['manifest_sha256']
    assert all(sha(Path(cp['marker']))==cp['marker_sha256'] for cp in m['policies'].values())
    cases={k:{} for k in LABELS}; ends={k:0 for k in LABELS}
    gpu=set();panels=controls=reset=journal=matched_starts=0
    for i in range(2):
        worker=FOLDER/f'node-{i:02d}'
        done=read(worker/'RESULT.json'); assert done['manifest_sha256']==hashed
        probe=read(worker/'GPU-PROBE.json'); assert probe['status']=='passed' and len(probe['devices'])==1
        gpu.add(probe['devices'][0]['cuda_uuid'])
        assert read(worker/'initial-parity.json')[0]['passed']
        calls=read(worker/'accounting/rank-00'/f'{worker.name}.json')['calls']
        assert all(r['resolved'] for r in calls)
        journal+=sum(r['executed_or_conservative'] for r in calls)
        for b in range(i,64,2):
            d=worker/'banks'/f'bank-{b:03d}'
            seeds=list(range(FIRST+8*b,FIRST+8*b+8))
            refs=read(d/'rng-reference.json');assert len(refs)==12
            canonical=d/'canonical-eval/rank-00/complete.json'
            assert sha(canonical.parent/'initial.pt')==read(canonical)['snapshot_sha256']
            assert read(canonical)['case']['seeds']==seeds
            for label in m['sequential_order'][b]:
                p=read(d/f'{label}.json');policy='SFT' if label=='repeat' else label
                assert p['load']['strict_load'] and p['load']['sha256']==m['policies'][policy]['sha256']
                assert p['sampler_seed']==95411000+b and p['seeds']==seeds
                ep=Path(p['episodes_path']);assert sha(ep)==p['episodes_sha256']
                ir=read(ep.with_name(ep.name.replace('-episodes.json','-initial.json')))
                assert ir['canonical']==str(canonical) and ir['report']['matched'] and ir['report']['physical_state_exact'] and ir['report']['public_state_exact']
                matched_starts+=8
                for call in range(1,13):
                    s=read(worker/'sampler'/f'{p["phase"]}-{call:02d}.json')
                    assert [s['before_rng_sha256'],s['after_rng_sha256']]==refs[call-1]
                    assert s['actions_sha256']==p['sampler_actions'][call-1] and s['actions_shape']==[8,50,12]
                rows=read(ep)['rows'];assert p['episodes']==rows and [r['seed'] for r in rows]==seeds
                for row in rows:
                    c=row['native_counts']; assert c['controls']-c['reset_controls']==600
                    controls+=600; reset+=c['reset_controls']
                    if label!='repeat':
                        assert row['seed'] not in cases[label]
                        cases[label][row['seed']]=int(row['success_once'])
                        ends[label]+=int(row['success_at_end'])
                panels+=1
    assert len(gpu)==2 and panels==193 and controls==926400 and journal==controls+reset==combined['charged_controls']
    assert all(set(a)==set(range(FIRST,FIRST+512)) for a in cases.values())
    totals={k:sum(v.values()) for k,v in cases.items()}
    assert totals==combined['first_successes'] and ends==combined['end_successes']
    tests={}
    for first,second,label in (('GRPO','PPO4','primary_PPO4_minus_GRPO'),('SFT','PPO4','PPO4_vs_SFT'),('SFT','GRPO','GRPO_vs_SFT')):
        delta=Counter(cases[second][seed]-cases[first][seed] for seed in cases[second])
        wins,losses=delta[1],delta[-1]
        exact=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
        r=combined['primary_PPO4_minus_GRPO'] if label=='primary_PPO4_minus_GRPO' else combined['descriptive_versus_SFT'][second]
        assert (wins,losses)==(r['wins'],r['losses']) and abs(exact-r['exact_McNemar_p'])<1e-12
        tests[label]=dict(wins=wins,losses=losses,exact_p=exact,role='primary' if label=='primary_PPO4_minus_GRPO' else 'descriptive')
    return dict(verified=True,manifest_sha256=hashed,combined_sha256=sha(FOLDER/'RESULT.json'),cases_per_policy=512,
                policy_episodes=panels*8,matched_physical_public_starts=matched_starts,matched_sampler_chunk_calls=panels*12,
                accounted_policy_controls=controls,reset_controls=reset,charged_controls=journal,distinct_accelerators=len(gpu),
                first_successes=totals,end_successes=ends,independent_paired_tests=tests)


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
