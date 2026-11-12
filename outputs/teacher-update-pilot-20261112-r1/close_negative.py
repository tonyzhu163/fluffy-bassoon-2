"""Close the zero-teacher-signal pilot without admitting incomplete comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ('run','judgments','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    data = args.run/'native04/data'
    calls = read(args.judgments)
    expected_calls = {(s,q,r,c) for s in range(120700,120704) for q in (100,250)
                      for r in range(3) for c in ('minus','plus','donor')}
    assert {(r['seed'],r['query'],r['repetition'],r['condition']) for r in calls} == expected_calls
    assert len(calls)==72 and all(r['valid'] and r['answer']['choice']=='abstain' and r['candidate'] is None for r in calls)
    contexts, item_types, image_checks = {},set(),0
    tokens = {'input_tokens':0,'cached_input_tokens':0,'output_tokens':0}
    for r in calls:
        folder = args.run/'teacher-records'/f"astra-{r['seed']}"/f"{r['seed']}-{r['query']}-r{r['repetition']}-{r['condition']}"
        assert sha(folder/'prompt.txt')==r['prompt_sha256']
        context = json.loads((folder/'prompt.txt').read_text().split('\n\n',1)[1])
        assert set(context)=={'task','current_control','control_budget','current_state','state_order',
                              'candidate_commands','action_order','controller','history'}
        history = context.pop('history')
        if r['condition']=='minus':
            assert isinstance(history,str) and 'unavailable' in history
        else:
            assert set(history)=={'relationship','task','image_controls','state_controls','states','command_start','commands'}
            assert len(history['commands'])==50 and len(history['states'])==51
            assert history['image_controls']==list(range(r['query']-50,r['query']+1,10))
            assert history['command_start']==r['query']-50
        contexts[(r['seed'],r['query'],r['repetition'],r['condition'])]=json.dumps(context,sort_keys=True)
        assert sha(data/str(r['seed'])/f"query-{r['query']}.json")==r['source_query_sha256']
        assert len(r['images'])==(1 if r['condition']=='minus' else 7)
        for item in r['images']:
            path = data.joinpath(*Path(item['path']).parts[-3:])
            assert sha(path)==item['sha256'] and int(path.stem)<=r['query']
            image_checks+=1
        for line in (folder/'events.jsonl').read_text().splitlines():
            event= json.loads(line)
            if 'item' in event:
                item_types.add(event['item']['type'])
            if event.get('type')=='turn.completed':
                for key in tokens:
                    tokens[key]+=event.get('usage',{}).get(key,0)
    assert item_types <= {'agent_message','reasoning'}, item_types
    for seed in range(120700,120704):
        for q in (100,250):
            for rep in range(3):
                group=[r for r in calls if (r['seed'],r['query'],r['repetition'])==(seed,q,rep)]
                assert len({tuple(r['order']) for r in group})==1
                assert len({r['images'][0]['sha256'] for r in group})==1
                assert len({contexts[(seed,q,rep,r['condition'])] for r in group})==1
    rows=read(data/'branches.json')
    assert len(rows)==13
    branches={(r['seed'],r['query'],r['candidate']):r for r in rows}
    pairs=[]
    for row in rows:
        assert row['fork_exact'] and row['global_rng_restored']
        assert row['baseline_continuation_exact']==(row['candidate']==0)
        raw=read(data/str(row['seed'])/f"query-{row['query']}.json")
        with np.load(data/str(row['seed'])/f"branch-{row['query']}-{row['candidate']}.npz") as trace:
            np.testing.assert_array_equal(trace['commands'][:5],raw['candidate_commands'][row['candidate']])
            assert len(trace['commands'])==row['controls']==500-row['query']
            assert bool(trace['successes'].any())==row['success_once']
            assert bool(trace['successes'][-1])==row['final_success']
            assert float(trace['rewards'].sum())==row['native_return']
    for seed in range(120700,120704):
        for q in (100,250):
            if all((seed,q,c) in branches for c in (0,1)):
                pair=[branches[(seed,q,c)] for c in (0,1)]
                pairs.append(dict(seed=seed,query=q,success=[r['success_once'] for r in pair],
                    native_return=[r['native_return'] for r in pair],success_before_query=pair[0]['success_before_query']))
    assert len(pairs)==6
    baselines=[dict(seed=seed,**{k:read(data/str(seed)/'baseline.json')[k]
                for k in ('success_once','final_success','native_return')}) for seed in range(120700,120704)]
    excluded=read(args.run/'native05/data/branches.json')
    runs={name:read(args.run/name/'exit.json') for name in
          ('native01','native02','native03','render-gc01','native04','native05')}
    summary=dict(status='closed_negative_with_incomplete_native_panel',planned_scenes=4,planned_queries=8,
        baseline_successes=sum(r['success_once'] for r in baselines),baselines=baselines,
        qualified_branches=13,complete_candidate_zero_replays=7,qualified_paired_queries=6,
        missing_qualified_branches=3,excluded_cross_process_branches=len(excluded),
        native_pairs=pairs,success_contrasts=sum(p['success'][0]!=p['success'][1] for p in pairs),
        native_return_contrasts=sum(p['native_return'][0]!=p['native_return'][1] for p in pairs),
        already_successful_qualified_queries=sum(p['success_before_query'] for p in pairs),
        teacher_calls=72,valid_teacher_calls=72,abstentions=72,estimable_teacher_updates=0,
        teacher_condition_counts={c:sum(r['condition']==c for r in calls) for c in ('minus','plus','donor')},
        student_updates=0,scope='Native OpenDrawer pretrain; 500-control pilot budget; four exploratory scenes.',
        interpretation='No teacher preference frequencies or paired policy-change labels can be estimated. '
            'Native05 restart changed inferred actions despite matched query observations and saved noise; '
            'its comparisons are excluded. The pilot does not test learned TTT transfer.')
    audit=dict(public_only_prompts=72,matched_context_triples=24,image_hash_checks=image_checks,
        teacher_item_types=sorted(item_types),external_teacher_tool_calls=0,teacher_tokens=tokens,
        prefix_state_checks=sum(r['prefix_state_checks'] for r in rows),candidate_prefix_checks=13,
        teacher_records_sha256=sha(args.judgments),worker_runs=runs,
        worker_gpu_hours=sum(r['ended']-r['started'] for r in runs.values())/3600,
        conservative_reservation_gpu_hours=(max(r['ended'] for r in runs.values())-min(r['started'] for r in runs.values()))/3600)
    for name,value in (('summary.json',summary),('audit.json',audit)):
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n')
    with (args.output/'diagnostic-pairs.jsonl').open('x') as stream:
        for pair in pairs:
            raw=read(data/str(pair['seed'])/f"query-{pair['query']}.json")
            public={k:raw[k] for k in ('prompt','current_state','candidate_commands','action_order','controller',
                                      'history_controls','history_commands','history_states','budget')}
            row=dict(seed=pair['seed'],query=pair['query'],context=public,
                delivered_action_mask=[True]*5+[False]*45,teacher_update_available=False,
                teacher_targets=dict(p_minus=None,p_plus=None,delta_teacher=None,
                                     abstentions=dict(minus=3,plus=3,donor=3)),future_outcome_labels=pair,
                note='Diagnostic only: all teacher calls abstained. Missing targets are not zero updates.')
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
