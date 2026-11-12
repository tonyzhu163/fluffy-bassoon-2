"""Descriptive scene-level evidence; repeated judgments are not independent scenes."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--judgments', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    native = json.loads((args.native/'complete.json').read_text())
    responses = json.loads((args.judgments/'complete.json').read_text())
    branches = {(r['seed'],r['query'],r['candidate']):r for r in native['branches']}
    keys = sorted({k[:2] for k in branches})
    assert len(keys) == 8 and len(branches) == 16 and len(responses) == 72
    conditions = ('minus','plus','donor')
    def utility(seed,q,c):
        b = branches[(seed,q,c)]
        return dict(success=float(b['success_once']), final=float(b['final_success']), native_return=b['native_return'])
    scored = []
    for r in responses:
        chosen = r['candidate'] if r.get('valid') and r.get('candidate') is not None else 0
        scored.append(dict(seed=r['seed'],query=r['query'],rep=r['repetition'],condition=r['condition'],
            valid=r['valid'],candidate=r.get('candidate'),fallback=chosen,**utility(r['seed'],r['query'],chosen)))
    query_results = []
    for seed,q in keys:
        a,b = branches[(seed,q,0)],branches[(seed,q,1)]
        preferences = {}
        for condition in conditions:
            rows = [r for r in scored if (r['seed'],r['query'],r['condition']) == (seed,q,condition)]
            choices = [r['candidate'] for r in rows if r['valid'] and r['candidate'] is not None]
            preferences[condition] = dict(choices=choices, abstentions=sum(r['valid'] and r['candidate'] is None for r in rows),
                invalid=sum(not r['valid'] for r in rows), p_candidate1=mean(choices) if choices else None,
                mean_success_with_base_fallback=mean(r['success'] for r in rows),
                mean_return_with_base_fallback=mean(r['native_return'] for r in rows))
        pminus,pplus = (preferences[c]['p_candidate1'] for c in ('minus','plus'))
        query_results.append(dict(seed=seed,query=q,success_before_query=a['success_before_query'],
            success=[a['success_once'],b['success_once']], final=[a['final_success'],b['final_success']],
            returns=[a['native_return'],b['native_return']],preferences=preferences,
            delta_teacher=None if pminus is None or pplus is None else pplus-pminus))
    def summarize(rows):
        return dict(n=len(rows), success=mean(r['success'] for r in rows), final=mean(r['final'] for r in rows),
                    native_return=mean(r['native_return'] for r in rows))
    aggregate = {condition:summarize([r for r in scored if r['condition']==condition]) for condition in conditions}
    aggregate['base_candidate0'] = summarize([utility(*key,0) for key in keys])
    aggregate['uniform_candidate'] = summarize([utility(*key,c) for key in keys for c in (0,1)])
    aggregate['oracle_success'] = mean(max(utility(*key,c)['success'] for c in (0,1)) for key in keys)
    scene_effects = []
    for seed in sorted({k[0] for k in keys}):
        m = {c:summarize([r for r in scored if r['seed']==seed and r['condition']==c]) for c in conditions}
        scene_effects.append(dict(seed=seed,means=m,
            plus_minus_success=m['plus']['success']-m['minus']['success'],
            plus_minus_return=m['plus']['native_return']-m['minus']['native_return']))
    report = dict(scenes=4,queries=8,repeated_calls=72,
        valid_responses=sum(r['valid'] for r in scored),
        abstentions=sum(r['valid'] and r['candidate'] is None for r in scored),
        distinct_success_outcomes=sum(r['success'][0] != r['success'][1] for r in query_results),
        distinct_final_outcomes=sum(r['final'][0] != r['final'][1] for r in query_results),
        distinct_native_returns=sum(r['returns'][0] != r['returns'][1] for r in query_results),
        already_successful_queries=sum(r['success_before_query'] for r in query_results),
        estimable_teacher_updates=sum(r['delta_teacher'] is not None for r in query_results),
        fallback='Use original candidate zero for abstentions or invalid responses; report their counts separately.',
        aggregate=aggregate,scene_effects=scene_effects,queries=query_results,
        native_wall_seconds=native['wall_seconds'], inference_wall_seconds_sum=sum(r['wall_seconds'] for r in responses),
        statistical_unit='scene, n=4; repeated choices are descriptive, not additional independent episodes')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    # Export only delivered action labels, with public context separated from
    # future native outcomes. Keep tied/unsuccessful pairs in the corpus.
    with args.output.with_name('training-pairs.jsonl').open('x') as stream:
        for row in query_results:
            seed,q = row['seed'],row['query']
            raw = json.loads((args.native/str(seed)/f'query-{q}.json').read_text())
            public_fields = ('prompt','current_state','candidate_commands','action_order','controller',
                             'history_controls','history_commands','history_states','budget')
            context = {k:raw[k] for k in public_fields}
            context['images'] = [str((args.native/str(seed)/'public'/f'{step:04d}.png').resolve())
                                 for step in raw['history_controls']]
            example = dict(format='teacher_evidence_pair_v1',seed=seed,query=q,context=context,
                delivered_action_mask=[True]*5+[False]*45,
                teacher_targets=dict(preferences={condition:{k:v for k,v in values.items()
                    if k in ('choices','abstentions','invalid','p_candidate1')}
                    for condition,values in row['preferences'].items()},delta_teacher=row['delta_teacher']),
                future_outcome_labels=dict(success=row['success'],final=row['final'],native_return=row['returns'],
                                          success_before_query=row['success_before_query']),
                success_contrast_available=row['success'][0] != row['success'][1],
                teacher_update_available=row['delta_teacher'] is not None,
                note='Outcome labels and teacher targets are offline supervision, never support inputs.')
            stream.write(json.dumps(example,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('queries','scene_effects')},indent=2))


if __name__ == '__main__':
    main()
