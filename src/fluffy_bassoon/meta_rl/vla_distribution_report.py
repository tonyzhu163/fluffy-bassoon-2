"""Small read-only terminal/partial-panel report for the fixed9 interventions.

Reads committed group indices and optional saved arrays only. No model imports,
construction, inference, allocation, checkpoint selection or worker writes.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import fcntl
import gzip
import json
from pathlib import Path
import pickle

import numpy as np

from .vla_distribution_plan import METHODS,OPERATIONS
from .vla_source40_noise_report import sha,read,member,require


FORMAT='source40_distribution_diagnostic_saved_report_v1'


def paired(records,left,right):
    a={r['task_id']:r for r in records if r['method']==left}
    b={r['task_id']:r for r in records if r['method']==right}
    cells={k:[] for k in ('both_success','both_failure','left_only','right_only')}
    for task in sorted(a.keys() & b.keys()):
        x,y=a[task]['success'],b[task]['success']
        cells['both_success' if x and y else 'left_only' if x else 'right_only' if y else 'both_failure'].append(task)
    return dict(left=left,right=right,matched_queries=sum(map(len,cells.values())),
        counts={k:len(v) for k,v in cells.items()},task_ids=cells)


def summarize_samples(samples):
    result={}
    for method,fields in samples.items():
        lane={}
        for key,values in fields.items():
            a=np.concatenate([np.asarray(x).reshape(-1) for x in values]).astype(np.float64)
            lane[key]=dict(values=int(a.size),mean=float(a.mean()),std=float(a.std()),
                mean_abs=float(np.abs(a).mean()),minimum=float(a.min()),maximum=float(a.max()))
        result[method]=lane
    return result


def report(root,plan_path,*,read_arrays=False):
    root,plan_path=Path(root).resolve(),Path(plan_path).resolve()
    experiment=root/'experiment';plan=read(plan_path);contract=read(experiment/'contract.json')
    require(sha(plan_path)==contract['plan_file_sha256'] and plan['diagnostic_sha256']==contract['diagnostic_sha256'],
        'Report plan differs from the immutable run contract')
    require(len(plan['groups'])==63 and [e['case']['task_id'] for e in plan['cases']]==list(range(0,90,10)),
        'Require the predeclared fixed9 seven-method panel')
    with ExitStack() as stack:
        lock=stack.enter_context((root/'launcher.lock').open('r'))
        try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Worker still active; use existing progress metadata and run this report after its attempt ends') from error
        index=read(experiment/'index.json') if (experiment/'index.json').exists() else None
        final_by_group={r['group']['group_id']:r for r in index.get('completed_groups',[])} if index else {}
        records=[];samples=defaultdict(lambda:defaultdict(list));committed_actions=0;identities=defaultdict(set)
        runtime_ids=set()
        for ordinal,group in enumerate(plan['groups']):
            path=experiment/f'group-{ordinal:03d}'/'index.json'
            if not path.exists():continue
            value=read(path)
            require(value.get('completed') is True and value.get('weights_unchanged') is True
                and value['case']==group['case'] and len(value['episodes'])==1 and len(value['trials'])==1,
                'Committed diagnostic group differs from its case/weights/episode contract')
            episode=value['episodes'][0];trial=value['trials'][0]
            require((episode['method'],episode['role'],episode['support_budget'])==(group['method'],'query',0)
                and (trial['method'],trial['support_budget'])==(group['method'],0)
                and bool(trial['query_success'])==bool(episode['success']), 'Actual query/method receipt differs')
            if group['group_id'] in final_by_group:
                require(sha(path)==final_by_group[group['group_id']]['result_sha256'], 'Final query index hash differs')
                runtime_ids.update(final_by_group[group['group_id']]['runtime_ids'])
            identity=(episode['episode_random_seed'],episode['initial_state_vector_sha256'],episode['initial_observation_sha256'])
            identities[group['case']['task_id']].add(identity)
            actions=episode['primitive_steps']+episode['settling_steps'];committed_actions+=actions
            recovery=path.parent/'evaluation-recovery.pkl'
            require(sha(recovery)==value['latest_recovery_sha256'], 'Complete query recovery checksum differs')
            record=dict(task_id=group['case']['task_id'],method=group['method'],success=bool(episode['success']),
                primitive_steps=episode['primitive_steps'],settling_steps=episode['settling_steps'],actions=actions,
                index_path=str(path),index_sha256=sha(path),episode_recovery_sha256=sha(recovery),
                trace_sha256=episode['trace_sha256'],query_seed=episode['episode_random_seed'],
                initial_observation_sha256=episode['initial_observation_sha256'])
            if group['method']!='native':record['operation']=OPERATIONS[group['method']]
            if read_arrays:
                trace=member(path.parent,episode['trace']);require(sha(trace)==episode['trace_sha256'],'Saved trace checksum differs')
                with gzip.open(trace,'rb') as stream:rows=pickle.load(stream)
                record['saved_chunks']=len(rows)
                for row in rows:
                    if group['method']=='native':
                        samples['native']['flow_epsilon'].append(row['native_initial_noise']);continue
                    d=row.get('steering_distribution')
                    require(d and d['random_draws']==1 and d['parameter_updates']==0,'Missing saved actual distribution record')
                    for key in ('base_mean','learned_std','applied_mean','applied_std','epsilon','proposed_noise','actual_executed_noise'):
                        samples[group['method']][key].append(d[key])
                    samples[group['method']]['clipped_coordinate'].append(np.abs(d['proposed_noise'])>1)
            records.append(record)
        require(all(len(values)==1 for values in identities.values()),'Recorded initial-state/query RNG differs within a case')
        complete=len(records)==63 and index is not None and index.get('completed') is True
        if index and index.get('completed') is True:require(complete,'Final index says complete but the63 query records are missing')
        scores={method:dict(successes=sum(r['success'] for r in records if r['method']==method),
            observed_queries=sum(r['method']==method for r in records),declared_queries=9) for method in METHODS}
        comparisons=[paired(records,m,'native') for m in METHODS if m!='native']
        for objective in ('ad','dpt'):
            comparisons += [paired(records,objective+'_zero_mean_same_std',objective+'_full'),
                paired(records,objective+'_original_mean_std1',objective+'_full')]
        attempts=[]
        for directory in sorted((root/'runtime-attempts').glob('*')):
            row=dict(runtime_id=directory.name)
            for filename,label in (('original-equivalence.json','original_equivalence'),('parameters-before.json','parameters_before'),('parameters-after.json','parameters_after')):
                path=directory/filename
                if path.exists():row[label]=dict(path=str(path),sha256=sha(path),metadata=read(path))
            attempts.append(row)
        equivalence=[r.get('original_equivalence',{}).get('metadata',{}).get('controllers',{}) for r in attempts if 'original_equivalence' in r]
        original_verified=bool(equivalence) and all(all(c.get('bitwise_actions') and c.get('bitwise_sampler_state') and c.get('bitwise_cache')
            for c in checks.values()) and set(checks)=={'ad','dpt'} for checks in equivalence)
        parameter_checks=[r['parameters_after']['metadata'].get('unchanged') for r in attempts if 'parameters_after' in r]
        unchanged=bool(parameter_checks) and all(parameter_checks)
        if complete:require(original_verified and unchanged and index.get('all_base_parameter_bytes_unchanged') is True,
            'Complete diagnostic lacks unchanged parameters/original sampler-cache equivalence')
        global_recovery=experiment/'comparison-recovery.pkl'
        exit_code=int((root/'exit-code.txt').read_text()) if (root/'exit-code.txt').exists() else None
        per_task=[dict(task_id=t,results={r['method']:r['success'] for r in records if r['task_id']==t},
            complete_methods=sum(r['task_id']==t for r in records),declared_methods=7) for t in range(0,90,10)]
        return dict(format=FORMAT,status='complete_fixed9_panel' if complete else 'partial_fixed9_panel',complete=complete,
            completed_queries=len(records),declared_queries=63,terminal_exit_code=exit_code,scores=scores,per_task=per_task,
            paired_discordances=comparisons,committed_query_actions=committed_actions,
            charged_total_actions=index.get('charged_total_actions') if index else None,
            action_cost_note='Committed actions exclude any interrupted/unknown RPCs; final driver charge retains those costs separately',
            original_stochastic_actions_cache_rng_verified=original_verified,all_base_parameter_bytes_unchanged=unchanged,
            actual_runtime_ids=sorted(runtime_ids),strict_common_runtime_primary_eligible=index.get('strict_common_runtime_primary_eligible') if index else None,
            plan_sha256=sha(plan_path),contract_sha256=sha(experiment/'contract.json'),
            final_index_sha256=sha(experiment/'index.json') if index else None,
            comparison_recovery_sha256=sha(global_recovery) if global_recovery.exists() else None,
            fixed_source500_models=contract['fixed500_models'],attempt_proofs=attempts,queries=records,
            descriptive_distribution_summaries=summarize_samples(samples) if read_arrays else None,
            distribution_summary_note='Saved chunk-weighted marginal summaries only; clipping mean is fraction clipped; no conditional residual/calibration optimum inferred',
            interpretation='Fixed9 distribution intervention diagnostic; no new checkpoint/hyperparameter selection or general meta-performance claim')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--read-arrays',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.output:
        require(not a.output.resolve().is_relative_to(a.run_root.resolve()),'Write report outside immutable worker output')
        require(not a.output.exists(),'Report output already exists')
    result=report(a.run_root,a.plan,read_arrays=a.read_arrays)
    if a.output:
        with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('queries','attempt_proofs','fixed_source500_models')},indent=2,allow_nan=False))

if __name__=='__main__':main()
