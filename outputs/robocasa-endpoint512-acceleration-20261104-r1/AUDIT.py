"""Independent raw audit across immutable endpoint execution migrations.

The per-panel state/image/sampler/journal checks are adapted from the original
FINAL-AUDIT.py (6c95116e). No model forward or simulator step is performed.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from scipy.stats import binomtest

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def wait_file(path,deadline):
    while not path.exists():
        assert time.time()<deadline, f'No completed artifact before deadline: {path}'
        time.sleep(2)
    return read(path)

def audit_one(canonical,entry,ctx):
    task=entry['task'];global_bank=canonical['bank'];step=canonical['step']
    episode=Path(canonical['episode_path'])
    panel_path=Path(canonical.get('source_panel_path',episode.parents[4]/'panels'/episode.name.replace('-i'+f'{step:03d}'+'-round00-episodes.json','.json')))
    if 'source_panel_sha256' in canonical:assert sha(panel_path)==canonical['source_panel_sha256']
    panel=read(panel_path);bank=panel['bank'];output=panel_path.parent.parent
    copy={k:v for k,v in canonical.items() if not k.startswith('source_')}
    assert copy==dict(panel,bank=global_bank), 'Canonical retention differs from the raw panel'
    assert step in [0,130] and 0<=global_bank<64 and (step,global_bank) not in ctx['seen']
    ctx['seen'].add((step,global_bank))
    assert panel['seeds']==entry['seeds'][global_bank*8:(global_bank+1)*8]
    assert panel['rng_seed']==entry['rng_seeds'][global_bank]
    phase=f'test-c{step:03d}-b{bank:03d}'
    assert panel_path.name==phase+'.json'
    if global_bank in ctx['pair_output']:assert ctx['pair_output'][global_bank]==str(output), 'Pair stitched across attempts'
    ctx['pair_output'][global_bank]=str(output)
    parity=read(output/'initial-parity.json')
    assert len(parity)==1 and parity[0]['passed'] and parity[0]['max_abs_chunk_log_ratio']==0
    journal=output.parent/'accounting/rank-00'/(output.name+'.json')
    if str(journal) not in ctx['journals']:ctx['journals'][str(journal)]=read(journal)
    calls=ctx['journals'][str(journal)]['calls']
    outcomes=ctx['outcomes'];pair_rng=ctx['pair_rng'];phase_starts=ctx['phase_starts'];bank_receipts=ctx['receipts']
    episode_path = Path(panel['episode_path'])
    assert episode_path.is_relative_to(output) and sha(episode_path) == panel['episode_sha256']
    episode = read(episode_path)
    assert episode['completed'] and episode['rows'] == panel['episodes']
    assert [r['seed'] for r in episode['rows']] == panel['seeds']
    assert len(episode['rows']) == 8
    assert sum(r['success_once'] for r in episode['rows']) == panel['first_success']
    assert sum(r['success_at_end'] for r in episode['rows']) == panel['success_at_end']
    for row in episode['rows']:
        assert row['episode_id'] == f"{task}/{row['seed']}"
        counts = row['native_counts']
        assert counts['controls'] - counts['reset_controls'] == 600
        if row['success_once']:
            assert 0 <= row['first_success_control'] <= 600
        assert row['seed'] not in outcomes[step]
        outcomes[step][row['seed']] = bool(row['success_once'])
        ctx['end_outcomes'][step][row['seed']] = bool(row['success_at_end'])

    initial_path = episode_path.with_name(episode_path.name.replace('-episodes.json', '-initial.json'))
    assert sha(initial_path) == panel['initial_report_sha256']
    initial = read(initial_path)
    current_path = Path(initial['initial']['path'])
    canonical_path = Path(initial['canonical'])
    assert current_path.is_relative_to(output) and canonical_path.is_relative_to(output)
    assert sha(current_path) == initial['initial']['sha256']
    canonical = read(canonical_path)
    reference_path = canonical_path.parent / 'initial.pt'
    assert sha(reference_path) == canonical['snapshot_sha256']
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    current = torch.load(current_path, map_location='cpu', weights_only=False)
    for key in ['format', 'case', 'physical', 'task_descriptions']:
        assert current[key] == reference[key], (task, bank, step, key)
    assert reference['case'] == canonical['case']
    a, b = np.asarray(reference['states']), np.asarray(current['states'])
    assert a.shape == b.shape and a.dtype == b.dtype and np.isfinite(a).all()
    assert np.array_equal(a, b)
    report = initial['report']
    assert report['matched'] and report['physical_state_exact'] and report['public_state_exact']
    assert not report['observations_modified']
    images_exact = True
    for key in ['main_images', 'wrist_images', 'extra_view_images']:
        a, b = np.asarray(reference['images'][key]), np.asarray(current['images'][key])
        assert a.shape == b.shape and a.dtype == b.dtype == np.uint8 and a.size
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        actual = report['images'][key]
        assert int(delta.max()) == actual['max_absolute_delta']
        assert int(np.count_nonzero(delta)) == actual['differing_components']
        assert int(delta.size) == actual['total_components']
        assert math.isclose(float(delta.mean()), actual['mean_absolute_delta'], abs_tol=1e-12)
        images_exact = images_exact and not np.any(delta)
    assert bool(images_exact) == panel['images_exact'] == report['images_exact']
    ctx['image_variation'] += not images_exact
    del current, reference

    sampler_paths = sorted((output / 'sampler').glob(phase + '-*.json'))
    assert len(sampler_paths) == 12
    assert [p.name for p in sampler_paths] == [phase + f'-{i:02d}.json' for i in range(1, 13)]
    assert {p.name: sha(p) for p in sampler_paths} == panel['sampler_sha256']
    sampler = [read(p) for p in sampler_paths]
    assert all(r['actions_shape'] == [8, 50, 12] and r['batch_size'] == 8 and r['chunk'] == 50
               for r in sampler)
    rng = [dict(before=r['before_rng_sha256'], after=r['after_rng_sha256']) for r in sampler]
    saved_rng = read(output / 'banks' / f'bank-{bank:03d}' / 'SAMPLER-RNG-REFERENCE.json')
    assert saved_rng['rng_seed'] == panel['rng_seed'] and saved_rng['rng_hashes'] == rng
    if global_bank in pair_rng:
        assert pair_rng[global_bank] == rng
    pair_rng[global_bank] = rng
    phase_calls = [r for r in calls if r['phase'] == phase]
    assert len(phase_calls) == 12
    assert all(r['resolved'] and r['requested'] == r['executed_or_conservative'] == 400
               for r in phase_calls)
    phase_starts[(step, global_bank)] = min(r['started_unix'] for r in phase_calls)
    bank_receipts.append(dict(task=task, step=step, bank=global_bank, local_bank=bank, source_output=str(output), panel_sha256=sha(panel_path),
        initial_snapshot_sha256=sha(current_path), canonical_snapshot_sha256=sha(reference_path),
        episode_sha256=sha(episode_path), sampler_receipts=12, controls=4800,
        physical_and_public_state_exact=True, images_exact=bool(images_exact)))

def independent_paired(initial,policy,seed,observed):
    assert set(initial)==set(policy) and len(initial)==512
    differences=np.array([int(policy[s])-int(initial[s]) for s in sorted(initial)])
    wins,losses=int(np.sum(differences==1)),int(np.sum(differences==-1))
    p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
    counts=np.array([np.sum(differences==d) for d in [-1,0,1]])
    boot=np.random.default_rng(seed).multinomial(512,counts/512,size=10000)
    interval=np.quantile((boot[:,2]-boot[:,0])/512,[.025,.975])*100
    assert observed['cases']==512 and observed['wins']==wins and observed['losses']==losses
    assert observed['initial_successes']==sum(initial.values()) and observed['policy_successes']==sum(policy.values())
    assert math.isclose(p,observed['exact_McNemar_p'],rel_tol=1e-10,abs_tol=1e-15)
    assert math.isclose(float(differences.mean()*100),observed['delta_percentage_points'],abs_tol=1e-12)
    np.testing.assert_allclose(interval,observed['paired_bootstrap95_percentage_points'],atol=1e-12,rtol=0)
    return dict(initial_successes=sum(initial.values()),policy_successes=sum(policy.values()),wins=wins,losses=losses,
        delta_percentage_points=float(differences.mean()*100),exact_McNemar_p=p,paired_bootstrap95_percentage_points=interval.tolist())

def main():
    parser=argparse.ArgumentParser()
    for key in ['manifest','source-wrapper','source-script','combined','release','destination']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    assert not args.destination.exists()
    m=read(args.manifest);prepared=Path(m['prepared']);original=Path(m['original']);entries=read(prepared)['tasks']
    assert sha(prepared)==m['prepared_sha256'] and sha(args.source_script)==m['endpoint_sha256']
    assert read(prepared)['maximum_policy_controls']==1228800 and len(entries)==2
    for folder,pin in [('flow10m-0988b5f9','0988b5f9e9bf31ac3db673cc2860959611b9ae12'),('robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf','8a1233c3fae6f5905adfda7684c455842b0ecfb8')]:
        path=original.parents[1]/'sources'/folder
        assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==pin
    contexts={}
    for entry in entries:
        task=entry['task']
        assert sha(entry['source_config'])==entry['source_config_sha256']
        assert sha(entry['initial_parity_fixture'])==entry['initial_parity_fixture_sha256']
        for cp in entry['checkpoints']:
            assert sha(cp['marker'])==cp['marker_sha256']
            ready=read(original/task/'staging-r1'/str(cp['step'])/'READY.json')
            assert ready['completed'] and ready['task']==task and ready['checkpoint']==cp
            assert ready['prepared_sha256']==sha(prepared) and ready['staging_script_sha256']==sha(args.source_script)
        contexts[task]=dict(seen=set(),outcomes={0:{},130:{}},end_outcomes={0:{},130:{}},pair_output={},
            pair_rng={},phase_starts={},image_variation=0,receipts=[],journals={})
        for panel in m['tasks'][task]['retained_panels']:
            audit_one(panel,entry,contexts[task])
        write(args.destination.with_name('AUDIT-PROGRESS.json'),dict(checked_unix=time.time(),retained_task=task,
            audited_policy_episodes=sum(len(c['seen'])*8 for c in contexts.values()),waiting_for_active_shards=True))
    for shard in m['shards']:
        root=Path(shard['output']);result=wait_file(root/'RESULT.json',args.deadline)
        assert result['completed'] and result['SGD']==result['unresolved_intents']==0
        paths=sorted((root/'panels').glob('*.json'))
        assert len(paths)==2*len(shard['banks']) and result['raw_receipt_sha256']=={p.name:sha(p) for p in paths}
        entry=next(e for e in entries if e['task']==shard['task'])
        for path in paths:
            panel=read(path);local=panel['bank']
            panel.update(source_panel_path=str(path),source_panel_sha256=sha(path),source_local_bank=local)
            panel['bank']=shard['banks'][local];audit_one(panel,entry,contexts[entry['task']])
    spec=importlib.util.spec_from_file_location('frozen_endpoint_completion',args.source_wrapper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not args.combined.exists():module.combine(args.manifest,args.combined)
    combined=read(args.combined);assert combined['completed'] and combined['migration_sha256']==sha(args.manifest)
    assert combined['valid_policy_controls']==1228800 and combined['valid_policy_episodes']==2048 and combined['SGD']==0
    summaries={};receipts=[]
    for entry in entries:
        task=entry['task'];c=contexts[task]
        assert c['seen']=={(s,b) for s in [0,130] for b in range(64)}
        assert all(set(values)==set(entry['seeds']) for values in c['outcomes'].values())
        for b in range(64):
            observed=sorted([0,130],key=lambda s:c['phase_starts'][(s,b)])
            assert observed==entry['bank_order'][b]==([0,130] if b%2==0 else [130,0])
        summaries[task]=dict(first_success=independent_paired(c['outcomes'][0],c['outcomes'][130],entry['bootstrap_seed'],combined['tasks'][task]['first_success']),
            success_at_end=independent_paired(c['end_outcomes'][0],c['end_outcomes'][130],entry['bootstrap_seed'],combined['tasks'][task]['success_at_end']),
            image_variation_panels=c['image_variation'],ABBA_passed=True,case_policy_episodes=1024)
        receipts.extend(c['receipts'])
    passing=True
    for i,task in enumerate(sorted(summaries,key=lambda t:summaries[t]['first_success']['exact_McNemar_p'])):
        p=summaries[task]['first_success'];threshold=.05/(2-i);passing=passing and p['exact_McNemar_p']<threshold
        assert combined['holm'][task]==dict(threshold=threshold,rejected_two_sided=passing,improvement=passing and p['wins']>p['losses'])
    journal_paths=set()
    for entry in entries:journal_paths.update((original/entry['task']/'accounting/rank-00').glob('*.json'))
    cursor=args.manifest
    while cursor:
        ancestor=read(cursor)
        for shard in ancestor['shards']:
            root=Path(shard['output']);path=root.parent/'accounting/rank-00'/(root.name+'.json')
            if path.exists():journal_paths.add(path)
        previous=ancestor.get('previous_manifest')
        if previous:assert sha(previous)==ancestor['previous_manifest_sha256']
        cursor=Path(previous) if previous else None
    ledgers={str(p):read(p) for p in journal_paths}
    charged=sum(row.get('executed_or_conservative',row['requested']) for j in ledgers.values() for row in j['calls'])
    unresolved=sum(not row['resolved'] for j in ledgers.values() for row in j['calls'])
    assert 1228800<=charged<=m['maximum_all_attempt_policy_controls']
    cost=wait_file(args.release,args.deadline)
    assert cost['completed'] and cost['all_attempt_gpu_seconds_upper']<=27600
    assert cost['prior_gpu_seconds_upper']==m['stopped']['all_attempt_gpu_seconds_upper']
    def recompute_charge(phase):
        elapsed=(datetime.datetime.fromisoformat(phase['released_shanghai'])-datetime.datetime.fromisoformat(phase['created_shanghai'])).total_seconds()
        result=elapsed*phase['GPUs']
        recorded=phase.get('gpu_seconds_upper',phase.get('creation_to_release_gpu_seconds_upper'))
        assert result==recorded
        return result
    prior=m['stopped']
    recalculated=sum(recompute_charge(p) for p in prior['jobs'])+prior['duplicate_failure_gpu_seconds']
    recalculated+=sum(recompute_charge(p) for p in prior.get('failed_acceleration_attempts',[]))
    recalculated+=sum(recompute_charge(p) for p in prior.get('released_phases',[]))
    assert recalculated==cost['prior_gpu_seconds_upper']
    new=cost['last_phase']
    delta=(datetime.datetime.fromisoformat(new['released_shanghai'])-datetime.datetime.fromisoformat(new['created_shanghai'])).total_seconds()
    assert delta*new['GPUs']==new['gpu_seconds_upper']
    assert new['GPUs']==m['maximum_gpus']
    assert cost['all_attempt_gpu_seconds_upper']==cost['prior_gpu_seconds_upper']+new['gpu_seconds_upper']
    value=dict(completed=True,whole_run_completed=True,checked_unix=time.time(),audit_source_sha256=sha(__file__),
        original_audit_reference='6c95116e',manifest_sha256=sha(args.manifest),prepared_sha256=sha(prepared),
        source_script_sha256=sha(args.source_script),source_wrapper_sha256=sha(args.source_wrapper),combined_sha256=sha(args.combined),
        tasks=summaries,holm=combined['holm'],policy_episodes=2048,valid_policy_controls=1228800,
        audited_policy_banks=len(receipts),sampler_receipts=12*len(receipts),all_attempt_controls_upper=charged,
        unresolved_abandoned_intents_charged=unresolved,all_attempt_journals={p:sha(p) for p in ledgers},
        cost_sha256=sha(args.release),cost=cost,banks=receipts,CPU_only=True,new_controls=0,new_GPU_seconds=0,SGD=0)
    assert len(receipts)==256
    with args.destination.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['banks','all_attempt_journals']}),flush=True)

if __name__=='__main__':main()
